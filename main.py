# main.py
import os
import torch
import torch.multiprocessing as mp
import time
from neural_network import ChessNet
#from self_play import run_parallel_self_play
from self_play_batch import run_parallel_self_play_batch
from train import train_network
from stockfish_processor import generate_stockfish_targets
from pgn_parser import parse_pgn_and_extract_positions
from board_encoder import board_to_tensor_torch 
import csv
import chess


import cProfile
import pstats
import io


# --- Configuration ---
NUM_ITERATIONS = 100         # Total training iterations (self-play + train)

TARGET_GAMES_PER_ITERATION = 250 # Target number of games per iteration
GAMES_RAMP_UP_ITERATIONS = 10 # Reach target games by iteration 10
INITIAL_GAMES_PER_ITERATION = 128 # Start with fewer games

EPOCHS_PER_ITERATION = 4    # Number of training epochs on the data from one iteration
BATCH_SIZE = 256            # Training batch size (adjust based on GPU memory)
LEARNING_RATE = 0.0005       # Training learning rate
NUM_WORKERS = 5            # Number of parallel workers for self-play (adjust based on CPU cores/GPU)
INFERENCE_BATCH_SIZE = 32   # Batch size for inference during self-play (adjust based on GPU memory)

# --- Dynamic MCTS Simulation Settings ---
INITIAL_MCTS_SIMULATIONS = 64  # Starting number of simulations
MAX_MCTS_SIMULATIONS = 256   # Target maximum simulations by the end

MODEL_CHECKPOINT = "trained_model.pth" # Path to save/load the model
SCRIPTED_MODEL_CHECKPOINT = "scripted_model.pt" # Path to save/load the scripted model
PGNS_TO_SAVE_PER_ITERATION = 10 # Save the first 10 games each iteration

STOCKFISH_ENGINE_PATH = "stockfish\stockfish-windows-x86-64-avx2.exe"

USE_PGNS = False # Set to True if you want to use PGNs for training
MAX_GAMES_TO_PROCESS = 1000 # Set to None to process all

PROFILE_SELF_PLAY = False # Set to True to profile one self-play game
PROFILE_OUTPUT_FILE = "self_play_profile.prof" # Output file for stats

USE_PUZZLES = True # Set to True if you want to use puzzles for training
PUZZLE_CSV_PATH = "Games\puzzle_chunks"


def create_initial_models(device):
    """Creates and saves initial .pth and .pt models if they don't exist."""
    pth_exists = os.path.exists(MODEL_CHECKPOINT)
    pt_exists = os.path.exists(SCRIPTED_MODEL_CHECKPOINT)

    if not pth_exists:
        print(f"No existing model found at {MODEL_CHECKPOINT}. Initializing random model.")
        initial_model = ChessNet().to(device)
        torch.save(initial_model.state_dict(), MODEL_CHECKPOINT)
        print(f"Initial random model saved to {MODEL_CHECKPOINT}")
        pth_exists = True # Mark as created
    else:
         initial_model = ChessNet().to(device) # Need instance even if loading below
         print(f"Found existing model at {MODEL_CHECKPOINT}.")

    if pth_exists and not pt_exists:
        print(f"Scripted model {SCRIPTED_MODEL_CHECKPOINT} not found. Creating from {MODEL_CHECKPOINT}...")
        try:
            # Load weights into a model instance
            initial_model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
            initial_model.eval()
            # Create example input
            example_input = torch.randn(1, 22, 8, 8, device=device)
            # Trace
            scripted_model = torch.jit.trace(initial_model, example_input)
            # Save
            torch.jit.save(scripted_model, SCRIPTED_MODEL_CHECKPOINT)
            print(f"Initial TorchScript model saved to {SCRIPTED_MODEL_CHECKPOINT}")
        except Exception as e:
            print(f"Failed to create initial TorchScript model: {e}")
            # Decide how to handle - maybe exit? For now, just warn.
    elif pt_exists:
         print(f"Found existing scripted model at {SCRIPTED_MODEL_CHECKPOINT}.")

    del initial_model # Free memory

def load_puzzle_samples(iteration, num_puzzles=1000, csv_path: str = PUZZLE_CSV_PATH, device: torch.device = torch.device("cpu")):
    """
    Load puzzle positions for the current iteration from a Lichess puzzle CSV,
    parsing the *entire puzzle line* of moves.

    Steps:
      1) We read 'num_puzzles' lines from the CSV, offset by iteration.
      2) For each row, parse the FEN (the position before the opponent's move),
         and the full 'Moves' string (all forced moves in UCI).
      3) Apply the first move to the board (the opponent's move),
         leaving the puzzle solver to move. This is puzzle start #1.
      4) Then for each subsequent move in the puzzle line, push it and record
         the resulting position. That means each step in the puzzle solution
         yields a new position (for both sides).
      5) Return a flat list of (board, board_tensor) for every step in every puzzle.

    Example usage:
        puzzle_positions = load_puzzle_positions_all_moves(
            iteration=5,
            num_puzzles=1000,
            csv_path="lichess_db_puzzle.csv",
            device=torch.device("cuda")
        )
        # Then pass puzzle_positions to generate_stockfish_targets(...)
    """
    puzzle_positions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        
        # Attempt to skip a header if it exists:
        header = next(reader, None)
        if header and "PuzzleId" in header[0]:
            pass  # We skipped the header
        else:
            # Possibly no header or unexpected format:
            # If you realize there's no header, uncomment the next line:
            # f.seek(0)
            # Or handle differently if needed
            pass


        for row in reader:
            
            if len(row) < 3:
                continue # Malformed row
            
            # CSV format:
            # PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
            fen = row[1].strip()
            moves_str = row[2].strip()
            if not fen or not moves_str:
                continue
            
            moves_list = moves_str.split()
            if len(moves_list) < 1:
                continue
            
            board = chess.Board(fen)
            
            # 1) The puzzle FEN is the position *before* the opponent's move.
            #    The first move in 'moves_list' is that opponent move.
            #    We push it so the solver is to move after that.
            first_move_uci = moves_list[0]
            try:
                board.push_uci(first_move_uci)
            except Exception as e:
                print(f"[load_puzzle_positions_all_moves] Invalid move {first_move_uci} for FEN {fen}: {e}")
                continue
            
            # Now the board is at the puzzle's "start" for the solver.
            puzzle_positions.append((board.copy(), board_to_tensor_torch(board, device)))
            
            for next_move_uci in moves_list[1:]:
                try:
                    board.push_uci(next_move_uci)
                except Exception as e:
                    print(f"[load_puzzle_positions_all_moves] Invalid forced move {next_move_uci} after first move: {e}")
                    break  # move on to next puzzle

                puzzle_positions.append((board.copy(), board_to_tensor_torch(board, device)))
                
    return puzzle_positions


    pass

if __name__ == "__main__":

    # --- Set up multiprocessing context ---
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing context already set.")

    # --- Initialize Model (if it doesn't exist) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

   # --- Initialize Models ---
    create_initial_models(device) # Create both .pth and .pt if needed

    # --- Define which model to use for each phase ---
    # Training always resumes from the standard .pth state_dict
    current_model_for_training = MODEL_CHECKPOINT
    # Self-play uses the optimized .pt TorchScript model
    current_model_for_self_play = SCRIPTED_MODEL_CHECKPOINT
    
    # --- Profiling Block ---
    if PROFILE_SELF_PLAY:
        print("\n" + "="*20 + " PROFILING MODE " + "="*20)
        print(f"Profiling one self-play game. Output will be saved to {PROFILE_OUTPUT_FILE}")
        print("Ensure NUM_WORKERS=0 for meaningful single-process profiling.")

        # Use settings for a single game, run sequentially
        profiler_num_games = 1
        profiler_num_workers = 0 # MUST be 0 for cProfile
        profiler_mcts_sims = MAX_MCTS_SIMULATIONS # Use configured sims

        # Create profiler object
        pr = cProfile.Profile()
        pr.enable() # Start profiling

        # Run the self-play function (which calls the worker internally)
        raw_positions, game_results, game_lengths = run_parallel_self_play_batch(
            num_games=profiler_num_games,
            model_path=current_model_for_self_play,
            num_workers=profiler_num_workers, # Force 0 workers
            mcts_simulations=profiler_mcts_sims,
            inference_batch_size=INFERENCE_BATCH_SIZE,
            iteration_num=0, # Iteration number doesn't matter much here
            num_pgns_to_save=0 # Don't save PGNs during profiling
        )

        pr.disable() # Stop profiling
        print("Profiling finished.")

        # Save stats to file
        pr.dump_stats(PROFILE_OUTPUT_FILE)
        print(f"Profiler stats saved to {PROFILE_OUTPUT_FILE}")

        # Print stats directly to console (can be very long)
        print("\n--- Profiler Stats Summary (Top 20 by cumulative time) ---")
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20) # Print top 20 functions by cumulative time
        print(s.getvalue())
        
        print(f"  Average Game Length: { sum(game_lengths):.2f} moves (plies)")
        print("--- End Profiler Stats Summary ---")

        print("Exiting after profiling.")
        exit() # Stop execution after profiling

    for iteration in range(65, NUM_ITERATIONS + 1):
        print(f"\n===== ITERATION {iteration}/{NUM_ITERATIONS} =====")

        if not USE_PGNS:
            # --- Calculate MCTS simulations for this iteration ---
            # Linear interpolation from INITIAL to MAX simulations over NUM_ITERATIONS
            if NUM_ITERATIONS <= 1:
                current_mcts_simulations = INITIAL_MCTS_SIMULATIONS
            else:
                # Interpolate linearly
                sims = INITIAL_MCTS_SIMULATIONS + iteration * INFERENCE_BATCH_SIZE
                current_mcts_simulations = int(round(sims))

            # Ensure we don't accidentally go below initial or above max due to rounding/edge cases
            current_mcts_simulations = max(INITIAL_MCTS_SIMULATIONS, min(MAX_MCTS_SIMULATIONS, current_mcts_simulations))

            print(f"Using {current_mcts_simulations} MCTS simulations for self-play this iteration.")
            
            if iteration >= GAMES_RAMP_UP_ITERATIONS:
                current_games_this_iteration = TARGET_GAMES_PER_ITERATION            
            else:
                # Linear ramp-up from INITIAL to TARGET games
                progress = max(0, iteration - 1) / max(1, GAMES_RAMP_UP_ITERATIONS - 1)
                games = INITIAL_GAMES_PER_ITERATION + (TARGET_GAMES_PER_ITERATION - INITIAL_GAMES_PER_ITERATION) * progress
                current_games_this_iteration = int(round(games))
                # Ensure it's at least the initial value
                current_games_this_iteration = max(INITIAL_GAMES_PER_ITERATION, current_games_this_iteration)

            # --- Step 1: Self-Play Data Generation ---
            print(f"Starting self-play phase ({current_games_this_iteration} games)...")
            start_time = time.time()

            raw_positions, game_results, game_lengths = run_parallel_self_play_batch(
                num_games=current_games_this_iteration,
                model_path=current_model_for_self_play,
                num_workers=NUM_WORKERS,
                mcts_simulations=current_mcts_simulations,
                inference_batch_size=INFERENCE_BATCH_SIZE,
                iteration_num=iteration,                 # Pass current iteration
                num_pgns_to_save=PGNS_TO_SAVE_PER_ITERATION # Pass save count
            )
            end_time = time.time()
            self_play_duration = end_time - start_time
            print(f"Self-play phase took {self_play_duration:.2f} seconds.")
            
            # --- Calculate and Print Statistics ---
            num_games_completed = len(game_results)
            if num_games_completed > 0:
                num_draws = game_results.count("1/2-1/2")
                num_white_wins = game_results.count("1-0")
                num_black_wins = game_results.count("0-1")
                num_other_results = num_games_completed - (num_draws + num_white_wins + num_black_wins) # Should be 0 usually

                draw_rate = (num_draws / num_games_completed) * 100
                avg_game_length = sum(game_lengths) / num_games_completed if game_lengths else 0

                print(f"Self-play finished. Completed {num_games_completed}/{current_games_this_iteration} games.")
                print(f"  Results: White Wins={num_white_wins}, Black Wins={num_black_wins}, Drawclears={num_draws}, Other={num_other_results}")
                print(f"  Draw Rate: {draw_rate:.2f}%")
                print(f"  Average Game Length: {avg_game_length:.2f} moves (plies)")
                print(f"  Generated {len(raw_positions)} training samples.")
            else:
                print("Self-play finished. WARNING: No games were completed successfully.")
                # Decide how to handle this - stop? continue?
                # break # Example: Stop if no games completed
        else: #USING PGNS
            PGN_FILE = f"Games/1-Split1000/{iteration}.pgn"
            start_time = time.time()
            raw_positions = parse_pgn_and_extract_positions(
                PGN_FILE,
                max_games=MAX_GAMES_TO_PROCESS
            )
            end_time = time.time()
            if raw_positions is not None:
                print(f"Data extraction took {end_time - start_time:.2f} seconds.")
            else:
                print("Data extraction failed.")
                
        if USE_PUZZLES:
            print("Loading puzzle data for this iteration...")
            path= PUZZLE_CSV_PATH+f"\{iteration}.csv"
            raw_puzzle= load_puzzle_samples(iteration, num_puzzles=1000, csv_path=path ,device=device)
            puzzle_samples = generate_stockfish_targets(raw_puzzle, STOCKFISH_ENGINE_PATH, workers=NUM_WORKERS, multipv=1, depth=12)

         
        # Generate the Stockfish-guided training samples:
        games_samples = generate_stockfish_targets(raw_positions, STOCKFISH_ENGINE_PATH, workers=NUM_WORKERS,multipv=15, depth=10)
        if not games_samples:
            print("!!! ERROR: No samples generated in self-play phase. Stopping.")
            break
        
        if USE_PUZZLES:
            # Combine the puzzle samples with the self-play samples
            samples = games_samples + puzzle_samples
        else:
            samples = games_samples
        

        # --- Step 2: Training ---
        print(f"Starting training phase ({EPOCHS_PER_ITERATION} epochs)...")
        updated_model_path = train_network(
            data_source=samples, # Pass the generated list of samples
            num_epochs=EPOCHS_PER_ITERATION,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            resume_from_checkpoint=current_model_for_training # Load the model we just used for self-play
        )
        # Update the path for the next iteration (train_network saves to a fixed path now)
        current_model_for_training = updated_model_path
        print(f"Training finished. Updated model saved to {current_model_for_training}")
        
        # --- <<< NEW: Create/Update TorchScript Model AFTER Training >>> ---
        print(f"Creating/Updating TorchScript model from {current_model_for_training}...")
        try:
            # Load the newly trained weights
            model = ChessNet().to(device)
            model.load_state_dict(torch.load(current_model_for_training, map_location=device))
            model.eval()
            # Create example input
            example_input = torch.randn(1, 22, 8, 8, device=device)
            # Trace
            scripted_model = torch.jit.trace(model, example_input)
            # Save (overwriting previous)
            torch.jit.save(scripted_model, SCRIPTED_MODEL_CHECKPOINT)
            print(f"TorchScript model saved to {SCRIPTED_MODEL_CHECKPOINT}")
            # Update the path for the next self-play iteration
            current_model_for_self_play = SCRIPTED_MODEL_CHECKPOINT
            del model # Free memory
            del scripted_model
        except Exception as e:
            print(f"Failed to trace or save TorchScript model after iteration {iteration}: {e}")
            print(f"Self-play for next iteration will use the standard model: {current_model_for_training}")
            # Fallback: use the .pth file for self-play if scripting fails
            current_model_for_self_play = current_model_for_training
        # --- <<< END NEW BLOCK >>> ---

    print("\n===== Training Loop Finished =====")