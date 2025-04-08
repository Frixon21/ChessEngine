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
from utils import load_config, update_config, pull_repo, push_repo, create_initial_models, process_downloaded_files, load_puzzle_samples

import cProfile
import pstats
import io

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
    
    
    # --- Load Basic Configuration ---
    master_config = load_config()
    MODEL_CHECKPOINT = master_config.get("MODEL_CHECKPOINT", "trained_model.pth")
    SCRIPTED_MODEL_CHECKPOINT = master_config.get("SCRIPTED_MODEL_CHECKPOINT", "scripted_model.pt")
    PROFILE_SELF_PLAY = master_config.get("PROFILE_SELF_PLAY", False)
    PROFILE_OUTPUT_FILE = master_config.get("PROFILE_OUTPUT_FILE", "self_play_profile.prof")
    MCTS_SIMULATIONS = master_config.get("MCTS_SIMULATIONS", 64)
    INFERENCE_BATCH_SIZE = master_config.get("INFERENCE_BATCH_SIZE", 32)
    NUM_ITERATIONS = master_config.get("NUM_ITERATIONS", 100)
    CURRENT_ITERATION = master_config.get("CURRENT_ITERATION", 0)
    

   # --- Initialize Models ---
    create_initial_models(device, MODEL_CHECKPOINT, SCRIPTED_MODEL_CHECKPOINT) # Create both .pth and .pt if needed

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
        profiler_mcts_sims = MCTS_SIMULATIONS # Use configured sims

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

    for iteration in range(CURRENT_ITERATION, NUM_ITERATIONS + 1):
        print(f"\n===== ITERATION {iteration}/{NUM_ITERATIONS} =====")
        
        # --- ReLoad Configuration ---
        master_config = load_config()
        NUM_ITERATIONS = master_config.get("NUM_ITERATIONS", 100)
        CURRENT_ITERATION = master_config.get("CURRENT_ITERATION", 82)
        GAMES_PER_ITERATION = master_config.get("GAMES_PER_ITERATION", 250)
        EPOCHS_PER_ITERATION = master_config.get("EPOCHS_PER_ITERATION", 4)
        BATCH_SIZE = master_config.get("BATCH_SIZE", 256)
        LEARNING_RATE = master_config.get("LEARNING_RATE", 0.0005)
        NUM_WORKERS = master_config.get("NUM_WORKERS", 5)
        INFERENCE_BATCH_SIZE = master_config.get("INFERENCE_BATCH_SIZE", 32)
        MCTS_SIMULATIONS = master_config.get("MCTS_SIMULATIONS", 256)
        MODEL_CHECKPOINT = master_config.get("MODEL_CHECKPOINT", "trained_model.pth")
        SCRIPTED_MODEL_CHECKPOINT = master_config.get("SCRIPTED_MODEL_CHECKPOINT", "scripted_model.pt")
        PGNS_TO_SAVE_PER_ITERATION = master_config.get("PGNS_TO_SAVE_PER_ITERATION", 10)
        STOCKFISH_ENGINE_PATH = master_config.get("STOCKFISH_ENGINE_PATH", "stockfish/stockfish-windows-x86-64-avx2.exe")
        engine_config = master_config.get("STOCKFISH_ENGINE_CONFIG", {})
        PUZZLE_DEPTH = engine_config.get("PUZZLE_DEPTH", 12)
        GAMES_DEPTH = engine_config.get("GAMES_DEPTH", 10)
        GAMES_MULTIPV = engine_config.get("GAMES_MULTIPV", 15)
        USE_PGNS = master_config.get("USE_PGNS", True)
        MAX_GAMES_TO_PROCESS = master_config.get("MAX_GAMES_TO_PROCESS", 1000)
        PROFILE_SELF_PLAY = master_config.get("PROFILE_SELF_PLAY", False)
        PROFILE_OUTPUT_FILE = master_config.get("PROFILE_OUTPUT_FILE", "self_play_profile.prof")
        USE_PUZZLES = master_config.get("USE_PUZZLES", True)
        PUZZLE_CSV_PATH = master_config.get("PUZZLE_CSV_PATH", "Games/puzzle_chunks")
        S3_BUCKET_NAME = master_config.get("S3_BUCKET_NAME", "chessgamegenerationpgns")
        S3_PREFIX = master_config.get("S3_PREFIX", "saved_games/")

        if not USE_PGNS:
            print(f"Using {MCTS_SIMULATIONS} MCTS simulations for self-play this iteration.")
            
            # --- Step 1: Self-Play Data Generation ---
            print(f"Starting self-play phase ({GAMES_PER_ITERATION} games)...")
            start_time = time.time()

            raw_positions, game_results, game_lengths = run_parallel_self_play_batch(
                num_games=GAMES_PER_ITERATION,
                model_path=current_model_for_self_play,
                num_workers=NUM_WORKERS,
                mcts_simulations=MCTS_SIMULATIONS,
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

                print(f"Self-play finished. Completed {num_games_completed}/{GAMES_PER_ITERATION} games.")
                print(f"  Results: White Wins={num_white_wins}, Black Wins={num_black_wins}, Drawclears={num_draws}, Other={num_other_results}")
                print(f"  Draw Rate: {draw_rate:.2f}%")
                print(f"  Average Game Length: {avg_game_length:.2f} moves (plies)")
                print(f"  Generated {len(raw_positions)} training samples.")
            else:
                print("Self-play finished. WARNING: No games were completed successfully.")
                # Decide how to handle this - stop? continue?
                # break # Example: Stop if no games completed
        else: #USING PGNS
            raw_positions = process_downloaded_files(S3_BUCKET_NAME, S3_PREFIX)
                
        if USE_PUZZLES:
            print("Loading puzzle data for this iteration...")
            path = os.path.join(PUZZLE_CSV_PATH, f"{iteration}.csv")
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
        pull_repo()
        push_repo(iteration, MODEL_CHECKPOINT, SCRIPTED_MODEL_CHECKPOINT)
        update_config({"CURRENT_ITERATION": iteration+1})
        print(f"Done, Waiting 10s")
        time.sleep(10)

    print("\n===== Training Loop Finished =====")
