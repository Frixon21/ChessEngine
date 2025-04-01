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

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# --- Configuration ---
NUM_ITERATIONS = 50         # Total training iterations (self-play + train)

TARGET_GAMES_PER_ITERATION = 256 # Target number of games per iteration
GAMES_RAMP_UP_ITERATIONS = 10 # Reach target games by iteration 10
INITIAL_GAMES_PER_ITERATION = 128 # Start with fewer games

EPOCHS_PER_ITERATION = 2    # Number of training epochs on the data from one iteration
BATCH_SIZE = 256            # Training batch size (adjust based on GPU memory)
LEARNING_RATE = 0.001       # Training learning rate
NUM_WORKERS = 5            # Number of parallel workers for self-play (adjust based on CPU cores/GPU)
INFERENCE_BATCH_SIZE = 32   # Batch size for inference during self-play (adjust based on GPU memory)

# --- Dynamic MCTS Simulation Settings ---
INITIAL_MCTS_SIMULATIONS = 64  # Starting number of simulations
MAX_MCTS_SIMULATIONS = 256   # Target maximum simulations by the end

MODEL_CHECKPOINT = "trained_model.pth" # Path to save/load the model
PGNS_TO_SAVE_PER_ITERATION = 10 # Save the first 10 games each iteration

STOCKFISH_ENGINE_PATH = "stockfish\stockfish-windows-x86-64-avx2.exe"

USE_PGNS = False # Set to True if you want to use PGNs for training
MAX_GAMES_TO_PROCESS = 1000 # Set to None to process all

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

    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"No existing model found at {MODEL_CHECKPOINT}. Initializing random model.")
        initial_model = ChessNet().to(device)
        torch.save(initial_model.state_dict(), MODEL_CHECKPOINT)
        print(f"Initial random model saved to {MODEL_CHECKPOINT}")
        del initial_model # Free memory
    else:
        print(f"Found existing model at {MODEL_CHECKPOINT}. Resuming training.")

    # --- Main Training Loop ---
    current_model_path = MODEL_CHECKPOINT

    
    for iteration in range(22, NUM_ITERATIONS + 1):
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
                model_path=current_model_path,
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

            
        # Generate the Stockfish-guided training samples:
        samples = generate_stockfish_targets(raw_positions, STOCKFISH_ENGINE_PATH)
        if not samples:
            print("!!! ERROR: No samples generated in self-play phase. Stopping.")
            break

        # --- Step 2: Training ---
        print(f"Starting training phase ({EPOCHS_PER_ITERATION} epochs)...")
        updated_model_path = train_network(
            data_source=samples, # Pass the generated list of samples
            num_epochs=EPOCHS_PER_ITERATION,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            resume_from_checkpoint=current_model_path # Load the model we just used for self-play
        )
        # Update the path for the next iteration (train_network saves to a fixed path now)
        current_model_path = updated_model_path
        print(f"Training finished. Updated model saved to {current_model_path}")

    print("\n===== Training Loop Finished =====")