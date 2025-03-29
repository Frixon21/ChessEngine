# main.py
import os
import torch
import torch.multiprocessing as mp
import math # Import math for ceiling function if needed

from neural_network import ChessNet
from self_play import run_parallel_self_play
from train import train_network


# --- Configuration ---
NUM_ITERATIONS = 50         # Total training iterations (self-play + train)
GAMES_PER_ITERATION = 256   # Number of games generated in each self-play phase (adjust based on time/memory)
EPOCHS_PER_ITERATION = 2    # Number of training epochs on the data from one iteration
BATCH_SIZE = 256            # Training batch size (adjust based on GPU memory)
LEARNING_RATE = 0.001       # Training learning rate
NUM_WORKERS = 5             # Number of parallel workers for self-play (adjust based on CPU cores/GPU)

# --- Dynamic MCTS Simulation Settings ---
INITIAL_MCTS_SIMULATIONS = 400  # Starting number of simulations
MAX_MCTS_SIMULATIONS = 1000   # Target maximum simulations by the end
# MCTS_SIMULATIONS = 50 # <--- Remove or comment out the old static value

MODEL_CHECKPOINT = "trained_model.pth" # Path to save/load the model


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

    for iteration in range(14, NUM_ITERATIONS + 1):
        print(f"\n===== ITERATION {iteration}/{NUM_ITERATIONS} =====")

        # --- Calculate MCTS simulations for this iteration ---
        # Linear interpolation from INITIAL to MAX simulations over NUM_ITERATIONS
        if NUM_ITERATIONS <= 1:
            current_mcts_simulations = INITIAL_MCTS_SIMULATIONS
        else:
            # Calculate the progress fraction (from 0.0 at iteration 1 to 1.0 at NUM_ITERATIONS)
            progress = (iteration - 1) / max(1, NUM_ITERATIONS - 1)
            # Interpolate linearly
            sims = INITIAL_MCTS_SIMULATIONS + (MAX_MCTS_SIMULATIONS - INITIAL_MCTS_SIMULATIONS) * progress
            # Round to the nearest integer for the simulation count
            current_mcts_simulations = int(round(sims))

        # Ensure we don't accidentally go below initial or above max due to rounding/edge cases
        current_mcts_simulations = max(INITIAL_MCTS_SIMULATIONS, min(MAX_MCTS_SIMULATIONS, current_mcts_simulations))

        print(f"Using {current_mcts_simulations} MCTS simulations for self-play this iteration.")

        # --- Step 1: Self-Play Data Generation ---
        print(f"Starting self-play phase ({GAMES_PER_ITERATION} games)...")

        samples, game_results, game_lengths = run_parallel_self_play(
            num_games=GAMES_PER_ITERATION,
            model_path=current_model_path,
            num_workers=NUM_WORKERS,
            mcts_simulations=current_mcts_simulations
        )
        
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
            print(f"  Results: White Wins={num_white_wins}, Black Wins={num_black_wins}, Draws={num_draws}, Other={num_other_results}")
            print(f"  Draw Rate: {draw_rate:.2f}%")
            print(f"  Average Game Length: {avg_game_length:.2f} moves (plies)")
            print(f"  Generated {len(samples)} training samples.")
        else:
            print("Self-play finished. WARNING: No games were completed successfully.")
            # Decide how to handle this - stop? continue?
            # break # Example: Stop if no games completed


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