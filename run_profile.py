# main.py
import os
import torch
import torch.multiprocessing as mp
import math # Import math for ceiling function if needed

from neural_network import ChessNet
from self_play_profile import run_parallel_self_play



# --- Configuration ---

GAMES_PER_ITERATION = 6   # Number of games generated in each self-play phase (adjust based on time/memory)
NUM_WORKERS = 6             # Number of parallel workers for self-play (adjust based on CPU cores/GPU)

MCTS_SIMULATIONS = 100 

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


    # --- Step 1: Self-Play Data Generation ---
    print(f"Starting self-play phase ({GAMES_PER_ITERATION} games)...")


    samples, game_results, game_lengths = run_parallel_self_play(
            num_games=GAMES_PER_ITERATION,
            model_path=current_model_path,
            num_workers=NUM_WORKERS,
            mcts_simulations=MCTS_SIMULATIONS
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


        