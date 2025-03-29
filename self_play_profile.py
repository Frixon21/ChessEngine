# self_play.py
import chess
import chess.pgn
import numpy as np
import torch
from mcts import mcts_push_pop_shared # Assuming this is your MCTS implementation
from neural_network import ChessNet
from board_encoder import board_to_tensor
from utils import move_to_index
from tqdm import tqdm
import torch.multiprocessing as mp
# from mcts_copy import mcts , MCTSNode # Keep your actual import
from datetime import datetime
import os
import cProfile # Import cProfile
import pstats   # Import pstats
import io       # Import io for redirecting output if needed later

def self_play_game(model_path, mcts_simulations=50, worker_id=0): # Added worker_id for unique filenames
    """
    Plays a single self-play game using the current model and returns training samples.
    Adds profiling for performance analysis.
    Each sample is a tuple (board_tensor, target_policy, outcome).
    """
    # --- Profiling Setup ---
    profiler = cProfile.Profile()
    print(f"[Worker {worker_id}/{os.getpid()}] Starting game, profiling enabled.") # Add process ID for clarity
    # --- End Profiling Setup ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the network (each process has its own copy)
    network = ChessNet()
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.to(device)
    network.eval()

    board = chess.Board()
    game_data = []  # Store (board_tensor, visit_distribution) for each move.

    # --- Start Profiling Core Logic ---
    profiler.enable()

    try: # Use try...finally to ensure profiler is disabled and saved
        # Play until game over.
        move_count = 0
        while not board.is_game_over():
            move_count += 1
            # --- Dynamic Temperature ---
            TEMPERATURE = 1.0
            TEMP_THRESHOLD_MOVES = 15 # Play proportionally for first 15 full moves (30 half-moves)

            # --- MCTS Search ---
            # Use MCTS to get best move and visit distribution.
            # This is likely the most time-consuming part per move.
            _best_move_greedy, visit_distribution = mcts_push_pop_shared(
                board,
                network,
                num_simulations=mcts_simulations,
                return_visit_distribution=True,
                device=device
            )

            # --- Move Selection ---
            if not visit_distribution:
                print(f"Warning [Worker {worker_id}/{os.getpid()}]: MCTS returned no distribution. FEN: {board.fen()}")
                legal_moves = list(board.legal_moves)
                if not legal_moves: break # Game ended unexpectedly?
                best_move = np.random.choice(legal_moves)
                visit_distribution = {} # Ensure it's a dict for data saving
            # Check board.ply() for half-moves, board.fullmove_number for full moves
            elif board.ply() < TEMP_THRESHOLD_MOVES * 2: # Use ply() for half-moves
                # Sample using Temperature
                moves = list(visit_distribution.keys())
                visits = np.array([visit_distribution[m] for m in moves], dtype=np.float32) # Use float32 for stability

                # Prevent overflow/underflow issues with large visit counts before exponentiation
                visits = visits - np.max(visits) # Normalize max to 0
                visit_powers = np.exp(visits / TEMPERATURE) # Use exp for softmax-like temp

                # Handle potential division by zero if all visit_powers are zero (e.g., extreme temp or all visits zero)
                sum_powers = np.sum(visit_powers)
                if sum_powers <= 0 or not np.isfinite(sum_powers):
                     print(f"Warning [Worker {worker_id}/{os.getpid()}]: Invalid probabilities during sampling (sum={sum_powers}). FEN: {board.fen()}. Falling back to greedy.")
                     best_move = max(visit_distribution, key=visit_distribution.get)
                else:
                    probabilities = visit_powers / sum_powers
                    # Final check for NaN just in case
                    if np.isnan(probabilities).any():
                         print(f"Warning [Worker {worker_id}/{os.getpid()}]: NaN probabilities encountered. Falling back to greedy.")
                         best_move = max(visit_distribution, key=visit_distribution.get)
                    else:
                         best_move = np.random.choice(moves, p=probabilities)

            else:
                # Play Greedily (Max Visits)
                best_move = max(visit_distribution, key=visit_distribution.get)

            # Handle case where sampling somehow failed or no move selected
            if best_move is None:
                print(f"Error [Worker {worker_id}/{os.getpid()}]: MCTS returned no move after processing distribution. FEN: {board.fen()}")
                break # Exit loop if no move can be made

            # --- Data Recording ---
            # Record current board state and the *original* MCTS visit distribution
            board_tensor = board_to_tensor(board) # Potential CPU cost
            # Use the raw visit_distribution from MCTS for the training target policy
            game_data.append((board_tensor, visit_distribution))

            # --- Make Move ---
            board.push(best_move) # Potential CPU cost

        # Game finished
        print(f"[Worker {worker_id}/{os.getpid()}] Game finished after {move_count} moves. Result: {board.result()}")

    finally: # Ensure this runs even if errors occur in the loop
        # --- Stop Profiling ---
        profiler.disable()
        # --- Save Profiling Stats ---
        profile_filename = f"selfplay_worker_{worker_id}_pid_{os.getpid()}.prof"
        # Sort stats by cumulative time spent in function and subfunctions
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.dump_stats(profile_filename)
        print(f"[Worker {worker_id}/{os.getpid()}] Profiling stats saved to {profile_filename}")
        # --- End Profiling ---


    # --- Post-Game Processing (Outside profiling scope unless you move enable/disable) ---
    result = board.result()
    if result == "1-0": outcome = 1.0
    elif result == "0-1": outcome = -1.0
    else: outcome = 0.0

    training_samples = []
    current_player = 1.0
    for board_tensor, visit_distribution in game_data:
        target_policy = np.zeros(4352, dtype=np.float32)
        total_visits = sum(visit_distribution.values())
        if total_visits > 0:
            for move, count in visit_distribution.items():
                try: # Add error handling for move_to_index just in case
                    index = move_to_index(move)
                    target_policy[index] = count / total_visits
                except Exception as e:
                    print(f"Error converting move {move} to index: {e}") # Should not happen with legal moves
        training_samples.append((board_tensor, target_policy, outcome * current_player))
        current_player *= -1

    return training_samples


# Modify run_parallel_self_play to pass worker_id
def run_parallel_self_play(num_games, model_path, num_workers=6, mcts_simulations=50):
    """
    Runs self-play games in parallel. Passes a worker_id to each worker.
    """
    # Ensure spawn method is set (might be redundant if set in main.py, but safe)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Already set

    pool = mp.Pool(processes=num_workers)
    results = []

    with tqdm(total=num_games, desc="Self-Play Games") as pbar:
        # Assign a unique worker_id to each task submitted
        for i in range(num_games):
            worker_id = i % num_workers # Simple way to assign repeating worker IDs
            async_result = pool.apply_async(
                self_play_game,
                args=(model_path, mcts_simulations, worker_id), # Pass worker_id
                callback=lambda _: pbar.update(1)
            )
            results.append(async_result)

        pool.close()
        pool.join()

    all_games_data = []
    for r in results:
        try:
            game_samples = r.get() # Use get() to retrieve result and potentially raise errors
            all_games_data.append(game_samples)
        except Exception as e:
             print(f"\n!!! Error retrieving result from a worker: {type(e).__name__}: {e}")
             # Optionally log the full traceback here
             # import traceback
             # print(traceback.format_exc())

    training_samples = []
    for game in all_games_data:
        if game: # Ensure the worker returned data
            training_samples.extend(game)
    return training_samples