# self_play_batch.py
import chess
import chess.pgn
import numpy as np
import torch
import os
import time
from tqdm import tqdm
import torch.multiprocessing as mp

# Import the new batched MCTS
from mcts_batch import run_simulations_batch, evaluate_terminal

from neural_network import ChessNet
from board_encoder import board_to_tensor # Keep using numpy version for storing samples initially
from utils import move_to_index


# --- Configuration (can be moved to main config if preferred) ---
TEMPERATURE = 1.0
TEMP_THRESHOLD_MOVES = 30 # Number of *plies* (half-moves)
MAX_GAME_MOVES = 200 # Max plies before declaring draw (avoids infinite games)
DRAW_SCORE = -0.1 # Score for draws (1/2-1/2)


def self_play_game_batch(model_path: str, mcts_simulations: int, inference_batch_size: int):
    """
    Plays a single self-play game using the current model and batched MCTS.
    Returns:
      - training_samples: List[(board_tensor_np, target_policy_np, outcome_float)]
      - final_result_str: String like '1-0', '0-1', '1/2-1/2'
      - moves_played_count: Integer ply count
      - pgn_string: String representation of the game PGN
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the network (each process needs its own instance in memory)
    network = ChessNet()
    try:
        # Load onto CPU first to avoid CUDA init issues in spawn context sometimes
        network.load_state_dict(torch.load(model_path, map_location='cpu'))
        network.to(device) # Move to target device
        network.eval()
        # print(f"Worker {mp.current_process().pid}: Model loaded successfully to {device}.") # Debug
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return [], "*", 0 # Return empty data on error
    except Exception as e:
        print(f"Error loading model {model_path} in worker: {e}")
        return [], "*", 0



    board = chess.Board()
    game_data = []  # Store (board_tensor_np, visit_distribution_dict) for each move
    moves_played_count = 0
    
    
    # --- PGN Setup ---
    game = chess.pgn.Game()
    game.headers["Event"] = "Self-Play Batch"
    game.headers["Site"] = "Local"
    game.headers["White"] = f"FrixBot_Sim{mcts_simulations}" 
    game.headers["Black"] = f"FrixBot_Sim{mcts_simulations}"
    pgn_node = game 
    
    outcome_obj = None
    max_moves_reached = False 
    
    while True: # Loop broken internally by game over or max moves
        
        current_outcome = board.outcome(claim_draw=True)
        if current_outcome is not None:
            outcome_obj = current_outcome
            break
        
        if board.ply() >= MAX_GAME_MOVES:
            print(f"DEBUG: Game reached max moves ({MAX_GAME_MOVES}).")
            max_moves_reached = True
            break # End game as draw

        
        # --- Run MCTS ---
        # Use Batched MCTS to get best move and visit distribution
        # The MCTS function now handles device and AMP internally if needed
        best_move_mcts, visit_distribution = run_simulations_batch(
            board,
            network,
            num_simulations=mcts_simulations,
            inference_batch_size=inference_batch_size,
            device=device,
            return_visit_distribution=True,
            c_puct=2.0
        )

        selected_move = None # Move actually played

        if best_move_mcts is None or not visit_distribution:
            # MCTS failed or returned no moves 
            legal_moves = list(board.legal_moves)
            if legal_moves:
                 print(f"  Fallback (MCTS fail): Picking random move from {len(legal_moves)}.")
                 selected_move = np.random.choice(legal_moves)
                 visit_distribution = {selected_move: 1} # Placeholder
            else:
                 print("  No legal moves found after MCTS fail. Ending game.")
                 outcome_obj = board.outcome(claim_draw=True) # Try get real outcome
                 break

        # --- Temperature Sampling / Greedy Selection ---
        # Decide whether to sample or play greedily based on temperature schedule
        elif board.ply() < TEMP_THRESHOLD_MOVES:
            # --- Sample using Temperature ---
            moves = list(visit_distribution.keys())
            # Filter out non-Move objects (shouldn't happen with current MCTS)
            valid_moves = [m for m in moves if isinstance(m, chess.Move)]

            if not valid_moves:
                 print(f"Warning: No valid chess.Move keys in distribution? FEN: {board.fen()}. Fallback.")
                 legal_moves = list(board.legal_moves)
                 if legal_moves: selected_move = np.random.choice(legal_moves)
                 else: 
                    outcome_obj = board.outcome(claim_draw=True)
                    break
            else:
                visits = np.array([visit_distribution.get(m, 0) for m in valid_moves], dtype=np.float32)
                visits = np.maximum(visits, 1e-9) # Ensure positivity for power/division

                if TEMPERATURE == 1.0:
                    probabilities = visits / np.sum(visits)
                else:
                    visit_powers = visits**(1.0 / TEMPERATURE)
                    probabilities = visit_powers / np.sum(visit_powers)

                # Validate probabilities before sampling
                if not (np.isnan(probabilities).any() or np.isinf(probabilities).any() or not np.isclose(np.sum(probabilities), 1.0)):
                    try:
                        selected_move = np.random.choice(valid_moves, p=probabilities)
                    except ValueError as ve:
                         print(f"Error during np.random.choice (probs sum issue?): {ve}. Probs: {probabilities}. Sum: {np.sum(probabilities)}")
                         # Fallback to greedy
                         selected_move = max(valid_moves, key=lambda m: visit_distribution.get(m, 0))
                else:
                    print(f"Warning: Invalid probabilities for sampling. FEN: {board.fen()}. Sum: {np.sum(probabilities)}. Fallback to greedy.")
                    selected_move = max(valid_moves, key=lambda m: visit_distribution.get(m, 0))

        else:
            # --- Play Greedily (Max Visits) ---
            selected_move = best_move_mcts # MCTS already gives the best move by visits

        # Final check if a move was selected
        if selected_move is None:
            print(f"Error: No move selected for FEN: {board.fen()}. Checking legal moves.")
            legal_moves = list(board.legal_moves)
            if legal_moves:
                print(f"  Fallback (Selection fail): Choosing random from {len(legal_moves)}.")
                selected_move = np.random.choice(legal_moves)
            else:
                print("  No legal moves, breaking.")
                outcome_obj = board.outcome(claim_draw=True)
                break
            if selected_move is None:
                 print("!!! CRITICAL: Could not select any move even with fallback.")
                 outcome_obj = board.outcome(claim_draw=True)
                 break

        # --- Record Data & Play Move ---
        # Record current board state (as numpy array) and the MCTS visit distribution
        # Store numpy arrays for simpler multiprocessing transfer initially
        board_tensor_np = board_to_tensor(board)
        game_data.append((board_tensor_np, visit_distribution))

        # Play the selected move
        try:
            # --- Add move to PGN ---
            pgn_node = pgn_node.add_main_variation(selected_move)
            # --- Push move onto board ---
            board.push(selected_move)
            moves_played_count += 1
        except Exception as e:
            print(f"!!! CRITICAL ERROR pushing move {selected_move.uci()} on FEN {board.fen()}: {e}")
            # Attempt to get result if game ended due to error, otherwise return '*'
            final_result_str = board.result(claim_draw=True)
            return [], final_result_str, moves_played_count


    # --- Game Finished ---
    termination_reason = "Unknown" # Default

         
    if max_moves_reached:
        final_result_str = "1/2-1/2" # Treat max moves as a draw
        termination_reason = "MaxMovesReached" # Custom reason for PGN
    else:
        # If max moves not reached, get outcome from board
        if outcome_obj is None:
             outcome_obj = board.outcome(claim_draw=True) # Get final state again

        if outcome_obj:
            final_result_str = outcome_obj.result()
            termination_reason = outcome_obj.termination.name
        else:
            final_result_str = "*" # Unknown if no standard outcome
            # Keep termination_reason as "Unknown"

    # Set PGN headers
    game.headers["Result"] = final_result_str
    game.headers["Termination"] = termination_reason # Use determined reason
    game.headers["PlyCount"] = str(moves_played_count)


    # --- Post-Game Processing: Assign Outcome and Create Training Samples ---
    if final_result_str == "1-0": outcome = 1.0  # White wins
    elif final_result_str == "0-1": outcome = -1.0 # Black wins
    elif final_result_str == "1/2-1/2": outcome = DRAW_SCORE # Draw ("1/2-1/2", "*", length limit)
    else: # Handle '*' or other unexpected results
        outcome = 0.0 # Treat errors/unknowns as neutral for training

    training_samples = []
    network_output_size = network.policy_fc.out_features
    # Iterate through the game history
    for i, (board_tensor_np, visit_distribution_dict) in enumerate(game_data):
        # Determine perspective: White is 1.0, Black is -1.0
        # Player whose turn it was for the recorded state
        # If i=0 (first move), board_tensor is initial state, White to move -> perspective = 1.0
        # If i=1, Black to move -> perspective = -1.0
        perspective = 1.0 if (i % 2) == 0 else -1.0

        # Create the target policy vector
        target_policy_np = np.zeros(network_output_size, dtype=np.float32)
        if isinstance(visit_distribution_dict, dict) and visit_distribution_dict:
            # Ensure keys are moves and values are numbers
            valid_visits = {m: v for m, v in visit_distribution_dict.items()
                            if isinstance(m, chess.Move) and isinstance(v, (int, float))}
            total_visits = sum(valid_visits.values())

            if total_visits > 0:
                for move, count in valid_visits.items():
                    try:
                        index = move_to_index(move)
                        target_policy_np[index] = count / total_visits
                    except IndexError:
                         print(f"Error: move_to_index({move}) out of bounds during policy creation.")
                         continue # Skip this move index
                    except Exception as e:
                         print(f"Error converting move {move} to index for policy: {e}")
                         continue
        # Else: target_policy remains all zeros if distribution was empty/invalid

        # Append sample: (state, policy_target, final_outcome_for_player_at_this_state)
        # outcome * perspective gives the game result from the view of the player whose turn it was
        training_samples.append((board_tensor_np, target_policy_np, outcome * perspective))

    # --- Convert PGN game object to string ---
    pgn_string = str(game) # Generate the PGN string representation


    return training_samples, final_result_str, moves_played_count, pgn_string


# --- Parallel Self-Play Runner ---

def self_play_game_worker(args):
    """Wrapper function for multiprocessing pool."""
    model_path, mcts_simulations, inference_batch_size, worker_id = args
    # print(f"Worker {worker_id} started.") # Debug
    try:
        result_tuple = self_play_game_batch(model_path, mcts_simulations, inference_batch_size)
        return result_tuple # Return the full tuple (samples, result, length, pgn_string)
    
    except Exception as e:
        print(f"!!! Unhandled exception in worker {worker_id}: {type(e).__name__}: {e}")
        import traceback
        print(traceback.format_exc())
        # Return empty data to avoid crashing the main process
        return [], "*", 0


def run_parallel_self_play_batch(
    num_games: int,
    model_path: str,
    num_workers: int,
    mcts_simulations: int,
    inference_batch_size: int, # Add batch size param
    iteration_num: int,         # For directory naming
    num_pgns_to_save: int       # How many PGNs to save
):
    """
    Runs self-play games in parallel using batched MCTS.

    Args:
        num_games: Total self-play games to generate.
        model_path: Path to the current model checkpoint.
        num_workers: Number of parallel worker processes.
        mcts_simulations: Number of MCTS simulations per move.
        inference_batch_size: Max batch size for NN inference within MCTS.

    Returns:
        Tuple: (all_training_samples, list_of_game_results, list_of_game_lengths)
    """
    # Ensure spawn start method is set (usually done in main.py)
    # try:
    #     mp.set_start_method('spawn', force=True)
    # except RuntimeError:
    #     pass # Already set

    if num_workers <= 0:
         print("Warning: num_workers set to 0 or less. Running sequentially.")
         tasks = [(model_path, mcts_simulations, inference_batch_size, 0) for _ in range(num_games)]
         results = [self_play_game_worker(task) for task in tqdm(tasks, desc="Self-Play (Sequential)")]
    else:
        pool = mp.Pool(processes=num_workers)
        tasks = [(model_path, mcts_simulations, inference_batch_size, i) for i in range(num_games)]
        results_async = []

        print(f"Starting parallel self-play with {num_workers} workers...")
        # Use tqdm for progress tracking based on completed tasks
        with tqdm(total=num_games, desc="Self-Play Games") as pbar:
            # Define callback inside the loop or ensure it doesn't capture loop variables incorrectly
            def update_pbar(_):
                pbar.update(1)

            for task in tasks:
                async_result = pool.apply_async(
                    self_play_game_worker,
                    args=(task,),
                    callback=update_pbar
                )
                results_async.append(async_result)

            pool.close() # No more tasks will be submitted
            pool.join()  # Wait for all tasks to complete

        # Gather results after pool has finished
        print("Gathering results from workers...")
        results = []
        for i, async_res in enumerate(tqdm(results_async, desc="Processing Results")):
             try:
                 results.append(async_res.get(timeout=60)) # Add timeout
             except mp.TimeoutError:
                 print(f"Warning: Timeout getting result from worker for game index {i}.")
                 results.append(([], "*", 0)) # Append dummy data on timeout
             except Exception as e:
                 print(f"\n!!! Error retrieving result from worker {i}: {type(e).__name__}: {e}")
                 results.append(([], "*", 0)) # Append dummy data on error

    # Process collected results
    all_training_samples = []
    list_of_game_results = []
    list_of_game_lengths = []
    list_of_pgns = []

    successful_games = 0
    for samples, result_str, num_moves, pgn_str in results:
        if samples is not None and result_str is not None and pgn_str is not None:
            all_training_samples.extend(samples)
            list_of_game_results.append(result_str)
            list_of_game_lengths.append(num_moves)
            list_of_pgns.append(pgn_str) # Store the PGN string

            if samples or num_moves > 0:
                 successful_games += 1
        else:
            print("Warning: Worker returned invalid data (None detected in tuple).")


    if successful_games < len(results):
         print(f"Warning: Only {successful_games}/{len(results)} games appear to have run or returned valid data.")
    if not all_training_samples:
         print("Warning: No training samples were collected from any games.")
    if not list_of_game_results:
         print("Warning: No game results were collected.")

     # --- Save the specified number of PGNs ---
    if num_pgns_to_save > 0:
        pgn_dir = f"self_play_pgns/iter_{iteration_num}"
        os.makedirs(pgn_dir, exist_ok=True)
        num_to_save = min(num_pgns_to_save, len(list_of_pgns)) # Don't try to save more than we have
        print(f"Saving first {num_to_save} PGNs to: {pgn_dir}")

        for i in range(num_to_save):
            pgn_str_to_save = list_of_pgns[i]
            if pgn_str_to_save: # Only save if the string is not empty
                pgn_filename = os.path.join(pgn_dir, f"game_{i:04d}.pgn") # Use game index for filename
                try:
                    with open(pgn_filename, "w", encoding='utf-8') as f: # Specify encoding
                        f.write(pgn_str_to_save)
                except Exception as e:
                    print(f"Error writing PGN file {pgn_filename}: {e}")
            else:
                 print(f"Warning: Skipping PGN save for game index {i} as PGN string was empty.")
    else:
         print("PGN saving disabled (num_pgns_to_save <= 0).")

    
    return all_training_samples, list_of_game_results, list_of_game_lengths