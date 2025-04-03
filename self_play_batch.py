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
from mcts_batch import run_simulations_batch # We still need MCTS to play the game

from neural_network import ChessNet
# Make sure board_to_tensor handles perspective correctly!
from board_encoder import board_to_tensor
from utils import move_to_index # Still needed for MCTS internal expansion if using indices


# --- Configuration (can be moved to main config if preferred) ---
TEMPERATURE = 1.0
TEMP_THRESHOLD_MOVES = 20 # Number of *plies* (half-moves)
MAX_GAME_MOVES = 200 # Max plies before declaring draw (avoids infinite games)

def self_play_game_batch(model_path: str, mcts_simulations: int, inference_batch_size: int):
    """
    Plays a single self-play game using the current model and batched MCTS.
    Collects position data (board object, board tensor) for later analysis.
    Corrected handling of run_simulations_batch return values.

    Returns:
        - position_data: List of tuples [(board_object_copy, board_tensor_np), ...]
        - final_result_str: String like '1-0', '0-1', '1/2-1/2', '*'
        - moves_played_count: Integer ply count
        - pgn_string: String representation of the game PGN
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = None # Initialize
    try:
        # print(f"Loading TorchScript model from {model_path}...") 
        network = torch.jit.load(model_path, map_location=device)
        network.to(device) # Ensure it's on the correct device
        network.eval()
        # print("TorchScript model loaded successfully.") 
    except Exception as e:
        print(f"Error loading TorchScript model {model_path}: {e}")
        # Handle error - maybe try loading the standard .pth file as fallback?
        # Or return error immediately
        return [], "*", 0, "" # Example error return

    board = chess.Board()
    position_data = []
    moves_played_count = 0

    game = chess.pgn.Game()
    game.headers["Event"] = "Self-Play Data Gen"
    game.headers["Site"] = "Local"
    game.headers["White"] = f"FrixBot_Sim{mcts_simulations}"
    game.headers["Black"] = f"FrixBot_Sim{mcts_simulations}"
    pgn_node = game

    outcome_obj = None
    max_moves_reached = False

    while True: # Game loop
        current_outcome = board.outcome(claim_draw=True)
        if current_outcome is not None:
            outcome_obj = current_outcome
            break
        if board.ply() >= MAX_GAME_MOVES:
            max_moves_reached = True
            break

        board_copy_for_sample = board.copy()
        board_tensor_np = board_to_tensor(board_copy_for_sample)
        position_data.append((board_copy_for_sample, board_tensor_np))

        # --- Run MCTS to decide the move ---
        # <<< FIX #1: Correctly assign single return value >>>
        # When return_visit_distribution=False, it returns only the best move
        best_move_mcts, visit_distribution = run_simulations_batch(
            board,
            network,
            num_simulations=mcts_simulations,
            inference_batch_size=inference_batch_size,
            device=device,
            return_visit_distribution=True,
            c_puct=1.25,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25
        )

        selected_move = None

        # --- Determine Move to Play ---
        if best_move_mcts is None: # MCTS failed or returned None
            legal_moves = list(board.legal_moves)
            if legal_moves:
                print(f"Warning: MCTS returned None for FEN: {board.fen()}. Picking random.")
                selected_move = np.random.choice(legal_moves)
            else:
                print(f"Warning: No legal moves found after MCTS fail/None. FEN: {board.fen()}. Ending game.")
                outcome_obj = board.outcome(claim_draw=True)
                break
        elif board.ply() < TEMP_THRESHOLD_MOVES and TEMPERATURE > 0: # Temperature sampling phase
             # Need visit distribution for sampling, so run MCTS again requesting distribution
             # When return_visit_distribution=True, it returns (best_move, distribution)

             if visit_distribution: # Check if distribution was returned
                 moves = list(visit_distribution.keys())
                 valid_moves = [m for m in moves if isinstance(m, chess.Move)]
                 if valid_moves:
                    visits = np.array([visit_distribution.get(m, 0) for m in valid_moves], dtype=np.float32)
                    visits = np.maximum(visits, 1e-9)
                    if TEMPERATURE == 1.0: probabilities = visits / np.sum(visits)
                    else: visit_powers = visits**(1.0 / TEMPERATURE); probabilities = visit_powers / np.sum(visit_powers)

                    probabilities /= np.sum(probabilities) # Renormalize
                    if not (np.isnan(probabilities).any() or np.isinf(probabilities).any() or not np.isclose(np.sum(probabilities), 1.0)):
                        try: selected_move = np.random.choice(valid_moves, p=probabilities)
                        except ValueError as ve:
                            print(f"Sampling Error: {ve}. Using MCTS best move.");
                            # Fallback to MCTS best move if sampling fails
                            selected_move = best_move_mcts if best_move_mcts in valid_moves else max(valid_moves, key=lambda m: visit_distribution.get(m, 0))
                    else:
                        print(f"Warning: Invalid probabilities. Using MCTS best move.");
                        selected_move = best_move_mcts if best_move_mcts in valid_moves else max(valid_moves, key=lambda m: visit_distribution.get(m, 0))
                 else: # No valid moves in distribution
                      print(f"Warning: No valid moves in sampling distribution. Fallback random."); legal_moves = list(board.legal_moves); selected_move = np.random.choice(legal_moves) if legal_moves else None
             else: # MCTS failed to return distribution for sampling
                 print(f"Warning: MCTS failed to return distribution for sampling. Fallback random."); legal_moves = list(board.legal_moves); selected_move = np.random.choice(legal_moves) if legal_moves else None

        else: # Greedy phase (ply >= threshold or TEMPERATURE == 0)
            # Use the best move already determined by the first MCTS call
            selected_move = best_move_mcts

        # --- Final Move Selection Check ---
        # (Error handling remains the same)
        if selected_move is None:
            print(f"Error: No move selected. FEN: {board.fen()}. Fallback random.")
            legal_moves = list(board.legal_moves)
            if legal_moves: selected_move = np.random.choice(legal_moves)
            else: print("No legal moves, breaking."); outcome_obj = board.outcome(claim_draw=True); break
            if selected_move is None: print(f"!!! CRITICAL: Could not select any move. FEN: {board.fen()}."); outcome_obj = board.outcome(claim_draw=True); break

        # --- Play the selected move ---
        # (Logic remains the same)
        try:
            if pgn_node is not None: pgn_node = pgn_node.add_main_variation(selected_move)
            else: print(f"Warning: pgn_node became None. Cannot add move {selected_move.uci()} to PGN.")
            board.push(selected_move)
            moves_played_count += 1
        except Exception as e:
            print(f"!!! CRITICAL ERROR pushing move {selected_move.uci()} on FEN {board.fen()}: {e}")
            final_result_str = board.result(claim_draw=True)
            return [], final_result_str, moves_played_count, ""

    # --- Game Finished ---
    # (Logic remains the same)
    termination_reason = "Unknown"
    if max_moves_reached: final_result_str = "1/2-1/2"; termination_reason = "MaxMovesReached"
    elif outcome_obj is not None: final_result_str = outcome_obj.result(); termination_reason = outcome_obj.termination.name
    else:
        final_outcome_check = board.outcome(claim_draw=True)
        if final_outcome_check: final_result_str = final_outcome_check.result(); termination_reason = final_outcome_check.termination.name
        else: final_result_str = "*"; termination_reason = "UnknownTermination"

    if game:
        game.headers["Result"] = final_result_str; game.headers["Termination"] = termination_reason
        game.headers["PlyCount"] = str(moves_played_count); pgn_string = str(game)
    else: pgn_string = ""

    # Clean up network instance
    del network
    if device.type == 'cuda': torch.cuda.empty_cache()

    # Return the collected position data
    return position_data, final_result_str, moves_played_count, pgn_string


# --- Parallel Self-Play Runner ---

def self_play_game_worker(args):
    """Wrapper function for multiprocessing pool."""
    model_path, mcts_simulations, inference_batch_size, worker_id = args
    try:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Expect position_data back from the batch function
        result_tuple = self_play_game_batch(model_path, mcts_simulations, inference_batch_size)
        return result_tuple # Returns (position_data, result, length, pgn_string)
    except Exception as e:
        print(f"!!! Unhandled exception in worker {worker_id}: {type(e).__name__}: {e}")
        import traceback
        print(traceback.format_exc())
        return [], "*", 0, "" # Return empty position data

def run_parallel_self_play_batch(
    num_games: int,
    model_path: str,
    num_workers: int,
    mcts_simulations: int,
    inference_batch_size: int,
    iteration_num: int,
    num_pgns_to_save: int
):
    """
    Runs self-play games in parallel using batched MCTS.
    Collects raw position data (board_obj, board_tensor) for later processing.
    Saves specified PGNs.

    Returns:
        Tuple: (all_raw_positions, list_of_game_results, list_of_game_lengths)
               where all_raw_positions is List[(board_obj, board_tensor_np)]
    """

    # (Multiprocessing setup remains the same)
    if num_workers <= 0:
        print("Warning: num_workers set to 0 or less. Running sequentially.")
        tasks = [(model_path, mcts_simulations, inference_batch_size, 0) for _ in range(num_games)]
        results = [self_play_game_worker(task) for task in tqdm(tasks, desc="Self-Play (Sequential)")]
    else:
        try: mp.set_start_method('spawn', force=True)
        except RuntimeError: pass
        pool = mp.Pool(processes=num_workers)
        tasks = [(model_path, mcts_simulations, inference_batch_size, i) for i in range(num_games)]
        results_async = []
        print(f"Starting parallel self-play with {num_workers} workers...")
        with tqdm(total=num_games, desc="Self-Play Games") as pbar:
            def update_pbar(_): pbar.update(1)
            for task in tasks:
                async_result = pool.apply_async(self_play_game_worker, args=(task,), callback=update_pbar)
                results_async.append(async_result)
            pool.close()
            pool.join()
        print("Gathering results from workers...")
        results = []
        for i, async_res in enumerate(results_async):
            try:
                results.append(async_res.get(timeout=300))
            except mp.TimeoutError:
                print(f"Warning: Timeout getting result from worker for game index {i}.")
                results.append(([], "*", 0, ""))
            except Exception as e:
                print(f"\n!!! Error retrieving result from worker {i}: {type(e).__name__}: {e}")
                results.append(([], "*", 0, ""))

    # Process results to collect raw positions instead of training samples
    all_raw_positions = [] # List to store (board_obj, board_tensor_np) tuples
    list_of_game_results = []
    list_of_game_lengths = []
    indexed_results_for_pgn = [] # Keep this for PGN saving

    successful_games = 0
    for i, result_tuple in enumerate(results):
         if isinstance(result_tuple, tuple) and len(result_tuple) == 4:
              # Unpack the results from the worker
              game_position_data, result_str, num_moves, pgn_str = result_tuple

              # Store data needed for PGN saving
              indexed_results_for_pgn.append({"index": i, "result": result_str, "moves": num_moves, "pgn": pgn_str})

              # Collect raw positions from this game
              if game_position_data: # Check if list is not empty
                    all_raw_positions.extend(game_position_data) # Add [(board_obj, tensor), ...]

              # Store game-level stats
              if result_str in ["1-0", "0-1", "1/2-1/2", "*"]:
                   list_of_game_results.append(result_str)
                   list_of_game_lengths.append(num_moves)
                   if game_position_data or num_moves > 0 or result_str != '*':
                        successful_games += 1
              else: # Handle unexpected result string
                   print(f"Warning: Worker {i} unexpected result: '{result_str}'.")
                   list_of_game_results.append("*")
                   list_of_game_lengths.append(0)
         else: # Handle malformed results
              print(f"Warning: Worker {i} malformed data: {type(result_tuple)}.")
              indexed_results_for_pgn.append({"index": i, "result": "*", "moves": 0, "pgn": ""})
              list_of_game_results.append("*")
              list_of_game_lengths.append(0)

    # (Status printing remains the same)
    if successful_games < len(results): print(f"Warning: Only {successful_games}/{len(results)} games valid.")
    if not all_raw_positions: print("Warning: No raw positions were collected.") # Changed warning message
    if not list_of_game_results: print("Warning: No game results were collected.")

    # (PGN saving logic remains the same as the last version)
    if indexed_results_for_pgn:
        pgn_dir = f"self_play_pgns/iter_{iteration_num}"; os.makedirs(pgn_dir, exist_ok=True)
        pgns_saved_count = 0; saved_indices = set()
        print(f"Saving PGNs (first {num_pgns_to_save} + all decisive) to: {pgn_dir}")
        # 1. Save initial PGNs
        for i in range(min(num_pgns_to_save, len(indexed_results_for_pgn))):
            game_data = indexed_results_for_pgn[i]; original_index = game_data["index"]; result_str = game_data["result"]; pgn_str_to_save = game_data["pgn"]
            if isinstance(pgn_str_to_save, str) and pgn_str_to_save.strip():
                safe_result_str = result_str.replace('/', '-'); pgn_filename = os.path.join(pgn_dir, f"game_{original_index:04d}_{safe_result_str}.pgn")
                try:
                    with open(pgn_filename, "w", encoding='utf-8') as f: f.write(pgn_str_to_save)
                    saved_indices.add(original_index); pgns_saved_count += 1
                except Exception as e: print(f"Error writing PGN {pgn_filename}: {e}")
        # 2. Save additional decisive PGNs
        # for game_data in indexed_results_for_pgn:
        #     original_index = game_data["index"]; result_str = game_data["result"]; pgn_str_to_save = game_data["pgn"]
        #     if result_str in ["1-0", "0-1"] and original_index not in saved_indices:
        #         if isinstance(pgn_str_to_save, str) and pgn_str_to_save.strip():
        #             safe_result_str = result_str.replace('/', '-'); pgn_filename = os.path.join(pgn_dir, f"game_{original_index:04d}_{safe_result_str}.pgn")
        #             try:
        #                 with open(pgn_filename, "w", encoding='utf-8') as f: f.write(pgn_str_to_save)
        #                 saved_indices.add(original_index); pgns_saved_count += 1
        #             except Exception as e: print(f"Error writing PGN {pgn_filename}: {e}")
        print(f"Saved {pgns_saved_count} PGN files in total.")
    elif num_pgns_to_save > 0: print("No game results available to save PGNs.")

    # Return the raw positions instead of MCTS training samples
    return all_raw_positions, list_of_game_results, list_of_game_lengths
