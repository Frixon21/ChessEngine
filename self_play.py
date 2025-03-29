# self_play.py
import chess
import chess.pgn 
import numpy as np
import torch
from mcts import mcts_push_pop_shared
from neural_network import ChessNet
from board_encoder import board_to_tensor
from utils import move_to_index
from tqdm import tqdm
import torch.multiprocessing as mp
# from mcts_copy import mcts , MCTSNode

def self_play_game(model_path, mcts_simulations=50):
    """
    Plays a single self-play game using the current model and returns training samples.
    Each sample is a tuple (board_tensor, target_policy, outcome).
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the network (each process has its own copy)
    network = ChessNet()    
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.to(device)
    network.eval()
    
    board = chess.Board()
    game_data = []  # Store (board_tensor, visit_distribution) for each move.
    moves_played_count = 0 
    
    # Play until game over.
    while not board.is_game_over():
        TEMPERATURE = 1.0
        TEMP_THRESHOLD_MOVES = 20 # Play proportionally for first 20 full moves
        
        # Use MCTS to get best move and visit distribution.
        _best_move_greedy, visit_distribution = mcts_push_pop_shared(board, network, num_simulations=mcts_simulations, return_visit_distribution=True, device=device)
        
        best_move = None
        if not visit_distribution:
            print("MCTS returned no distribution, game might be over or error occurred.")
            # Maybe pick random legal move if any exist?
            legal_moves = list(board.legal_moves)
            if not legal_moves: break
            best_move = np.random.choice(legal_moves)
            visit_distribution = {} 
        
        # Use board.ply() which counts half-moves
        elif board.ply() < TEMP_THRESHOLD_MOVES * 2:
            # Sample using Temperature
            moves = list(visit_distribution.keys())
            # Filter out potential non-Move objects if any crept in (shouldn't happen)
            valid_moves = [m for m in moves if isinstance(m, chess.Move)]
            if not valid_moves:
                 print(f"Warning: No valid moves in distribution? FEN: {board.fen()}. Fallback.")
                 legal_moves = list(board.legal_moves)
                 if not legal_moves: break
                 best_move = np.random.choice(legal_moves)
            else:
                visits = np.array([visit_distribution[m] for m in valid_moves], dtype=np.float32)
                # Ensure visits are positive for power calculation if temp != 1
                visits = np.maximum(visits, 1e-6)

                if TEMPERATURE == 1.0:
                    sum_visits = np.sum(visits)
                    if sum_visits == 0: # Avoid division by zero if all visits are zero/negative
                         print(f"Warning: Sum of visits is zero during sampling. FEN: {board.fen()}. Fallback.")
                         best_move = max(visit_distribution, key=visit_distribution.get) if visit_distribution else np.random.choice(valid_moves)
                    else:
                         probabilities = visits / sum_visits
                else:
                    visit_powers = visits**(1.0 / TEMPERATURE)
                    sum_powers = np.sum(visit_powers)
                    if sum_powers == 0:
                         print(f"Warning: Sum of visit powers is zero during sampling. FEN: {board.fen()}. Fallback.")
                         best_move = max(visit_distribution, key=visit_distribution.get) if visit_distribution else np.random.choice(valid_moves)
                    else:
                        probabilities = visit_powers / sum_powers

                # Ensure probabilities are valid before sampling
                if 'probabilities' in locals() and not (np.isnan(probabilities).any() or not np.isclose(np.sum(probabilities), 1.0)):
                    best_move = np.random.choice(valid_moves, p=probabilities)
                else:
                    # Fallback to greedy if probabilities are invalid
                    print(f"Warning: Invalid probabilities during sampling. FEN: {board.fen()}. Fallback to greedy.")
                    best_move = max(visit_distribution, key=visit_distribution.get) if visit_distribution else np.random.choice(valid_moves)
        
        
        
        else:
            # --- Play Greedily (Max Visits) ---
            if visit_distribution:
                # Filter for valid moves before finding max
                valid_moves_dist = {m:v for m,v in visit_distribution.items() if isinstance(m, chess.Move)}
                if valid_moves_dist:
                     best_move = max(valid_moves_dist, key=valid_moves_dist.get)
                else:
                     print(f"Warning: Greedy choice but no valid moves in distribution? FEN: {board.fen()}. Fallback.")
                     legal_moves = list(board.legal_moves)
                     if not legal_moves: break
                     best_move = np.random.choice(legal_moves)
            else:
                 print(f"Warning: Greedy selection with empty distribution?. FEN: {board.fen()}")
                 legal_moves = list(board.legal_moves)
                 if not legal_moves: break
                 best_move = np.random.choice(legal_moves)

        # Handle case where sampling somehow failed or no move selected
        if best_move is None:
            print(f"Error: No move selected for FEN: {board.fen()}. Checking legal moves.")
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                print("No legal moves. Game Over.")
                break
            else:
                print(f"Fallback: Choosing random from {len(legal_moves)} legal moves.")
                best_move = np.random.choice(legal_moves)
            if best_move is None: # If even random choice fails
                 print("!!! CRITICAL: Could not select any move.")
                 break # Exit game loop

        # Record current board state and the *original* MCTS visit distribution
        board_tensor = board_to_tensor(board)
        # Use the raw visit_distribution from MCTS for the training target policy
        game_data.append((board_tensor, visit_distribution))

        try:
            board.push(best_move)
            moves_played_count += 1
        except Exception as e:
            print(f"!!! CRITICAL ERROR pushing move {best_move.uci()} on FEN {board.fen()}: {e}")
            # Return current state as failure
            return [], board.result() if board.is_game_over() else "*", moves_played_count
            
    
    # --- Game Finished ---
    final_result_str = board.result()
    # print(f"Game finished. Result: {final_result_str}, Moves: {moves_played_count}") # Optional debug

    # --- Post-Game Processing ---
    if final_result_str == "1-0": outcome = 1.0
    elif final_result_str == "0-1": outcome = -1.0
    else: outcome = 0.0 # Including "1/2-1/2" and "*" if game ended unexpectedly
        

    # Create training samples with perspective correction.
    training_samples = []
    current_player = 1.0  # White starts.
    for i, (board_tensor, visit_distribution) in enumerate(game_data):
        target_policy = np.zeros(4352, dtype=np.float32)
        # Check if visit_distribution is a dict and has values
        if isinstance(visit_distribution, dict) and visit_distribution:
             # Ensure moves are valid before processing
             valid_moves_in_dist = {m: v for m, v in visit_distribution.items() if isinstance(m, chess.Move)}
             total_visits = sum(valid_moves_in_dist.values())

             if total_visits > 0:
                 for move, count in valid_moves_in_dist.items():
                     try:
                         index = move_to_index(move)
                         target_policy[index] = count / total_visits
                     except Exception as e:
                         print(f"Error converting move {move} to index: {e}")
        # Else: target_policy remains all zeros if no valid distribution

        # Append sample: (state, policy_target, final_outcome_for_player_at_this_state)
        training_samples.append((board_tensor, target_policy, outcome * current_player))
        current_player *= -1 # Flip perspective for next state

    return training_samples, final_result_str, moves_played_count



def run_parallel_self_play(num_games, model_path, num_workers=6, mcts_simulations=50):
    """
    Runs self-play games in parallel.
    
    Parameters:
      num_games: Total self-play games to generate.
      model_path: Path to the current model checkpoint.
      num_workers: Number of parallel worker processes.
      
    Returns:
      A list of training samples from all games.
    """
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(processes=num_workers)
    results_async = []
    
    # We'll track completion of each game with a tqdm bar
    with tqdm(total=num_games, desc="Self-Play Games") as pbar:
        for i in range(num_games):
            # Apply async so we can use a callback to update the progress bar
            async_result = pool.apply_async(
                self_play_game, 
                args=(model_path,mcts_simulations),
                # Each time a worker is done, update our progress bar by 1
                callback=lambda _: pbar.update(1)
            )
            results_async.append(async_result)

        # Close the pool to new tasks and wait for everything to finish
        pool.close()
        pool.join()

    # Now gather the results
    all_training_samples = []
    list_of_game_results = []
    list_of_game_lengths = []

    
    print("Gathering results from workers...")
    for async_result in tqdm(results_async, desc="Processing Results"):
        try:
            # Each worker returns (training_samples, final_result_str, moves_played_count)
            samples, result_str, num_moves = async_result.get()

            if samples is not None and result_str is not None: # Check for valid return
                all_training_samples.extend(samples)
                list_of_game_results.append(result_str)
                list_of_game_lengths.append(num_moves)
            else:
                print("Warning: Worker returned invalid data (None).")

        except Exception as e:
            print(f"\n!!! Error retrieving result from a worker: {type(e).__name__}: {e}")
            # Optionally log traceback
            # import traceback
            # print(traceback.format_exc())

    if not all_training_samples:
         print("Warning: No training samples were collected from any games.")
    if not list_of_game_results:
         print("Warning: No game results were collected.")


    return all_training_samples, list_of_game_results, list_of_game_lengths