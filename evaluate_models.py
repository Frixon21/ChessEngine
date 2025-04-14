import chess
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import time
import os
from neural_network import ChessNet
from mcts_batch import run_simulations_batch
import math

# --- Configuration ---
DEFAULT_SIMULATIONS = 800
DEFAULT_INF_BATCH_SIZE = 32 # Batch size for NN calls *during MCTS*
DEFAULT_C_PUCT = 1.25       # Exploration constant for MCTS during evaluation

# --- Helper Functions ---

def load_model(checkpoint_path, device):
    """
    Load a ChessNet model from a checkpoint file onto the specified device.
    """
    # Ensure the path exists before trying to load
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ChessNet().to(device) # Create model instance first
    try:
        # Load the state dict
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval() # Set to evaluation mode
        # print(f"Successfully loaded model from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model state_dict from {checkpoint_path}: {e}")
        raise e # Re-raise the exception
    return model

def play_game(model_white, model_black, simulations=DEFAULT_SIMULATIONS, inference_batch_size=DEFAULT_INF_BATCH_SIZE, c_puct=DEFAULT_C_PUCT):
    """
    Play a single game between model_white (White) and model_black (Black)
    using the new run_simulations_batch MCTS.

    Returns: +1 if White wins, -1 if Black wins, 0 for draw.
    """
    # Ensure models are on the same device, get device
    device_w = next(model_white.parameters()).device
    device_b = next(model_black.parameters()).device
    if device_w != device_b:
        # This shouldn't happen if loaded correctly in worker, but good check
        print("Warning: Models are on different devices!")
        # Attempt to move black model to white's device
        try:
            model_black.to(device_w)
            device = device_w
            print(f"Moved model_black to {device}")
        except Exception as e:
            print(f"Could not move model_black to device {device_w}: {e}")
            # Fallback or raise error - let's try CPU as fallback
            try:
                 device = torch.device("cpu")
                 model_white.to(device)
                 model_black.to(device)
                 print("Warning: Falling back to CPU due to device mismatch.")
            except Exception as cpu_e:
                 print(f"Could not move models to CPU: {cpu_e}")
                 raise RuntimeError("Model device mismatch and cannot resolve.")

    else:
        device = device_w # Both models are on the same device

    board = chess.Board()
    current_model = model_white # White to move first

    while not board.is_game_over(claim_draw=True): # Check for draws too
        if board.ply() > 250: # Add a ply limit to prevent overly long games
             print("Game exceeded ply limit, declaring draw.")
             return 0

        # --- Use the new MCTS function ---
        # No need to create root node explicitly
        best_move = run_simulations_batch(
            root_board=board, # Pass the current board state
            network=current_model,
            num_simulations=simulations,
            inference_batch_size=inference_batch_size, # Pass this parameter
            device=device,
            c_puct=c_puct, # Use configured C_PUCT
            dirichlet_alpha=0.0, # No noise for evaluation
            dirichlet_epsilon=0.0, # No noise for evaluation
            return_visit_distribution=False # We only need the best move
        )

        if best_move is None:
            # MCTS failed to return a move (e.g., no legal moves from root)
            print(f"Warning: MCTS returned None for FEN: {board.fen()}. Game likely over?")
            break # Exit loop, result will be determined below

        try:
            board.push(best_move)
        except Exception as e:
            print(f"!!! CRITICAL ERROR pushing move {best_move.uci()} on FEN {board.fen()}: {e}")
            # If pushing fails, game state is uncertain, declare draw? Or based on last state?
            # Let's try to get result from board before push failed
            outcome = board.outcome(claim_draw=True)
            if outcome: return outcome.result() == '1-0' and 1 or outcome.result() == '0-1' and -1 or 0
            else: return 0 # Fallback draw

        # Swap sides
        current_model = model_black if current_model == model_white else model_white

    # Evaluate final result
    outcome = board.outcome(claim_draw=True)
    if outcome:
        if outcome.winner == chess.WHITE:
            return 1
        elif outcome.winner == chess.BLACK:
            return -1
        else:
            return 0 # Draw
    else:
        # Should only happen if loop exited unexpectedly without game end
        print(f"Warning: Game loop finished but no outcome found? FEN: {board.fen()}")
        return 0 # Treat as draw

# --- Worker Function for Multiprocessing ---

def match_single_game(args):
    """
    Worker function to play one game.
    Args tuple: (model_path_a, model_path_b, a_is_white, simulations, inference_batch_size)
    Returns result from Model A's perspective (+1 win, 0 draw, -1 loss).
    """
    model_path_a, model_path_b, a_is_white, simulations, inference_batch_size = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load both models in this worker
    try:
        model_a = load_model(model_path_a, device)
        model_b = load_model(model_path_b, device)
    except Exception as e:
        print(f"Worker Error: Failed to load models ({model_path_a}, {model_path_b}). Error: {e}")
        # Return a draw or error indicator if models can't load
        # Returning 0 (draw) might skew results less than -1 or +1
        return 0

    # Play one game
    result = 0 # Default to draw in case of error in play_game
    try:
        if a_is_white:
            # Model A is White
            result = play_game(model_a, model_b, simulations, inference_batch_size)
            # result is +1 for White (A) win, 0 draw, -1 for Black (B) win
            return result
        else:
            # Model A is Black
            result = play_game(model_b, model_a, simulations, inference_batch_size)
            # result is +1 if White (B) won, 0 draw, -1 if Black (A) won
            # Flip the sign for Model A's perspective
            return -result
    except Exception as e:
        print(f"Worker Error: Exception during play_game: {e}")
        # Fallback to returning draw on game error
        return 0
    finally:
        # Clean up models from memory / GPU memory
        del model_a
        del model_b
        if device.type == 'cuda':
            torch.cuda.empty_cache()


# --- Main Match Function (Parallelized) ---

def match_models_multiprocess(
    model_path_a,
    model_path_b,
    num_games=50,
    simulations=DEFAULT_SIMULATIONS,
    inference_batch_size=DEFAULT_INF_BATCH_SIZE, # Added parameter
    num_workers=5
):
    """
    Play a match between two models in parallel using the new MCTS.
    """
    # Ensure even number of games if possible for fair color assignment
    if num_games % 2 != 0:
        print(f"Warning: num_games ({num_games}) is odd. Playing {num_games-1} games for even color split.")
        num_games -= 1
    if num_games == 0: return 0.0, 0.0

    games_per_color = num_games // 2

    # Build task list: (model_path_a, model_path_b, a_is_white, simulations, inference_batch_size)
    tasks = []
    tasks.extend([(model_path_a, model_path_b, True, simulations, inference_batch_size)] * games_per_color)
    tasks.extend([(model_path_a, model_path_b, False, simulations, inference_batch_size)] * games_per_color)

    # Set start method before creating pool if not already set
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    results = []
    pool = mp.Pool(processes=num_workers)

    print(f"Starting match: {model_path_a} vs {model_path_b} ({num_games} games)")
    with tqdm(total=num_games, desc="Match Progress") as pbar:
        results_async = []
        def update_pbar(*args): pbar.update(1)
        def error_callback(e): print(f"Error from worker: {e}") # Basic error callback

        for task in tasks:
            res = pool.apply_async(
                match_single_game,
                args=(task,),
                callback=update_pbar,
                error_callback=error_callback # Add error callback
            )
            results_async.append(res)

        pool.close()
        pool.join()

    # Gather results safely
    all_outcomes = []
    print("Gathering match results...")
    for i, r in enumerate(tqdm(results_async, desc="Collecting Results")):
        try:
            all_outcomes.append(r.get(timeout=60)) # Add timeout
        except Exception as e:
            print(f"Error getting result for game {i}: {e}")
            all_outcomes.append(0) # Count as draw if result retrieval fails

    # Summarize outcomes (Score from A's perspective: Win=1, Draw=0.5, Loss=0)
    score_a = 0.0
    wins_a = 0
    draws = 0
    losses_a = 0

    for outcome in all_outcomes:
        if outcome == 1:
            score_a += 1.0
            wins_a += 1
        elif outcome == 0:
            score_a += 0.5
            draws += 1
        elif outcome == -1:
            # score_a += 0.0 # No points for loss
            losses_a += 1
        else:
             print(f"Warning: Unexpected outcome value received: {outcome}") # Should not happen

    # Calculate score_b based on score_a and total games
    score_b = float(num_games) - score_a

    return score_a, score_b, wins_a, draws, losses_a

# --- Example Match Execution ---
if __name__ == "__main__":
    # --- Configuration for the match ---
    MODEL_A_PATH = "trained_model_itter60.pth" # e.g., the older model
    MODEL_B_PATH = "trained_model.pth"     # e.g., the current/new model

    NUM_GAMES = 100 # Total games (will be adjusted to be even)
    SIMULATIONS_PER_MOVE = 400 # MCTS simulations per move for evaluation games
    INF_BATCH_SIZE_EVAL = 32 # Inference batch size used during MCTS evaluation
    NUM_WORKERS_EVAL = 5 # Number of parallel games

    # --- Run the match ---
    start_match_time = time.time()
    score_A, score_B, wins_A, draws, losses_A = match_models_multiprocess(
        MODEL_A_PATH,
        MODEL_B_PATH,
        num_games=NUM_GAMES,
        simulations=SIMULATIONS_PER_MOVE,
        inference_batch_size=INF_BATCH_SIZE_EVAL,
        num_workers=NUM_WORKERS_EVAL
    )
    end_match_time = time.time()

    # --- Print Results ---
    actual_games_played = wins_A + draws + losses_A
    print("\n===== Match Results =====")
    print(f"Model A: {MODEL_A_PATH}")
    print(f"Model B: {MODEL_B_PATH}")
    print(f"Games Played: {actual_games_played}")
    print(f"Simulations per Move: {SIMULATIONS_PER_MOVE}")
    print(f"Final Score: A = {score_A:.1f}, B = {score_B:.1f}")
    print(f"Detailed: A Wins = {wins_A}, Draws = {draws}, B Wins = {losses_A}")
    
    if actual_games_played > 0:
            win_rate_A = (wins_A + 0.5 * draws) / actual_games_played
            print(f"Model A Score %: {win_rate_A * 100:.2f}%")
            # Calculate Elo difference based on Model A's win rate
            # This estimates Elo of A relative to B. Positive means A is stronger.
            if win_rate_A > 0 and win_rate_A < 1:
                 elo_diff = -400 * math.log10(1 / win_rate_A - 1)
                 print(f"Estimated Elo difference (A - B): {elo_diff:+.1f}")
            else:
                 print("Estimated Elo difference: N/A (score is 0% or 100%)")

    if score_A > score_B:
        print(f"\nResult: Model A ({MODEL_A_PATH}) is stronger.")
    elif score_B > score_A:
        print(f"\nResult: Model B ({MODEL_B_PATH}) is stronger.")
    else:
        print("\nResult: It's a tie!")

    print(f"Match duration: {end_match_time - start_match_time:.2f} seconds")
