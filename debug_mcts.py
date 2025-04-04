import chess
import torch
import time
import os
import io
import chess.pgn

# --- Imports from your project ---
try:
    from neural_network import ChessNet
    # Use the MCTS version with the log_details parameter
    from mcts_batch import run_simulations_batch
    # Need the board encoder for MCTS internal calls
    from board_encoder import board_to_tensor_torch
    from utils import move_to_index 
except ImportError as e:
    print(f"Error importing project files: {e}")
    print("Make sure neural_network.py, mcts_batch_log_param.py, board_encoder.py are accessible.")
    exit()

# --- Configuration ---
# TODO: Set the FEN string of the position you want to debug
# Example: Position before White missed mate in the game log
# --- Configuration ---
# PGN String of the game ending in repetition
# (Using the one provided earlier)
PGN_STRING_TO_DEBUG = """
[Event "Self-Play Data Gen"]
[Site "Local"]
[Date "????.??.??"]
[Round "?"]
[White "FrixBot_Sim256"]
[Black "FrixBot_Sim256"]
[Result "1/2-1/2"]
[Termination "THREEFOLD_REPETITION"]
[PlyCount "99"]

1. Nf3 a6 2. e4 d5 3. e5 e6 4. d4 Bd7 5. Nbd2 Ra7 6. c3 h5 7. Nb3 Ra8 8. Be2 c5 9. Nxc5 Bxc5 10. dxc5 Qh4 11. Nxh4 Ne7 12. O-O Nf5 13. Nxf5 exf5 14. Kh1 h4 15. Bf4 h3 16. g3 g5 17. Bxg5 f6 18. Bxf6 Rf8 19. Qd3 Rxf6 20. exf6 Nc6 21. Qxd5 f4 22. Qe4+ Kf7 23. Qxf4 Be6 24. Qd6 Bd5+ 25. Kg1 Rg8 26. Bh5+ Rg6 27. Rfe1 Ne7 28. Rxe7+ Kf8 29. Re6+ Kg8 30. Qxd5 Rg5 31. Ree1+ Rxd5 32. Bg6 Rd7 33. Re6 Rd5 34. Rae1 Rxc5 35. Re7 Rd5 36. c4 Rd6 37. Bh7+ Kh8 38. Bc2 Rd1 39. Re6 Rxe1+ 40. Rxe1 a5 41. Bg6 b6 42. g4 b5 43. Bd3 bxc4 44. Bxc4 a4 45. f3 a3 46. b3 Kh7 47. Re7+ Kg6 48. Rg7+ Kxf6 49. Rc7 Kg6 50. Re7 Kf6 51. Re4 Kg6 52. Re8 Kg7 53. Ra8 Kf6 54. Ra7 Kg6 55. Rc7 Kg5 56. Re7 Kf4 57. Re1 Kxf3 58. g5 Kg4 59. g6 Kf3 60. g7 Kf4 61. Ba6 Kf3 62. Bb7+ Kf4 63. Re8 Kf5 64. Re1 Kf6 65. g8=Q Kf5 66. Kf1 Kf6 67. Re8 Kf5 68. Qf7+ Kg5 69. Qg6+ Kf4 70. Qf7+ Kg5 71. Qe7+ Kf4 72. Qd6+ Kf5 73. Kg1 Kg5 74. Kf2 Kg4 75. Qe7 Kf4 76. Qd6+ Kf5 77. b4 Kg5 78. Re7 Kf5 79. b5 Kg5 80. b6 Kh4 81. Qc7 Kh5 82. Kg3 Kg5 83. Bh1 Kg6 84. Qd7 Kh5 85. Qc7 Kg5 86. Qc6 Kh5 87. Qc7 Kg5 88. Qb7 Kh5 1/2-1/2"""


# TODO: Set the path to the model checkpoint you want to debug
MODEL_PATH_TO_DEBUG = "trained_model.pth" # Use the checkpoint from Iteration 30

# --- MCTS Parameters for Debugging ---
# Use the same parameters as your self-play or evaluation
SIMULATIONS_TO_DEBUG = 256
INF_BATCH_SIZE_DEBUG = 32
C_PUCT_DEBUG = 1.5

def load_model_for_debug(checkpoint_path, device):
    """ Loads a model for debugging. """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return None
    model = ChessNet().to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"Successfully loaded model from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading model state_dict from {checkpoint_path}: {e}")
        return None

# --- Main Debugging Logic ---
def debug_mcts_for_position(pgn_string, model_path, simulations, inf_batch_size, c_puct):
    """ Loads model, runs MCTS with logging, prints results for a FEN """
    print("=" * 60)
    print(f"Debugging MCTS Output for final position of provided PGN")
    print(f"Model: {model_path}, Sims: {simulations}, Batch: {inf_batch_size}, C_PUCT: {c_puct}")
    print("-" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = load_model_for_debug(model_path, device)
    if model is None:
        return

    board = None
    # Create Board
    try:
        pgn = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn)
        if game is None:
            print("Error: Could not read game from PGN string.")
            return
        board = game.board()
        print("Replaying game moves...")
        move_count = 0
        for move in game.mainline_moves():
            board.push(move)
            move_count += 1
        print(f"Reached position after {move_count} half-moves.")
        print(f"Final FEN: {board.fen()}")
        print("Board State:")
        print(board)
    except ValueError:
        print("Error: Invalid FEN string.")
        return

    if board is None:
        print("Failed to set up board.")
        return
    
    if board.is_game_over():
        print("Game is already over for this position.")
        return
    
     # --- Check Specific Priors ---
    print("\n--- Checking Initial Policy Priors ---")
    initial_policy_probs = None
    try:
        board_tensor = board_to_tensor_torch(board, device).unsqueeze(0)
        with torch.no_grad():
            policy_logits, _ = model(board_tensor) # Only need policy logits here
            # Use float() for softmax stability if not using autocast here
            initial_policy_probs = torch.softmax(policy_logits.float(), dim=1).squeeze(0).cpu().numpy()

        # Define key moves to check (adjust if needed for the specific FEN)
        moves_to_check = {}
        if board.turn == chess.WHITE: # Example for the FEN after 50...Kh4
             moves_to_check = {
                 "Qb7+(leads to Threefold)": chess.Move.from_uci("c7b7"),
                 "Re6+": chess.Move.from_uci("e7e6"),
                 "Qd6+": chess.Move.from_uci("c7d6"),
             }


        print("Priors for key moves:")
        if not moves_to_check:
             print("  (No specific moves defined for this turn in debug script)")
        for name, move in moves_to_check.items():
            if board.is_legal(move):
                try:
                    idx = move_to_index(move)
                    # Ensure index is valid before accessing probability
                    if 0 <= idx < len(initial_policy_probs):
                         prior = initial_policy_probs[idx]
                         print(f"  {name:<5} ({move.uci()}): {prior:.6f} (Index: {idx})")
                    else:
                         print(f"  {name:<5} ({move.uci()}): Index {idx} out of bounds!")
                except Exception as e:
                    print(f"  Error getting index/prior for {name} ({move.uci()}): {e}")
            else:
                print(f"  {name:<5} ({move.uci()}): Illegal")

    except Exception as e:
        print(f"Error during prior checking: {e}")
    # --- END Prior Check ---


    # Run MCTS with Logging Enabled
    print("\nRunning MCTS search with detailed logging...")
    start_time = time.time()
    best_move, visit_distribution = run_simulations_batch(
        root_board=board,
        network=model,
        num_simulations=simulations,
        inference_batch_size=inf_batch_size,
        device=device,
        c_puct=c_puct,
        dirichlet_alpha=0.0, # No noise for debugging specific move choice
        dirichlet_epsilon=0.0,
        return_visit_distribution=True, # Get distribution too
        log_details=True # <<< ENABLE LOGGING >>>
    )
    end_time = time.time()
    print(f"MCTS search took {end_time - start_time:.2f} seconds.")

    # --- Print Results ---
    print("\n" + "=" * 20 + " MCTS RESULTS " + "=" * 20)
    if best_move:
        print(f"Chosen Best Move: {best_move.uci()}")
    else:
        print("MCTS did not return a best move.")

    if visit_distribution:
        print("\nVisit Distribution (Top 10 by visits):")
        sorted_dist = sorted(visit_distribution.items(), key=lambda item: item[1], reverse=True)
        for i, (move, visits) in enumerate(sorted_dist):
            if i >= 10: break
            print(f"  Move: {move.uci():<7} Visits: {visits}")
    else: print("No visit distribution returned.")
    print("=" * 60)


# --- Run the debug function ---
if __name__ == "__main__":
    # Ensure prerequisites are met (paths, imports)
    if not os.path.exists(MODEL_PATH_TO_DEBUG):
         print(f"ERROR: Model file not found: {MODEL_PATH_TO_DEBUG}")
    else:
         debug_mcts_for_position(
             PGN_STRING_TO_DEBUG,
             MODEL_PATH_TO_DEBUG,
             SIMULATIONS_TO_DEBUG,
             INF_BATCH_SIZE_DEBUG,
             C_PUCT_DEBUG
         )

