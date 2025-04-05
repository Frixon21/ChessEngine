import chess
import chess.engine
import numpy as np
import time
import math
import os
from stockfish_processor import score_to_probability
from utils import move_to_index


# TODO: Set the path to your Stockfish executable
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe" # IMPORTANT: CHANGE THIS



NUM_ACTIONS = 4352 # User updated - assumed correct
STOCKFISH_VALUE_SCALE_CONST = 0.005 # OK
STOCKFISH_MULTIPV = 10 # OK
POLICY_TEMPERATURE = 0.2 # OK
ANALYSIS_LIMIT = chess.engine.Limit(depth=10) # 


# --- Main Debugging Logic ---
def debug_single_position(fen, stockfish_path, analysis_limit):
    """Analyzes a single FEN and prints Stockfish targets."""
    print("-" * 60)
    print(f"Debugging FEN: {fen}")

    board = None
    try:
        board = chess.Board(fen)
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        print(board) # Print the board state
    except ValueError:
        print("Error: Invalid FEN string provided.")
        return

    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        # Configure threads/hash if you did during generation
        engine.configure({"Threads": 5, "Hash": 2048})
        print(f"\nStockfish engine initialized (MultiPV={STOCKFISH_MULTIPV})")

        # --- Get Stockfish Value & Policy Info ---
        print(f"Running analysis (Limit: {analysis_limit})...")
        start_time = time.time()
        info_list = engine.analyse(board, analysis_limit, multipv=STOCKFISH_MULTIPV)
        end_time = time.time()
        print(f"Analysis took {end_time - start_time:.2f} seconds.")

        if not info_list:
            print("Stockfish returned no analysis info.")
            return

        # --- Calculate Value Target ---
        stockfish_value = 0.0
        score_obj = info_list[0].get("score")
        print("\n--- Value Target Calculation ---")
        print(f"Raw Score (PV 1): {score_obj}")
        if score_obj is not None:
            pov_score = score_obj.pov(board.turn)
            print(f"Score from Current Player Pov: {pov_score}")
            if pov_score.is_mate():
                stockfish_value = 1.0 if pov_score.mate() > 0 else -1.0
            else:
                cp_score = pov_score.score()
                if cp_score is None: stockfish_value = 0.0
                else:
                    stockfish_value = float(np.clip(2 / (1 + np.exp(-STOCKFISH_VALUE_SCALE_CONST * cp_score)) - 1, -1.0, 1.0))
            print(f"Calculated Value Target: {stockfish_value:.4f}")
        else:
            print("No score found, Value Target defaults to 0.0")

        # --- Calculate Policy Target ---
        print("\n--- Policy Target Calculation ---")
        pv_moves = []; pv_scores = []
        print(f"Raw MultiPV Moves & Scores (Player Pov):")
        for i, info in enumerate(info_list):
            pv = info.get("pv"); score = info.get("score")
            if pv and score:
                move = pv[0]
                if move in board.legal_moves:
                    pov_score = score.pov(board.turn)
                    pv_moves.append(move); pv_scores.append(pov_score)
                    score_str = f"Mate({pov_score.mate()})" if pov_score.is_mate() else f"CP({pov_score.score()})"
                    print(f"  PV {i+1}: {move.uci():<7} ({score_str})")

        if not pv_moves:
            print("Could not extract valid PV moves.")
            return

        print(f"\nCalculating probabilities with Temperature = {POLICY_TEMPERATURE}...")
        probabilities = score_to_probability(pv_moves, pv_scores, temp=POLICY_TEMPERATURE)

        if probabilities is None:
            print("Failed to calculate probabilities reliably.")
            return

        print("\nFinal Policy Target (Non-Zero Probabilities):")
        stockfish_policy_vector = np.zeros(NUM_ACTIONS, dtype=np.float32)
        found_valid = False
        prob_sum = 0.0
        for i, move in enumerate(pv_moves):
            try:
                move_idx = move_to_index(move)
                if 0 <= move_idx < NUM_ACTIONS:
                    prob = probabilities[i]
                    stockfish_policy_vector[move_idx] = prob
                    if prob > 1e-5: # Only print if probability is noticeable
                        print(f"  Move: {move.uci():<7} Index: {move_idx:<5} Probability: {prob:.4f}")
                    prob_sum += prob
                    found_valid = True
            except Exception as e:
                print(f"  Error getting index for move {move.uci()}: {e}")

        if not found_valid:
            print("  No valid policy entries could be mapped.")
        else:
            # Final normalization check (should be close to 1)
            final_sum = np.sum(stockfish_policy_vector)
            print(f"  (Sum of probabilities assigned: {prob_sum:.4f}, Final vector sum after potential internal normalization: {final_sum:.4f})")
            if not np.isclose(final_sum, 1.0):
                 print("  Warning: Final policy vector does not sum close to 1.0 after assignment!")

    except chess.engine.EngineTerminatedError:
        print("FATAL: Stockfish engine terminated unexpectedly.")
    except Exception as e:
        print(f"An error occurred: {type(e).__name__} - {e}")
    finally:
        if engine:
            engine.quit()
            print("\nStockfish engine closed.")
    print("-" * 60)


# --- Run the debug function ---
if __name__ == "__main__":
    # Make sure prerequisites are met (paths, NUM_ACTIONS, move_to_index)
    # debug_single_position(FEN_TO_DEBUG, STOCKFISH_PATH, ANALYSIS_LIMIT)

    # Example: Debug another position
    FEN_TO_DEBUG_2 = "3r4/ppk2pp1/2p1p3/4b3/P3n1P1/8/KPP2PN1/3rBR1R w - - 2 31"
    debug_single_position(FEN_TO_DEBUG_2, STOCKFISH_PATH, ANALYSIS_LIMIT)
