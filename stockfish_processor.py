import chess
import chess.engine
import numpy as np
from tqdm import tqdm
import time
import math
import os

# --- Assumed Imports ---
try:
    from utils import move_to_index # Assuming this exists
except ImportError:
    print("Warning: Could not import move_to_index from utils. Using placeholder.")
    def move_to_index(move):
        raise NotImplementedError("Please implement or import your move_to_index function")

# --- Constants ---
NUM_ACTIONS = 4352 
STOCKFISH_VALUE_SCALE_CONST = 0.005 
STOCKFISH_MULTIPV = 20 
POLICY_TEMPERATURE = 0.2 


def score_to_probability(pv_moves, pv_scores, temp=0.1):
    """
    Converts a list of Stockfish PovScore objects into probabilities
    using softmax with temperature, prioritizing score types.

    Args:
        pv_moves: List of chess.Move objects corresponding to pv_scores.
        pv_scores: List of chess.engine.PovScore objects.
        temp: Temperature for softmax scaling.

    Returns:
        A numpy array of probabilities corresponding to pv_moves,
        or None if probabilities cannot be computed reliably.
    """
    if not pv_scores or not pv_moves or len(pv_moves) != len(pv_scores):
        return None

    winning_mates = []
    cp_scores = []
    losing_mates = []
    move_indices = list(range(len(pv_moves))) # Keep track of original index

    # Separate scores by type
    for i, score in enumerate(pv_scores):
        if score.is_mate():
            mate_val = score.mate()
            if mate_val > 0:
                winning_mates.append({"index": i, "score": mate_val}) # Store mate number
            else:
                losing_mates.append({"index": i, "score": mate_val}) # Store mate number
        else:
            cp_val = score.score()
            if cp_val is not None:
                cp_scores.append({"index": i, "score": cp_val})

    scaled_scores_dict = {} # Store scaled scores keyed by original index

    # --- Prioritize Score Types ---
    if winning_mates:
        # Higher score = faster mate
        winning_mates.sort(key=lambda x: x["score"], reverse=True)
        best_mate_score = winning_mates[0]["score"]
        # Scale based on how many moves faster than the slowest winning mate
        mate_value_scale = 100.0 # Smaller scale for relative diffs
        relative_mate_scores = [mate_value_scale * (m["score"] - best_mate_score) for m in winning_mates]
        # Apply temperature scaling
        safe_temp = max(temp, 1e-6)
        scaled_rel_mates = [s / (safe_temp + 1e-9) for s in relative_mate_scores] # Simpler scaling for mates
        for i, m_info in enumerate(winning_mates):
             scaled_scores_dict[m_info["index"]] = scaled_rel_mates[i]
        category_indices = [m["index"] for m in winning_mates]

    elif cp_scores:
        # Sort by CP score descending
        cp_scores.sort(key=lambda x: x["score"], reverse=True)
        best_cp = cp_scores[0]["score"]
        relative_cp_scores = [s["score"] - best_cp for s in cp_scores]
        # Scale relative scores by temperature
        safe_temp = max(temp, 1e-6)
        scaled_rel_cp = [s / (safe_temp * 100 + 1e-9) for s in relative_cp_scores] # Scale by temp*100
        for i, cp_info in enumerate(cp_scores):
            scaled_scores_dict[cp_info["index"]] = scaled_rel_cp[i]
        category_indices = [cp["index"] for cp in cp_scores]

    elif losing_mates:
        # Sort by "least bad" mate (largest negative number / smallest abs value)
        losing_mates.sort(key=lambda x: x["score"], reverse=True)
        least_bad_mate = losing_mates[0]["score"] # e.g., -4 is better than -1
        # Score relative to least bad mate (will be 0 or negative)
        relative_losing_mates = [m["score"] - least_bad_mate for m in losing_mates]
        # Scale relative scores by temperature
        safe_temp = max(temp, 1e-6)
        # Negative scores scaled by temp -> smaller negatives become relatively larger after exp
        scaled_rel_losing_mates = [s / (safe_temp * 10 + 1e-9) for s in relative_losing_mates] # Smaller scaling factor?
        for i, m_info in enumerate(losing_mates):
            scaled_scores_dict[m_info["index"]] = scaled_rel_losing_mates[i]
        category_indices = [m["index"] for m in losing_mates]

    else: # No valid scores found at all
        return None

    # --- Apply Softmax only to the relevant category ---
    final_scaled_scores = [scaled_scores_dict.get(i, -np.inf) for i in category_indices] # Get scores for moves in this category

    try:
        exp_scores = np.exp(np.array(final_scaled_scores, dtype=np.float32))
        sum_exp_scores = np.sum(exp_scores)

        if not np.isfinite(sum_exp_scores) or sum_exp_scores < 1e-9:
            if pv_scores: print(f"\nWarning [Post-Pri]: Sum invalid/zero ({sum_exp_scores}). Scores: {pv_scores}")
            return None

        probabilities_subset = exp_scores / sum_exp_scores

        # Create the final probability vector for all original pv_moves
        final_probabilities = np.zeros(len(pv_moves), dtype=np.float32)
        for i, original_index in enumerate(category_indices):
             final_probabilities[original_index] = probabilities_subset[i]

        if np.isnan(final_probabilities).any():
             print(f"\nWarning: NaN detected in final probabilities. Scores: {pv_scores}")
             return None

        return final_probabilities # Return numpy array

    except Exception as e:
        print(f"\nError during final softmax calculation: {e}. Scores: {pv_scores}")
        return None


def generate_stockfish_targets(raw_positions_data, stockfish_path, analysis_limit=None, workers=5, multipv=STOCKFISH_MULTIPV, depth=10):
    """
    Analyzes board positions using Stockfish (MultiPV) to generate
    value and richer policy targets. Includes robustness check in probability calc.
    Processes unique positions. Uses prioritized score handling.
    """
    if analysis_limit is None: 
        analysis_limit = chess.engine.Limit(depth=depth)
    stockfish_training_samples = []; engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": workers, "Hash": 2048})
        print(f"Stockfish engine initialized from: {stockfish_path} (Requesting MultiPV={multipv})")
    except Exception as e: print(f"FATAL: Error initializing Stockfish: {e}"); return None

    print(f"Analyzing ~{len(raw_positions_data)} positions with Stockfish...")
    unique_positions = {pos_data[0].fen(): pos_data for pos_data in raw_positions_data}.values()
    print(f"Processing {len(unique_positions)} unique positions...")
    analysis_iterator = tqdm(list(unique_positions), desc="Stockfish Analysis", unit="pos") # Convert view to list for tqdm
    processed_count = 0; skipped_count = 0; start_time = time.time()

    for i, (board_obj, board_tensor) in enumerate(analysis_iterator):
        try:
            # --- Get Stockfish Value ---
            info_list = engine.analyse(board_obj, analysis_limit, multipv=multipv)
            if not info_list: skipped_count += 1; continue

            score_obj = info_list[0].get("score")
            if score_obj is not None:
                pov_score = score_obj.pov(board_obj.turn)
                if pov_score.is_mate(): stockfish_value = 1.0 if pov_score.mate() > 0 else -1.0
                else:
                    cp_score = pov_score.score()
                    if cp_score is None: stockfish_value = 0.0
                    else: stockfish_value = float(np.clip(2 / (1 + np.exp(-STOCKFISH_VALUE_SCALE_CONST * cp_score)) - 1, -1.0, 1.0))
            else: stockfish_value = 0.0

            # --- Get Stockfish Policy ---
            pv_moves = []; pv_scores = []
            temp_move_score_map = {} # Store scores associated with moves temporarily
            for info in info_list:
                pv = info.get("pv"); score = info.get("score")
                if pv and score:
                    move = pv[0]
                    if move in board_obj.legal_moves:
                         # Only add move once, associate with its score object
                         if move not in temp_move_score_map:
                              temp_move_score_map[move] = score.pov(board_obj.turn)

            # Create lists in corresponding order
            pv_moves = list(temp_move_score_map.keys())
            pv_scores = [temp_move_score_map[m] for m in pv_moves]

            if not pv_moves: skipped_count += 1; continue

            # Convert scores to probabilities (returns numpy array or None)
            probabilities = score_to_probability(pv_moves, pv_scores, temp=POLICY_TEMPERATURE)

            if probabilities is None: # Check if None was returned
                 # Warning printed inside score_to_probability
                 skipped_count += 1; continue

            # --- Populate and Append Sample ---
            stockfish_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
            valid_policy_entry = False
            for j, move in enumerate(pv_moves): # Iterate using original pv_moves list
                try:
                    move_idx = move_to_index(move)
                    if 0 <= move_idx < NUM_ACTIONS:
                        # Assign probability from the final_probabilities array
                        stockfish_policy[move_idx] = probabilities[j]
                        valid_policy_entry = True
                except Exception as e: print(f"\nWarning: move_to_index failed for {move.uci()}: {e}")

            if valid_policy_entry:
                 policy_sum = np.sum(stockfish_policy)
                 # Check sum before normalizing (should be close to 1 already if softmax worked)
                 if policy_sum > 1e-6:
                      # Re-normalize just in case, although ideally not needed if softmax is correct
                      # stockfish_policy /= policy_sum
                      stockfish_training_samples.append((board_tensor, stockfish_policy, stockfish_value))
                      processed_count += 1
                 else: skipped_count += 1
            else: skipped_count += 1

        except chess.engine.EngineTerminatedError: print("FATAL: Stockfish engine terminated."); engine = None; break
        except Exception as e: print(f"\nError during analysis for FEN {board_obj.fen()}: {type(e).__name__} - {e}. Skipping."); skipped_count += 1

        # Update postfix
        current_elapsed_time = time.time() - start_time
        avg_rate = (i + 1) / current_elapsed_time if current_elapsed_time > 0 else 0
        analysis_iterator.set_postfix(avg_rate=f"{avg_rate:.1f} pos/s", skipped=skipped_count)

    if engine: engine.quit(); print("\nStockfish engine closed.")
    print(f"Stockfish analysis complete. Generated {processed_count} samples, skipped {skipped_count}.")
    return stockfish_training_samples