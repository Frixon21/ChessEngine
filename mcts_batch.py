# mcts_batch.py
import chess
import math
import numpy as np
import torch
import random # Added for potential sampled logging if needed later
from board_encoder import board_to_tensor_torch
from utils import move_to_index


#Flag to enable/disable detailed MCTS logging
DETAILED_MCTS_LOGGING = False 
#How many top moves/policy outputs to log
LOG_TOP_N = 5

# --- Constants ---
DEBUG_PRINT_REPETITION_DETAILS = False # Keep debug flag if needed
DEBUG_PRINT_TOP_N_MOVES = 8
VIRTUAL_LOSS_VALUE = 1.0

def evaluate_terminal(board: chess.Board):
    """Checks if the board state is terminal and returns the score from White's perspective."""
    if board.is_checkmate():
        # Value is from White's perspective: +1 if White won, -1 if Black won
        return 1.0 if board.turn == chess.BLACK else -1.0 # White won if it's Black's turn now
    if board.is_stalemate() or board.is_insufficient_material() or \
       board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0.0
    # Game is not over
    return None

class MCTSNode:
    """Represents a node in the Monte Carlo Tree Search."""
    def __init__(self, parent=None, move=None, prior=0.0):
        self.parent = parent
        self.move = move  # The move that led to this node
        self.children = {}  # Maps chess.Move to MCTSNode
        self.visits = 0
        self.value_sum = 0.0 # Accumulated value from simulations passing through here
        self.prior = prior # Prior probability from network (potentially noise-augmented at root)
        self._is_expanded = False # Flag to avoid redundant expansion checks
        self.virtual_loss_count = 0

    def value(self) -> float:
        """Returns the average value of this node (from this node player's perspective)."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_expanded(self) -> bool:
        return self._is_expanded

    def expand(self, board: chess.Board, policy_probs: np.ndarray):
        """Expands the node by creating children for all legal moves."""
        if self._is_expanded: return
        self._is_expanded = True
        current_legal_moves = list(board.legal_moves)
        if not current_legal_moves:
            return

        for move in current_legal_moves:
            if move not in self.children:
                try:
                    move_idx = move_to_index(move)
                    if 0 <= move_idx < len(policy_probs):
                        prior = policy_probs[move_idx]
                        self.children[move] = MCTSNode(parent=self, move=move, prior=prior)
                    else:
                        print(f"Error expanding: move_to_index({move})={move_idx} out of bounds ({len(policy_probs)}). FEN: {board.fen()}")
                except Exception as e:
                     print(f"Error creating child node for move {move} (Index: {move_idx if 'move_idx' in locals() else 'N/A'}): {e}. FEN: {board.fen()}")


    def select_child(self, c_puct: float, parent_N_eff: int) -> 'MCTSNode':
        """Selects the child node with the highest PUCT score."""
        best_score = -float('inf')
        best_child = None
        parent_sqrt_N_eff = math.sqrt(max(1, parent_N_eff))

        for child in self.children.values():
            N_real = child.visits
            VLC = child.virtual_loss_count
            N_eff = N_real + VLC
            W_real = child.value_sum
            W_eff = W_real - (VLC * VIRTUAL_LOSS_VALUE)
            Q_eff_child = W_eff / N_eff if N_eff > 0 else 0.0 # Value from child's perspective
            prior_term = child.prior if child.prior > 1e-6 else 1e-6
            U_eff = c_puct * prior_term * parent_sqrt_N_eff / (1 + N_eff)
            score = -Q_eff_child + U_eff # Score from parent's perspective

            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None and self.children:
             print(f"Warning: select_child found no best_child despite children existing. FEN: {self.parent.move if self.parent else 'Root'}")

        return best_child

    def backpropagate(self, value: float, release_virtual_loss: bool):
        """
        Backpropagates the simulation result up the tree.
        'value' is the evaluation from the perspective of the node *where the evaluation occurred*.
        """
        node = self
        current_value_perspective = value
        while node is not None:
            if release_virtual_loss:
                if node.virtual_loss_count > 0:
                    node.virtual_loss_count -= 1

            node.visits += 1
            node.value_sum += current_value_perspective # Add value from the incoming perspective

            node = node.parent
            # Flip perspective for the parent node
            current_value_perspective *= -1.0


# --- Main Batched MCTS Function ---

def run_simulations_batch(
    root_board: chess.Board,
    network: torch.nn.Module,
    num_simulations: int,
    inference_batch_size: int,
    device: torch.device,
    c_puct: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    return_visit_distribution: bool = False
):
    """ MCTS with batch inference and corrected terminal backpropagation. """
    root_node = MCTSNode()
    num_actions = network.policy_fc.out_features
    root_legal_moves = list(root_board.legal_moves)

    if not root_legal_moves:
        print("Warning: MCTS called on a board with no legal moves.")
        if return_visit_distribution: return None, {}
        else: return None

    try:
        # Initial evaluation
        root_tensor = board_to_tensor_torch(root_board, device).unsqueeze(0)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            policy_logits, value_tensor = network(root_tensor)
            policy_probs_gpu = torch.softmax(policy_logits, dim=1)
            initial_value = value_tensor.item()
        initial_policy_probs = policy_probs_gpu.cpu().numpy()[0]

        if DETAILED_MCTS_LOGGING:
            print(f"\n--- MCTS Initial Eval: FEN: {root_board.fen()} | Turn: {'W' if root_board.turn else 'B'} ---")
            print(f"  Network Raw Value Prediction: {initial_value:.4f}")
            top_indices = np.argsort(initial_policy_probs)[-LOG_TOP_N:][::-1]
            print(f"  Top {LOG_TOP_N} Policy Predictions (Raw):")
            for idx in top_indices: print(f"    Index {idx}: {initial_policy_probs[idx]:.4f}")
            print("--- End Initial Eval ---")

        # Apply noise & expand root
        policy_for_expansion = initial_policy_probs
        if dirichlet_epsilon > 0 and len(root_legal_moves) > 1:
            noise = np.random.dirichlet([dirichlet_alpha] * num_actions)
            policy_for_expansion = (1 - dirichlet_epsilon) * initial_policy_probs + dirichlet_epsilon * noise
            policy_for_expansion /= np.sum(policy_for_expansion)
        root_node.expand(root_board, policy_for_expansion)

    except Exception as e:
        print(f"!!! Exception during initial network eval/expansion: {type(e).__name__}: {e}")
        print(f"    Initial Eval FEN: {root_board.fen()}")
        if not root_node.is_expanded(): # Fallback expansion
            uniform_policy = np.ones(num_actions) / num_actions
            try: root_node.expand(root_board, uniform_policy); print("    Fallback: Expanded root with uniform policy.")
            except Exception as exp_e: print(f"    Fallback expansion failed: {exp_e}"); return (None, {}) if return_visit_distribution else None

    # --- Batch Simulation Loop ---
    pending_evaluations: list[tuple[MCTSNode, chess.Board]] = []
    completed_sims = 0
    while completed_sims < num_simulations:
        current_node = root_node
        sim_board = root_board.copy()
        current_node.virtual_loss_count += 1
        release_vl_for_this_path = True

        while current_node.is_expanded():
            terminal_value = evaluate_terminal(sim_board)
            if terminal_value is not None:
                if DETAILED_MCTS_LOGGING: print(f"--- MCTS Terminal Found during Selection: FEN: {sim_board.fen()} | Result(W): {terminal_value:.1f} ---")

                # Determine the player whose turn it was at the node *before* the terminal state
                current_node_turn = not sim_board.turn # Player who made the move leading to sim_board state
                if terminal_value == 0.0:
                    value_for_backprop = 0.0
                elif current_node_turn == chess.WHITE:
                    # If White made the move, backprop value from White's perspective
                    value_for_backprop = terminal_value
                else: # Black made the move
                    # If Black made the move, backprop value from Black's perspective
                    value_for_backprop = -terminal_value

                current_node.backpropagate(value_for_backprop, release_virtual_loss=True)
                completed_sims += 1
                current_node = None
                release_vl_for_this_path = False
                break # Exit selection loop

            parent_N_eff = current_node.visits + current_node.virtual_loss_count
            best_child = current_node.select_child(c_puct, parent_N_eff)

            if best_child is None: # Handle selection failure
                print(f"Warning: select_child returned None. FEN: {sim_board.fen()}")
                current_node.backpropagate(0.0, release_virtual_loss=True) # Backprop draw as fallback
                completed_sims += 1
                current_node = None
                release_vl_for_this_path = False
                break

            best_child.virtual_loss_count += 1
            try:
                sim_board.push(best_child.move)
                current_node = best_child
            except Exception as e: # Handle push error
                print(f"!!! Error pushing move {best_child.move.uci()} during selection: {e} FEN: {sim_board.fen()}")
                failed_node_parent = best_child.parent
                if failed_node_parent: failed_node_parent.backpropagate(0.0, release_virtual_loss=True)
                completed_sims += 1
                current_node = None
                release_vl_for_this_path = False
                break

        if current_node is not None: # Reached a leaf node
            terminal_value = evaluate_terminal(sim_board)
            if terminal_value is not None: # Leaf is terminal
                if DETAILED_MCTS_LOGGING: print(f"--- MCTS Terminal Found at Leaf: FEN: {sim_board.fen()} | Result(W): {terminal_value:.1f} ---")

                current_node_turn = not sim_board.turn # Player who made the move leading to sim_board state
                if terminal_value == 0.0:
                    value_for_backprop = 0.0
                elif current_node_turn == chess.WHITE:
                    value_for_backprop = terminal_value
                else: # Black's turn
                    value_for_backprop = -terminal_value

                current_node.backpropagate(value_for_backprop, release_virtual_loss=True)
                completed_sims += 1
                release_vl_for_this_path = False
            else: # Leaf needs expansion
                pending_evaluations.append((current_node, sim_board.copy()))
                release_vl_for_this_path = False # VL handled by batch processing

        # Process batch if needed
        process_now = len(pending_evaluations) >= inference_batch_size or \
                      (completed_sims + len(pending_evaluations) >= num_simulations and pending_evaluations)

        if process_now:
            # (Batch processing logic remains the same as before)
            states_to_evaluate = [item[1] for item in pending_evaluations]
            nodes_in_batch = [item[0] for item in pending_evaluations]
            try:
                batch_tensors = torch.stack([board_to_tensor_torch(b, device) for b in states_to_evaluate])
                batch_tensors = batch_tensors.contiguous().float().to(device)
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    policy_logits_batch, value_batch = network(batch_tensors)
                    policy_probs_batch_gpu = torch.softmax(policy_logits_batch, dim=1)
                policy_batch_cpu = policy_probs_batch_gpu.cpu().numpy()
                value_batch_cpu = value_batch.cpu().numpy().squeeze(-1)

                for i, node_to_expand in enumerate(nodes_in_batch):
                    expansion_board_state = states_to_evaluate[i]
                    if not node_to_expand.is_expanded():
                         node_to_expand.expand(expansion_board_state, policy_batch_cpu[i])
                    # NN value is from perspective of player in expansion_board_state
                    nn_value_perspective = value_batch_cpu[i]
                    node_to_expand.backpropagate(nn_value_perspective, release_virtual_loss=True)
                    completed_sims += 1
            except Exception as e: # Handle batch errors
                print(f"!!! Batch Inference/Expansion Exc: {type(e).__name__}: {e}")
                fen_list_str = "Could not retrieve FENs"
                try:
                    max_fen_print = 3; fen_list = [b.fen() for b in states_to_evaluate[:max_fen_print]]; fen_list_str = "; ".join(fen_list);
                    if len(states_to_evaluate) > max_fen_print: fen_list_str += "..."
                except Exception as fen_e: print(f"  (Error getting FENs for debug: {fen_e})")
                print(f"    ERROR FENs (Batch Eval): {fen_list_str}")
                failed_count = len(pending_evaluations); print(f"    Failed batch size {failed_count}. Backprop 0 & release VL.")
                for node_to_fail in nodes_in_batch: node_to_fail.backpropagate(0.0, release_virtual_loss=True); completed_sims += 1
            pending_evaluations.clear()

        if completed_sims >= num_simulations: break # Exit main loop

    # --- Simulation phase finished ---

    if not root_node.children: # Handle no children case
        print(f"Warning: Root node has no children after {num_simulations} simulations! FEN:", root_board.fen())
        fallback_move = root_legal_moves[0] if root_legal_moves else None
        return (fallback_move, {}) if return_visit_distribution else fallback_move

    final_legal_moves_set = {m for m in root_board.legal_moves}
    valid_children = {move: child for move, child in root_node.children.items() if move in final_legal_moves_set}

    if not valid_children: # Handle no valid children case
        print(f"Warning: Root node children moves are all illegal/invalid now. FEN:", root_board.fen())
        fallback_move = root_legal_moves[0] if root_legal_moves else None
        return (fallback_move, {}) if return_visit_distribution else fallback_move

    if DETAILED_MCTS_LOGGING: # Log final stats
        print(f"\n--- MCTS Final Stats: FEN: {root_board.fen()} | Turn: {'W' if root_board.turn else 'B'} | Total Root Visits: {root_node.visits} ---")
        move_stats = []
        parent_total_visits = max(1, root_node.visits); parent_sqrt_visits = math.sqrt(parent_total_visits)
        for move, child in valid_children.items():
            N = child.visits; Q_child = child.value(); Q_parent = -Q_child; P = child.prior
            if N == 0: U = c_puct * P * parent_sqrt_visits
            else: U = c_puct * P * parent_sqrt_visits / (1 + N)
            PUCT_final = Q_parent + U
            move_stats.append({"Move": move.uci(), "N": N, "Q_parent": Q_parent, "P": P, "U_final": U, "PUCT_final": PUCT_final})
        move_stats.sort(key=lambda x: x["N"], reverse=True)
        print(f"  Top {LOG_TOP_N} Moves by Visits:"); print(f"  {'Move':<10} {'N':<8} {'Q(Parent)':<12} {'P(Prior)':<12} {'U(Final)':<12} {'PUCT(Final)':<12}"); print("-" * 70)
        for i, stats in enumerate(move_stats):
            if i >= LOG_TOP_N: break
            print(f"  {stats['Move']:<10} {stats['N']:<8} {stats['Q_parent']:<12.4f} {stats['P']:<12.4f} {stats['U_final']:<12.4f} {stats['PUCT_final']:<12.4f}")
        print("--- End Final Stats ---")

    # Select move (random tie-breaking among most visited)
    max_visits = -1; top_moves_with_max_visits = []
    for child in valid_children.values(): max_visits = max(max_visits, child.visits)
    if max_visits >= 0: top_moves_with_max_visits = [m for m, c in valid_children.items() if c.visits == max_visits]

    best_move = None
    if len(top_moves_with_max_visits) == 1: best_move = top_moves_with_max_visits[0]
    elif top_moves_with_max_visits: best_move = random.choice(top_moves_with_max_visits)
    else: # Fallback if all visits are 0
         print(f"Warning: No moves found with max_visits >= 0. FEN: {root_board.fen()}. Selecting first valid child.")
         if valid_children: best_move = next(iter(valid_children.keys()))
         else: best_move = root_legal_moves[0] if root_legal_moves else None

    # Return results
    if return_visit_distribution:
        visit_distribution = {move: child.visits for move, child in valid_children.items()}
        return best_move, visit_distribution
    else:
        return best_move