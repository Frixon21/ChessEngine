# mcts_batch.py
import chess
import math
import numpy as np
import torch
import collections
import time # For potential debugging/timeouts

from board_encoder import board_to_tensor_torch # Use the torch version for efficiency
from utils import move_to_index

def evaluate_terminal(board: chess.Board):
    """Checks if the board state is terminal and returns the score from White's perspective."""
    if board.is_checkmate():
        # If Black is checkmated (White wins), return 1.0
        # If White is checkmated (Black wins), return -1.0
        return 1.0 if board.turn == chess.BLACK else -1.0
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

    def value(self) -> float:
        """Returns the average value of this node."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def is_expanded(self) -> bool:
        return self._is_expanded

    def expand(self, board: chess.Board, policy_probs: np.ndarray):
        """Expands the node by creating children for all legal moves."""
        if self._is_expanded: return # Already expanded
        self._is_expanded = True
        current_legal_moves = list(board.legal_moves)
        if not current_legal_moves:
            # This can happen if we reach a terminal state just before expansion
            # Or potentially due to a bug elsewhere. Mark as expanded but with no children.
             # print(f"Warning: Expanding node with no legal moves. FEN: {board.fen()}")
            return

        for move in current_legal_moves:
            if move not in self.children: # Defensive check
                try:
                    move_idx = move_to_index(move)
                    prior = policy_probs[move_idx]
                    self.children[move] = MCTSNode(parent=self, move=move, prior=prior)
                except IndexError:
                     print(f"Error: move_to_index({move}) out of bounds for policy array (size {len(policy_probs)}).")
                     # Skip this move or handle error appropriately
                     continue
                except Exception as e:
                     print(f"Error creating child node for move {move}: {e}")
                     continue # Skip problematic move

    def select_child(self, c_puct: float) -> 'MCTSNode':
        """Selects the child node with the highest UCT score."""
        best_score = -float('inf')
        best_child = None
        parent_sqrt_visits = math.sqrt(self.visits) # Cache sqrt

        # --- UCT Calculation ---
        # Q(s,a): Average value for the child (action) from the child's perspective.
        #         We use -child.value() because the parent wants to maximize its own value,
        #         which is the negative of the child's value.
        # U(s,a): Exploration bonus: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        #         P(s,a): Prior probability of the action (child.prior)
        #         N(s): Visit count of the parent (self.visits)
        #         N(s,a): Visit count of the child (child.visits)

        for child in self.children.values():
            prior_term = child.prior if child.prior > 0 else 1e-6 # Avoid math errors with zero prior
            uct_explore = c_puct * prior_term * parent_sqrt_visits / (1 + child.visits)
            uct_value = -child.value() # Value from parent's perspective
            score = uct_value + uct_explore

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def backpropagate(self, value: float):
        """Backpropagates the simulation result up the tree."""
        node = self
        # The value needs to be flipped at each step as we go up,
        # representing the value from the perspective of the player whose turn it was at that node.
        current_value = value
        while node is not None:
            node.visits += 1
            node.value_sum += current_value
            node = node.parent
            current_value *= -1.0 # Flip perspective for the parent

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
    """
    Runs MCTS simulations using batch inference.

    Args:
        root_board: The starting board state.
        network: The neural network model.
        num_simulations: The total number of simulations to run.
        inference_batch_size: Max number of states to batch for network evaluation.
        device: The torch device ('cuda' or 'cpu').
        c_puct: Exploration constant for UCT.
        dirichlet_alpha: Alpha parameter for Dirichlet noise at the root.
        dirichlet_epsilon: Epsilon parameter (weight) for Dirichlet noise.
        return_visit_distribution: If True, returns (best_move, visit_dict), else just best_move.

    Returns:
        Best move found, and optionally the visit distribution dictionary.
    """
    root_node = MCTSNode()
    num_actions = network.policy_fc.out_features # Get output size from network layer

    # --- Initial Network Call & Dirichlet Noise (Optional) ---
    initial_policy_probs = None
    initial_value = 0.0
    root_legal_moves = list(root_board.legal_moves)

    if not root_legal_moves:
        # Handle case where the game is already over at the root
        print("Warning: MCTS called on a board with no legal moves.")
        if return_visit_distribution:
            return None, {}
        else:
            return None

    try:
        # Single initial evaluation for the root state
        root_tensor = board_to_tensor_torch(root_board, device).unsqueeze(0)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            policy_logits, value_tensor = network(root_tensor)
            policy_probs_gpu = torch.softmax(policy_logits, dim=1)
            initial_value = value_tensor.item() # Value estimate for the root state

        initial_policy_probs = policy_probs_gpu.cpu().numpy()[0]

        # Apply Dirichlet noise if enabled
        if dirichlet_epsilon > 0 and len(root_legal_moves) > 1: # Need >1 move for noise application
            noise = np.random.dirichlet([dirichlet_alpha] * num_actions)
            noisy_policy = (1 - dirichlet_epsilon) * initial_policy_probs + dirichlet_epsilon * noise
            root_node.expand(root_board, noisy_policy)
        else:
            # Expand root using raw policy if noise disabled or not applicable
            root_node.expand(root_board, initial_policy_probs)

        # Backpropagate initial root value estimate (optional, but can slightly help)
        # root_node.visits += 1 # Count this initial eval? Debatable.
        # root_node.value_sum += initial_value

    except Exception as e:
        print(f"!!! Exception during initial network eval/expansion: {type(e).__name__}: {e}")
        print(f"    Root FEN: {root_board.fen()}")
        # Fallback: Expand with uniform priors if network fails initially?
        if not root_node.is_expanded():
            uniform_policy = np.ones(num_actions) / num_actions
            try:
                root_node.expand(root_board, uniform_policy)
                print("    Fallback: Expanded root with uniform policy.")
            except Exception as exp_e:
                 print(f"    Fallback expansion failed: {exp_e}")
                 # Cannot proceed if root cannot be expanded
                 if return_visit_distribution: return None, {}
                 else: return None
        # If already expanded (e.g., noise applied before error), continue cautiously


    # --- Batch Simulation Loop ---
    pending_evaluations: list[tuple[MCTSNode, chess.Board]] = [] # Nodes waiting for NN eval
    completed_sims = 0
    sim_attempts = 0

    while completed_sims < num_simulations:
        # 1. Start Selection Phase from Root
        current_node = root_node
        # Create a temporary board copy for this simulation path
        # Important: Use a copy that reflects the state *before* the MCTS simulation starts
        sim_board = root_board.copy()

        while current_node.is_expanded():
            # Check for terminal state along the path *before* selecting child
            # This uses the sim_board which reflects the state *at current_node*
            terminal_value = evaluate_terminal(sim_board)
            if terminal_value is not None:
                # Found terminal state during selection -> backpropagate and end this sim path
                current_node.backpropagate(terminal_value)
                completed_sims += 1
                current_node = None # Signal that this path is done
                break # Exit inner while loop (selection phase)

            # Select best child using UCT
            best_child = current_node.select_child(c_puct)
            if best_child is None:
                 print(f"Warning: select_child returned None despite node being expanded. FEN: {sim_board.fen()}")
                 # This might happen if all children have illegal moves somehow (should be rare)
                 # Treat as if we hit a leaf node that needs eval? Or backprop draw?
                 # Backpropagating draw might be safer.
                 current_node.backpropagate(0.0) # Backpropagate draw value
                 completed_sims += 1
                 current_node = None # Signal path done
                 break # Exit inner while loop

            # Move down the tree
            try:
                sim_board.push(best_child.move) # Update the simulation board
                current_node = best_child
            except Exception as e:
                 print(f"!!! Error pushing move {best_child.move.uci()} during selection: {e}")
                 print(f"    Board FEN before push: {sim_board.fen()}")
                 # Treat as sim failure? Backpropagate 0?
                 current_node.backpropagate(0.0)
                 completed_sims += 1
                 current_node = None # Signal path done
                 break # Exit inner while loop

        # If loop finished normally (didn't break early due to terminal/error)
        if current_node is not None:
            # Reached a leaf node (not expanded yet) or hit terminal state *exactly* at leaf
            terminal_value = evaluate_terminal(sim_board)
            if terminal_value is not None:
                # Leaf node is actually terminal -> backpropagate
                current_node.backpropagate(terminal_value)
                completed_sims += 1
            else:
                # Leaf node needs expansion -> add to batch queue
                # Pass the node and a *copy* of the board state *at that node*
                pending_evaluations.append((current_node, sim_board.copy()))

        # 2. Process Batch if Full or End of Simulations Reached
        # Check if we need to process the batch now
        # Process if:
        #   - Batch is full
        #   - OR We've finished selection for all simulations needed AND there are pending evals
        process_now = len(pending_evaluations) >= inference_batch_size or \
                      (completed_sims + len(pending_evaluations) >= num_simulations and pending_evaluations)

        if process_now:
            # Prepare batch tensor
            states_to_evaluate = [item[1] for item in pending_evaluations] # Get board objects
            # Convert boards to tensors efficiently
            batch_tensors = torch.stack([board_to_tensor_torch(b, device) for b in states_to_evaluate])

            # Run network inference
            try:
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    policy_logits_batch, value_batch = network(batch_tensors)
                    policy_probs_batch_gpu = torch.softmax(policy_logits_batch, dim=1)

                policy_batch_cpu = policy_probs_batch_gpu.cpu().numpy()
                value_batch_cpu = value_batch.cpu().numpy().squeeze(-1) # Ensure shape (batch_size,)

                # Expand nodes and backpropagate results
                for i, (node_to_expand, _) in enumerate(pending_evaluations):
                    # Ensure the node wasn't somehow expanded concurrently (shouldn't happen in this model)
                    if not node_to_expand.is_expanded():
                        # Use the board state that was saved with the node for expansion context
                        expansion_board_state = states_to_evaluate[i]
                        node_to_expand.expand(expansion_board_state, policy_batch_cpu[i])

                    # Backpropagate the value estimated by the network
                    node_to_expand.backpropagate(value_batch_cpu[i])
                    completed_sims += 1 # Count simulation as completed after backpropagation

            except Exception as e:
                print(f"!!! Exception during batch network eval/backprop: {type(e).__name__}: {e}")
                # How to recover? Maybe backpropagate 0 for all failed evals?
                failed_count = len(pending_evaluations)
                print(f"    Failed to evaluate batch of size {failed_count}. Backpropagating 0 for these.")
                for node_to_fail, _ in pending_evaluations:
                     # Avoid double-counting visits if backprop fails partially
                     if node_to_fail.visits == 0: # Simple check, might not be perfect
                         node_to_fail.backpropagate(0.0) # Backpropagate draw as fallback
                         completed_sims += 1 # Still count as a completed sim attempt
                # Continue to next simulation attempt

            # Clear the processed batch
            pending_evaluations.clear()

        # Safety break if simulations somehow exceed target (shouldn't happen)
        if completed_sims >= num_simulations:
            break

    # --- Simulation phase finished ---

    # --- Select Move Based on Visits ---
    if not root_node.children:
        print("Warning: Root node has no children after simulations! FEN:", root_board.fen())
        # Fallback: Pick first legal move if possible
        fallback_move = root_legal_moves[0] if root_legal_moves else None
        if return_visit_distribution:
            return fallback_move, {}
        else:
            return fallback_move

    # Filter children based on current legal moves (defense against rare state mismatches)
    final_legal_moves_set = {m for m in root_board.legal_moves}
    valid_children = {move: child for move, child in root_node.children.items() if move in final_legal_moves_set}

    if not valid_children:
        print("Warning: Root node children moves are all illegal/invalid now. FEN:", root_board.fen())
        fallback_move = root_legal_moves[0] if root_legal_moves else None
        if return_visit_distribution:
            return fallback_move, {}
        else:
            return fallback_move

    # Find best move among the valid children based on visit count
    # Using child.visits directly is standard (robust selection)
    best_move = max(valid_children, key=lambda move: valid_children[move].visits)

    if return_visit_distribution:
        # Return distribution based only on valid children considered for best_move
        visit_distribution = {move: child.visits for move, child in valid_children.items()}
        return best_move, visit_distribution
    else:
        return best_move