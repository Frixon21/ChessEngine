# mcts.py
import chess
import math
import numpy as np
import torch
from board_encoder import board_to_tensor
from utils import move_to_index

# --- Constants for Dirichlet Noise (can be moved to main config) ---
# DIRICHLET_ALPHA = 0.3  # Controls noise shape (lower = more peaked) - common value for chess
# DIRICHLET_EPSILON = 0.25 # Controls noise amount (0 = none, 1 = only noise) - standard AlphaZero value

def evaluate_terminal(board):
    # Add a check for draw by insufficient material?
    if board.is_insufficient_material(): return 0.0
    result = board.result()
    if result == "1-0": return 1.0
    elif result == "0-1": return -1.0
    elif result != "*": return 0.0 # Draw claimed or detected
    else: return None # Game not over


class MCTSNode:
    def __init__(self, parent=None, move=None):
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0 # This will store the potentially noise-augmented prior

    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0

    def is_expanded(self):
        return bool(self.children)


def uct_score(child, parent_visits, c_puct=1.0):
    Q = child.value()
    # Ensure child.prior is non-negative, add small epsilon if prior is zero to avoid math errors if needed
    # Although noise should prevent zero priors for legal moves if epsilon > 0
    prior_term = child.prior if child.prior > 0 else 1e-6
    U = c_puct * prior_term * math.sqrt(parent_visits) / (1 + child.visits)
    # Use negative Q for parent's selection perspective
    return -Q + U

# --- The Recursive Helper (NO CHANGES NEEDED HERE for noise) ---
def mcts_recursive_search(node, board, network, device, c_puct, root_fen):
    """
    Performs one simulation step recursively using push/pop on a SHARED board.
    Returns the evaluated value from the perspective of the player
    whose turn it was at the state represented by 'node'.
    NO CHANGES HERE - uses priors set by the caller (root) or during its own expansion.
    """

    # 1. Check for terminal state FIRST using the current board state
    terminal_value = evaluate_terminal(board)
    if terminal_value is not None:
        # Backpropagate terminal value
        # Need to ensure visits/value_sum are updated correctly relative to player turn
        # This implementation seems okay: returns value for current player at terminal state
        node.visits += 1 # Visit the terminal node itself
        node.value_sum += terminal_value # Store value relative to player at node
        return terminal_value # Return value for the current player at this state


    # 2. Check if node is expanded
    if not node.is_expanded():
        # 2a. Evaluate leaf node with network and expand
        # NOTE: Priors calculated here are RAW network priors, noise is only applied at the root
        try:
            board_tensor = board_to_tensor(board)
            board_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                policy_logits, value_tensor = network(board_tensor)
                policy_probs_gpu = torch.softmax(policy_logits, dim=1)
                value = value_tensor.item()

            policy_cpu = policy_probs_gpu.cpu().numpy()[0]

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                 terminal_value = evaluate_terminal(board)
                 if terminal_value is None:
                      print(f"Warning: No legal moves but not terminal? FEN: {board.fen()}")
                      # Fallback: treat as a draw? Or raise error? Let's return 0 for safety.
                      terminal_value = 0.0
                      # raise RuntimeError(f"Corrupt state: No legal moves but not terminal. FEN: {board.fen()}")
                 node.visits += 1
                 node.value_sum += terminal_value
                 return terminal_value

            for move in legal_moves:
                move_idx = move_to_index(move)
                child_node = MCTSNode(parent=node, move=move)
                # Use the RAW policy from the network for non-root expansions
                child_node.prior = policy_cpu[move_idx]
                node.children[move] = child_node

            # Backpropagate the network value from this node
            node.visits += 1
            node.value_sum += value # Store value relative to current player
            return value

        except Exception as e:
             print(f"!!! Exception during NN eval/expansion: {type(e).__name__}: {e}")
             print(f"    Board FEN: {board.fen()}")
             raise e

    # 3. Node is already expanded - Select best child and recurse
    else:
        best_child = None
        best_score = -float('inf')
        parent_visits = node.visits # Use current node's visits for UCT calculation

        # Ensure moves being considered are legal *now*
        # This check was previously removed for debugging, let's ensure it's robust
        current_legal_moves_set = {m for m in board.legal_moves}
        valid_children = {m: child for m, child in node.children.items() if m in current_legal_moves_set}

        if not valid_children:
            # If no children correspond to legal moves (e.g., state mismatch or terminal)
            # Re-evaluate terminal state as a safeguard
            terminal_value = evaluate_terminal(board)
            if terminal_value is not None:
                # print(f"Info: No valid children found, node is terminal ({terminal_value}). FEN: {board.fen()}")
                node.visits += 1 # Count visit
                node.value_sum += terminal_value
                return terminal_value
            else:
                # This indicates a potential state mismatch if not terminal
                print(f"!!! Error: Node expanded but no valid children match legal moves, and not terminal.")
                print(f"    Board FEN: {board.fen()}")
                print(f"    Node Children Moves: {[m.uci() for m in node.children.keys()]}")
                print(f"    Board Legal Moves: {[m.uci() for m in current_legal_moves_set]}")
                # Fallback: treat as a draw to avoid crashing simulation?
                node.visits += 1
                node.value_sum += 0.0 # Assign draw value
                return 0.0 # Return draw value
                # Or raise error:
                # raise RuntimeError("State mismatch: Expanded node children don't match board legal moves.")


        for move, child in valid_children.items():
            score = uct_score(child, parent_visits, c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
             # This case should ideally be prevented by the 'if not valid_children' block above
             print(f"!!! Error: No best child selected from valid children. FEN: {board.fen()}")
             # Fallback: Treat as draw
             terminal_value = 0.0
             node.visits += 1
             node.value_sum += terminal_value
             return terminal_value
             # raise RuntimeError("Logical error: No best child found from valid children.")


        # Recurse: Make the move, search deeper, then undo the move
        try:
            board.push(best_child.move)
            value_from_child = mcts_recursive_search(best_child, board, network, device, c_puct, root_fen)
            board.pop()

            # Backpropagate
            node.visits += 1
            node.value_sum += (-value_from_child) # Negate value from child's perspective
            return -value_from_child # Return value for the node above 'node'

        except Exception as e:
             # Handle errors during recursion/push/pop
             print(f"!!! Exception during recursion/push/pop: {type(e).__name__}: {e}")
             print(f"    Board FEN before push: {board.fen()}")
             print(f"    Move attempted: {best_child.move.uci() if best_child else 'None'}")
             print(f"    Node considered: Move {node.move.uci() if node.move else 'ROOT'}")
             # Attempt recovery only if feasible, otherwise re-raise
             # Check if the move is actually on the stack before popping
             if board.move_stack and board.peek() == best_child.move:
                  try:
                      board.pop()
                      print("    Attempted pop recovery.")
                  except Exception as pop_e:
                      print(f"    Pop recovery failed: {pop_e}")
             raise e



# --- The Main MCTS Entry Point ---
def mcts_push_pop_shared(root_board, network, num_simulations=100, c_puct=1.0,
                         return_visit_distribution=False, device='cpu',
                         dirichlet_alpha=0.3, dirichlet_epsilon=0.25): # Add noise parameters
    """
    Run MCTS using push/pop optimization with a SHARED board object.
    Optionally adds Dirichlet noise to root node priors for exploration.
    """
    root_node = MCTSNode()
    num_actions = 4352 # Assuming this is the size of your policy output

    # --- Apply Dirichlet Noise to Root Node Priors ---
    if dirichlet_epsilon > 0:
        try:
            # 1. Get initial policy from network for the root board
            root_tensor = board_to_tensor(root_board)
            root_tensor = torch.tensor(root_tensor, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                policy_logits, _ = network(root_tensor) # Don't need value here
                policy_probs_gpu = torch.softmax(policy_logits, dim=1)

            policy_cpu = policy_probs_gpu.cpu().numpy()[0]

            # 2. Generate Dirichlet noise
            noise = np.random.dirichlet([dirichlet_alpha] * num_actions)

            # 3. Mix policy and noise
            mixed_policy = (1 - dirichlet_epsilon) * policy_cpu + dirichlet_epsilon * noise

            # 4. Create root children and assign MIXED priors
            root_legal_moves = list(root_board.legal_moves)
            if root_legal_moves: # Only expand if there are moves
                for move in root_legal_moves:
                    move_idx = move_to_index(move)
                    child_node = MCTSNode(parent=root_node, move=move)
                    # Use the mixed policy for root children priors
                    child_node.prior = mixed_policy[move_idx]
                    root_node.children[move] = child_node
            # Root node is now considered "expanded" with noise-augmented priors
            # The first recursive call will select among these children using UCT

        except Exception as e:
            print(f"!!! Error applying Dirichlet noise: {type(e).__name__}: {e}")
            # Fallback: Proceed without noise if application fails? Or raise?
            # Let's proceed without noise to avoid crashing entirely.
            root_node.children = {} # Ensure it's not marked as expanded incorrectly

    # If noise is disabled (epsilon=0) or failed, the root node will be expanded
    # during the first simulation that calls mcts_recursive_search on the root,
    # using the raw network priors as before.

    # --- Run Simulations ---
    search_board = root_board.copy()
    initial_fen = root_board.fen()

    for sim_idx in range(num_simulations):
        search_board.reset()
        search_board.set_fen(initial_fen)
        try:
            mcts_recursive_search(root_node, search_board, network, device, c_puct, initial_fen)
        except Exception as e:
            print(f"--- Simulation {sim_idx + 1} Failed: {type(e).__name__}: {e} ---")
            # Maybe break loop if too many simulations fail? For now, continue.

    # --- Select Move Based on Visits ---
    if not root_node.children:
        # This can happen if the root state is terminal or if noise application failed AND no simulations ran/expanded
        print("Warning: Root node has no children after simulations!")
        legal_moves = list(root_board.legal_moves)
        fallback_move = legal_moves[0] if legal_moves else None
        # Ensure we return the correct format based on return_visit_distribution
        if return_visit_distribution:
            return fallback_move, {}
        else:
            return fallback_move

    # Get final legal moves from the original root board state
    final_legal_moves_set = {m for m in root_board.legal_moves}
    # Filter children to only those corresponding to currently legal moves
    valid_children = {move: child for move, child in root_node.children.items() if move in final_legal_moves_set}

    if not valid_children:
        print("Warning: Root node children moves are all illegal/invalid now.")
        legal_moves = list(root_board.legal_moves)
        fallback_move = legal_moves[0] if legal_moves else None
        if return_visit_distribution:
            return fallback_move, {}
        else:
            return fallback_move

    # Find best move among the valid children based on visit count
    best_move = max(valid_children, key=lambda move: valid_children[move].visits)

    if return_visit_distribution:
        # Return distribution based only on valid children considered for best_move
        visit_distribution = {move: child.visits for move, child in valid_children.items()}
        return best_move, visit_distribution
    else:
        return best_move