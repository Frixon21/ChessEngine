# mcts_batch.py
import chess
import math
import numpy as np
import torch

from board_encoder import board_to_tensor_torch # Use the torch version for efficiency
from utils import move_to_index


# --- Constants ---
# Set this to True to enable the detailed printing when a repeating move is chosen
DEBUG_PRINT_REPETITION_DETAILS = False
DEBUG_PRINT_TOP_N_MOVES = 8 # How many top moves to show stats for
VIRTUAL_LOSS_VALUE = 1.0

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
        self.virtual_loss_count = 0

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
                    if 0 <= move_idx < len(policy_probs):
                         prior = policy_probs[move_idx]
                         self.children[move] = MCTSNode(parent=self, move=move, prior=prior)
                    else: print(f"Error expanding: move_to_index({move})={move_idx} out of bounds ({len(policy_probs)}). FEN: {board.fen()}")
                except Exception as e: print(f"Error creating child {move}: {e}")

    def select_child(self, c_puct: float, parent_N_eff: int) -> 'MCTSNode':
        """
        Selects the child node with the highest PUCT score,
        incorporating virtual loss. Needs effective parent visit count.
        """
        best_score = -float('inf')
        best_child = None
        parent_sqrt_N_eff = math.sqrt(max(1, parent_N_eff))

        # --- UCT Calculation ---
        # Q(s,a): Average value for the child (action) from the child's perspective.
        #         We use -child.value() because the parent wants to maximize its own value,
        #         which is the negative of the child's value.
        # U(s,a): Exploration bonus: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        #         P(s,a): Prior probability of the action (child.prior)
        #         N(s): Visit count of the parent (self.visits)
        #         N(s,a): Visit count of the child (child.visits)

        for child in self.children.values():
            N_real = child.visits
            VLC = child.virtual_loss_count
            N_eff = N_real + VLC

            # Calculate effective Q value (penalized by virtual losses)
            W_eff = child.value_sum - (VLC * VIRTUAL_LOSS_VALUE)
            Q_eff = W_eff / N_eff if N_eff > 0 else 0.0

            # Calculate exploration bonus using effective visits
            prior_term = child.prior if child.prior > 0 else 1e-6
            U_eff = c_puct * prior_term * parent_sqrt_N_eff / (1 + N_eff)

            # Score from parent's perspective (Maximize -Q + U)
            score = -Q_eff + U_eff

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def backpropagate(self, value: float, release_virtual_loss: bool):
        """
        Backpropagates the simulation result up the tree.
        If release_virtual_loss is True, decrements virtual loss counts.
        """
        node = self
        # The value needs to be flipped at each step as we go up,
        # representing the value from the perspective of the player whose turn it was at that node.
        current_value = value
        while node is not None:
            # Decrement virtual loss if this path is releasing it
            if release_virtual_loss:
                if node.virtual_loss_count > 0:
                    node.virtual_loss_count -= 1
                # else: # Optional: Warn if trying to decrement below zero
                #     if node.visits > 0: # Don't warn for root before first real visit?
                #          print(f"Warning: Tried to decrement VL below zero for node {node.move}")

            # Update real statistics
            node.visits += 1
            node.value_sum += current_value

            # Prepare for parent
            node = node.parent
            current_value *= -1.0 # Flip perspective

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
        print(f"    ERROR FEN (Initial Eval): {root_board.fen()}")
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

    while completed_sims < num_simulations:
        # 1. Start Selection Phase from Root
        current_node = root_node
        # Create a temporary board copy for this simulation path
        # Important: Use a copy that reflects the state *before* the MCTS simulation starts
        sim_board = root_board.copy()
        path_taken = [current_node]
        
         # <<< Increment VL for root at start of path >>>
        current_node.virtual_loss_count += 1
        release_vl_for_this_path = True # Assume we need to release unless handled otherwise

        while current_node.is_expanded():
            # Check for terminal state along the path *before* selecting child
            # This uses the sim_board which reflects the state *at current_node*
            terminal_value = evaluate_terminal(sim_board)
            if terminal_value is not None:
                # Found terminal state during selection -> backpropagate and end this sim path
                current_node.backpropagate(terminal_value, release_virtual_loss=True)
                completed_sims += 1
                current_node = None # Signal that this path is done
                release_vl_for_this_path = False # VL already released by backprop
                break # Exit inner while loop (selection phase)
            
            # Calculate effective parent visits for select_child
            parent_N_eff = current_node.visits + current_node.virtual_loss_count
            best_child = current_node.select_child(c_puct, parent_N_eff) # Pass effective N

            # Select best child using UCT
            if best_child is None:
                 print(f"Warning: select_child returned None despite node being expanded. FEN: {sim_board.fen()}")
                 # This might happen if all children have illegal moves somehow (should be rare)
                 # Treat as if we hit a leaf node that needs eval? Or backprop draw?
                 # Backpropagating draw might be safer.
                 current_node.backpropagate(0.0, release_virtual_loss=True) # Backpropagate draw value
                 completed_sims += 1
                 current_node = None # Signal path done
                 release_vl_for_this_path = False
                 break # Exit inner while loop
             
            # <<< Increment VL for chosen child before descending >>>
            best_child.virtual_loss_count += 1
            path_taken.append(best_child)

            # Move down the tree
            try:
                sim_board.push(best_child.move) # Update the simulation board
                current_node = best_child
            except Exception as e:
                 print(f"!!! Error pushing move {best_child.move.uci()} during selection: {e}")
                 print(f"    Board FEN before push: {sim_board.fen()}")
                 # Treat as sim failure? Backpropagate 0?
                 dummy_leaf = path_taken[-1] # The node that failed to push
                 dummy_leaf.backpropagate(0.0, release_virtual_loss=True) # Backpropagate draw value
                 completed_sims += 1
                 current_node = None # Signal path done
                 release_vl_for_this_path = False # VL released
                 break # Exit inner while loop

        # If loop finished normally (didn't break early due to terminal/error)
        if current_node is not None:
            # Reached a leaf node (not expanded yet) or hit terminal state *exactly* at leaf
            terminal_value = evaluate_terminal(sim_board)
            if terminal_value is not None:
                # Leaf node is actually terminal -> backpropagate
                current_node.backpropagate(terminal_value, release_virtual_loss=True)
                completed_sims += 1
                release_vl_for_this_path = False # VL released
            else:
                # Leaf node needs expansion -> add to batch queue
                # Pass the node and a *copy* of the board state *at that node*
                pending_evaluations.append((current_node, sim_board.copy()))
                release_vl_for_this_path = False 
                
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
            nodes_in_batch = [item[0] for item in pending_evaluations] # Keep track of nodes        

            # Run network inference
            try:
                # Convert boards to tensors efficiently
                batch_tensors = torch.stack([board_to_tensor_torch(b, device) for b in states_to_evaluate])
                
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    policy_logits_batch, value_batch = network(batch_tensors)
                    policy_probs_batch_gpu = torch.softmax(policy_logits_batch, dim=1)

                policy_batch_cpu = policy_probs_batch_gpu.cpu().numpy()
                value_batch_cpu = value_batch.cpu().numpy().squeeze(-1) # Ensure shape (batch_size,)

                # Expand nodes and backpropagate results, releasing VL
                for i, node_to_expand in enumerate(nodes_in_batch):
                    if not node_to_expand.is_expanded():
                        expansion_board_state = states_to_evaluate[i]
                        node_to_expand.expand(expansion_board_state, policy_batch_cpu[i])

                    # <<< Backpropagate NN value AND release virtual loss >>>
                    node_to_expand.backpropagate(value_batch_cpu[i], release_virtual_loss=True)
                    completed_sims += 1

            except Exception as e:
                print(f"!!! Batch Exc: {e}")
                # <<< Add FEN Print (needs care - might error on multiple boards) >>>
                # Print FENs of boards in the *failed* batch if possible
                fen_list_str = "Could not retrieve FENs"
                try:
                    # Only print first few FENs to avoid spamming
                    max_fen_print = 5
                    fen_list = [b.fen() for b in states_to_evaluate[:max_fen_print]]
                    fen_list_str = "; ".join(fen_list)
                    if len(states_to_evaluate) > max_fen_print:
                        fen_list_str += "..."
                except Exception as fen_e:
                    print(f"  (Error getting FENs for debug: {fen_e})")
                print(f"    ERROR FENs (Batch Eval): {fen_list_str}")
                # <<< End Add FEN Print >>>
                print(f"!!! Batch Exc: {e}"); failed_count = len(pending_evaluations)
                print(f"    Failed batch {failed_count}. Backprop 0 & release VL.")
                # <<< Release VL even on error >>>
                for node_to_fail in nodes_in_batch:
                     # Backpropagate 0 and release VL for the failed path
                     node_to_fail.backpropagate(0.0, release_virtual_loss=True)
                     completed_sims += 1 # Still count sim attempt
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
    # best_move = max(valid_children, key=lambda move: valid_children[move].visits)
    
    
    # --- More Robust Move Selection ---
    # 1. Find the maximum visit count
    max_visits = -1
    for child in valid_children.values():
        if child.visits > max_visits:
            max_visits = child.visits

    # 2. Collect all moves that achieved the maximum visit count
    top_moves_with_max_visits = []
    for move, child in valid_children.items():
        if child.visits == max_visits:
            top_moves_with_max_visits.append(move)

    # 3. Choose the best move among the ties
    if len(top_moves_with_max_visits) == 1:
        # No tie, the choice is clear
        best_move = top_moves_with_max_visits[0]
    elif top_moves_with_max_visits:
        # --- Tie-breaking ---
        
        # Option A: Break ties using Q-value (choose move with highest Q among ties)
        # best_move = max(top_moves_with_max_visits, key=lambda move: valid_children[move].value()) # Note: Use child.value() directly

        # Option B: Randomly choose among tied top moves (simple, promotes diversity)
        import random
        best_move = random.choice(top_moves_with_max_visits)

        # Option C: Keep original behaviour (first encountered) - This is what happens implicitly now
        # best_move = top_moves_with_max_visits[0] # (Equivalent to original max())
        
        
    
      # ======================================================================== #
    # <<< INSERT DEBUGGING CHECKPOINT HERE >>>                                 #
    # This block runs *after* all simulations are complete and the             #
    # best_move (by visits) has been determined, but *before* returning.       #
    # ======================================================================== #
    if DEBUG_PRINT_REPETITION_DETAILS:
        move_stats = []
        parent_total_visits = max(1, root_node.visits) # Avoid division by zero
        parent_sqrt_visits = math.sqrt(parent_total_visits)

        for move, child in valid_children.items():
            is_repeating_draw = False
            try:
                board_copy = root_board.copy()
                board_copy.push(move)
                is_repeating_draw = board_copy.can_claim_draw() # Check if *this specific move* enables a draw claim
            except Exception as e:
                print(f"Debug Check Error: Pushing {move.uci()} failed: {e}")

            Q = -child.value() # Value from parent's perspective
            N = child.visits
            P = child.prior
            if N == 0: U = c_puct * P * parent_sqrt_visits
            else: U = c_puct * P * parent_sqrt_visits / (1 + N)
            PUCT = Q + U

            move_stats.append({
                "Move": move.uci(), "N": N, "Q": Q, "P": P, "U": U, "PUCT": PUCT, "Repeats?": is_repeating_draw
            })

        move_stats.sort(key=lambda x: x["N"], reverse=True)

        chosen_move_repeats = False
        for stats in move_stats:
            if stats["Move"] == best_move.uci():
                chosen_move_repeats = stats["Repeats?"]
                break

        # --- Print only if the chosen move leads to repetition draw ---
        if chosen_move_repeats:
            print(f"\n--- MCTS DEBUG (Repetition Choice) | FEN: {root_board.fen()} ---")
            print(f"Chosen Move: {best_move.uci()} (Leads to drawable repetition)")
            print(f"Root Node Visits: {root_node.visits}")
            print("-" * 80)
            print(f"{'Move':<10} {'N':<8} {'Q':<10} {'P':<10} {'U':<10} {'PUCT':<10} {'Repeats?':<10}")
            print("-" * 80)
            for i, stats in enumerate(move_stats):
                if i >= DEBUG_PRINT_TOP_N_MOVES: break
                print(f"{stats['Move']:<10} {stats['N']:<8} {stats['Q']:<10.4f} {stats['P']:<10.4f} {stats['U']:<10.4f} {stats['PUCT']:<10.4f} {stats['Repeats?']:<10}")
            print("-" * 80)
    # <<< END DEBUGGING BLOCK >>>
    # ======================================================================== #
    

    if return_visit_distribution:
        # Return distribution based only on valid children considered for best_move
        visit_distribution = {move: child.visits for move, child in valid_children.items()}
        return best_move, visit_distribution
    else:
        return best_move