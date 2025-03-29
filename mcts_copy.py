#mcts_copy.py
import chess
import math
import numpy as np
import torch
from board_encoder import board_to_tensor  # our function to convert board to tensor
from utils import move_to_index

def evaluate_terminal(board):
    """
    Simple terminal evaluation:
    +1 for a win for the current player,
    -1 for a loss,
    0 for a draw.
    """
    result = board.result()
    if result == "1-0":
        return 1
    elif result == "0-1":
        return -1
    else:
        return 0

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board              # Current chess.Board state.
        self.parent = parent            # Parent node in the tree.
        self.move = move                # Move that led to this node.
        self.children = {}              # Dict of moves to child nodes.
        self.visits = 0                 # How many times this node was visited.
        self.value_sum = 0.0            # Total value of this node.
        self.prior = 0.0                # Prior probability (from the NN) for this move.
        
    def value(self):
        # Average value of this node.
        return self.value_sum / self.visits if self.visits > 0 else 0

def uct_score(child, parent_visits, c_puct=1.0):
    """
    UCT (Upper Confidence bound for Trees) score to balance exploration and exploitation.
    """
    Q = child.value()
    U = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
    return Q + U

def mcts(root, network, num_simulations=100, c_puct=1.0, return_visit_distribution=False, device='cpu'):
    """
    Run MCTS from the root node using the neural network for evaluation.
    
    Parameters:
      root: MCTSNode corresponding to the current board position.
      network: Your trained neural network (or untrained at the start).
      num_simulations: Number of simulations to run.
      c_puct: Constant to balance exploration vs. exploitation.
      return_visit_distribution: If True, return a dictionary of visit counts for root moves.
      
    Returns:
        If return_visit_distribution is False:
          The move from the root node with the highest visit count.
        Else:
          A tuple (best_move, visit_distribution) where visit_distribution is a dict
          mapping each legal move (from the root) to its visit count.
    """
    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # 1. Selection: Traverse down the tree until a leaf node.
        while node.children:
            # Choose the child with the highest UCT score.
            node = max(node.children.values(), key=lambda n: uct_score(n, node.visits, c_puct))
            search_path.append(node)
        
        # 2. Evaluation / Expansion:
        # If the node is terminal, evaluate it directly.
        if node.board.is_game_over():
            reward = evaluate_terminal(node.board)
        else:
            with torch.no_grad():
                # Convert board to tensor, then use the network for evaluation.
                board_tensor = board_to_tensor(node.board)  # Shape: (12, 8, 8)
                board_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension.
                policy_logits, value = network(board_tensor)
                # Convert raw scores to probabilities.
                policy = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]
                reward = value.item()
                
                # Expand node: create children for each legal move.
                for move in node.board.legal_moves:
                    new_board = node.board.copy()
                    new_board.push(move)
                    child_node = MCTSNode(new_board, parent=node, move=move)
                    index = move_to_index(move)  # Map move to an index.
                    child_node.prior = policy[index]
                    node.children[move] = child_node

        # 3. Backpropagation: Update the nodes along the search path.
        for node in reversed(search_path):
            node.visits += 1
            node.value_sum += reward
            # Flip the reward for the opponent.
            reward = -reward

    # After simulations, select the move with the most visits.
    best_child = max(root.children.values(), key=lambda n: n.visits)
    best_move = best_child.move
    
    if return_visit_distribution:
        # Create a dictionary mapping moves to their visit counts.
        visit_distribution = {move: child.visits for move, child in root.children.items()}
        return best_move, visit_distribution

    return best_move
