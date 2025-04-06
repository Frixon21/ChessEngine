"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import chess
from chess.engine import PlayResult, Limit
import chess.gaviota
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE
import logging

import os
import sys
import torch

# Add the parent directory to sys.path so that mcts_batch.py can be imported.
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from mcts_batch import run_simulations_batch
from neural_network import ChessNet

# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)

# Load the tablebase
TABLEBASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tables/3-4-5-Gaviota"))
tb = chess.gaviota.open_tablebase(TABLEBASE_PATH)

def try_tablebase_mate_line(board: chess.Board, tb_obj, max_depth=50):
    """
    Try to find a mate move using the tablebase.
    This function probes the DTM (distance-to-mate) for the current board.
    It then iterates through legal moves and returns the first move that
    decreases the absolute mate distance.
    
    Note: Gaviota's probe_dtm returns a mate distance from the perspective of the side to move.
    For winning positions, a smaller absolute value means a faster mate.
    
    Returns:
        A chess.Move if a mate move is found, else None.
    """
    try:
        dtm = tb_obj.probe_dtm(board)
    except chess.gaviota.TablebaseError:
        return None

    # If mate distance is 0, we're at the terminal position.
    if dtm == 0:
        return None

    # For a winning position for the side to move, a good move should reduce the absolute mate distance.
    for move in board.legal_moves:
        board.push(move)
        try:
            new_dtm = tb_obj.probe_dtm(board)
        except chess.gaviota.TablebaseError:
            board.pop()
            continue
        # Check if the absolute mate distance is reduced.
        # (You might add additional checks, e.g. if new_dtm equals 1 then it's nearly mate.)
        if abs(new_dtm) < abs(dtm):
            board.pop()
            return move
        board.pop()
    return None



class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""



class FrixBot(ExampleEngine):
    """
    A homemade bot using MCTS and a neural network (ChessNet).
    This bot uses run_simulations_batch() to search the position and returns the best move.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripted_model.pt"))
        self.network = torch.jit.load(model_path, map_location=self.device)
        self.network.to(self.device)
        self.network.eval()
        logger.info(f"FrixBot (TorchScript) initialized on device: {self.device}")
        
    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """
        Given a board position, run MCTS with your neural network to choose the best move.
        The method returns a PlayResult containing the chosen move.
        """
        if len(board.piece_map()) <= 5:
            mate_move = try_tablebase_mate_line(board, tb)
            if mate_move is not None:
                logger.info(f"Tablebase mate override: {mate_move.uci()} found for FEN {board.fen()}")
                return PlayResult(mate_move, None)
        num_simulations = 1600  # Adjust as needed.
        inference_batch_size = 32
        c_puct = 1.25
        dirichlet_alpha = 0.3
        dirichlet_epsilon = 0.25

        best_move = run_simulations_batch(
            board,
            self.network,
            num_simulations=num_simulations,
            inference_batch_size=inference_batch_size,
            device=self.device,
            return_visit_distribution=False,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon
        )
        # If no move is found, return a null move.
        if best_move is None:
            best_move = chess.Move.null()
        # Log the chosen move (optional).
        logger.info(f"Search completed, selected move: {best_move.uci()} for FEN: {board.fen()}")
        return PlayResult(best_move, None)