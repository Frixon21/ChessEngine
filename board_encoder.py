import chess
import numpy as np
import torch

# Mapping from piece symbol to channel index
piece_to_channel = {
    'P': 0,  # White Pawn
    'N': 1,  # White Knight
    'B': 2,  # White Bishop
    'R': 3,  # White Rook
    'Q': 4,  # White Queen
    'K': 5,  # White King
    'p': 6,  # Black Pawn
    'n': 7,  # Black Knight
    'b': 8,  # Black Bishop
    'r': 9,  # Black Rook
    'q': 10, # Black Queen
    'k': 11  # Black King
}

def board_to_tensor(board: chess.Board) -> np.ndarray:
   
    """
    Converts a python-chess board into a tensor with shape (12, 8, 8).
    Each channel corresponds to a piece type.
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Iterate over all squares on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            symbol = piece.symbol()
            channel = piece_to_channel[symbol]
            row = chess.square_rank(square)
            col = chess.square_file(square)
            tensor[channel, row, col] = 1.0
            
    return tensor


def board_to_tensor_torch(board, device="cpu"):
    """
    Efficiently converts a chess.Board into a torch.Tensor on the given device.
    Uses board.piece_map() to iterate only over occupied squares.
    Output shape is (12, 8, 8) with dtype float32.
    """
    arr = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        # Get channel from piece symbol.
        channel = piece_to_channel[piece.symbol()]
        row = chess.square_rank(square)
        col = chess.square_file(square)
        arr[channel, row, col] = 1.0
    # Use torch.from_numpy to avoid an extra copy
    tensor = torch.from_numpy(arr)
    return tensor.to(device, non_blocking=True)
