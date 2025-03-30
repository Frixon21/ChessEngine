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

TOTAL_PLANES = 22 # 12 pieces + 2 reps + 1 turn + 1 ply + 4 castling + 1 fifty + 1 ep
# Scale ply count to roughly fit in [0, 1] range, assuming max ~100 moves (200 ply)
PLY_SCALE_FACTOR = 1.0 / 200.0
# 50-move rule counter maxes at 100 (or 150 for 75-move rule)
FIFTY_MOVE_SCALE_FACTOR = 1.0 / 100.0

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Converts a python-chess board into a tensor with shape (TOTAL_PLANES, 8, 8).
    Includes piece positions, repetitions, turn, ply count, castling,
    50-move count, and en passant square.
    """
    
    tensor = np.zeros((TOTAL_PLANES, 8, 8), dtype=np.float32)
    
    
    # --- Planes 0-11: Piece Positions ---
    for square, piece in board.piece_map().items():
        symbol = piece.symbol()
        channel = piece_to_channel[symbol]
        row = chess.square_rank(square)
        col = chess.square_file(square)
        tensor[channel, row, col] = 1.0
    
    # --- Plane 12: Repetition Count 1 ---
    # If the current position has occurred once before (is a 2nd occurrence)
    if board.is_repetition(count=2):
        tensor[12, :, :] = 1.0

    # --- Plane 13: Repetition Count 2 ---
    # If the current position has occurred twice before (is a 3rd occurrence)
    if board.is_repetition(count=3):
        tensor[13, :, :] = 1.0
    
    # --- Plane 14: Side to Move ---
    if board.turn == chess.WHITE:
        tensor[14, :, :] = 1.0
    # else: # Implicitly 0.0 for Black
    
    # --- Plane 15: Total Ply Count ---
    # Scale the ply count
    scaled_ply = min(1.0, board.ply() * PLY_SCALE_FACTOR) # Cap at 1.0
    tensor[15, :, :] = scaled_ply
    
    # --- Planes 16-19: Castling Rights ---
    if board.has_kingside_castling_rights(chess.WHITE): tensor[16, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[17, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): tensor[18, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[19, :, :] = 1.0
    
    # --- Plane 20: No-Progress Count (50-Move Rule Counter) ---
    scaled_fifty = min(1.0, board.halfmove_clock * FIFTY_MOVE_SCALE_FACTOR) # Cap at 1.0
    tensor[20, :, :] = scaled_fifty

    # --- Plane 21: En Passant Square ---
    if board.ep_square is not None:
        row = chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        tensor[21, row, col] = 1.0 # Mark only the specific square
            
    return tensor


def board_to_tensor_torch(board: chess.Board, device="cpu") -> torch.Tensor:
    """
    Efficiently converts a chess.Board into a torch.Tensor on the given device.
    Includes all 22 feature planes.
    Output shape is (TOTAL_PLANES, 8, 8) with dtype float32.
    """
     # Use numpy array first for easier indexing
    arr = np.zeros((TOTAL_PLANES, 8, 8), dtype=np.float32)

    # --- Planes 0-11: Piece Positions ---
    for square, piece in board.piece_map().items():
        channel = piece_to_channel[piece.symbol()]
        row = chess.square_rank(square)
        col = chess.square_file(square)
        arr[channel, row, col] = 1.0

    # --- Plane 12: Repetition Count 1 ---
    if board.is_repetition(count=2): arr[12, :, :] = 1.0
    # --- Plane 13: Repetition Count 2 ---
    if board.is_repetition(count=3): arr[13, :, :] = 1.0
    # --- Plane 14: Side to Move ---
    if board.turn == chess.WHITE: arr[14, :, :] = 1.0
    # --- Plane 15: Total Ply Count ---
    scaled_ply = min(1.0, board.ply() * PLY_SCALE_FACTOR); arr[15, :, :] = scaled_ply
    # --- Planes 16-19: Castling Rights ---
    if board.has_kingside_castling_rights(chess.WHITE): arr[16, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): arr[17, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): arr[18, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): arr[19, :, :] = 1.0
    # --- Plane 20: No-Progress Count ---
    scaled_fifty = min(1.0, board.halfmove_clock * FIFTY_MOVE_SCALE_FACTOR); arr[20, :, :] = scaled_fifty
    # --- Plane 21: En Passant Square ---
    if board.ep_square is not None:
        row = chess.square_rank(board.ep_square); col = chess.square_file(board.ep_square); arr[21, row, col] = 1.0

    # Convert the final numpy array to a tensor
    tensor = torch.from_numpy(arr)
    return tensor.to(device, non_blocking=True)