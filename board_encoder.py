import chess
import numpy as np
import torch

# --- Constants ---
# Piece types (order matters for flipping)
P, N, B, R, Q, K = range(6)
PIECE_TYPES = [P, N, B, R, Q, K]
NUM_PIECE_TYPES = 6

# Plane indices (keeps code readable)
# 0-5: Friendly pieces (P, N, B, R, Q, K)
# 6-11: Opponent pieces (P, N, B, R, Q, K)
F_PAWN, F_KNIGHT, F_BISHOP, F_ROOK, F_QUEEN, F_KING = range(0, 6)
O_PAWN, O_KNIGHT, O_BISHOP, O_ROOK, O_QUEEN, O_KING = range(6, 12)

# 12: Repetition count 1
# 13: Repetition count 2
REP_1 = 12
REP_2 = 13

# 14: Side to Move (Always 1.0 from current player's perspective)
TURN = 14

# 15: Total Ply Count (Scaled)
PLY = 15

# 16-19: Castling Rights (Friendly K, Q | Opponent K, Q)
F_CASTLE_K, F_CASTLE_Q, O_CASTLE_K, O_CASTLE_Q = range(16, 20)

# 20: No-Progress Count (50-Move Rule Counter) (Scaled)
FIFTY_MOVE = 20

# 21: En Passant Square (Relative to current player)
EP_SQUARE = 21

TOTAL_PLANES = 22 # 6 friendly + 6 opponent + 2 reps + 1 turn + 1 ply + 4 castling + 1 fifty + 1 ep

# Scale ply count to roughly fit in [0, 1] range, assuming max ~100 moves (200 ply)
PLY_SCALE_FACTOR = 1.0 / 200.0
# 50-move rule counter maxes at 100 (or 150 for 75-move rule)
FIFTY_MOVE_SCALE_FACTOR = 1.0 / 100.0

# --- Helper ---
def _flip_rank(rank: int) -> int:
    """Flips the rank index (0 becomes 7, 1 becomes 6, etc.)."""
    return 7 - rank

# --- Main Encoding Functions ---

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Converts a python-chess board into a numpy tensor with shape (TOTAL_PLANES, 8, 8),
    representing the state from the perspective of the current player.
    """
    # Determine perspective
    player_color = board.turn # chess.WHITE or chess.BLACK

    # Initialize tensor
    tensor = np.zeros((TOTAL_PLANES, 8, 8), dtype=np.float32)

    # --- Planes 0-11: Piece Positions (Perspective Adjusted) ---
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type - 1 # chess.PAWN = 1 -> 0, etc.
        piece_color = piece.color

        # Determine row and column, flip rank if black to move
        row = chess.square_rank(square)
        col = chess.square_file(square)
        if player_color == chess.BLACK:
            row = _flip_rank(row) # Flip row perspective

        # Determine channel based on piece type and color relative to player
        if piece_color == player_color:
            # Friendly piece
            channel = piece_type # 0-5
        else:
            # Opponent piece
            channel = piece_type + NUM_PIECE_TYPES # 6-11

        tensor[channel, row, col] = 1.0

    # --- Plane 12 & 13: Repetition Counts ---
    # These are independent of perspective
    if board.is_repetition(count=2): tensor[REP_1, :, :] = 1.0
    if board.is_repetition(count=3): tensor[REP_2, :, :] = 1.0

    # --- Plane 14: Side to Move ---
    # Always 1.0 because the representation is *always* from the current player's view
    tensor[TURN, :, :] = 1.0

    # --- Plane 15: Total Ply Count ---
    scaled_ply = min(1.0, board.ply() * PLY_SCALE_FACTOR)
    tensor[PLY, :, :] = scaled_ply

    # --- Planes 16-19: Castling Rights (Perspective Adjusted) ---
    # Friendly rights
    if board.has_kingside_castling_rights(player_color): tensor[F_CASTLE_K, :, :] = 1.0
    if board.has_queenside_castling_rights(player_color): tensor[F_CASTLE_Q, :, :] = 1.0
    # Opponent rights
    opponent_color = not player_color
    if board.has_kingside_castling_rights(opponent_color): tensor[O_CASTLE_K, :, :] = 1.0
    if board.has_queenside_castling_rights(opponent_color): tensor[O_CASTLE_Q, :, :] = 1.0

    # --- Plane 20: No-Progress Count ---
    # Independent of perspective
    scaled_fifty = min(1.0, board.halfmove_clock * FIFTY_MOVE_SCALE_FACTOR)
    tensor[FIFTY_MOVE, :, :] = scaled_fifty

    # --- Plane 21: En Passant Square (Perspective Adjusted) ---
    if board.ep_square is not None:
        row = chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        if player_color == chess.BLACK:
            row = _flip_rank(row) # Flip row perspective
        tensor[EP_SQUARE, row, col] = 1.0

    return tensor


def board_to_tensor_torch(board: chess.Board, device="cpu") -> torch.Tensor:
    """
    Efficiently converts a chess.Board into a torch.Tensor on the given device,
    representing the state from the perspective of the current player.
    Output shape is (TOTAL_PLANES, 8, 8) with dtype float32.
    """
    # Use the numpy function first, then convert
    np_tensor = board_to_tensor(board)

    # Convert the final numpy array to a tensor
    tensor = torch.from_numpy(np_tensor).contiguous()
    return tensor.to(device, non_blocking=True)

