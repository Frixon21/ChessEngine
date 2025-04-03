import chess

def move_to_index(move: chess.Move) -> int:
    """
    Maps a chess.Move object to an index in the fixed move space.
    
    We use a simple encoding scheme:
    
    - For non-promotion moves:
        index = from_square * 64 + to_square.
        This yields 64 * 64 = 4096 possible indices.
    
    - For promotion moves:
        We reserve indices starting at 4096.
        index = 4096 + from_square * 4 + promotion_offset,
        where promotion_offset is assigned as follows:
            Knight: 0, Bishop: 1, Rook: 2, Queen: 3.
        This gives 64 * 4 = 256 indices for promotion moves.
    
    In total, this simple scheme covers 4096 + 256 = 4352 moves.
    
    Parameters:
      move: a chess.Move object.
      
    Returns:
      An integer index between 0 and (at least) 4351.
    """
    if move.promotion:
        # Mapping for promotion moves.
        promo_map = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3}
        promotion_offset = promo_map.get(move.promotion, 0)
        return 4096 + move.from_square * 4 + promotion_offset
    else:
        # For non-promotion moves.
        return move.from_square * 64 + move.to_square


def lightweight_board_copy(board: chess.Board) -> chess.Board:
    new_board = chess.Board()  # start from the standard initial position
    for move in board.move_stack:
         new_board.push(move)
    new_board.halfmove_clock = board.halfmove_clock
    new_board.fullmove_number = board.fullmove_number
    return new_board
