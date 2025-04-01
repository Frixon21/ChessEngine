import chess
import chess.pgn
import numpy as np
from tqdm import tqdm
import time
import os
from board_encoder import board_to_tensor


def parse_pgn_and_extract_positions(pgn_file_path, max_games=None, sample_every_n_moves=1):
    """
    Parses a PGN file, extracts board positions, and generates tensor representations.

    Args:
        pgn_file_path (str): Path to the PGN file.
        max_games (int, optional): Maximum number of games to process from the file.
                                   Defaults to None (process all games).
        sample_every_n_moves (int, optional): Extract position every N moves.
                                             Defaults to 1 (extract every position).

    Returns:
        list: A list of tuples [(chess.Board object, board_tensor_np), ...],
              or None if the file cannot be opened.
    """
    if not os.path.exists(pgn_file_path):
        print(f"Error: PGN file not found at {pgn_file_path}")
        return None

    raw_positions_data = []
    games_processed = 0

    print(f"Opening PGN file: {pgn_file_path}")
    try:
        with open(pgn_file_path, 'r', encoding='utf-8') as pgn_file:
            # Use tqdm to show progress based on estimated number of games if possible,
            # otherwise just show iterations. Estimating games accurately is hard.
            game_reader = chess.pgn.read_game(pgn_file)
            pbar = tqdm(desc="Processing Games", unit=" game")

            while True:
                if max_games is not None and games_processed >= max_games:
                    print(f"\nReached max_games limit: {max_games}")
                    break

                game = None
                try:
                    # Read the next game header and moves
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break # End of file
                except ValueError as ve:
                    print(f"\nWarning: Skipping game due to ValueError during read: {ve}")
                    # Try to find the next game header or break if it's persistent
                    # This basic skip might not always work perfectly for badly formed PGNs
                    continue
                except Exception as e:
                    print(f"\nWarning: Skipping game due to unexpected error during read: {type(e).__name__} - {e}")
                    continue # Skip to next game

                pbar.update(1)
                games_processed += 1

                # --- Process Game ---
                board = game.board() # Get starting position from headers (handles FEN)
                move_counter = 0
                try:
                    # Iterate through moves in the main line
                    for node in game.mainline(): # Use mainline() iterator
                        move = node.move
                        if move is None: continue # Skip null moves if any

                        # --- Extract position BEFORE the move ---
                        if move_counter % sample_every_n_moves == 0:
                            board_obj = board.copy()
                            # Generate tensor (ensure board_to_tensor handles perspective)
                            board_tensor = board_to_tensor(board_obj)
                            raw_positions_data.append((board_obj, board_tensor))

                        # --- Make the move ---
                        board.push(move)
                        move_counter += 1

                except Exception as e:
                     print(f"\nError processing moves in game {games_processed}: {type(e).__name__} - {e}")
                     # Continue to the next game

            pbar.close()

    except FileNotFoundError:
        print(f"Error: PGN file not found at {pgn_file_path}")
        return None
    except Exception as e:
        print(f"Error opening or reading PGN file {pgn_file_path}: {e}")
        return None

    print(f"\nFinished processing. Extracted {len(raw_positions_data)} positions from {games_processed} games.")
    return raw_positions_data

# --- Example Usage ---
if __name__ == "__main__":
    # IMPORTANT: Set the correct path to your PGN file
    PGN_FILE = "Games/1.pgn"
    # Optional: Limit number of games to process for testing
    MAX_GAMES_TO_PROCESS = 10000 # Set to None to process all
    # Optional: Sample positions (1 = every position, 5 = every 5th position)
    SAMPLE_RATE = 1

    start_time = time.time()
    extracted_data = parse_pgn_and_extract_positions(
        PGN_FILE,
        max_games=MAX_GAMES_TO_PROCESS,
        sample_every_n_moves=SAMPLE_RATE
    )
    end_time = time.time()

    if extracted_data is not None:
        print(f"Data extraction took {end_time - start_time:.2f} seconds.")
        # Now you can pass extracted_data to your Stockfish processor function
        # Example:
        # STOCKFISH_PATH = "..."
        # stockfish_samples = generate_stockfish_targets(extracted_data, STOCKFISH_PATH)
        # if stockfish_samples:
        #     # Proceed to training...
        #     pass
        print("Example: First 5 extracted data points (Board object, Tensor shape):")
        for i, (board_obj, board_tensor) in enumerate(extracted_data[:5]):
            print(f"  {i+1}: {board_obj.fen()} | Tensor shape: {board_tensor.shape}")
    else:
        print("Data extraction failed.")
