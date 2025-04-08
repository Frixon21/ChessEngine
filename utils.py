import chess
import json
import subprocess
from pgn_parser import parse_pgn_and_extract_positions 
import os
from neural_network import ChessNet
import torch
import csv
from board_encoder import board_to_tensor_torch 
import boto3
import time


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

def load_config(config_file="config.json"):
    """Load config values from a JSON file and return them as a dict."""
    with open(config_file, "r") as f:
        data = json.load(f)
    return data

def update_config(new_values, config_file="config.json"):
    """
    Update the configuration file with new_values.
    
    Parameters:
      new_values (dict): A dictionary of keys and values to update in the config.
      config_file (str): The path to the configuration file.
    """
    # Load the current configuration
    config = load_config(config_file)
    
    # Update the dictionary with new values
    config.update(new_values)
    
    # Write the updated configuration back to the file
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
        
def pull_repo():
    """
    Runs a git pull to update your local repository.
    """
    try:
        print("Running git pull to update PGN files...")
        subprocess.run(["git", "pull"], check=True)
        print("Git pull completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Git pull failed:", e)
        
def push_repo(itteration, model_checkpoint, scripted_model_checkpoint):
    """
    Runs a git push to update your remote repository.
    """
    try:
        print("Running git push to update trained_model...")
        subprocess.run(["git", "add",  model_checkpoint, scripted_model_checkpoint, "saved_games"], check=True)
        commit_message = f"Update model Iteration {itteration}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push"], check=True)
        print("Git push completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Git push failed:", e)
        
def create_initial_models(device, model_checkpoint, scripted_model_checkpoint):
    """Creates and saves initial .pth and .pt models if they don't exist."""
    pth_exists = os.path.exists(model_checkpoint)
    pt_exists = os.path.exists(scripted_model_checkpoint)

    if not pth_exists:
        print(f"No existing model found at {model_checkpoint}. Initializing random model.")
        initial_model = ChessNet().to(device)
        torch.save(initial_model.state_dict(), model_checkpoint)
        print(f"Initial random model saved to {model_checkpoint}")
        pth_exists = True # Mark as created
    else:
         initial_model = ChessNet().to(device) # Need instance even if loading below
         print(f"Found existing model at {model_checkpoint}.")

    if pth_exists and not pt_exists:
        print(f"Scripted model {scripted_model_checkpoint} not found. Creating from {model_checkpoint}...")
        try:
            # Load weights into a model instance
            initial_model.load_state_dict(torch.load(model_checkpoint, map_location=device))
            initial_model.eval()
            # Create example input
            example_input = torch.randn(1, 22, 8, 8, device=device)
            # Trace
            scripted_model = torch.jit.trace(initial_model, example_input)
            # Save
            torch.jit.save(scripted_model, scripted_model_checkpoint)
            print(f"Initial TorchScript model saved to {scripted_model_checkpoint}")
        except Exception as e:
            print(f"Failed to create initial TorchScript model: {e}")
            # Decide how to handle - maybe exit? For now, just warn.
    elif pt_exists:
         print(f"Found existing scripted model at {scripted_model_checkpoint}.")

    del initial_model # Free memory

def load_puzzle_samples(iteration, num_puzzles=1000, csv_path: str = "Games/puzzle_chunks", device: torch.device = torch.device("cpu")):
    """
    Load puzzle positions for the current iteration from a Lichess puzzle CSV,
    parsing the *entire puzzle line* of moves.

    Steps:
      1) We read 'num_puzzles' lines from the CSV, offset by iteration.
      2) For each row, parse the FEN (the position before the opponent's move),
         and the full 'Moves' string (all forced moves in UCI).
      3) Apply the first move to the board (the opponent's move),
         leaving the puzzle solver to move. This is puzzle start #1.
      4) Then for each subsequent move in the puzzle line, push it and record
         the resulting position. That means each step in the puzzle solution
         yields a new position (for both sides).
      5) Return a flat list of (board, board_tensor) for every step in every puzzle.

    Example usage:
        puzzle_positions = load_puzzle_positions_all_moves(
            iteration=5,
            num_puzzles=1000,
            csv_path="lichess_db_puzzle.csv",
            device=torch.device("cuda")
        )
        # Then pass puzzle_positions to generate_stockfish_targets(...)
    """
    puzzle_positions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        
        # Attempt to skip a header if it exists:
        header = next(reader, None)
        if header and "PuzzleId" in header[0]:
            pass  # We skipped the header
        else:
            # Possibly no header or unexpected format:
            # If you realize there's no header, uncomment the next line:
            # f.seek(0)
            # Or handle differently if needed
            pass


        for row in reader:
            
            if len(row) < 3:
                continue # Malformed row
            
            # CSV format:
            # PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
            fen = row[1].strip()
            moves_str = row[2].strip()
            if not fen or not moves_str:
                continue
            
            moves_list = moves_str.split()
            if len(moves_list) < 1:
                continue
            
            board = chess.Board(fen)
            
            # 1) The puzzle FEN is the position *before* the opponent's move.
            #    The first move in 'moves_list' is that opponent move.
            #    We push it so the solver is to move after that.
            first_move_uci = moves_list[0]
            try:
                board.push_uci(first_move_uci)
            except Exception as e:
                print(f"[load_puzzle_positions_all_moves] Invalid move {first_move_uci} for FEN {fen}: {e}")
                continue
            
            # Now the board is at the puzzle's "start" for the solver.
            puzzle_positions.append((board.copy(), board_to_tensor_torch(board, device)))
            
            for next_move_uci in moves_list[1:]:
                try:
                    board.push_uci(next_move_uci)
                except Exception as e:
                    print(f"[load_puzzle_positions_all_moves] Invalid forced move {next_move_uci} after first move: {e}")
                    break  # move on to next puzzle

                puzzle_positions.append((board.copy(), board_to_tensor_torch(board, device)))
                
    return puzzle_positions


    pass

def download_and_delete_s3_objects(local_dir, bucket_name, prefix):
    """
    Downloads all objects under the given S3 prefix to a local directory,
    then deletes them from S3.
    """
    s3 = boto3.client('s3')
    os.makedirs(local_dir, exist_ok=True)
    
    # List all objects under the specified prefix.
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    while 'Contents' not in response:
        print("No objects found in S3 bucket under the prefix., waiting 60s")
        time.sleep(60)
    
    local_files = []
    for obj in response['Contents']:
        key = obj['Key']
        # If the key ends with a slash, assume it is a directory marker and skip it.
        if key.endswith('/'):
            continue
        
        # Create a local filename using only the base name of the key.
        local_file = os.path.join(local_dir, os.path.basename(key))
        print(f"Downloading {key} to {local_file}...")
        try:
            s3.download_file(bucket_name, key, local_file)
        except Exception as e:
            print(f"Error downloading {key}: {e}")
            continue
        
        local_files.append(local_file)
        
        # After a successful download, delete the object from S3.
        print(f"Deleting {key} from S3...")
        try:
            s3.delete_object(Bucket=bucket_name, Key=key)
        except Exception as e:
            print(f"Error deleting {key} from S3: {e}")
        
    return local_files

def process_downloaded_files(bucket_name, prefix):
    local_dir = "downloaded_pgns"
    # Download files from S3
    local_files = download_and_delete_s3_objects(local_dir, bucket_name, prefix)
    
    all_positions = []
    for pgn_file in local_files:
        print(f"Processing file: {pgn_file}")
        positions = parse_pgn_and_extract_positions(pgn_file)
        if positions is not None:
            all_positions.extend(positions)
        else:
            print(f"Data extraction failed for {pgn_file}")
        # Optionally, delete the local file after processing.
        try:
            os.remove(pgn_file)
        except Exception as e:
            print(f"Error deleting local file {pgn_file}: {e}")
    print(f"Total positions extracted: {len(all_positions)}")
    return all_positions
