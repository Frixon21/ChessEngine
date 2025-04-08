#!/usr/bin/env python3
import os
import time
import multiprocessing as mp
from self_play_batch import self_play_game_worker  # Adjust this import as needed
from tqdm import tqdm
import subprocess
import boto3
import json
from utils import load_config




def upload_to_s3(local_file, bucket, object_key):
    """Uploads local_file to S3 bucket with the given object key."""
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file, bucket, object_key)
        print(f"Uploaded {local_file} to s3://{bucket}/{object_key}")
    except Exception as e:
        print("Failed to upload file to S3:", e)

def git_pull():
    """
    Stage changes, commit them with a timestamp message, push to GitHub,
    and then pull the latest changes.
    """
    try:
        print("Running git pull to update PGN files...")
        subprocess.run(["git", "pull"], check=True)
        print("Git pull completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Git pull failed:", e)

def main():
    
    # Load configuration values from a JSON file.
    config = load_config("config.json")
    S3_BUCKET_NAME = config.get("S3_BUCKET_NAME", "chessgamegenerationpgns")
    S3_PREFIX = config.get("S3_PREFIX", "saved_games/")
    NUM_GAMES = config.get("GAMES_TO_GENERATE", 250)
    NUM_WORKERS = config.get("NUM_WORKERS_GAME_GEN", 8)
    LOCAL_OUTPUT_DIR = config.get("LOCAL_OUTPUT_DIR", "saved_games")
    # Ensure the output directory exists.
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    
    # Create a filename based on current time.
    timestamp = int(time.time())
    local_output_file = os.path.join(LOCAL_OUTPUT_DIR, f"games_{timestamp}.pgn")
    
    print(f"Generating {NUM_GAMES} games using {NUM_WORKERS} workers...")
    print(f"PGNs will be saved to {local_output_file}")
   
    # Define necessary variables for self-play
    current_model_for_self_play = "./scripted_model.pt"  # Update path as needed
    current_mcts_simulations = 256
    INFERENCE_BATCH_SIZE = 64

   
    # Create tasks as a list of tuples, not just integers.
    
    # Create a pool of worker processes.
    
    try: 
        mp.set_start_method('spawn', force=True)
    except RuntimeError: 
        pass
    pool = mp.Pool(processes=NUM_WORKERS)
    tasks = [(current_model_for_self_play, current_mcts_simulations, INFERENCE_BATCH_SIZE, i) for i in range(NUM_GAMES)]
    
    
    # Open the output file in append mode.
    with open(local_output_file, "w", encoding="utf-8") as f:
        game_count = 0
        
        # Use imap_unordered to get results as soon as they are ready.
        for result in tqdm(pool.imap_unordered(self_play_game_worker, tasks),
                           total=NUM_GAMES,
                           desc="Self-Play Games"):
            # Assume result is a tuple: (position_data, result_str, moves_played_count, pgn_string)
            try:
                pgn_string = result[3]
            except IndexError:
                print("Warning: Received unexpected result format, skipping game.")
                continue
            if pgn_string and pgn_string.strip():
                # Write the PGN to the file.
                f.write(pgn_string.strip() + "\n\n")
                f.flush()  # Ensure data is written immediately.
                game_count += 1
            else:
                print("Warning: Received empty PGN for a game, skipping.")
            
    pool.close()
    pool.join()        
    print(f"Finished generating games. Total games saved: {game_count}")
    
    # Upload the local file to S3.
    s3_object_key = os.path.join(S3_PREFIX, os.path.basename(local_output_file))
    upload_to_s3(local_output_file, S3_BUCKET_NAME, s3_object_key)

    # Optionally, remove the local file if you wish.
    os.remove(local_output_file)
    print(f"Local file {local_output_file} deleted after upload.")
    
    git_pull()

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(10)
