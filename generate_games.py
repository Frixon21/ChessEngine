#!/usr/bin/env python3
import os
import time
import multiprocessing as mp
from self_play_batch import self_play_game_worker  # Adjust this import as needed
from tqdm import tqdm
import subprocess
# Configuration: adjust these as needed.
NUM_GAMES = 100          # Total number of games to generate in this run.
NUM_WORKERS = 8          # Number of parallel worker processes.
OUTPUT_DIR = "saved_games"  # Directory where PGNs will be saved.


def git_push_and_pull():
    """
    Stage changes, commit them with a timestamp message, push to GitHub,
    and then pull the latest changes.
    """
    try:
        # Stage the output directory (or you can stage all changes with '.')
        subprocess.run(["git", "add", OUTPUT_DIR], check=True)
        
        # Create a commit message with the current time.
        commit_message = f"Update self-play games at {time.ctime()}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # Push the commit.
        subprocess.run(["git", "push"], check=True)
        
        # Pull the latest changes (if any).
        subprocess.run(["git", "pull"], check=True)
        
        print("Git push and pull completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Git command failed:", e)

def main():
    # Ensure the output directory exists.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create a filename based on current time.
    timestamp = int(time.time())
    output_file = os.path.join(OUTPUT_DIR, f"games_{timestamp}.pgn")
    
    print(f"Generating {NUM_GAMES} games using {NUM_WORKERS} workers...")
    print(f"PGNs will be saved to {output_file}")
   
    # Define necessary variables for self-play
    current_model_for_self_play = "/home/jupyter/project/ChessEngine/scripted_model.pt"  # Update path as needed
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
    with open(output_file, "w", encoding="utf-8") as f:
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
    
    git_push_and_pull()

if __name__ == "__main__":
    main()
