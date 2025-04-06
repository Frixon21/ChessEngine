#!/usr/bin/env python3
import os
import time
import multiprocessing
from self_play_batch import self_play_game_worker  # Adjust this import as needed

# Configuration: adjust these as needed.
NUM_GAMES = 500          # Total number of games to generate in this run.
NUM_WORKERS = 16          # Number of parallel worker processes.
OUTPUT_DIR = "saved_games"  # Directory where PGNs will be saved.
SAVE_INTERVAL = 10       # After how many games to log progress.

def main():
    # Ensure the output directory exists.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create a filename based on current time.
    timestamp = int(time.time())
    output_file = os.path.join(OUTPUT_DIR, f"games_{timestamp}.pgn")
    
    print(f"Generating {NUM_GAMES} games using {NUM_WORKERS} workers...")
    print(f"PGNs will be saved to {output_file}")
    
    # Create a pool of worker processes.
    pool = multiprocessing.Pool(processes=NUM_WORKERS)
    
    # Open the output file in append mode.
    with open(output_file, "w", encoding="utf-8") as f:
        game_count = 0
        
        # Use imap_unordered to get results as soon as they are ready.
        for result in pool.imap_unordered(self_play_game_worker, range(NUM_GAMES)):
            # Assume result is a tuple:
            #   (position_data, result_str, moves_played_count, pgn_string)
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
                if game_count % SAVE_INTERVAL == 0:
                    print(f"{game_count} games saved so far.")
            else:
                print("Warning: Received empty PGN for a game, skipping.")
        
    pool.close()
    pool.join()
    print(f"Finished generating games. Total games saved: {game_count}")

if __name__ == "__main__":
    main()
