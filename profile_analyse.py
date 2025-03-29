import pstats
import os

# --- Configuration ---
PROFILE_DIR = "."  # Directory where .prof files are saved (current directory)
NUM_STATS_TO_SHOW = 30 # How many lines of stats to display
# ---

profile_files = [f for f in os.listdir(PROFILE_DIR) if f.endswith(".prof") and f.startswith("selfplay_worker")]

if not profile_files:
    print("No profile files found in the current directory.")
else:
    # Analyze the first profile file found (or loop through all)
    profile_to_analyze = profile_files[0]
    print(f"\n--- Analyzing: {profile_to_analyze} ---")

    stats = pstats.Stats(os.path.join(PROFILE_DIR, profile_to_analyze))

    # --- Useful Sorting and Printing Options ---

    # Sort by 'cumulative' time (total time spent in function + all subfunctions)
    # This is often the most useful view to find overall bottlenecks
    print(f"\n--- Top {NUM_STATS_TO_SHOW} Functions by Cumulative Time ---")
    stats.sort_stats('cumulative').print_stats(NUM_STATS_TO_SHOW)

    # Sort by 'tottime' (total time spent ONLY in the function itself, excluding subfunctions)
    # Useful for finding functions that are inherently slow, not just calling other slow functions
    print(f"\n--- Top {NUM_STATS_TO_SHOW} Functions by Total Time (tottime) ---")
    stats.sort_stats('tottime').print_stats(NUM_STATS_TO_SHOW)

    # Sort by number of calls
    # print(f"\n--- Top {NUM_STATS_TO_SHOW} Functions by Number of Calls ---")
    # stats.sort_stats('ncalls').print_stats(NUM_STATS_TO_SHOW)

    # --- You can also look for specific functions ---
    # print("\n--- Stats for specific functions (example) ---")
    # stats.print_callers('mcts_push_pop_shared') # Who called this function?
    # stats.print_callees('board_to_tensor')    # What functions did this one call?