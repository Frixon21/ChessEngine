import chess
import torch
import multiprocessing
from tqdm import tqdm
from mcts import MCTSNode, mcts
from neural_network import ChessNet

###############################################################################
# 1) Helper functions: loading models, playing a single game, etc.
###############################################################################

def load_model(checkpoint_path, device):
    """
    Load a ChessNet model from a checkpoint file onto the specified device.
    """
    model = ChessNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def play_game(model_white, model_black, simulations=25):
    """
    Play a single game between model_white (White) and model_black (Black) using MCTS.
    Returns: +1 if White wins, -1 if Black wins, 0 for draw.
    """
    device = next(model_white.parameters()).device
    board = chess.Board()
    current_model = model_white  # White to move first

    while not board.is_game_over():
        root = MCTSNode(board.copy())
        best_move = mcts(
            root, 
            current_model,
            num_simulations=simulations,
            c_puct=1.0,
            return_visit_distribution=False,
            device=device  # Ensure the model is on the correct device (CPU or GPU
        )
        board.push(best_move)
        # Swap sides
        current_model = model_black if current_model == model_white else model_white

    # Evaluate final result
    result = board.result()  # '1-0', '0-1', or '1/2-1/2'
    if result == '1-0':
        return +1
    elif result == '0-1':
        return -1
    else:
        return 0

###############################################################################
# 2) A single-game worker function for multiprocessing
###############################################################################

def match_single_game(args):
    """
    Worker function to play one game:
      - model_path_a, model_path_b: file paths to each model
      - a_is_white: boolean indicating if Model A is playing White
      - simulations: number of MCTS sims per move

    Returns an integer from the perspective of Model A:
      +1 if Model A wins
      0 if draw
      -1 if Model A loses
    """
    model_path_a, model_path_b, a_is_white, simulations = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load both models in this worker
    model_a = load_model(model_path_a, device)
    model_b = load_model(model_path_b, device)

    # Play one game
    if a_is_white:
        # Model A is White
        result = play_game(model_a, model_b, simulations)
        # result is +1 for White, 0 for draw, -1 for Black
        return result
    else:
        # Model A is Black
        result = play_game(model_b, model_a, simulations)
        # play_game(...) returns +1 if model_b (White) won, 
        # so from Model A's perspective we need to flip the sign
        return -result

###############################################################################
# 3) The main match function, parallelized, with a tqdm progress bar
###############################################################################

def match_models_multiprocess(
    model_path_a, 
    model_path_b, 
    num_games=50, 
    simulations=25, 
    num_workers=4
):
    """
    Play a match between two models in parallel.
      - 'num_games' total games,
      - Half with Model A as White, half with Model B as White.
      - 'simulations' controls MCTS playouts per move.
      - 'num_workers' is the # of processes in the pool.

    Returns: (score_a, score_b) after all games.
    """
    games_as_white_for_a = num_games // 2
    games_as_white_for_b = num_games - games_as_white_for_a

    # Build task list: (model_path_a, model_path_b, a_is_white, simulations)
    tasks = []
    # Model A as White
    for _ in range(games_as_white_for_a):
        tasks.append((model_path_a, model_path_b, True, simulations))
    # Model A as Black
    for _ in range(games_as_white_for_b):
        tasks.append((model_path_a, model_path_b, False, simulations))

    manager = multiprocessing.Manager()
    results = []
    pool = multiprocessing.Pool(processes=num_workers)

    # We use tqdm to track the overall progress of all games
    with tqdm(total=num_games, desc="Match Progress") as pbar:
        # For each task, we do apply_async. The callback updates the progress bar.
        for task in tasks:
            r = pool.apply_async(
                match_single_game,
                (task,),
                callback=lambda _: pbar.update(1)  # increment bar after a game
            )
            results.append(r)

        pool.close()
        pool.join()

    # Gather results
    all_outcomes = [r.get() for r in results]

    # Summarize outcomes
    # +1 => model A wins
    #  0 => draw
    # -1 => model A loses
    score_a = 0.0
    score_b = 0.0

    for outcome in all_outcomes:
        if outcome == +1:
            score_a += 1.0
        elif outcome == 0:
            score_a += 0.5
            score_b += 0.5
        else:
            score_b += 1.0

    return score_a, score_b

###############################################################################
# 4) Run an example match
###############################################################################
if __name__ == "__main__":
    old_model_path = "trained_model_60k.pth"
    new_model_path = "trained_model.pth"

    num_games = 100
    simulations = 50
    num_workers = 6

    score_a, score_b = match_models_multiprocess(
        old_model_path, 
        new_model_path,
        num_games=num_games,
        simulations=simulations,
        num_workers=num_workers
    )
    print(f"Final score for {num_games} games (A vs B): A={score_a}, B={score_b}")

    if score_a > score_b:
        print("Old Model (A) is stronger" if old_model_path == "trained_model_60k.pth" else "New Model is stronger!")
    elif score_b > score_a:
        print("New Model (B) is stronger!" if new_model_path == "trained_model.pth" else "Old Model is stronger!")
    else:
        print("It's a tie!")
