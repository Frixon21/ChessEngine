import os
import sys
import chess
import torch
from mcts import MCTSNode, mcts
from neural_network import ChessNet

def uci_loop():
       
    def get_model_path(filename):
        # When running as a PyInstaller one-file executable,
        # sys._MEIPASS contains the temporary folder path.
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, filename)
        return filename

    board = chess.Board()
    network = ChessNet()
    model_path = get_model_path("trained_model.pth")
    network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    network.eval()
    
    while True:
        # Read a command from standard input.
        command = input()

        # UCI initialization commands.
        if command == "uci":
            print("id name FrixBot")
            print("id author Alex")
            print("uciok")

        elif command == "isready":
            print("readyok")

        elif command.startswith("position"):
            # Example command: "position startpos moves e2e4 e7e5"
            tokens = command.split()
            if "startpos" in tokens:
                board.reset()
                if "moves" in tokens:
                    moves_index = tokens.index("moves") + 1
                    for move_str in tokens[moves_index:]:
                        try:
                            move = chess.Move.from_uci(move_str)
                            board.push(move)
                        except Exception as e:
                            pass  # Skip illegal moves.

        elif command.startswith("go"):
            # Run MCTS from the current position.
            root = MCTSNode(board.copy())
            best_move = mcts(root, network, num_simulations=200)  # Adjust simulation count as desired.
            if best_move is None:
                best_move = chess.Move.null()  # In case no move is found.
            print("bestmove", best_move.uci())

        elif command == "quit":
            break

        elif command == "ucinewgame":
            board.reset()

if __name__ == "__main__":
    uci_loop()
