# FrixBot

**FrixBot** is an open‑source deep learning chess engine inspired by AlphaZero and Leela Chess Zero. It combines a Monte Carlo Tree Search (MCTS) with a neural network (implemented in PyTorch) to learn chess by self‑play and puzzles, using Stockfish analysis to create training targets.

## Features

- **Self‑Play Data Generation:**  
  Uses batched MCTS (see `mcts_batch.py`) and self‑play routines to generate high‑quality chess games.
- **Neural Network Training:**  
  Trains a neural network (defined in `neural_network.py`) on training samples derived from self‑play positions and puzzles.
- **Stockfish Integration:**  
  Probes positions with Stockfish to train the neural network on the self played games.
- **PGN Parsing:**  
  Parses PGN files using `pgn_parser.py` to extract positions for training.
- **Flexible Configuration:**  
  Easily adjust the number of MCTS simulations, self‑play games, training epochs, batch sizes, and more.
- **Cloud & Local Support:**  
  Designed for self‑play on local machines and cloud VMs with spot pricing, with support for GPU acceleration and multiprocessing.
- **Git Integration:**  
  Automates saving and pushing generated game data and model checkpoints to GitHub.

## Requirements

- Python 3.8+
- PyTorch (GPU support recommended)
- python‑chess
- tqdm
- (Additional dependencies are listed in the `*req.txt` files )

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Frixon21/ChessEngine.git
   cd ChessEngine
   
2. **Set Up a Virtual Environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies:**

    ```bash
    pip install -r gpu_req.txt

4. **Configure Stockfish:**
Place your Stockfish binary (e.g. stockfish-windows-x86-64-avx2.exe) in the folder defined by the configuration (typically the stockfish/ directory).


## Usage

### Self-Play and Training

ChessEngine operates in an iterative loop that combines self-play and training:

1. **Self-Play Data Generation:**  
   - Use the `generate_games.py` script (or the equivalent self‑play routine) to run games using batched MCTS and your neural network.
   - The generated games are saved as PGN files in the `saved_games/` directory.

2. **Training:**  
   - Run `main.py` to process the self‑play PGNs (and optionally puzzle data) using Stockfish to generate training samples.
   - The training script trains the neural network for a configurable number of epochs.
   - The updated model is saved as `trained_model.pth`, and a TorchScript model (`scripted_model.pt`) is generated for fast inference.

3. **Git Integration:**  
   - At the end of each training iteration, the script automatically stages, commits, and pushes the updated model files and saved game data to GitHub, ensuring your progress is backed up.

To run the main training loop, execute:

    python main.py  

### Using Puzzles

You can supplement your training data with puzzles (e.g. from Lichess puzzles). To enable puzzle training:

- Set `USE_PUZZLES = True` in the configuration.
- Update `PUZZLE_CSV_PATH` to point to the directory containing your puzzle CSV files.

The script will load puzzle positions—parsing the entire sequence of forced moves—and incorporate these into the training sample generation using Stockfish.

### UCI Interface

The repository includes a UCI interface implemented in `FrixBot.py`. This allows your engine to interact with chess GUIs or online platforms like Lichess. To run the engine in UCI mode, simply execute:

    python FrixBot.py

### Cloud Deployment

ChessEngine is designed to scale on cloud virtual machines, making it suitable for both local development and large-scale self-play:

- **Spot Instance Support:**  
  Generate self-play games cost-effectively by leveraging spot pricing.
- **GPU Acceleration:**  
  The engine supports GPU acceleration to speed up both self-play and training.
- **Dynamic Resource Management:**  
  Tools are provided for managing persistent disks, snapshots, and even changing instance configurations (for example, migrating disks between zones).

### Contributing

Contributions are welcome! If you have improvements, bug fixes, or new features, please open an issue or submit a pull request.  
Please ensure your changes follow the project's style guidelines and include tests where appropriate.

### License

This project is released under the [MIT License](LICENSE).

### Acknowledgments

- **Inspiration:**  
  Inspired by the methods used in AlphaZero and Leela Chess Zero.
- **Community:**  
  Special thanks to the open‑source community for tools like python‑chess and PyTorch.
- **Data Sources:**  
  Thanks to Lichess for providing free puzzles and game data.

