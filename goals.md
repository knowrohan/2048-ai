# Project Goals: 2048 AI with MCTS and Pygame

## Phase 1: Core Game Engine
* **Board:** 4x4 matrix representing the 2048 grid.
* **Mechanics:** Implement Up, Down, Left, Right shifts with tile merging and score tracking.
* **Spawning:** After every valid move, spawn a new tile in a random empty spot (90% chance for '2', 10% chance for '4').
* **State Detection:** Identify terminal states (when no valid moves remain).
* **Headless Design:** Ensure the core game logic is entirely decoupled from the UI to allow for rapid, headless simulations during training.

## Phase 2: AI Architecture (MCTS + Neural Network)

* **State Representation:** Process the 4x4 grid into a format suitable for the neural network (e.g., applying a $\log_2$ transformation to the tile values).
* **MCTS Simulator:** Implement a search tree that looks ahead by simulating future moves and random tile spawns (incorporating Expectimax logic for the random spawns).
* **Neural Network Evaluator:** Design a network to predict the value (expected future score or win probability) of board states at the leaf nodes of the search tree.
* **Action Selection:** Output the optimal move based on the MCTS evaluation.

## Phase 3: Training Pipeline
* **Self-Play:** Automate a loop where the agent plays games against the headless engine to generate training data.
* **Experience Replay:** Store board states, chosen actions, and eventual outcomes/rewards.
* **Model Updating:** Periodically train the neural network on this stored data to improve its evaluation accuracy, thereby improving future MCTS iterations.

## Phase 4: Pygame Visualization
* **Interface:** Create a clean, grid-based UI displaying the tiles, current score, and current AI action.
* **Aesthetics:** Implement standard 2048 color-coding for different tile values.
* **Integration:** Write a run loop that allows the user to watch the trained AI play the game in real-time, ideally with adjustable game speeds.