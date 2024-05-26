# Tetris AI

![AI plays tetris](images/ai_tetris.png)

## About

A Deep Q-Network AI model trained to play Tetris. *Unofficially dubbed the TetrisMaster3000.*

This repository is a fork of tristanrussell's [gym-simpletetris](https://github.com/tristanrussell/gym-simpletetris) environment.

https://topey2x.github.io/TetrisAI/

## Installation
TBD

## Usage
TBD

## Environment Space

### Action Space

- Move
 - [0] Left
 - [1] Right
 - [2] Hard Drop (all the way down)
 - [3] Soft Drop (one step)

- Rotate
 - [4] Left
 - [5] Right

- Other
 - [6] Idle

### Observation Space

- Board Info
 - Width * Height (flattened)
  - [0-199] = 10x20

- Piece Info
 - [200] Block Type
 - [201] Rotation
 - [202-203] Origin Location (x,y)

## Training Parameters 

### Batch Size

Uses a batch size of 128. This is to provide more data for stable weight updates, as well as a better representation of training data.

### Neurons in MLP's Hidden Layers

Uses two hidden layers, the first has 128 neurons and the second is double that with 256. This is in hopes to increase the model's ability to learn complex patterns, improves learning rate and helps the network deal with noisy data.

## Rewards

To encourage the model to clear lines effectively, the following rewards were chosen:

- **Reward Step** = +1 for every step that does not end the game.
- **Penalise Height** = Negative reward equivalent to tower height once block is placed.
- **Penalise Height Increase** = Negative reward equivalent to height increase.
- **Advanced Clears** = 100 for a single line clear, 250 for double line clear, 750 for a triple line clear and 3000 for a tetris.
- **High Scoring** = Overwrites advanced clears and gives 1000 reward per line clear.
:w

## Evaluation Metrics

- Steps (longer game is better)
- Score (higher score is better)
