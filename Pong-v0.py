import numpy as np
import gym

num_games = 1
hidden_size = 300
learning_rate = 1e-4
gamma = 0.99

def main():
    env = gym.make('Pong-v0')
    input_dim = 80*80*3
    output_dim = 6 #Action space size

def sigmoid(x):
    return 1./(1. + np.exp(-x))

main()
