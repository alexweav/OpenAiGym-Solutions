import gym
import numpy as np
import time
from numerical_gradient import *

#Displays numerical evidence that backprop gradients are correct
#Slows down performance dramatically and doesnt affect outcome, so disabled by default
check_gradient = False

#Some useful numerical values
num_hidden_neurons = 10
learning_rate = 1e-4
gamma = 0.99
num_games = 1
win_reward_threshold = 195.0

def main():
    env = gym.make('CartPole-v0')
    input_dim = env.observation_space.low.shape[0]

    model = init_model(input_dim, num_hidden_neurons)
    rmsprop_cache = init_rmsprop_cache(model)

    for game in range(num_games):
        observation = env.reset()
        observation = observation.reshape(1, input_dim)
        done = False
        reward = 0.0
        observations = [] #List of all observations made throughout game
        hidden_activations = [] #Store hidden layer activations at every stage for backprop
        d_log_probs = [] #Store the derivative of the loss function at every stage
        num_game_steps = 0
        total_reward = 0.0
        #while not done:
        for i in range(5):
            #Single step of the game
            env.render()
            probability, hidden_activation = eval_model(model, observation)
            action = decide_action(probability)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            observation = observation.reshape(1, input_dim)
            d_log_probability = action - probability

            #Store all the values for this game step and proceed
            observations += [observation]
            hidden_activations += [hidden_activation]
            d_log_probs += [d_log_probability]
            num_game_steps += 1
        print("Game ", game, " final reward: ", total_reward)

        #1 if we won, -1 otherwise
        win_modifier = 1 if total_reward >= win_reward_threshold else -1

        observations = np.vstack(observations)
        d_log_probs = np.vstack(d_log_probs)
        hidden_activations = np.vstack(hidden_activations)

        #Model derivatives for frame 0 of the episode
        #How to get deritaves for all frames without loop?
        model_derivatives = backprop(hidden_activations, d_log_probs, model, observations)
        #model = update(model, model_derivatives, rmsprop_cache)


#Initiates model and returns it in the form of a dict
def init_model(input_dim, num_hidden_neurons):
    model = {}
    model['W1'] = np.random.randn(input_dim, num_hidden_neurons) / np.sqrt(input_dim)
    model['b1'] = np.zeros((1, num_hidden_neurons))
    model['W2'] = np.random.randn(num_hidden_neurons) / np.sqrt(num_hidden_neurons)
    model['b2'] = np.zeros(1)
    return model

def init_rmsprop_cache(model):
    rmsprop_cache = {}
    for key, params in model.items():
        rmsprop_cache[key] = np.zeros_like(params)


#Standard sigmoid function
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

#Applies rectified linear activation to input
def relu(x):
    x[x<0] = 0
    return x
    
#Performs a feedforward pass, returns the final output probability and hidden layer activations
def eval_model(model, data):
    hidden_scores = np.dot(data, model['W1']) + model['b1']
    hidden_activations = relu(hidden_scores)
    final_scores = np.dot(hidden_activations, model['W2']) + model['b2']
    probability = sigmoid(final_scores)
    return probability, hidden_activations

#Decides an action to take given the probability of the "1" action
def decide_action(probability):
    if np.random.uniform() < probability:
        return 1
    else:
        return 0

#Backpropagation of a single frame
def backprop(hidden_activations, d_log_prob, model, episode_observations):
    N = episode_observations.shape[0]
    d_b2 = np.sum(d_log_prob, axis=0)
    d_W2 = np.dot(hidden_activations.T, d_log_prob).ravel()
    d_hidden_activations = (model['W2'] * d_log_prob).reshape(N, num_hidden_neurons)
    d_hidden_activations[hidden_activations <= 0] = 0 #ReLU backprop, trivial, no need to check
    d_b1 = np.sum(d_hidden_activations, axis=0)
    d_W1 = np.dot(episode_observations.T, d_hidden_activations)
    
    if check_gradient: 
        d_b2_num = numerical_gradient_layer(lambda b : np.dot(hidden_activations, model['W2']) + b, model['b2'], d_log_prob)
        d_W2_num = numerical_gradient_layer(lambda w : np.dot(hidden_activations, w) + model['b2'], model['W2'], d_log_prob)
        #d_hidden_activations_num = numerical_gradient_layer(lambda x : np.dot(x, model['W2']) + model['b2'], hidden_activations, d_log_prob)
        print('d_b2 error:', np.max(relative_error(d_b2, d_b2_num)))
        #print(d_b2)
        #print(d_b2_num)
        print('d_W2 error:', np.max(relative_error(d_W2, d_W2_num)))
        #print(d_W2)
        #print(d_W2_num)
        d_b1_num = numerical_gradient_layer(lambda b : np.dot(episode_observations, model['W1']) + b, model['b1'], d_hidden_activations)
        d_W1_num = numerical_gradient_layer(lambda w : np.dot(episode_observations, w) + model['b1'], model['W1'], d_hidden_activations) 
        print('d_b1 error:', np.max(relative_error(d_b1, d_b1_num)))
        print('d_W1 error:', np.max(relative_error(d_W1, d_W1_num)))
    
    return {'W1':d_W1, 'b1':d_b1, 'W2':d_W2, 'b2':d_b2}

#RMSProp update of a single matrix
def rmsprop(theta, dtheta, error, learning_rate, decay):
    eps = 1e-8
    error = decay * error + (1 - decay) * dtheta**2
    return theta - learning_rate * dtheta / (np.sqrt(error) + eps)

main()
