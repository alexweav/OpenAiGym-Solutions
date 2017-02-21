import numpy as np
import gym
import time

num_games = 1

def main():
    env = gym.make('Pong-v0')
    input_dim = 80*80*3
    output_dim = 6 #Action space size

    architecture = [input_dim, 500, 100, output_dim]
    model = init_model(architecture)

    for game in range(num_games):
        observation = env.reset()
        reward = 0.0
        observations = []
        hidden1_activations = []
        hidden2_activations = []
        d_log_probs = []
        rewards = []
        num_game_steps = 0
        done = False
        while not done:
            env.render()
            cropped = crop_frame(observation)
            downsampled = downsample(cropped)
            observation = downsampled.reshape(1, np.prod(downsampled.shape))
            scores, h1a, h2a = eval_model(model, observation)
            prob = softmax(scores)
            action = [make_choice(prob)]
            print(action)
            observation, reward, done, info = env.step(action)
            rewards.append(reward)

        

#The frame has unnecessary pixels on it (i.e. scores at top)
#Crop these out, resulting in a 160x160x3 image 
def crop_frame(frame):
    frame = frame[35:195]
    return frame

#Takes an image as a 3D numpy tensor of shape (width, height, channels)
#Downsamples tensor by factor of 2 along width and height
def downsample(image):
    image = image[::2, ::2, :]
    return image
    
#Creates a feedforward neural network
def init_model(architecture):
    model = {}
    model['W1'] = np.random.randn(architecture[0], architecture[1]) / np.sqrt(architecture[0])
    model['b1'] = np.zeros((1, architecture[1]))
    model['W2'] = np.random.randn(architecture[1], architecture[2]) / np.sqrt(architecture[1])
    model['b2'] = np.zeros((1, architecture[2]))
    model['W3'] = np.random.randn(architecture[2], architecture[3]) / np.sqrt(architecture[2])
    model['b3'] = np.zeros((1, architecture[3]))
    return model

def relu(x):
    x[x<0] = 0
    return x

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)

def eval_model(model, data):
    hidden1_scores = np.dot(data, model['W1']) + model['b1']
    hidden1_activations = relu(hidden1_scores)
    hidden2_scores = np.dot(hidden1_activations, model['W2']) + model['b2']
    hidden2_activations = relu(hidden2_scores)
    final_scores = np.dot(hidden2_activations, model['W3']) + model['b3']
    return final_scores, hidden1_activations, hidden2_activations

def make_choice(probabilities):
    return np.random.choice(np.arange(np.prod(probabilities.shape)), p=probabilities.ravel())


main()
