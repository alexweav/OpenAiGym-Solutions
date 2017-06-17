import numpy as np
import gym

num_games = 1
hidden_dim = 300
learning_rate = 1e-4
gamma = 0.99

def main():
    env = gym.make('Pong-v0')
    input_dim = 80*80*3
    output_dim = 6 #Action space size
    model = init_model(input_dim, hidden_dim, output_dim)
    env = gym.make("Pong-v0")
    observation = env.reset()
    for game in num_games:
        env.render()

#Karpathy's preprocess method.
#Crops out everything except playing field,
#Collapses into a single-channel image
def crop_frame(frame):
    frame = frame[35:195]
    frame = downsample(frame)
    frame[frame == 144] = 0
    frame[frame == 109] = 0
    frame[frame != 0] = 1
    return frame

def downsample(image):
    image = image[::2, ::2, :]
    return image

def init_model(input_dim, hidden_dim, output_dim):
    model = {}
    model['W1'] = np.random.randn(input_dim, hidden_dim)/np.sqrt(input_dim)
    model['W2'] = np.random.randn(hidden_dim, 1)/np.sqrt(hidden_dim)
    return model

def relu(x):
    x[x<0] = 0
    return x

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def eval_model(observation, model):
    hidden_score = np.dot(observation, model['W1'])
    hidden_act = relu(hidden_score)
    log_prob = np.dot(hidden_act, model['W2'])
    prob = sigmoid(log_prob)
    return prob, hidden_act

main()
