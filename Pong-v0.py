import numpy as np
import gym
import time

num_games = 1
print_every = 10
gamma = 0.99

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
        total_reward = 0.0
        num_game_steps = 0
        done = False
        while not done:
            env.render()
            cropped = crop_frame(observation)
            downsampled = downsample(cropped)
            observation = downsampled.reshape(1, np.prod(downsampled.shape))
            observations += [observation]
            scores, h1a, h2a = eval_model(model, observation)
            hidden1_activations += [h1a]
            hidden2_activations += [h2a]
            prob = softmax(scores)
            action = [make_choice(prob)]
            print(action)
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
            total_reward += reward
            d_log_probability = prob.copy()
            d_log_probability[0, action[0]] -= 1
            d_log_probs += [d_log_probability]
            num_game_steps += 1

        if game % print_every == 0:
            pass

        observations = np.vstack(observations)
        d_log_probs = np.vstack(d_log_probs)
        hidden1_activations = np.vstack(hidden1_activations)
        hidden2_activations = np.vstack(hidden2_activations)
        rewards = np.vstack(rewards)
        
        accumulated_rewards = accumulate_reward(rewards)
        accumulated_rewards -= np.mean(accumulated_rewards)
        accumulated_rewards /= np.std(accumulated_rewards)
        
        print("Num steps", num_game_steps)
        print("Observations shape", observations.shape)
        print("Accumulated rewards shape", accumulated_rewards.shape)
        print("D_log_probs shape", d_log_probs.shape)
        
        

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

def accumulate_reward(rewards):
    accumulated_reward = np.zeros_like(rewards)
    accumulator = 0
    for i in range(rewards.shape[0]):
        accumulator = gamma * accumulator * rewards[i]
        accumulated_reward[i] = accumulator
    return accumulated_reward

main()
