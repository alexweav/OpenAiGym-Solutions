import numpy as np
import gym

num_steps = 1000000
hidden_dim = 300
learning_rate = 1e-4
gamma = 0.99

def main():
    env = gym.make('Pong-v0')
    input_dim = 80*80*3
    output_dim = 6 #Action space size
    model = init_model(input_dim, hidden_dim, output_dim)
    env = gym.make("Pong-v0")
    prev_obs = None
    observation = env.reset()
    observations, hidden_states, d_log_ps, d_rewards = [], [], [], []
    reward_sum = 0.0
    for step in range(num_steps):
        env.render()
        obs = crop_frame(observation)
        obs, prev_obs = advance_observation(obs, prev_obs, input_dim)
        prob, hidden_act = eval_model(obs, model)
        action = 2 if np.random.uniform() < prob else 3
        observations.append(obs)
        hidden_states.append(hidden_act)
        target = 1 if action == 2 else 0
        d_log_ps.append(target - prob)
        observation, reward, done, _ = env.step(action)
        reward_sum += reward
        d_rewards.append(reward)

        if done:
            episode_observations = np.vstack(observations)
            episode_hidden_states = np.vstack(hidden_states)
            episode_d_log_ps = np.vstack(d_log_ps)
            episode_rewards = np.vstack(d_rewards)
            observations, hidden_states, d_log_ps, d_rewards = [], [], [], []
            accumulated_reward = accumulate_rewards(episode_rewards)
            accumulated_reward -= np.mean(accumulated_reward)
            accumulated_reward /= np.std(accumulated_reward)
            episode_d_log_ps *= accumulated_reward

#Karpathy's preprocess method.
#Crops out everything except playing field,
#Collapses into a single-channel image
def crop_frame(frame):
    frame = frame[35:195]
    frame = downsample(frame)
    frame[frame == 144] = 0
    frame[frame == 109] = 0
    frame[frame != 0] = 1
    return frame.ravel()

def downsample(image):
    image = image[::2, ::2, :]
    return image

def advance_observation(current_obs, prev_obs, input_dim):
    obs = current_obs - prev_obs if prev_obs is not None else np.zeros(input_dim)
    prev_obs = current_obs
    return obs, prev_obs

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

def accumulate_rewards(rewards):
    ar = np.zeros_like(rewards)
    accumulator = 0
    for r in reversed(range(0, rewards.size)):
        accumulator = accumulator * gamma + rewards[r]
        ar[r] = accumulator
    return ar

main()
