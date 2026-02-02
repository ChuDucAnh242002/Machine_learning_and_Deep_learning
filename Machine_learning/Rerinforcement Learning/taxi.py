import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode='ansi')
env.reset()
print(env.render())

action_size = env.action_space.n
print("Action size: ", action_size)
state_size = env.observation_space.n
print("State size: ", state_size)

qtable = np.zeros((state_size, action_size))
episodes = 1000
interactions = 100 # interactions or steps
epsilon = 0.1
alpha = 0.5
gamma = 0.9
debug = 0
hist = []

def eval_policy(env_, pi_, gamma_, t_max_, episodes_):
    v_pi_rep = np.empty(episodes_) # N trials
    for e in range(episodes_):
        s_t = env.reset()[0]
        v_pi = 0
        for t in range(t_max_):
            a_t = pi_[s_t]
            s_t, r_t, done, truncated, info = env_.step(a_t) 
            v_pi += gamma_**t*r_t 
            t += 1
            if done:
                break
        v_pi_rep[e] = v_pi
        env.close()
    return np.mean(v_pi_rep), np.min(v_pi_rep), np.max(v_pi_rep), np.std(v_pi_rep)

for episode in range(episodes):

    state = env.reset()[0]
    done = False
    
    for interact in range(interactions):
        # exploitation vs. exploratin by e-greedy sampling of actions
        if np.random.uniform(0, 1) < epsilon:
            action = np.argmax(qtable[state,:])
        else:
            action = np.random.randint(0, action_size)

        # Observe
        new_state, reward, done, truncated, info = env.step(action)
        
        # Update Q-table
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
                
        # Our new state is state
        state = new_state

        # Check if terminated
        if done == True: 
            break
    
    #Put thing in hist
    if episode % 10 == 0 or episode == 1:
        pi = np.argmax(qtable, axis=1)
        val_mean, val_min, val_max, val_std= eval_policy(env, pi, gamma, interactions, episodes)
        hist.append([episode, val_mean,val_min,val_max,val_std])
        if debug == True:
            print(val_mean)

env.reset()

hist = np.array(hist)
print(hist.shape)
plt.plot(hist[:,0],hist[:,1])
plt.show()

test_episidoes = 10
test_total_rewards = 0
test_total_actions = 0

for episode in range(test_episidoes):
    state = env.reset()[0]
    done = False
    episode_reward = 0
    action_count = 0
    
    for interact in range(interactions):
        action = np.argmax(qtable[state, :])

        new_state, reward, done, truncated, info = env.step(action)

        episode_reward += reward  
        action_count += 1

        state = new_state

        if done or truncated:
            break

    test_total_rewards += episode_reward
    test_total_actions += action_count

average_reward = test_total_rewards / test_episidoes
average_actions = test_total_actions / test_episidoes

print(f'Average Total Reward: {average_reward:.2f}')
print(f'Average Number of Actions: {average_actions:.2f}')