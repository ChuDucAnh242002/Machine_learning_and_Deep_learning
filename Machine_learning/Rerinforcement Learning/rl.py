import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode='ansi')
# env.reset()
# print(env.render())

action_size = env.action_space.n
print("Action size: ", action_size)
state_size = env.observation_space.n
print("State size: ", state_size)

done = False
env.reset()
while not done:
    print(env.render())
    action = int(input('0/down 1/up 2/right 3/left 4/pickup 5/dropoff:'))
    new_state, reward, done, truncated, info = env.step(action)
    time.sleep(1.0)
    print('')
    print(f'Observations: S_t+1={new_state}, R_t+1={reward}, done={done}')
