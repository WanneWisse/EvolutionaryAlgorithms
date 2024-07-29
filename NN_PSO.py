from NN import NN
from PSO import PSO 
import numpy as np
import gymnasium as gym

population_size = 100
nn = NN(4,2,[5,5])
iterations = 1000
env = gym.make('CartPole-v1')

state_size = len(nn.flatten())
individuals = PSO(state_size,0,1,population_size)
weights = individuals.generate_start_solution()


def run_sim(nn):
    observation, info = env.reset(seed=42)
    total_reward = 0
    for _ in range(1000):
        observation = np.array([observation])
        action = np.argmax(nn.predict(observation))
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            observation, info = env.reset()
            #assume PSO minimisation
            return -total_reward

for _ in range(iterations):
    scores = []
    for weight in weights:
        nn.reconstruct(weight)
        score = run_sim(nn)
        scores.append(score)
    print(np.average(scores))
    weights = individuals.move_one_step(scores)

