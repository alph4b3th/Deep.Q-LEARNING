import gymnasium as gym
import numpy as np
from agent import Agent
from neural import DeepQNetwork
from gymnasium.wrappers import TimeLimit

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="human", max_episode_steps=128 )
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                  eps_end=1e-2, 
                  input_dims=[8], lr=1e-4)
    
    scores, eps_history = [], []
    n_games = 5000


    for eps in range (n_games):
        score = 0
        cycles = 0
        done = False
        observation, _ = env.reset()

        while True:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            if truncated or done:
                break

            score += reward
            agent.store_transaction(observation, action, reward, observation_, done)
            agent.learn()
            observation=observation_
        
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_scores = np.mean(scores[-100:])

        print(f"episode: {eps}, score: {score:.2f}, avg score: {avg_scores:.2f}, epsilon: {agent.epsilon}")

