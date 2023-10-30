import gymnasium as gym
import numpy as np
from agent import Agent
from neural import DeepQNetwork
from gymnasium.wrappers import  RecordVideo

def trigger_episode(episode_id):
    return episode_id % 2 == 0 

if __name__ == "__main__":
    env = gym.make("LunarLander-v2",
                    render_mode="rgb_array",
                    max_episode_steps=500,
                    gravity = -11,
                    wind_power= 20.0,
                    turbulence_power = 2.0,
                  )
    
    agent = Agent(gamma=0.80, 
                  epsilon=8e-2,
                  batch_size=64,
                  n_actions=4,
                  eps_end=8e-2, 
                  input_dims=[8],
                  lr=1e-3,
                  max_memory_size=50_000)
    
    env = RecordVideo(env=env, 
                      episode_trigger=trigger_episode, 
                      video_folder="Z:\iatreinando\LunarLander-v2-pt"
                     )
   
    scores, eps_history = [], []
    n_games = 100_000


    for eps in range (n_games):
        score = 0
        cycles = 0
        done = False
        observation, _ = env.reset()

        while True:
            env.render()
         
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

    env.close() 