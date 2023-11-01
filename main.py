import gymnasium as gym
import torch
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
    
    state, _ = env.reset()
    n_observations = len(state)
    agent = Agent(observations=n_observations, n_actions=env.action_space.n)
    
    env = RecordVideo(env=env, 
                      episode_trigger=trigger_episode, 
                      video_folder="Z:\iatreinando\LunarLander-v2-pt-7"
                     )
   
    scores, eps_history = [], []
    n_games = 16000



    for eps in range (n_games):
        score = 0
        done = False
        state, _ = env.reset()
        device = agent.policy_net.device
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        while not done:
            env.render()
            action = agent.choose_action(state, env)
            agent.decay_eps()
            observation, reward, terminated, truncated, _ = env.step(action.item())
            score += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            
            next_state = None
            if not terminated:
                next_state = torch.tensor(
                                        observation,
                                        dtype=torch.float32, 
                                        device=device
                            )
                
                next_state = next_state.unsqueeze(0)
                
            agent.memory.push(state, action, next_state, reward)
            state = next_state
            agent.learn()
            agent.target_net_state_dict = agent.target_net.state_dict()
            agent.policy_net_state_dict = agent.policy_net.state_dict()
            for key in agent.policy_net_state_dict:
                agent.target_net_state_dict[key] =\
                agent.policy_net_state_dict[key]*agent.tau +\
                agent.target_net_state_dict[key]*(1-agent.tau)
                
                agent.target_net.load_state_dict(agent.target_net_state_dict)

            
            
            
        
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_scores = np.mean(scores[-100:])

        print(f"episode: {eps}, score: {score:.2f}, avg score: {avg_scores:.2f}, epsilon: {agent.epsilon}")

    env.close() 