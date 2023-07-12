# import gym
from haptic.gym.envs.box2d.lunar_lander import LunarLander
from haptic.learning_algorithm.dqn import Agent
import numpy as np
import torch as th

if __name__ == "__main__":
    # env = gym.make("LunarLander-v2")
    env = LunarLander()
    scores, eps_history = [], []
    n_games = 500
    model = th.load("trials/models/DQN_Lunar")
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        observation = th.tensor(observation).to(device="cuda")
        while not done:
            env.render()
            actions = model.forward(observation).cpu().data.numpy()
            action = np.argmax(actions).item()
            observation_, reward, done, info = env.step(action=action)
            score += reward
            observation = th.tensor(observation_).to(device="cuda")
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(
            "episode",
            i,
            f"score {score}",
            f"avg_score {avg_score}",
        )
