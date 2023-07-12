from haptic.gym.envs.box2d.lunar_lander import LunarLander
from haptic.learning_algorithm.dqn import Agent
import numpy as np
import torch as th
import matplotlib.pyplot as plt

LOAD_MODEL = False

if __name__ == "__main__":
    env = LunarLander()
    agent = Agent(
        gamma=0.99,
        epsilon=1,
        batch_size=64,
        n_actions=4,
        eps_end=0.01,
        input_dims=[8],
        lr=0.003,
        max_mem_size=5000,
        max_q_target_iter=500,
    )
    if LOAD_MODEL:
        model = th.load("DQN_Lunar")
        agent.Q_pred = model
        print("\n model loaded successfully \n")
    scores, eps_history, avg_scores = [], [], []
    n_games = 1000
    total_steps = 0
    pilot = th.load("DQN_Lunar")
    max_avg_score = -np.inf
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        episode_steps = 0
        while not done:
            episode_steps += 1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action=action)
            score += reward
            agent.store_transitions(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        total_steps += episode_steps

        print(
            "episode",
            i,
            f"score {score}",
            f"avg_score {avg_score}",
            f"epsilon {agent.epsilon}",
            f"episode_steps {episode_steps}",
            f"total_steps {total_steps}",
        )

        if avg_score > max_avg_score:
            max_avg_score = max(avg_score, max_avg_score)
            model = agent.Q_pred
            th.save(
                model,
                "trials/models/DQN_Lunar",
            )
            print("\n saving best model \n")

        # build the plot
        plt.plot(avg_scores)
        plt.xlabel("timesteps")
        plt.ylabel("average score")
        plt.title("average score during training")
        # plt.show()
        plt.savefig(f"trials/graphs/fully_auto_training.png")
        # plt.close()


# from stable_baselines3 import DQN
# from stable_baselines3.dqn import MlpPolicy
# from haptic.gym.envs.box2d.lunar_lander import LunarLander

# env = LunarLander()
# DQNmodel = DQN(MlpPolicy, env, verbose=1, buffer_size=10000)
# DQNmodel.learn(1000_000)
# DQNmodel.save("trials/models/SB_DQN_Lunar")
