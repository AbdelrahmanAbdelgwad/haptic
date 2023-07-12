from haptic.gym.envs.box2d.car_racing import CarRacing
from haptic.learning_algorithm.dqn_cnn import Agent
import numpy as np
import torch as th
import matplotlib.pyplot as plt

STATE_W = 96
STATE_H = 96
frames_per_state = 4

if __name__ == "__main__":
    env = CarRacing(
        allow_reverse=False,
        grayscale=1,
        show_info_panel=1,
        discretize_actions="smooth",
        num_tracks=2,
        num_lanes=2,
        num_lanes_changes=4,
        max_time_out=5,
        frames_per_state=frames_per_state,
    )
    agent = Agent(
        gamma=0.99,
        epsilon=0,
        batch_size=64,
        n_actions=15,
        eps_end=0,
        input_dims=(96, 96, frames_per_state),
        lr=0.003,
        max_mem_size=5000,
        max_q_target_iter=300,
        observation_space=env.observation_space,
    )
    model = th.load("trials/models/final_model_custom_DQN_Car_Racer")
    agent.Q_pred = model
    print("\n model loaded successfully \n")
    scores, eps_history, avg_scores = [], [], []
    n_games = 20
    total_steps = 0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        episode_steps = 0
        while not done:
            # if episode_steps >= 500:
            #     break
            env.render()
            episode_steps += 1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
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
