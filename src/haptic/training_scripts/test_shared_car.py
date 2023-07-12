from haptic.gym.envs.box2d.car_racing import CarRacingShared
from haptic.learning_algorithm.shared_dqn_cnn import Agent
from stable_baselines3 import DQN
import numpy as np
import torch as th
import matplotlib.pyplot as plt

ALPHA = 0.4
STATE_W = 96
STATE_H = 96
frames_per_state = 4
n_actions = 15
RANDOM_ACTION_PROB = 0.5


if __name__ == "__main__":
    env = CarRacingShared(
        allow_reverse=False,
        grayscale=1,
        show_info_panel=1,
        discretize_actions="smooth",  # n_actions = 5
        num_tracks=2,
        num_lanes=2,
        num_lanes_changes=4,
        max_time_out=5,
        frames_per_state=4,
    )
    agent = Agent(
        gamma=0.99,
        epsilon=0,
        batch_size=64,
        n_actions=n_actions,
        eps_end=0,
        input_dims=(96, 96, frames_per_state + 1),
        lr=0.003,
        max_mem_size=5000,
        max_q_target_iter=300,
        alpha=ALPHA,
        observation_space=env.observation_space,
        cuda_index=1,
    )
    model = th.load("trials/models/final_model_DQN_Car_Racer_alpha_0.6")
    agent.Q_pred = model
    print("\n model loaded successfully \n")
    scores, eps_history, avg_scores = [], [], []
    n_games = 500
    total_steps = 0
    pilot = DQN.load("trials/models/FINAL_MODEL_SMOOTH_CAR")
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
            state = (
                th.tensor(observation[:, :, 0:4])
                .to(agent.Q_pred.device)
                .cpu()
                .data.numpy()
            )

            pi_action, _ = pilot.predict(state)
            if np.random.random() < RANDOM_ACTION_PROB:
                pi_action = env.action_space.sample()
            pi_frame = pi_action * np.ones((STATE_W, STATE_H))
            observation[:, :, 4] = pi_frame
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(
                action=action, pi_action=pi_action
            )
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
