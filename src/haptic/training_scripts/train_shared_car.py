from haptic.gym.envs.box2d.car_racing import CarRacingShared
from haptic.learning_algorithm.shared_dqn_cnn import Agent
from stable_baselines3 import DQN
import numpy as np
import torch as th
import matplotlib.pyplot as plt

LOAD_MODEL = False
ALPHA = 0.6
STATE_W = 96
STATE_H = 96
frames_per_state = 4
n_actions = 15
RANDOM_ACTION_PROB = 0
RANDOM_ACTION_PROB_INC = 0
action_space = [i for i in range(n_actions)]

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
        epsilon=1,
        batch_size=64,
        n_actions=n_actions,
        eps_end=0.01,
        input_dims=(96, 96, frames_per_state + 1),
        lr=0.003,
        max_mem_size=5000,
        max_q_target_iter=300,
        alpha=ALPHA,
        observation_space=env.observation_space,
        cuda_index=1,
    )
    if LOAD_MODEL:
        model = th.load("trials/models/best_model_DQN_Car_Racer_alpha_0.6")
        agent.Q_pred = model
        print("\n model loaded successfully \n")
    scores, eps_history, avg_scores = [], [], []
    n_games = 500
    total_steps = 0
    pilot = DQN.load("trials/models/FINAL_MODEL_SMOOTH_CAR")
    max_avg_score = -np.inf
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        episode_steps = 0
        while not done:
            if total_steps % 50_000 == 0:
                RANDOM_ACTION_PROB+=RANDOM_ACTION_PROB_INC
            episode_steps += 1
            # pi_action = env.action_space.sample()
            state = (
                th.tensor(observation[:, :, 0:4])
                .to(agent.Q_pred.device)
                .cpu()
                .data.numpy()
            )

            pi_action, _ = pilot.predict(state)
            if np.random.random() < RANDOM_ACTION_PROB:
                pi_action = np.random.choice(action_space)
            pi_frame = pi_action * np.ones((STATE_W, STATE_H))
            observation[:, :, 4] = pi_frame
            # print(flattened_obs.shape)
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(
                action=action, pi_action=pi_action
            )
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
        if avg_scores[i] > max_avg_score:
            model = agent.Q_pred
            th.save(
                model,
                "trials/models/best_model_DQN_Car_Racer_alpha_0.6",
            )
            print("\n saving best model \n")
            max_avg_score = avg_scores[i]
        if total_steps > 1000_000:
            break

        if total_steps % 5000 == 0:
            th.save(
                model,
                "trials/models/final_model_DQN_Car_Racer_alpha_0.6",
            )
            print("\n saving model every 5000 steps \n")

        # build the plot
        plt.plot(avg_scores)
        plt.xlabel("timesteps")
        plt.ylabel("average score")
        plt.title("average score during training")
        # plt.show()
        plt.savefig(f"trials/graphs/DQN_Car_Racer_alpha_0.6.png")
        # plt.close()

    model = agent.Q_pred
    th.save(
        model,
        "trials/models/final_model_DQN_Car_Racer_alpha_0.6",
    )
    print("\n saving final model \n")
