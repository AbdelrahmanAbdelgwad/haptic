from haptic.gym.envs.box2d.car_racing import CarRacingShared
from haptic.learning_algorithm.shared_dqn import Agent
from stable_baselines3 import DQN
import numpy as np
import torch as th
import matplotlib.pyplot as plt

LOAD_MODEL = False
ALPHA = 0.4
STATE_W = 96
STATE_H = 96

if __name__ == "__main__":
    env = CarRacingShared(
        allow_reverse=False,
        grayscale=1,
        show_info_panel=1,
        discretize_actions="smooth",
        num_tracks=2,
        num_lanes=2,
        num_lanes_changes=4,
        max_time_out=2,
        frames_per_state=4,
    )
    agent = Agent(
        gamma=0.99,
        epsilon=1,
        batch_size=64,
        n_actions=15,
        eps_end=0.01,
        input_dims=[STATE_W * STATE_H * 5],
        lr=0.003,
        max_mem_size=5000,
        max_q_target_iter=300,
        alpha=ALPHA,
    )
    if LOAD_MODEL:
        model = th.load("trials/models/DQN_Car_Racer_alpha_0.4")
        agent.Q_pred = model
        print("\n model loaded successfully \n")
    scores, eps_history, avg_scores = [], [], []
    n_games = 100
    total_steps = 0
    pilot = DQN.load("trials/models/fully_autonomous_research_paper_model")
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        episode_steps = 0
        while not done:
            # if episode_steps >= 500:
            #     break
            episode_steps += 1
            # pi_action = env.action_space.sample()
            env.render()
            state = (
                th.tensor(observation[:, :, 0:4])
                .to(agent.Q_pred.device)
                .cpu()
                .data.numpy()
            )

            pi_action, _ = pilot.predict(state)
            pi_frame = pi_action * np.ones((STATE_W, STATE_H))
            observation[:, :, 4] = pi_frame
            flattened_obs = th.flatten(th.tensor(observation))
            # print(flattened_obs.shape)
            action = agent.choose_action(flattened_obs)
            observation_, reward, done, info = env.step(
                action=action, pi_action=pi_action
            )
            score += reward
            flattened_obs_ = th.flatten(th.tensor(observation_))
            agent.store_transitions(flattened_obs, action, reward, flattened_obs_, done)
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
        if avg_scores[i] > avg_scores[i - 1]:
            model = agent.Q_pred
            th.save(
                model,
                "trials/models/best_model_DQN_Car_Racer_alpha_0.4",
            )
            print("\n saving best model \n")

        # build the plot
        plt.plot(avg_scores)
        plt.xlabel("timesteps")
        plt.ylabel("average score")
        plt.title("average score during training")
        # plt.show()
        plt.savefig(f"trials/graphs/DQN_Car_Racer_alpha_0.4.png")
        # plt.close()

    model = agent.Q_pred
    th.save(
        model,
        "trials/models/final_model_DQN_Car_Racer_alpha_0.4",
    )
    print("\n saving final model \n")