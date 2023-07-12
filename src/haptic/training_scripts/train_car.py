from haptic.gym.envs.box2d.car_racing import CarRacing
from haptic.learning_algorithm.dqn_cnn import Agent
import numpy as np
import torch as th
import matplotlib.pyplot as plt

LOAD_MODEL = True
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
        epsilon=1,
        batch_size=64,
        n_actions=15,
        eps_end=0.05,
        eps_dec=5e-6,
        input_dims=(96, 96, frames_per_state),
        lr=0.0001,
        max_mem_size=5000,
        max_q_target_iter=10000,
        observation_space=env.observation_space,
        cuda_index=1
    )
    if LOAD_MODEL:
        model = th.load("trials/models/final_model_custom_DQN_Car_Racer")
        agent.Q_pred = model
        print("\n model loaded successfully \n")
    scores, eps_history, avg_scores = [], [], []
    n_games = 1000
    total_steps = 0
    max_avg_score = -np.inf
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        episode_steps = 0
        while not done:
            # if episode_steps >= 500:
            #     break
            episode_steps += 1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
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
                "trials/models/best_model_custom_DQN_Car_Racer",
            )
            print("\n saving best model \n")
            max_avg_score = avg_scores[i]
        if total_steps > 1000_000:
            break

        # build the plot
        plt.plot(avg_scores)
        plt.xlabel("timesteps")
        plt.ylabel("average score")
        plt.title("average score during training")
        # plt.show()
        plt.savefig(f"trials/graphs/custom_DQN_Car_Racer_2.png")
        # plt.close()

    model = agent.Q_pred
    th.save(
        model,
        "trials/models/final_model_custom_DQN_Car_Racer",
    )
    print("\n saving final model \n")
