from haptic.gym.envs.box2d.lunar_lander import LunarLanderShared
from haptic.learning_algorithm.shared_dqn import Agent
import numpy as np
import torch as th

ALPHA = 0.2
# MAX_EPISODE_STEPS = 500

if __name__ == "__main__":
    env = LunarLanderShared()
    agent = Agent(
        gamma=0.99,
        epsilon=0,
        batch_size=64,
        n_actions=4,
        eps_end=0,
        input_dims=[9],
        lr=0.003,
        max_mem_size=5000,
        alpha=ALPHA,
    )
    model = th.load(
        "trials/models/DQN_Lunar_Shared_alpha_0.4_with_pretrained_model_as_pilot_workstation"
    )
    pilot = th.load("trials/models/DQN_Lunar")
    agent.Q_pred = model
    print("\n model loaded successfully \n")
    scores, eps_history = [], []
    n_games = 500
    total_steps = 0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        episode_steps = 0
        while not done:
            if episode_steps >= 1000:
                break
            episode_steps += 1
            env.render()
            state = th.tensor(observation[:8]).to(agent.Q_pred.device)
            pi_q_values = pilot.forward(state).cpu().data.numpy()
            pi_action = np.argmax(pi_q_values).item()
            # pi_action = env.action_space.sample()
            observation[8] = pi_action
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(
                action=action, pi_action=pi_action
            )
            score += reward
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
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
