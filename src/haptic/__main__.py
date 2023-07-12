import torch
import haptic.gym as gym

# import gnwrapper
from haptic.gym import logger as gymlogger

from haptic.gym.wrappers import Monitor
from haptic.utils.parser import *

gymlogger.set_level(30)

from IPython.display import HTML
from IPython import display as ipythondisplay
from haptic.gym.envs.box2d.car_racing import CarRacing
import time
from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy


def add(x, y):
    return x + y


def main():
    """Main script implementation"""
    args = parse_args()
    agent_config, eval_config = generate_agent_config(
        args.agent_config_path,
        args.eval_config_path,
    )

    if args.render_each > -1:
        agent_config.set("render", "render_each", value=args.render_each)

    if args.render_each_eval > -1:
        eval_config.set("render", "render_each", value=args.render_each_eval)

    if (args.mode == "test") or (args.mode == "train"):
        agent_config.set("mode", "mode", value=args.mode)

    total_timesteps = agent_config.getint("timesteps", "total_timesteps")
    max_episode_timesteps = agent_config.getint("timesteps", "max_episode_timesteps")
    model_save_path = agent_config.get("paths", "model_save_path")

    if agent_config.get("mode", "mode") == "train":
        # if gpu is to be used
        # pylint: disable=E1101
        device = torch.device(args.device)
        # pylint: enable=E1101
        print("using", device)

        # env = gnwrapper.Animation(CarRacingDiscrete())
        # env = CarRacingDiscrete()
        env = CarRacing(
            allow_reverse=False,
            grayscale=1,
            show_info_panel=1,
            discretize_actions=args.action_disc_level,
            num_tracks=2,
            num_lanes=2,
            num_lanes_changes=4,
            max_time_out=2,
            frames_per_state=4,
        )
        if args.initial_model != "none":
            DQNmodel = DQN.load(args.initial_model, env=env)
        else:
            DQNmodel = DQN(CnnPolicy, env, verbose=1, buffer_size=10000)
            print("CONTINUE DQN MODEL TRAINING")

        t1 = time.time()
        # Train model
        print(f"\n training will start for {total_timesteps} timesteps \n")
        DQNmodel.learn(
            total_timesteps=total_timesteps,
            log_interval=agent_config.getint("statistics", "log_interval"),
        )
        t2 = time.time()
        dt = t2 - t1
        # Save model
        DQNmodel.save(model_save_path)
        time_in_hours = ((dt / 1000) / 60) / 60
        print("\n", "training time was", time_in_hours, "\n")

    elif agent_config.get("mode", "mode") == "test":
        env = CarRacing(
            allow_reverse=False,
            grayscale=1,
            show_info_panel=1,
            discretize_actions=args.action_disc_level,
            num_tracks=2,
            num_lanes=2,
            num_lanes_changes=4,
            max_time_out=0,
            frames_per_state=4,
        )
        # Uncomment following line to save video of our Agent interacting in this environment
        # This can be used for debugging and studying how our agent is performing
        env = gym.wrappers.Monitor(env, "./video/", force=True)
        model = DQN.load(args.initial_model)
        t = 0
        done = False
        episode_reward = 0
        observation = env.reset()
        # Notice that episodes here are very small due to the way that the environment is structured
        for episode in range(round(total_timesteps / max_episode_timesteps)):
            while not done:
                t += 1
                env.render()
                action, _ = model.predict(observation)
                observation, reward, done, info = env.step(action)
                episode_reward += reward
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                break
            print(episode_reward)
            episode_reward = 0
        env.close()


if __name__ == "__main__":
    main()
