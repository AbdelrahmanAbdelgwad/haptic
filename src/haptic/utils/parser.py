"""Parser implementation for HighRL library"""
from typing import Tuple
from os import path, getcwd
import os
import argparse
import getpass
from configparser import RawConfigParser
from rich_argparse import RichHelpFormatter
from haptic.configs import *

# from haptic import __version__


def parse_args() -> argparse.Namespace:
    """Crease argument parser interface

    Returns:
        argparse.Namespace: namespace of input arguments
    """
    parser = argparse.ArgumentParser(
        prog="Parse arguments",
        description="Parse arguments to train shared autonomy agent",
        epilog="Enjoy the training! \N{slightly smiling face}",
        formatter_class=RichHelpFormatter,
    )
    # parser.add_argument(
    #     "-v",
    #     "--version",
    #     version=__version__,
    #     action="version",
    # )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        dest="device",
        help="What device to use for training (cpu/cuda) (default: %(default)s)",
    )
    parser.add_argument(
        "--initial-model",
        type=str,
        default="none",
        dest="initial_model",
        help="Path of initial shared autonomy model used in training (default: %(default)s)",
    )

    parser.add_argument(
        "--agent-config",
        type=str,
        default="none",
        dest="agent_config_path",
        help="path of configuration file of agent environment",
    )

    parser.add_argument(
        "--eval-config",
        type=str,
        default="none",
        dest="eval_config_path",
        help="Path of configuration file of agent evaluation environment",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        dest="mode",
        choices=["train", "test"],
        help="Whether to run model for training or inference (default: %(default)s)",
    )
    parser.add_argument(
        "--action-disc-level",
        type=str,
        default="smooth",
        dest="action_disc_level",
        choices=["hard", "soft", "smooth"],
        help="What level of action discretization to use (default: %(default)s)",
    )
    parser.add_argument(
        "--training-algorithm",
        type=str,
        default="DQN",
        dest="training_algorithm",
        choices=["DQN", "Custom_DQN"],
        help="which environment to use through training/testing (default: %(default)s)",
    )

    parser.add_argument(
        "--render-each",
        type=int,
        default=-1,
        dest="render_each",
        help="the frequency of rendering for agent environment (default: %(default)s)",
    )

    parser.add_argument(
        "--render-each-eval",
        type=int,
        default=-1,
        dest="render_each_eval",
        help="the frequency of rendering for agent eval environment (default: %(default)s)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="desktop",
        dest="output_dir",
        help="relative path to output results for robot mode (default: %(default)s)",
    )

    args = parser.parse_args()

    return args


def generate_agent_config(
    agent_config_path: str,
    eval_config_path: str,
) -> Tuple[RawConfigParser, ...]:
    """Generates the agent configs

    Args:
        agent_config_path (str): path of the config file for agent training env
        eval_config_path (str): path of the config file for agent eval env

    Returns:
        Tuple[RawConfigParser, ...]: Tuple of config objects for robot and eval envs
    """
    agent_config = None
    eval_config = None
    if agent_config_path != "none":
        agent_config_path = path.join(getcwd(), agent_config_path)
        assert path.exists(
            agent_config_path
        ), f"path {agent_config_path} does not exist"
        robot_config = RawConfigParser()
        robot_config.read(agent_config_path)
    else:
        robot_config = RawConfigParser()
        robot_config.read_string(agent_config_str)

    if eval_config_path != "none":
        eval_config_path = path.join(getcwd(), eval_config_path)
        assert path.exists(eval_config_path), f"path {eval_config_path} does not exist"
        eval_config = RawConfigParser()
        eval_config.read(eval_config_path)
    else:
        eval_config = RawConfigParser()
        eval_config.read_string(eval_config_str)

    return (robot_config, eval_config)


def handle_output_dir(args: argparse.Namespace) -> argparse.Namespace:
    """Parse output dir from user and create output folders

    Args:
        args (argparse.Namespace): input args namespace.

    Returns:
        argparse.Namespace: args namespace with adjusted output path.
    """
    username = getpass.getuser()
    if args.output_dir == "desktop":
        args.output_dir = f"/home/{username}/Desktop"
    else:
        args.output_dir = path.join(getcwd(), args.output_dir)

    output_dir_path = path.join(args.output_dir, "output_dir")
    env_render_path = path.join(output_dir_path, "env_render")

    saved_models_path = path.join(output_dir_path, "saved_models")
    agent_models_path = path.join(saved_models_path, "robot")

    logs_path = path.join(output_dir_path, "logs")
    agent_logs_path = path.join(logs_path, "robot")
    output_paths = {
        "output_dir_path": output_dir_path,
        "env_render_path": env_render_path,
        "saved_models_path": saved_models_path,
        "robot_models_path": agent_models_path,
        "logs_path": logs_path,
        "robot_logs_path": agent_logs_path,
    }

    for path_name, output_path in output_paths.items():
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        args.__setattr__(path_name, output_path)

    return args
