"""Configurations for training and inference"""
agent_config_str = """
[mode]
mode = train

[paths]
model_save_path = FINAL_MODEL_SMOOTH_CAR

[timesteps]
max_episode_timesteps = 1000
total_timesteps = 1000_000


[render]
render_each = 1

[statistics]
log_interval = 50
"""

# ---------------------EVALUATIONS CONFIGURATIONS-----------------#
eval_config_str = """

[timesteps]
max_episode_steps = 100
max_session_steps = 200

[reward]
collision_score = -25
reached_goal_score = 100
minimum_velocity = 0.1
minimum_distance = 0.1
maximum_distance = 1470
velocity_std = 2.0
alpha = 0.4
progress_discount = 0.4

[render]
render_each = 1
save_to_file = False

[statistics]
collect_statistics = True
scenario = train
"""
