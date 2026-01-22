from gym.envs.registration import register

register(
    id="RandomMOMDP-v0",
    entry_point="environments.random_momdp:RandomMOMDPEnv",
    max_episode_steps=200,
)
