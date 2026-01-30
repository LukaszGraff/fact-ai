from gym.envs.registration import register

register(
    id = 'MO-FourRoom-v2',
    entry_point = 'environments.MO-Four-Room:FourRoomEnv',
    max_episode_steps=200,
)

register(
    id = 'MO-FourRoom-Gymnasium-v0',
    entry_point = 'environments.MO-Four-Room-gymnasium:MoFourRoomGymEnv',
    max_episode_steps=200,
)

register(
    id = 'MO-NineRoom-v2',
    entry_point = 'environments.MO-Nine-Room:NineRoomEnv',
    max_episode_steps=400,
)
