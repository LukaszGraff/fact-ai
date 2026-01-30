from gym.envs.registration import register

register(
    id = 'MO-Ant-v2',
    entry_point = 'environments.ant:AntEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v2',
    entry_point = 'environments.hopper:HopperEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v3',
    entry_point = 'environments.hopper_v3:HopperEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-HalfCheetah-v2',
    entry_point = 'environments.half_cheetah:HalfCheetahEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Walker2d-v2',
    entry_point = 'environments.walker2d:Walker2dEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Swimmer-v2',
    entry_point = 'environments.swimmer:SwimmerEnv',
    max_episode_steps=500,
)

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
