from gym.envs.registration import register

#### Image-based follower envs. ####
register(
    id='Sprites-v0',
    entry_point='sprites_env.envs.sprites:SpritesEnv',
    kwargs={'n_distractors': 0}
)

register(
    id='Sprites-v1',
    entry_point='sprites_env.envs.sprites:SpritesEnv',
    kwargs={'n_distractors': 1}
)

register(
    id='Sprites-v2',
    entry_point='sprites_env.envs.sprites:SpritesEnv',
    kwargs={'n_distractors': 2}
)

#### State-based follower envs. ####
register(
    id='SpritesState-v0',
    entry_point='sprites_env.envs.sprites:SpritesStateEnv',
    kwargs={'n_distractors': 0}
)

register(
    id='SpritesState-v1',
    entry_point='sprites_env.envs.sprites:SpritesStateEnv',
    kwargs={'n_distractors': 1}
)


register(
    id='SpritesState-v2',
    entry_point='sprites_env.envs.sprites:SpritesStateEnv',
    kwargs={'n_distractors': 2}
)

#### Image-based repel envs. ####
register(
    id='SpritesRepel-v0',
    entry_point='sprites_env.envs.sprites:SpritesRepelEnv',
    kwargs={'n_distractors': 0}
)

register(
    id='SpritesRepel-v1',
    entry_point='sprites_env.envs.sprites:SpritesRepelEnv',
    kwargs={'n_distractors': 1}
)

register(
    id='SpritesRepel-v2',
    entry_point='sprites_env.envs.sprites:SpritesRepelEnv',
    kwargs={'n_distractors': 2}
)

#### State-based repel envs. ####
register(
    id='SpritesRepelState-v0',
    entry_point='sprites_env.envs.sprites:SpritesRepelStateEnv',
    kwargs={'n_distractors': 0}
)

register(
    id='SpritesRepelState-v1',
    entry_point='sprites_env.envs.sprites:SpritesRepelStateEnv',
    kwargs={'n_distractors': 1}
)

register(
    id='SpritesRepelState-v2',
    entry_point='sprites_env.envs.sprites:SpritesRepelStateEnv',
    kwargs={'n_distractors': 2}
)

