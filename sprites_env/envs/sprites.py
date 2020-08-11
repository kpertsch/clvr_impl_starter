import gym
from gym.spaces import Box
import numpy as np
import cv2
import os
from general_utils import AttrDict
from sprites_datagen.utils.template_blender import TemplateBlender


class SpritesEnv(gym.Env):
    SHAPES = ['rectangle', 'circle', 'tri_right', 'tri_bottom', 'tri_left', 'tri_top']

    def __init__(self, follow=True, **kwarg):
        self.shapes = None

        self.n_distractors = kwarg['n_distractors'] if kwarg else 1
        self.n_dim = self._n_dim = 2
        self._n_state = 2 * self.n_dim

        self.base_shape_idx_list = [1, 0]
        self.max_ep_len = 40

        self.follow = follow
        self.repel = not self.follow
        self.max_speed = 0.05
        self.obj_size = 0.2
        self.resolution = 64

        self.pos_bounds = [[self.obj_size/2, 1 - self.obj_size/2]] * 2
        bounds = list(self.pos_bounds) + [[-self.max_speed, self.max_speed]] * 2
        if bounds is not None:
            bounds = np.asarray(bounds)
            assert bounds.ndim == 2
            assert bounds.shape[0] == self._n_state
            assert bounds.shape[1] == 2
        self._bounds = bounds

        self._sprite_res = int(self.obj_size * self.resolution)
        self._shape_sprites = self._get_shape_sprites()  # generate geometric shape templates
        self._template_blender = TemplateBlender((self.resolution, self.resolution))

        self.observation_space = Box(low=0.0, high=1.0,
                shape=(self.resolution, self.resolution),
                dtype=np.float32)

        self.action_space = Box(low=-1.0, high=1.0,
                shape=(2,),
                dtype=np.float32
                )

    def set_config(self, spec):
        self._spec = spec
        self.resolution = self._spec.resolution
        self.max_speed = self._spec.max_speed
        self.max_ep_len = self._spec.max_ep_len
        self.pos_bounds = [[self._spec.obj_size/2, 1 - self._spec.obj_size/2]] * 2
        bounds = list(self.pos_bounds) + [[-self.max_speed, self.max_speed]] * 2
        if bounds is not None:
            bounds = np.asarray(bounds)
            assert bounds.ndim == 2
            assert bounds.shape[0] == self._n_state
            assert bounds.shape[1] == 2
        self._bounds = bounds

        self._sprite_res = int(self._spec.obj_size * self._spec.resolution)
        self._shape_sprites = self._get_shape_sprites()  # generate geometric shape templates
        self._template_blender = TemplateBlender((self.resolution, self.resolution))

        self.follow = self._spec.follow
        self.repel = not self.follow

        self.observation_space = Box(low=0.0, high=1.0,
                shape=(self.resolution, self.resolution),
                dtype=np.float32)

    def _clip(self, state):
        return np.clip(state, self._bounds[:, 0], self._bounds[:, 1])

    def _forward(self, state):
        """ Assuming that state is [shape_idx, 4] for [position, velocity] """
        pos, vel = np.split(state, 2, -1)

        pos += vel

        # Enables shapes bouncing off of walls
        for d in range(self._n_dim):
            too_small = np.less(pos[:, d], self._bounds[d, 0])
            too_big = np.greater(pos[:, d], self._bounds[d, 1])

            pos[too_small, d] = 2 * self._bounds[d, 0] - pos[too_small, d]
            pos[too_big, d] = 2 * self._bounds[d, 1] - pos[too_big, d]
            vel[np.logical_or(too_small, too_big), d] *= -1

        state = np.concatenate((pos, vel), -1)
        return state

    def forward(self, state):
        state = self._clip(state)
        state = self._clip(self._forward(state))
        return state[:, :self._n_dim].copy(), state

    def reset(self):
        self.ep_len = 0
        self.distractor_shape_idx_list = np.random.choice(np.arange(2, len(self.SHAPES)), size=self.n_distractors)
        self.all_idxs = np.array(self.base_shape_idx_list + list(self.distractor_shape_idx_list))
        self.shapes = np.asarray(self.SHAPES)[self.all_idxs]
        state = np.random.uniform(size=(self.n_distractors + 2, self._n_state))

        if self._bounds is not None:
            min_value = self._bounds[np.newaxis, :, 0]
            max_value = self._bounds[np.newaxis, :, 1]
            span = max_value - min_value

            state = min_value + state * span
        pos_state, self._state = self.forward(state)
        im = self._render(np.expand_dims(pos_state, 0), self.shapes).squeeze(0)
        return im / 255

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, action):
        vel = np.array(action) * self.max_speed
        state = self._state.copy()
        state[0,2:] = vel
        pos_state, self._state = self.forward(state)

        im = self._render(np.expand_dims(pos_state, 0), self.shapes).squeeze(0)
        reward = self._reward(self._state)

        self.ep_len += 1
        done = (self.ep_len >= self.max_ep_len)
        info = {}

        return im / 255, reward, done, info

    def _reward(self, state):
        agent_pos = state[0, :2]
        target_pos = state[1, :2]

        if self.follow:
            return 1. - np.sqrt(((target_pos - agent_pos) ** 2).sum()) / np.sqrt(2)
        else:
            return np.sqrt(((target_pos - agent_pos) ** 2).sum()) / np.sqrt(2)

    def _render(self, trajectories, shapes):
        sprites = [self._shape_sprites[shape] for shape in shapes]
        return self._template_blender.create((trajectories * (self.resolution - 1)).astype(int), sprites)

    def render(self, mode='rgb_array'):
        pos_state = self._state[:, :self._n_dim].copy()
        im = self._render(np.expand_dims(pos_state, 0), self.shapes).squeeze(0)
        return im

    def _get_shape_sprites(self):
        shapes = AttrDict()
        canvas = np.zeros((self._sprite_res, self._sprite_res), np.uint8)
        shapes.rectangle = cv2.rectangle(canvas.copy(), (1, 1), (self._sprite_res - 2, self._sprite_res - 2), 255, -1)
        shapes.circle = cv2.circle(canvas.copy(), (int(self._sprite_res / 2), int(self._sprite_res / 2)),
                                   int(self._sprite_res / 3), 255, -1)
        shapes.tri_right = cv2.fillConvexPoly(canvas.copy(),
                                              np.array([[[1, 1], [1, self._sprite_res - 2],
                                                         [self._sprite_res - 2, int(self._sprite_res / 2)]]]), 255)
        shapes.tri_bottom = cv2.fillConvexPoly(canvas.copy(),
                                               np.array([[[1, 1], [self._sprite_res - 2, 1],
                                                          [int(self._sprite_res / 2), self._sprite_res - 2]]]), 255)
        shapes.tri_left = cv2.fillConvexPoly(canvas.copy(),
                                             np.array([[[self._sprite_res - 2, 1], [self._sprite_res - 2, self._sprite_res - 2],
                                                        [1, int(self._sprite_res / 2)]]]), 255)
        shapes.tri_top = cv2.fillConvexPoly(canvas.copy(),
                                            np.array([[[1, self._sprite_res - 2], [self._sprite_res - 2, self._sprite_res - 2],
                                                       [int(self._sprite_res / 2), 1]]]), 255)
        return shapes


class SpritesStateEnv(SpritesEnv):
    def __init__(self, follow=True, **kwarg):
        super().__init__(follow=follow, **kwarg)
        # only return pos_state
        self.observation_space = Box(low=0.0, high=1.0,
                shape=((self.n_distractors + 2) * self._n_dim, ),
                dtype=np.float32)

    def set_config(self, spec):
        super().set_config(spec)
        self.observation_space = Box(low=0.0, high=1.0,
                shape=((self.n_distractors + 2) * self._n_dim, ),
                dtype=np.float32)

    def reset(self):
        super().reset()
        return self._state[:, :self._n_dim].copy().flatten()

    def step(self, action):
        _, reward, done, info = super().step(action)
        return self._state[:, :self._n_dim].copy().flatten(), reward, done, info


class SpritesRepelEnv(SpritesEnv):
    def __init__(self, **kwarg):
        super().__init__(follow=False, **kwarg)


class SpritesRepelStateEnv(SpritesStateEnv):
    def __init__(self, **kwarg):
        super().__init__(follow=False, **kwarg)


if __name__  == '__main__':
    data_spec = AttrDict(
        resolution=64,
        max_ep_len=40,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        follow=True,
    )
    env = SpritesEnv()
    env.set_config(data_spec)
    obs = env.reset()
    cv2.imwrite("test_rl.png", 255 * np.expand_dims(obs, -1))
    obs, reward, done, info = env.step([0, 0])
    cv2.imwrite("test_rl_1.png", 255 * np.expand_dims(obs, -1))
