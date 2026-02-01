import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

def make_sim_env(task_name):
    """
    Environment for simulated XArm6 manipulation, with joint position control
    Action space:      [arm_qpos (6),            # absolute joint position
                        gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ arm_qpos (6),          # absolute joint position
                                        gripper_qpos (1)]      # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ arm_qvel (6),          # absolute joint velocity (rad)
                                        gripper_velocity (1)]  # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_pick_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'xarm6_pick_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class XArm6Task(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        arm_action = action[:6]
        normalized_gripper_action = action[6]

        gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_gripper_action)
        full_gripper_action = [gripper_action, -gripper_action]

        env_action = np.concatenate([arm_action, full_gripper_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        arm_qpos = qpos_raw[:6]
        gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos_raw[6])]
        return np.concatenate([arm_qpos, gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        arm_qvel = qvel_raw[:6]
        gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(qvel_raw[6])]
        return np.concatenate([arm_qvel, gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        # raise NotImplementedError
        pass

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        # raise NotImplementedError
        pass


class PickCubeTask(XArm6Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:7] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[7:]
        return env_state

    def get_reward(self, physics):
        # return whether gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        def _touches(geom_name):
            return (("red_box", geom_name) in all_contact_pairs) or ((geom_name, "red_box") in all_contact_pairs)

        touch_gripper = _touches("right_finger_mesh") or _touches("left_finger_mesh")
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_gripper:
            reward = 2
        if touch_gripper and not touch_table:  # lifted
            reward = 4
        return reward


def print_mujoco_info():
    env = make_sim_env('sim_pick_cube')
    env.reset()
    physics = env._physics
    print(f"qpos: {physics.data.qpos}, shape: {physics.data.qpos.shape}")
    print(f"qvel: {physics.data.qvel}, shape: {physics.data.qvel.shape}")
    print(f"drive joint idx: {physics.model.name2id('drive_joint', 'joint')}")

if __name__ == '__main__':
    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]
    print_mujoco_info()