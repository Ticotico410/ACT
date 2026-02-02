import numpy as np
import collections
import os

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from utils import sample_box_pose, sample_insertion_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython
e = IPython.embed


def make_ee_sim_env(task_name):
    """
    Environment for simulated XArm6 manipulation, with end-effector control.
    Action space:      [arm_pose (7),             # position and quaternion for end effector
                        gripper_positions (1),]   # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ arm_qpos (6),          # absolute joint position
                                        gripper_position (1)]  # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ arm_qvel (6),          # absolute joint velocity (rad)
                                        gripper_velocity (1)]  # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_pick_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'xarm6_ee_pick_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCubeEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT, n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class XArm6EETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        # set mocap position and qua
        np.copyto(physics.data.mocap_pos[0], action[:3])
        np.copyto(physics.data.mocap_quat[0], action[3:7])

        # set gripper
        g_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action[7])
        np.copyto(physics.data.ctrl, np.array([g_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:7] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['gripper_base_link']
        #     get env._physics.named.data.xquat['gripper_base_link']
        np.copyto(physics.data.mocap_pos[0], [0.3010574, 0.49999854, 0.43614391])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([PUPPET_GRIPPER_POSITION_CLOSE])
        np.copyto(physics.data.ctrl, close_gripper_control)

        # forward physics to update positions after setting qpos
        physics.forward()

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
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        # used in scripted policy to obtain starting pose
        obs['mocap_pose'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        # raise NotImplementedError
        pass


class PickCubeEETask(XArm6EETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        # cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        # np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[12:]
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
    env = make_ee_sim_env('sim_pick_cube')
    env.reset()
    physics = env._physics
    print(f"mocap pos: {physics.named.data.xpos['gripper_base_link']}")   # [0.22734068, 0.49999763, 0.5206962]
    print(f"mocap quat: {physics.named.data.xquat['gripper_base_link']}") # [3.99999997e-04, 3.67235628e-06, -9.99999920e-01, -2.11985174e-06]
    print(f"qpos: {physics.data.qpos}, shape: {physics.data.qpos.shape}")
    print(f"qvel: {physics.data.qvel}, shape: {physics.data.qvel.shape}")
    print(f"drive joint idx: {physics.model.name2id('drive_joint', 'joint')}")

if __name__ == '__main__':
    print_mujoco_info()