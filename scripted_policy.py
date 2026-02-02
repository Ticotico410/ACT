import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint: dict, next_waypoint: dict, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.trajectory[0]['t'] == self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)
        next_waypoint = self.trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        xyz, quat, gripper = self.interpolate(self.curr_waypoint, next_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            xyz = xyz + np.random.uniform(-scale, scale, xyz.shape)

        action = np.concatenate([xyz, quat, [gripper]])

        self.step_count += 1
        return action


class PickPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose = ts_first.observation['mocap_pose']
        # print(f"init_mocap_pose: {init_mocap_pose}")
        box_info = np.array(ts_first.observation['env_state'])
        # print(f"box_info: {box_info}")
        box_xyz = box_info[:3]
        box_quat = box_info[3:]

        gripper_pick_quat = Quaternion(init_mocap_pose[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=30)

        # Original trajectory (commented out):
        # self.trajectory = [
        #     {"t": 0, "xyz": init_mocap_pose[:3], "quat": init_mocap_pose[3:], "gripper": 0}, # sleep
        #     {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1},     # approach the cube
        #     {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1},  # go down
        #     {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0},  # close gripper
        #     {"t": 310, "xyz": box_xyz + np.array([0, 0, 0.015]), "quat": gripper_pick_quat.elements, "gripper": 1},   # go up
        #     {"t": 360, "xyz": box_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1},     # move to right
        #     {"t": 400, "xyz": box_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1},     # stay
        # ]

        self.trajectory = [
            {"t":   0, "xyz": init_mocap_pose[:3], "quat": init_mocap_pose[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": box_xyz + np.array([0, 0, 0.20]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 400, "xyz": init_mocap_pose[:3], "quat": init_mocap_pose[3:], "gripper": 0},     # sleep
        ]

def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_pick_cube' in task_name:
        env = make_ee_sim_env('sim_pick_cube')
        physics = env._physics
    else:
        raise NotImplementedError

    for episode_idx in range(1):
        ts = env.reset()
        episode = [ts]
        # Print initial qpos (ts=0)
        # print(f"\n========== Episode {episode_idx} ==========")
        # print(f"reset")
        # print(f"qpos: {ts.observation['qpos']}")
        # print(f"mocap pose: {ts.observation['mocap_pose']}")
        # print(f"reset gripper base link pos: {physics.named.data.xpos['gripper_base_link']}\n")
        # print(f"tracking error: {np.abs(ts.observation['mocap_pose'][2] - physics.named.data.xpos['gripper_base_link'][2])}\n")

        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()
            plt.show()
            plt.pause(0.5)

        policy = PickPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if step % 10 == 0:
                print(f"ts={step}")
                # print(f"qpos: {ts.observation['qpos']}")
                # print(f"action: {action}")
                # print(f"gripper base link pos: {physics.named.data.xpos['gripper_base_link']}\n")
                err_x = np.abs(ts.observation['mocap_pose'][0] - physics.named.data.xpos['gripper_base_link'][0])
                err_y = np.abs(ts.observation['mocap_pose'][1] - physics.named.data.xpos['gripper_base_link'][1])
                err_z = np.abs(ts.observation['mocap_pose'][2] - physics.named.data.xpos['gripper_base_link'][2])
                print(f"mocap error: x: {err_x:.3f}, y: {err_y:.3f}, z: {err_z:.3f}\n")
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.005)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_pick_cube_scripted'
    test_policy(test_task_name)
