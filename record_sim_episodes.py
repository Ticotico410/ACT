import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import SIM_TASK_CONFIGS
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import XARM6_GRIPPER_POSITION_NORMALIZE_FN
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy

import IPython
e = IPython.embed

try:
    import mujoco.viewer
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False


def main(args):
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    use_viewer = args.get('use_viewer', False)
    inject_noise = False
    render_cam_name = 'angle'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    elif task_name == 'xarm6_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    else:
        raise NotImplementedError

    success = []
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)
        # setup plotting
        viewer_context = None
        if onscreen_render and use_viewer and VIEWER_AVAILABLE:
            # 使用 mujoco viewer（可以用鼠标调节视角）
            try:
                viewer_context = mujoco.viewer.launch_passive(
                    env.physics.model, 
                    env.physics.data
                )
            except:
                # 如果上面的方式不行，尝试访问底层对象
                viewer_context = mujoco.viewer.launch_passive(
                    env.physics.model._model, 
                    env.physics.data._data
                )
        
        if onscreen_render and not (use_viewer and VIEWER_AVAILABLE):
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                if viewer_context is not None:
                    viewer_context.sync()
                    time.sleep(0.01)
                else:
                    plt_img.set_data(ts.observation['images'][render_cam_name])
                    plt.pause(0.002)
        
        if viewer_context is not None:
            viewer_context.close()
        if onscreen_render and viewer_context is None:
            plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            if 'xarm6' in task_name:
                # xarm6: ctrl[0] is left drive_joint, ctrl[1] is right drive_joint
                left_ctrl = XARM6_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                right_ctrl = XARM6_GRIPPER_POSITION_NORMALIZE_FN(ctrl[1])
                joint[6] = left_ctrl
                joint[6+7] = right_ctrl
            else:
                # viperx: ctrl[0] is left_finger, ctrl[2] is right_finger
                left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
                joint[6] = left_ctrl
                joint[6+7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0

        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()

        episode_replay = [ts]
        # setup plotting
        replay_viewer_context = None
        if onscreen_render and use_viewer and VIEWER_AVAILABLE:
            try:
                replay_viewer_context = mujoco.viewer.launch_passive(
                    env.physics.model,
                    env.physics.data
                )
            except:
                replay_viewer_context = mujoco.viewer.launch_passive(
                    env.physics.model._model,
                    env.physics.data._data
                )
        
        if onscreen_render and not (use_viewer and VIEWER_AVAILABLE):
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                if replay_viewer_context is not None:
                    replay_viewer_context.sync()
                    time.sleep(0.01)
                else:
                    plt_img.set_data(ts.observation['images'][render_cam_name])
                    plt.pause(0.02)
        
        if replay_viewer_context is not None:
            replay_viewer_context.close()
        if onscreen_render and replay_viewer_context is None:
            plt.close()

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        """
        For each timestep:
        observations
        - images
        - each_cam_name         (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--use_viewer', action='store_true')
    
    main(vars(parser.parse_args()))

