import os
import argparse
from constants import XML_DIR


def _apply_keyframe_or_reset(physics):
    if physics.model.nkey > 0:
        key_id = 0
        key_qpos = physics.model.key(key_id).qpos
        if key_qpos.size > 0:
            physics.data.qpos[:key_qpos.size] = key_qpos[:]
        key_qvel = physics.model.key(key_id).qvel
        if key_qvel.size > 0:
            physics.data.qvel[:key_qvel.size] = key_qvel[:]
        physics.forward()
    else:
        physics.reset()


def visualize_xml(xml_filename, use_viewer=True):
    # 支持：仅文件名（在 XML_DIR 下）、或相对路径 assets/xxx.xml、或绝对路径
    if os.path.isabs(xml_filename):
        xml_path = xml_filename
    elif xml_filename.startswith("assets" + os.sep) or xml_filename.startswith("assets/"):
        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), xml_filename)
    else:
        xml_path = os.path.join(XML_DIR, xml_filename)
    xml_path = os.path.normpath(xml_path)
    if not os.path.exists(xml_path):
        print(f"文件不存在: {xml_path}")
        return

    from dm_control import mujoco
    from dm_control.rl import control
    from dm_control.suite import base

    with open(xml_path, 'r') as f:
        if f.read().strip().startswith('<mujocoinclude>'):
            print(f"跳过 include 片段（非完整模型）: {xml_path}")
            return

    physics = mujoco.Physics.from_xml_path(xml_path)
    _apply_keyframe_or_reset(physics)

    if use_viewer:
        class MinimalTask(base.Task):
            def initialize_episode(self, physics):
                super().initialize_episode(physics)
                _apply_keyframe_or_reset(physics)

            def get_observation(self, physics):
                return {}

            def get_reward(self, physics):
                return 0

        task = MinimalTask()
        env = control.Environment(
            physics,
            task,
            time_limit=float('inf'),
            control_timestep=0.01,
            flat_observation=False,
        )
        from dm_control import viewer
        viewer.launch(env)
    else:
        import matplotlib.pyplot as plt
        pixels = physics.render(height=480, width=640)
        plt.figure(figsize=(10, 8))
        plt.imshow(pixels)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize a MuJoCo XML environment.")
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    visualize_xml(args.filename, use_viewer=True)
