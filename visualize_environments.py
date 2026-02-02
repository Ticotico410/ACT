import os
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
    xml_path = os.path.join(XML_DIR, xml_filename)
    if not os.path.exists(xml_path):
        return

    from dm_control import mujoco
    from dm_control.rl import control
    from dm_control.suite import base

    with open(xml_path, 'r') as f:
        if f.read().strip().startswith('<mujocoinclude>'):
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
    filename = 'xarm6_ee_pick_cube.xml'
    # filename = 'xarm6_viz.xml'
    use_viewer = True
    visualize_xml(filename, use_viewer=use_viewer)
