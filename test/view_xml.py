import os
import argparse
from constants import XML_DIR

"""
python view_xml.py --xml viperx/bimanual_viperx_transfer_cube.xml
python view_xml.py --xml viperx/bimanual_viperx_insertion.xml
python view_xml.py --xml viperx/bimanual_viperx_ee_transfer_cube.xml
python view_xml.py --xml viperx/bimanual_viperx_ee_insertion.xml
"""

def visualize_xml(xml_filename, use_viewer=True):
    """Visualize MuJoCo XML file using dm_control viewer"""
    xml_path = os.path.join(XML_DIR, xml_filename)
    
    print(f"Loading XML from: {xml_path}")
    
    # Check if file exists before trying to load
    if not os.path.exists(xml_path):
        print(f" Error: XML file not found at {xml_path}")
        print(f" XML_DIR is set to: {XML_DIR}")
        print(f" Requested file: {xml_filename}")
        # Suggest correct path if it's in viperx folder
        viperx_path = os.path.join(XML_DIR, 'viperx', os.path.basename(xml_filename))
        if os.path.exists(viperx_path):
            print(f"  ✓ Found file at: {viperx_path}")
            print(f"  Try using: --xml viperx/{os.path.basename(xml_filename)}")
        return
    
    try:
        # Use dm_control (same as the rest of the codebase)
        from dm_control import mujoco
        from dm_control.rl import control
        from dm_control.suite import base
        
        # Check if this is a mujocoinclude file (partial XML)
        with open(xml_path, 'r') as f:
            content = f.read().strip()
            if content.startswith('<mujocoinclude>'):
                print(f"✗ Error: This is a partial XML file (mujocoinclude), not a complete MuJoCo model.")
                print(f"  Partial files like '{xml_filename}' are meant to be included in other XML files.")
                print(f"  Try viewing a complete model instead, e.g.:")
                print(f"    python view_xml.py --xml viperx/bimanual_viperx_transfer_cube.xml")
                return
        
        physics = mujoco.Physics.from_xml_path(xml_path)
        
        print("✓ Successfully loaded XML file")
        print(f"  - Number of joints: {physics.model.njnt}")
        print(f"  - Number of bodies: {physics.model.nbody}")
        print(f"  - Number of actuators: {physics.model.nu}")
        print(f"  - Number of DOF: {physics.model.nv}")
        print(f"  - Number of qpos: {physics.model.nq}")
        
        # List available cameras
        camera_names = []
        for i in range(physics.model.ncam):
            camera_names.append(physics.model.id2name(i, 'camera'))
        print(f"\nAvailable cameras: {camera_names}")
        
        # Apply keyframe if available (before any rendering)
        if physics.model.nkey > 0:
            key_id = 0  # Use first keyframe
            print(f"  - Found {physics.model.nkey} keyframe(s), applying keyframe {key_id} for initial pose")
            key_qpos = physics.model.key(key_id).qpos
            if key_qpos.size > 0:
                physics.data.qpos[:key_qpos.size] = key_qpos[:]
            key_qvel = physics.model.key(key_id).qvel
            if key_qvel.size > 0:
                physics.data.qvel[:key_qvel.size] = key_qvel[:]
            # Forward kinematics to update positions
            physics.forward()
        else:
            print("  - No keyframe found, using default initial pose")
            physics.reset()
        
        if use_viewer:
            # Test render first to ensure cameras work
            print("\nTesting camera rendering...")
            if camera_names:
                test_camera = camera_names[0]
                test_img = physics.render(height=480, width=640, camera_id=test_camera)
                print(f"  Test render from '{test_camera}' camera: shape={test_img.shape}, mean={test_img.mean():.2f}")
            
            print("\nOpening dm_control viewer...")
            print("Controls:")
            print("  - Mouse: Drag to rotate, scroll to zoom, right-click to pan")
            print("  - Space: Pause/Resume simulation")
            print("  - Esc: Exit viewer")
            if camera_names:
                print(f"  - Press 'C' to cycle through cameras: {', '.join(camera_names)}")
            
            # Create a minimal task for the environment
            class MinimalTask(base.Task):
                def __init__(self):
                    super().__init__()
                
                def initialize_episode(self, physics):
                    super().initialize_episode(physics)
                    # Apply keyframe if available (keyframe 0 is the default initial state)
                    if physics.model.nkey > 0:
                        key_id = 0  # Use first keyframe
                        key_qpos = physics.model.key(key_id).qpos
                        if key_qpos.size > 0:
                            physics.data.qpos[:key_qpos.size] = key_qpos[:]
                        key_qvel = physics.model.key(key_id).qvel
                        if key_qvel.size > 0:
                            physics.data.qvel[:key_qvel.size] = key_qvel[:]
                        # Forward kinematics to update positions
                        physics.forward()
                    else:
                        physics.reset()
                
                def get_observation(self, physics):
                    return {}
                
                def get_reward(self, physics):
                    return 0
                
                def before_step(self, action, physics):
                    pass
            
            # Create environment with minimal task
            task = MinimalTask()
            env = control.Environment(physics, task, time_limit=float('inf'), 
                                     control_timestep=0.01, flat_observation=False)
            
            # Launch dm_control viewer (interactive GUI)
            from dm_control import viewer
            viewer.launch(env)
        else:
            # Fallback to matplotlib render
            import matplotlib.pyplot as plt
            pixels = physics.render(height=480, width=640)
            plt.figure(figsize=(10, 8))
            plt.imshow(pixels)
            plt.title(f'MuJoCo: {xml_filename}')
            plt.axis('off')
            plt.show()
            print("\nVisualization complete! Close the window to exit.")
            
    except FileNotFoundError:
        print(f"✗ Error: XML file not found at {xml_path}")
        print(f"  XML_DIR is set to: {XML_DIR}")
    except Exception as e:
        print(f"✗ Error loading XML: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize MuJoCo XML file with native viewer')
    parser.add_argument('--xml', type=str, default='viperx/bimanual_viperx_transfer_cube.xml',
                        help='XML filename relative to XML_DIR (default: viperx/bimanual_viperx_transfer_cube.xml)')
    parser.add_argument('--no-viewer', action='store_true',
                        help='Use matplotlib instead of native viewer (fallback)')
    
    args = parser.parse_args()
    visualize_xml(args.xml, use_viewer=not args.no_viewer)

