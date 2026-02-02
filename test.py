import numpy as np
from ee_sim_env import make_ee_sim_env
from constants import START_ARM_POSE
from dm_control import viewer

def test_ee_sim_env():
    # 创建环境
    print("正在创建环境...")
    env = make_ee_sim_env('sim_pick_cube')
    
    # 重置环境（这会调用 initialize_robots，设置 qpos 为 START_ARM_POSE）
    print(f"正在重置环境，初始 qpos 设置为: {START_ARM_POSE}")
    ts = env.reset()
    # 验证 qpos 是否正确设置
    physics = env._physics
    current_qpos = physics.data.qpos[:7].copy()
    print(f"当前 qpos: {current_qpos}")
    print(f"qpos 匹配: {np.allclose(current_qpos, START_ARM_POSE)}")
    
    # 获取观察
    obs = ts.observation
    print(f"观察空间键: {obs.keys()}")
    print(f"qpos 观察: {obs['qpos']}")
    
    # 可视化环境
    print("正在启动可视化查看器...")
    print("提示：关闭查看器窗口以退出程序")
    viewer.launch(env)

if __name__ == '__main__':
    test_ee_sim_env()

