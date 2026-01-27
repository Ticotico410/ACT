# xArm6 双臂 Transfer Cube（EE 控制）轨迹规划替换方案（对齐原版 ViperX Pipeline）

> 目标：在 **不引入新范式**（仍然是 scripted waypoint + mocap+weld 拖末端）的前提下，把原版 ACT 的 **ViperX 双臂 transfer cube（EE 控制）** 正确替换为 **xArm6 双臂**，使 scripted_policy 生成的 EE 轨迹能稳定抓取/交接方块并用于数据采集。

---

## 0. 原则与成功条件

### 0.1 必须满足的 3 个“强耦合”条件
1. **Weld 绑定对象必须是 TCP（抓取中心）**  
   `mocap_left/right` 必须 weld 到 `xarm6_left/right/gripper_tcp`（而不是 gripper_base_link）。

2. **Reset 时 mocap 必须与 TCP 零误差对齐**  
   reset 后必须满足：
   - `mocap_pos == xpos(gripper_tcp)`
   - `mocap_quat == xquat(gripper_tcp)`

3. **轨迹参数必须按 TCP 几何重算**  
   ViperX 的 `box_grasp_z`/`approach_z` 是按它的 `gripper_link` 经验写死的。  
   xArm6 焊到 TCP 后，`grasp/approach` 必须按 **cube 尺寸 + 安全间隙**重新定义。

### 0.2 “成功”定义（最小验收）
- 可视化中：绿色 TCP site 和红色 mocap 十字一致移动；
- scripted_policy rollout 中：右手能抓起 cube（z 抬升明显），并在 meet 点附近交接/放下；
- `record_sim_episodes` 不再因 EE rollout 失败而无法采集轨迹。

---

## 1. 文件改动清单（你需要改这些）

- `xarm6_left.xml` / `xarm6_right.xml`
  - 在 `xarm_gripper_base_link` 下新增 `gripper_tcp` body + 可视化 site
- `bimanual_xarm6_ee_transfer_cube.xml`
  - weld 约束 body2 改为 `.../gripper_tcp`
  - 确保 mocap body 上存在红色十字 site（可选但强烈建议）
- `ee_sim_env.py`
  - `initialize_robots()` 中：qpos 写入后 `physics.forward()`，动态读取 TCP 的 `xpos/xquat` 对齐 mocap
- `scripted_policy.py`
  - xarm6 分支：用 “cube 尺寸 + clearance” 定义 approach/grasp 点
  - pick 姿态：先走“标定常量 PICK_QUAT_LEFT/RIGHT”，不要盲乘 ViperX 的经验角
  - quat 插值：改成 slerp（或至少 normalize）

---

## 2. Step 1：在 xarm6_left/right.xml 新增 TCP body（抓取中心）

> 位置：放在 `xarm6_*/xarm_gripper_base_link` 下。  
> 初值偏移 `pos="0 0 0.05"` 只是起点：后面会用校准步骤精调到“指尖中点”。

**xarm6_left.xml（右臂同理，注意改 name/site 名）：**
```xml
<!-- under xarm6_left/xarm_gripper_base_link -->
<body name="xarm6_left/gripper_tcp" pos="0 0 0.05" quat="1 0 0 0">
  <site name="tcp_left" pos="0 0 0" size="0.006" type="sphere" rgba="0 1 0 1"/>
</body>

<!-- under xarm6_left/xarm_gripper_base_link -->
<body name="xarm6_right/gripper_tcp" pos="0 0 0.05" quat="1 0 0 0">
  <site name="tcp_right" pos="0 0 0" size="0.006" type="sphere" rgba="0 1 0 1"/>
</body>
```

## 3. Step 2：在 EE 场景 XML 中 weld 到 TCP （mocap to tcp）
```xml
<!--bimanual_xarm6_ee_transfer_cube.xml-->
<equality>
  <weld body1="mocap_left"  body2="xarm6_left/gripper_tcp"  solref="0.01 1" solimp=".25 .25 0.001" />
  <weld body1="mocap_right" body2="xarm6_right/gripper_tcp" solref="0.01 1" solimp=".25 .25 0.001" />
</equality>
```

### 3.1 确保 mocap 有红色十字可视化:
复用 ViperX 的“红色十字”，确保 mocap body 上挂了 3 个红色 site（或 geom）
```xml
<body name="mocap_left" mocap="true">
  <site name="mocap_left_x" type="box" size="0.03 0.002 0.002" rgba="1 0 0 1"/>
  <site name="mocap_left_y" type="box" size="0.002 0.03 0.002" rgba="1 0 0 1"/>
  <site name="mocap_left_z" type="box" size="0.002 0.002 0.03" rgba="1 0 0 1"/>
</body>
```
若仍看不到红十字，通常是 dm_control 渲染没开 site 显示（见 Step 6 Debug）。

# 4. Step3：EE 环境 reset 对齐 TCP（关键：open-loop 从零误差起步）

ee_sim_env.py（或BimanualXarm6EETask.initialize_robots 所在文件）
```xml
def initialize_robots(self, physics):
    # 1) set joint pose
    physics.named.data.qpos[:24] = START_XARM6_POSE

    # 2) forward kinematics
    physics.forward()

    # 3) align mocap with TCP
    tcpL = 'xarm6_left/gripper_tcp'
    tcpR = 'xarm6_right/gripper_tcp'

    np.copyto(physics.data.mocap_pos[0],  physics.named.data.xpos[tcpL])
    np.copyto(physics.data.mocap_quat[0], physics.named.data.xquat[tcpL])
    np.copyto(physics.data.mocap_pos[1],  physics.named.data.xpos[tcpR])
    np.copyto(physics.data.mocap_quat[1], physics.named.data.xquat[tcpR])

    # 4) reset gripper (example)
    physics.data.ctrl[:] = np.array([
        XARM6_GRIPPER_POSITION_CLOSE,
        XARM6_GRIPPER_POSITION_CLOSE
    ])
```

## 4.1:对齐自检（强烈建议加）
```xml
errL = np.linalg.norm(physics.data.mocap_pos[0] - physics.named.data.xpos[tcpL])
errR = np.linalg.norm(physics.data.mocap_pos[1] - physics.named.data.xpos[tcpR])
print("[RESET] mocap-tcp err:", errL, errR)
```

# 5. Step 4: scripted_policy（xarm6）正确生成 EE 轨迹
## 5.1 用 cube 尺寸 + clearance 定义 approach / grasp
假设 cube geom 半边长 CUBE_HALF = 0.02（如果你的 cube 尺寸不同，改这里）。
推荐参数（先保证能抓到，再微调）：

- 1. APPROACH_CLEAR = 0.08 （上方 8cm）

- 2. GRASP_CLEAR = 0.005 （顶面上方 5mm，之后根据指尖厚度调整）

- 3. MEET_DROP = 0.10 （比初始 TCP z 低 10cm 作为 meet 高度）


```xml
CUBE_HALF = 0.02
APPROACH_CLEAR = 0.08
GRASP_CLEAR = 0.005

approach_xyz = box_xyz + np.array([0, 0, CUBE_HALF + APPROACH_CLEAR])
grasp_xyz    = box_xyz + np.array([0, 0, CUBE_HALF + GRASP_CLEAR])

```

## 5.2 pick 姿态：先使用“标定常量 PICK_QUAT_LEFT/RIGHT”
不要沿用 ViperX 的 “绕某轴 -45°/-60°” 经验值。xArm6 TCP 坐标系不一样，盲乘会导致夹爪侧着撞、IK 不稳定或夹不到。建议先在 scripted_policy.py 顶部添加：
```xml
PICK_QUAT_LEFT  = np.array([1, 0, 0, 0])  # placeholder, will calibrate
PICK_QUAT_RIGHT = np.array([1, 0, 0, 0])  # placeholder, will calibrate
```

## 5.3 xarm6 版 waypoint 模板（可直接替换 generate_trajectory 的 xarm6 分支）
```xml
if self.robot_type == 'xarm6':
    initL = init_mocap_pose_left
    initR = init_mocap_pose_right
    box_xyz = box_info[:3]

    CUBE_HALF = 0.02
    APPROACH_CLEAR = 0.08
    GRASP_CLEAR = 0.005

    init_z = 0.5 * (initL[2] + initR[2])
    meet_xyz = np.array([0.0, 0.5, init_z - 0.10])  # MEET_DROP=0.10

    approach_xyz = box_xyz + np.array([0,0,CUBE_HALF + APPROACH_CLEAR])
    grasp_xyz    = box_xyz + np.array([0,0,CUBE_HALF + GRASP_CLEAR])

    qL = PICK_QUAT_LEFT
    qR = PICK_QUAT_RIGHT

    # Right arm: pick -> move to meet -> release
    self.right_trajectory = [
      {"t":0,   "xyz":initR[:3], "quat":initR[3:], "gripper":0},
      {"t":80,  "xyz":approach_xyz, "quat":qR, "gripper":1},
      {"t":120, "xyz":grasp_xyz,    "quat":qR, "gripper":1},
      {"t":160, "xyz":grasp_xyz,    "quat":qR, "gripper":0},
      {"t":210, "xyz":meet_xyz + np.array([+0.06,0,0]), "quat":qR, "gripper":0},
      {"t":240, "xyz":meet_xyz, "quat":qR, "gripper":0},
      {"t":310, "xyz":meet_xyz, "quat":qR, "gripper":1},
      {"t":360, "xyz":meet_xyz + np.array([+0.12,0,0]), "quat":qR, "gripper":1},
    ]

    # Left arm: arrive at meet -> close to receive
    self.left_trajectory = [
      {"t":0,   "xyz":initL[:3], "quat":initL[3:], "gripper":0},
      {"t":120, "xyz":meet_xyz + np.array([-0.12,0,0]), "quat":qL, "gripper":1},
      {"t":240, "xyz":meet_xyz + np.array([-0.02,0,0]), "quat":qL, "gripper":1},
      {"t":310, "xyz":meet_xyz + np.array([-0.02,0,0]), "quat":qL, "gripper":0},
      {"t":360, "xyz":meet_xyz + np.array([-0.12,0,0]), "quat":qL, "gripper":0},
    ]
```
解释：这依然是“world-frame open-loop + offset”的原版逻辑，只是把 grasp/approach 定义成 TCP 的物理意义（顶面 clearance），并把姿态用可标定常量固定住。

# 6. Step5: 四元数插值修复（slerp），防止姿态发散/翻转
原版常见问题：quat 线性插值且不 normalize，会导致 mocap 姿态不合法，weld 约束抽风。在 scripted_policy.py 的 interpolate() 改成 slerp：
```xml
from pyquaternion import Quaternion

@staticmethod
def interpolate(curr_wp, next_wp, t):
    t0, t1 = curr_wp["t"], next_wp["t"]
    frac = (t - t0) / (t1 - t0 + 1e-8)

    xyz = curr_wp["xyz"] + (next_wp["xyz"] - curr_wp["xyz"]) * frac

    q0 = Quaternion(curr_wp["quat"])
    q1 = Quaternion(next_wp["quat"])
    quat = Quaternion.slerp(q0, q1, amount=frac).elements  # wxyz

    g = curr_wp["gripper"] + (next_wp["gripper"] - curr_wp["gripper"]) * frac
    return xyz, quat, g
```

# 7. Step6: PICK_QUAT 标定（20 秒搞定，不再猜坐标系
## 7.1 目的

获得在你当前 xarm6 模型中“夹爪对着 cube 正向抓取”的稳定姿态：

- 1. PICK_QUAT_LEFT
- 2. PICK_QUAT_RIGHT

## 7.2 最省事的标定流程（推荐）

- 1. 运行 EE 环境可视化（任意能 step 的脚本）

- 2. 让机械臂处在你认为正确的抓取姿态（可以临时把 START_XARM6_POSE 改到那个姿态；或在运行中用 mocap 手动拖到 cube 上方并旋转）

- 3. 在该姿态时打印 TCP 的 xquat，作为 PICK_QUAT 常量

## 7.3 代码片段（放在 reset 后或任意 step 时打印一次）
```xml
tcpL = 'xarm6_left/gripper_tcp'
tcpR = 'xarm6_right/gripper_tcp'
print("CALIB PICK_QUAT_LEFT =", physics.named.data.xquat[tcpL].copy())
print("CALIB PICK_QUAT_RIGHT=", physics.named.data.xquat[tcpR].copy())
```

把输出复制到 scripted_policy.py 顶部：
```xml
PICK_QUAT_LEFT  = np.array([w, x, y, z])
PICK_QUAT_RIGHT = np.array([w, x, y, z])
```
这一步是“最稳且最快”的姿态对齐方法：不需要你理解 TCP 局部轴方向，直接用仿真里正确姿态的真实 quat 作为常量。

# 8. Debug 工具（建议都加上，定位问题会快 10 倍）
## 8.1 强制显示 site / mocap（如果红十字或 TCP 球看不到）
在渲染处传 scene_option 打开 mjVIS_SITE / mjVIS_MOCAP：
```xml
import mujoco as mj

def make_vis_option():
    opt = mj.MjvOption()
    opt.flags[mj.mjtVisFlag.mjVIS_SITE] = 1
    opt.flags[mj.mjtVisFlag.mjVIS_MOCAP] = 1
    return opt

VIS_OPT = make_vis_option()

img = physics.render(height=480, width=640, camera_id='front_close', scene_option=VIS_OPT)
```

## 8.2 pos-only 模式（快速判断问题在 quat 还是 xyz）
在 before_step() 暂时不写 mocap_quat，只写 pos：
```xml
np.copyto(physics.data.mocap_pos[0], action_left[:3])
np.copyto(physics.data.mocap_pos[1], action_right[:3])
# mocap_quat 保持 reset 时的值
```
- 1. 如果 pos-only 能靠近并夹住 cube：问题在 PICK_QUAT（姿态不适配）
- 2. 如果 pos-only 也抓不到：问题在 TCP 偏移或 grasp/approach 高度