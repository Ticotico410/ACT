# ViperX 双臂 Transfer Cube：EE 控制轨迹生成与执行 Pipeline

## 总览 Pipeline（端到端流程）

```
1. env.reset() → initialize_episode()
2. initialize_robots(): 设置 START_ARM_POSE (16维) + mocap_pos/quat (硬编码对齐 gripper_link)
3. initialize_episode(): sample_box_pose() → 写入 red_box_joint 的 qpos[16:23]
4. get_observation(): 返回 mocap_pose_left/right (从 physics.data.mocap_pos/quat 读取)
5. policy.generate_trajectory(ts_first): 读取 mocap_pose + env_state[:3] → 生成 waypoint 列表
6. policy.__call__(ts): step_count 索引 waypoint → interpolate() → 输出 16-dim action
7. task.before_step(action): action → mocap_pos/quat[0,1] + ctrl (gripper)
8. MuJoCo step: weld 约束驱动 vx300s_left/right/gripper_link 跟随 mocap
9. get_reward(): 检测接触对 → 返回 0-4 级奖励
```

---

## 【A. EE action 的语义与坐标系】

### 1) Policy 输出的 action 结构

**代码位置**: `scripted_policy.py:60-64`

```python
action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])  # 8-dim: [x,y,z, qw,qx,qy,qz, gripper]
action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])
return np.concatenate([action_left, action_right])  # 16-dim total
```

- **xyz**: 3维，world frame 下的位置（米）
- **quat**: 4维，world frame 下的四元数 `[w, x, y, z]`
- **gripper**: 1维，归一化值 `[0, 1]`，0=闭合，1=张开

### 2) 坐标系：World Frame

所有 xyz/quat 都在 **MuJoCo world frame** 下定义。XML 中的 `worldbody` 是原点，z 向上。

### 3) Gripper 映射到 MuJoCo actuator ctrl

**代码位置**: `ee_sim_env.py:74-90`

```python
def before_step(self, action, physics):
    action_left = action[:8]   # [xyz(3), quat(4), gripper(1)]
    action_right = action[8:16]
    
    # mocap 位置和姿态
    np.copyto(physics.data.mocap_pos[0], action_left[:3])      # 左臂 mocap 位置
    np.copyto(physics.data.mocap_quat[0], action_left[3:7])    # 左臂 mocap 四元数
    np.copyto(physics.data.mocap_pos[1], action_right[:3])
    np.copyto(physics.data.mocap_quat[1], action_right[3:7])
    
    # 夹爪控制：归一化 [0,1] → 物理位置值
    g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])   # [0,1] → [0.01844, 0.05800]
    g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
    np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))
    #                                                                  ↑ ViperX 使用两个相反方向的 slide joint
```

**关键点**:
- ViperX 夹爪使用两个 slide joint（left_finger, right_finger），控制值相反（`[pos, -pos]`）
- `PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN`: `x * (0.05800 - 0.01844) + 0.01844`

---

## 【B. Reset 时为什么 mocap 要对齐末端？（open-loop 轨迹成功的前提）】

### 1) initialize_robots() 设置关节初始 qpos

**代码位置**: `ee_sim_env.py:92-115`

```python
def initialize_robots(self, physics):
    # 设置 16 维关节初始位置（左右各 8：6 arm + 2 gripper）
    physics.named.data.qpos[:16] = START_ARM_POSE
    # START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,   # left
    #                   0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]    # right
    
    # 硬编码 mocap 位置，对齐到末端 link
    np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])   # left mocap
    np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])                     # 单位四元数
    np.copyto(physics.data.mocap_pos[1], [0.31718881, 0.49999888, 0.29525084])  # right mocap
    np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])
    
    # 重置夹爪控制（闭合）
    close_gripper_control = np.array([
        PUPPET_GRIPPER_POSITION_CLOSE, -PUPPET_GRIPPER_POSITION_CLOSE,  # left
        PUPPET_GRIPPER_POSITION_CLOSE, -PUPPET_GRIPPER_POSITION_CLOSE,  # right
    ])
    np.copyto(physics.data.ctrl, close_gripper_control)
```

### 2) Mocap 初值的来源

**注释说明** (`ee_sim_env.py:96-101`):

```python
# to obtain these numbers:
# (1) make an ee_sim env and reset to the same start_pose
# (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
#     get env._physics.named.data.xquat['vx300s_left/gripper_link']
#     repeat the same for right side
```

**关键点**:
- 先用 joint space 环境 reset 到 `START_ARM_POSE`
- 读取 `vx300s_left/gripper_link` 的 world frame 位置和四元数
- 将这些值硬编码为 mocap 的初始值

### 3) XML 必须存在的末端 body 名字

**XML 位置**: `bimanual_viperx_ee_transfer_cube.xml:6-7`

```xml
<equality>
    <weld body1="mocap_left" body2="vx300s_left/gripper_link" ... />
    <weld body1="mocap_right" body2="vx300s_right/gripper_link" ... />
</equality>
```

**必须存在**: `vx300s_left/gripper_link` 和 `vx300s_right/gripper_link`（在 `vx300s_left.xml` / `vx300s_right.xml` 中定义）

### 4) 为什么保证了 init_mocap_pose 等于真实末端位姿？

- `START_ARM_POSE` 决定了 `gripper_link` 的初始世界坐标
- `mocap_pos/quat` 硬编码为与 `gripper_link` 相同的值
- `weld` 约束在 reset 后立即生效，强制 `mocap` 与 `gripper_link` 重合
- 因此 `obs['mocap_pose_left/right']` 读取的值就是真实的末端位姿

---

## 【C. 方块 box_xyz 从哪来？与 XML 场景参数怎么耦合？】

### 1) initialize_episode() 采样方块 pose

**代码位置**: `ee_sim_env.py:174-183`

```python
def initialize_episode(self, physics):
    self.initialize_robots(physics)
    # 采样方块位置（7维：xyz + quat）
    cube_pose = sample_box_pose()  # 返回 [x, y, z, qw, qx, qy, qz]
    box_start_idx = physics.model.name2id('red_box_joint', 'joint')
    # 写入 qpos 的索引范围（16 个关节之后，即索引 16:23）
    np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
```

### 2) sample_box_pose() 的采样范围

**代码位置**: `utils.py:133-142`

```python
def sample_box_pose():
    x_range = [0.0, 0.2]    # 世界坐标 x 范围
    y_range = [0.4, 0.6]    # 世界坐标 y 范围（桌面中央区域）
    z_range = [0.05, 0.05]  # 固定 z=0.05（桌面高度 + 方块半高）
    
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])  # 单位四元数（无旋转）
    return np.concatenate([cube_position, cube_quat])
```

### 3) 写入哪个 joint 的 qpos

- **Joint 名字**: `red_box_joint`（在 XML 中定义，type="free"，7 DOF）
- **qpos 索引**: `[16:23]`（前 16 个是机器人关节，后面 7 个是 box 的 pose）

### 4) Policy 参数与 XML 的耦合

**Policy 代码**: `scripted_policy.py:94-104`

```python
self.right_trajectory = [
    {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), ...},   # approach: z+0.08
    {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), ...}, # grasp: z-0.015
    ...
]
```

**耦合点**:
- `sample_box_pose()` 的 `z_range=[0.05, 0.05]` 必须匹配 XML 中桌面的实际高度（table 的 z 位置 + box 半高 0.02）
- Policy 中的 `0.08`（approach 高度）和 `-0.015`（grasp 偏移）是硬编码的，假设 box 在 `z=0.05`
- 如果 XML 改变桌面高度或 box 尺寸，这些偏移量需要相应调整

---

## 【D. PickAndTransferPolicy.generate_trajectory：轨迹如何由 "init_mocap_pose + box_xyz" 拼出来】

### 1) generate_trajectory() 从观测中读取的字段

**代码位置**: `scripted_policy.py:69-76`

```python
def generate_trajectory(self, ts_first):
    # 从观测中读取初始 mocap 位姿（7维：xyz + quat）
    init_mocap_pose_right = ts_first.observation['mocap_pose_right']  # [x,y,z, qw,qx,qy,qz]
    init_mocap_pose_left = ts_first.observation['mocap_pose_left']
    
    # 从 env_state 读取方块位置（前 3 维是 xyz）
    box_info = np.array(ts_first.observation['env_state'])  # [x,y,z, qw,qx,qy,qz]
    box_xyz = box_info[:3]  # 只取位置，忽略旋转
    box_quat = box_info[3:]
```

### 2) Waypoint 的数据结构

**代码位置**: `scripted_policy.py:85-104`

```python
self.left_trajectory = [
    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
    {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1},
    ...
]
```

每个 waypoint 包含：
- `t`: 时间步索引（整数）
- `xyz`: 3维位置（world frame）
- `quat`: 4维四元数（world frame）
- `gripper`: 归一化夹爪值 [0, 1]

### 3) 右手关键阶段的 waypoint

**代码位置**: `scripted_policy.py:94-104`

```python
self.right_trajectory = [
    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
    {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1},  # approach
    {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1},  # grasp
    {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0},  # close
    {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0},  # lift + approach meet
    {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0},  # meet
    {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1},  # release
    ...
]
```

**关键阶段**:
- **approach** (t=90): `box_xyz + [0,0,0.08]` - 在方块上方 8cm
- **grasp** (t=130): `box_xyz + [0,0,-0.015]` - 下降 1.5cm 抓取
- **close** (t=170): 位置不变，gripper=0（闭合）
- **lift + meet** (t=200-220): 移动到 `meet_xyz = [0, 0.5, 0.25]`（世界坐标固定点）
- **release** (t=310): 位置不变，gripper=1（张开）

### 4) 强调：World-frame 几何偏移的 open-loop 轨迹

**关键点**:
- **不是 IK/MPC**：轨迹是硬编码的 world-frame 偏移，不求解逆运动学
- **对初始对齐敏感**：如果 `init_mocap_pose` 与真实末端不重合，整个轨迹会偏移
- **对 box_xyz 范围敏感**：`sample_box_pose()` 的范围必须与 policy 的偏移量匹配
- **对坐标系定义敏感**：`gripper_link` 的局部坐标系决定了 quat 的物理含义（例如，`quat=[1,0,0,0]` 意味着什么朝向）

---

## 【E. 逐步执行：step_count 选 waypoint → 插值 → 写 mocap】

### 1) __call__() 如何用 step_count 选择当前段

**代码位置**: `scripted_policy.py:36-64`

```python
def __call__(self, ts):
    # 第一步：生成轨迹（只在 step_count==0 时调用一次）
    if self.step_count == 0:
        self.generate_trajectory(ts)
    
    # 获取当前和下一个 waypoint
    if self.left_trajectory[0]['t'] == self.step_count:
        self.curr_left_waypoint = self.left_trajectory.pop(0)  # 弹出当前段
    next_left_waypoint = self.left_trajectory[0]  # 下一个段（还在列表中）
    
    # 右臂同理
    if self.right_trajectory[0]['t'] == self.step_count:
        self.curr_right_waypoint = self.right_trajectory.pop(0)
    next_right_waypoint = self.right_trajectory[0]
    
    # 在当前段和下一段之间插值
    left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
    right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)
    
    # 拼接 action
    action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
    action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])
    
    self.step_count += 1
    return np.concatenate([action_left, action_right])
```

### 2) interpolate() 对 xyz、quat、gripper 的处理

**代码位置**: `scripted_policy.py:22-34`

```python
@staticmethod
def interpolate(curr_waypoint, next_waypoint, t):
    # 计算时间分数（当前时间步在段内的比例）
    t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
    
    # 线性插值 xyz
    curr_xyz = curr_waypoint['xyz']
    next_xyz = next_waypoint['xyz']
    xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
    
    # 线性插值 quat（注意：这是近似，真正的四元数应该用 slerp）
    curr_quat = curr_waypoint['quat']
    next_quat = next_waypoint['quat']
    quat = curr_quat + (next_quat - curr_quat) * t_frac
    
    # 线性插值 gripper
    curr_grip = curr_waypoint['gripper']
    next_grip = next_waypoint['gripper']
    gripper = curr_grip + (next_grip - curr_grip) * t_frac
    
    return xyz, quat, gripper
```

**关键点**:
- **xyz**: 线性插值（正确）
- **quat**: 线性插值（近似，理论上应用 slerp，但在小角度下可接受）
- **gripper**: 线性插值（正确）

### 3) before_step() 写入 mocap 并由 MuJoCo weld 驱动整臂

**代码位置**: `ee_sim_env.py:74-90`（已在【A.3】中展示）

**流程**:
1. `before_step()` 将 action 写入 `physics.data.mocap_pos[0,1]` 和 `mocap_quat[0,1]`
2. `physics.data.ctrl` 写入夹爪控制值
3. MuJoCo 的 `weld` 约束（在 XML 中定义）强制 `vx300s_left/gripper_link` 与 `mocap_left` 重合
4. MuJoCo 求解器通过调整 arm joints 的 qpos，使得 `gripper_link` 跟随 `mocap`
5. 整个手臂被"拖拽"到目标位姿（forward kinematics 自动满足）

---

## 【F. XML 的核心耦合点清单（必须点名）】

### 1) Mocap body 必须存在

**XML 位置**: `bimanual_viperx_ee_transfer_cube.xml:15-24`

```xml
<body mocap="true" name="mocap_left" pos="0.095 0.50 0.425">
    <site pos="0 0 0" size="0.003 0.003 0.03" type="box" name="mocap_left_site1" rgba="1 0 0 1"/>
    ...
</body>
<body mocap="true" name="mocap_right" pos="-0.095 0.50 0.425">
    ...
</body>
```

- `mocap="true"` 表示这是 MuJoCo 的 mocap body（可通过 `physics.data.mocap_pos/quat` 直接控制）
- 名字必须是 `mocap_left` 和 `mocap_right`（与 `before_step()` 中的索引对应：`mocap_pos[0]` = left，`mocap_pos[1]` = right）

### 2) Equality weld 约束：mocap → gripper_link

**XML 位置**: `bimanual_viperx_ee_transfer_cube.xml:5-8`

```xml
<equality>
    <weld body1="mocap_left" body2="vx300s_left/gripper_link" solref="0.01 1" solimp=".25 .25 0.001" />
    <weld body1="mocap_right" body2="vx300s_right/gripper_link" solref="0.01 1" solimp=".25 .25 0.001" />
</equality>
```

- `weld` 约束强制两个 body 的位置和姿态完全一致
- `solref` 和 `solimp` 控制约束的刚性和求解参数
- 这是整个 EE 控制的核心：mocap 移动 → weld 约束 → gripper_link 跟随 → 整臂运动

### 3) gripper_link 的局部坐标系决定 quat 的物理含义

- `gripper_link` 的局部坐标系（在 `vx300s_left.xml` 中定义）决定了：
  - `quat=[1,0,0,0]` 对应什么朝向（通常是默认朝向）
  - policy 中的 `gripper_pick_quat` 旋转（例如绕 y 轴 -60 度）的物理效果
- 如果 XML 中 `gripper_link` 的局部坐标系改变，policy 中的 quat 需要相应调整

### 4) 夹爪 actuator 的 ctrl 顺序/维度必须与 before_step() 一致

**XML 位置**: `bimanual_viperx_ee_transfer_cube.xml:34-40`

```xml
<actuator>
    <position ... joint="vx300s_left/left_finger" ... />
    <position ... joint="vx300s_left/right_finger" ... />
    <position ... joint="vx300s_right/left_finger" ... />
    <position ... joint="vx300s_right/right_finger" ... />
</actuator>
```

**代码位置**: `ee_sim_env.py:90`

```python
np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))
#                                   ↑ left_finger    ↑ right_finger   ↑ left_finger    ↑ right_finger
```

- XML 中 actuator 的顺序必须与 `ctrl` 数组的索引对应（左左、左右、右左、右右）
- ViperX 使用两个相反的 slide joint，因此控制值是 `[pos, -pos]`

### 5) Cube joint 名字与 env_state 的布局决定 box_xyz 的含义

**XML 位置**: `bimanual_viperx_ee_transfer_cube.xml:26-27`

```xml
<body name="box" pos="0.2 0.5 0.05">
    <joint name="red_box_joint" type="free" frictionloss="0.01" />
```

**代码位置**: `ee_sim_env.py:179, 186-188`

```python
box_start_idx = physics.model.name2id('red_box_joint', 'joint')
np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)  # 写入

def get_env_state(physics):
    env_state = physics.data.qpos.copy()[16:]  # 读取（假设 box joint 在索引 16 开始）
    return env_state
```

- Joint 名字必须是 `red_box_joint`
- `type="free"` 表示 7 DOF（3 位置 + 4 四元数）
- `get_env_state()` 返回 `qpos[16:]`，假设 box joint 在索引 16（前 16 个是机器人关节）
- Policy 从 `env_state[:3]` 读取 `box_xyz`

### 6) Camera/site 可视化（红色十字）如何挂在 mocap body 上

**XML 位置**: `bimanual_viperx_ee_transfer_cube.xml:16-18, 21-23`

```xml
<body mocap="true" name="mocap_left" pos="...">
    <site pos="0 0 0" size="0.003 0.003 0.03" type="box" name="mocap_left_site1" rgba="1 0 0 1"/>
    <site pos="0 0 0" size="0.003 0.03 0.003" type="box" name="mocap_left_site2" rgba="1 0 0 1"/>
    <site pos="0 0 0" size="0.03 0.003 0.003" type="box" name="mocap_left_site3" rgba="1 0 0 1"/>
</body>
```

- 三个 `site` 元素（小方块）形成红色十字，用于可视化 mocap 位置
- `rgba="1 0 0 1"` 表示红色
- 这些 site 是可视化辅助，不影响物理仿真

---

## 总结：四个关键耦合点

1. **World frame**：所有 xyz/quat 都在 MuJoCo world frame 下，policy 的硬编码偏移量必须匹配 XML 场景的实际尺寸
2. **Mocap/weld**：`mocap_left/right` → `weld` → `vx300s_left/right/gripper_link`，这是 EE 控制的物理机制
3. **gripper_link**：末端 link 的名字和局部坐标系决定 quat 的物理含义，以及 reset 时 mocap 的对齐值
4. **box_xyz**：`sample_box_pose()` 的范围和 policy 的偏移量（如 `+[0,0,0.08]`）必须匹配 XML 中的桌面高度和 box 尺寸

