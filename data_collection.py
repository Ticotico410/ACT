#!/usr/bin/env python3
import argparse
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import mujoco
import mujoco.viewer
import numpy as np
from utils import sample_box_pose

"""
python data_collection.py --model assets/panda_ee_pick_cube.xml --save-dir datasets/sim_pick_cube_scripted --camera-names top,angle
"""

SPACE_KEYS = {32}
ENTER_KEYS = {13, 257}
ESC_KEYS = {27, 256}
GRIPPER_OPEN_KEYS = {69, 101}   # E: Open 
GRIPPER_CLOSE_KEYS = {82, 114}  # R: Close
MAX_GRIPPER_STEP = 0.0002       # Maximum gripper step size

def _find_mocap_id(model: mujoco.MjModel) -> int:
    for body_id in range(model.nbody):
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id >= 0:
            return mocap_id
    raise RuntimeError("No mocap body found in the model.")


def _find_mocap_body_id(model: mujoco.MjModel) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap")
    if bid >= 0:
        return bid
    for body_id in range(model.nbody):
        if int(model.body_mocapid[body_id]) >= 0:
            return body_id
    return -1


def _enable_robot_gravcomp(model: mujoco.MjModel, root_body_name: str = "link0", value: float = 1.0) -> None:
    """Enable gravity compensation for robot subtree only (keep cube under gravity)."""
    root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if root_bid < 0:
        return

    for bid in range(model.nbody):
        cur = bid
        in_subtree = False
        while cur >= 0:
            if cur == root_bid:
                in_subtree = True
                break
            if cur == 0:
                break
            cur = int(model.body_parentid[cur])
        if in_subtree:
            model.body_gravcomp[bid] = value


def _find_ee_target(model: mujoco.MjModel, preferred_site: str) -> Tuple[str, int, str]:
    if preferred_site:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, preferred_site)
        if sid >= 0:
            return "site", sid, preferred_site

    for name in ("ee", "grasp", "tool_site", "tcp"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            return "site", sid, name

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    if bid >= 0:
        return "body", bid, "link7"

    # fallback: any body welded to mocap if possible, otherwise first movable body after world
    for bid in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if "gripper" in bname or "ee" in bname or "tool" in bname:
            return "body", bid, bname

    raise RuntimeError("未找到末端 site/body（可用 --ee-site 指定）。")


def _site_quat(data: mujoco.MjData, site_id: int) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return quat


def _get_ee_pose(data: mujoco.MjData, ee_kind: str, ee_id: int) -> Tuple[np.ndarray, np.ndarray]:
    if ee_kind == "site":
        return data.site_xpos[ee_id].copy(), _site_quat(data, ee_id)
    return data.xpos[ee_id].copy(), data.xquat[ee_id].copy()


def _next_episode_index(save_dir: Path) -> int:
    max_idx = -1
    for p in save_dir.glob("episode_*.hdf5"):
        stem = p.stem
        parts = stem.split("_")
        if not parts:
            continue
        tail = parts[-1]
        if tail.isdigit():
            max_idx = max(max_idx, int(tail))
    return max_idx + 1


def _gripper_range(model: mujoco.MjModel) -> Tuple[float, float]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    if jid < 0:
        return 0.0, 1.0
    low, high = model.jnt_range[jid]
    return float(low), float(high)


def _randomize_red_box_pose(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Randomize red_box free-joint pose using utils.sample_box_pose()."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_box_joint")
    if jid < 0:
        return False

    qpos_adr = int(model.jnt_qposadr[jid])
    dof_adr = int(model.jnt_dofadr[jid])
    pose = sample_box_pose().astype(np.float64)  # [x, y, z, qw, qx, qy, qz]
    data.qpos[qpos_adr : qpos_adr + 7] = pose
    data.qvel[dof_adr : dof_adr + 6] = 0.0
    return True


def _save_episode_hdf5(
    save_path: Path,
    camera_names: List[str],
    data_buf: Dict[str, List[np.ndarray]],
    height: int,
    width: int,
) -> None:
    # 严格对齐 ACT baseline 的保存结构（参考 record_sim_episodes.py）
    max_timesteps = len(data_buf["action"])
    if max_timesteps == 0:
        print(f"[save] 跳过空 episode: {save_path.name}")
        return

    qpos_arr = np.asarray(data_buf["qpos"], dtype=np.float64)
    qvel_arr = np.asarray(data_buf["qvel"], dtype=np.float64)
    action_arr = np.asarray(data_buf["action"], dtype=np.float64)
    qpos_dim = qpos_arr.shape[1]
    qvel_dim = qvel_arr.shape[1]
    action_dim = action_arr.shape[1]

    with h5py.File(str(save_path), "w", rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs["sim"] = True
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in camera_names:
            cam_arr = np.asarray(data_buf[f"images/{cam_name}"], dtype=np.uint8)
            ds = image.create_dataset(
                cam_name,
                (max_timesteps, height, width, 3),
                dtype="uint8",
                chunks=(1, height, width, 3),
            )
            ds[...] = cam_arr

        qpos_ds = obs.create_dataset("qpos", (max_timesteps, qpos_dim))
        qvel_ds = obs.create_dataset("qvel", (max_timesteps, qvel_dim))
        action_ds = root.create_dataset("action", (max_timesteps, action_dim))

        qpos_ds[...] = qpos_arr
        qvel_ds[...] = qvel_arr
        action_ds[...] = action_arr

    print(f"[save] 已保存: {save_path}")
    print(f"[save] episode_len={max_timesteps}")


def _init_episode_writer(
    save_path: Path,
    camera_names: List[str],
    height: int,
    width: int,
    qpos_dim: int = 8,
    qvel_dim: int = 8,
    action_dim: int = 8,
) -> Dict[str, object]:
    root = h5py.File(str(save_path), "w", rdcc_nbytes=1024 ** 2 * 2)
    root.attrs["sim"] = True
    obs = root.create_group("observations")
    image = obs.create_group("images")
    image_ds: Dict[str, h5py.Dataset] = {}
    for cam_name in camera_names:
        image_ds[cam_name] = image.create_dataset(
            cam_name,
            shape=(0, height, width, 3),
            maxshape=(None, height, width, 3),
            dtype="uint8",
            chunks=(1, height, width, 3),
        )
    qpos_ds = obs.create_dataset(
        "qpos", shape=(0, qpos_dim), maxshape=(None, qpos_dim), dtype="float64", chunks=(1, qpos_dim)
    )
    qvel_ds = obs.create_dataset(
        "qvel", shape=(0, qvel_dim), maxshape=(None, qvel_dim), dtype="float64", chunks=(1, qvel_dim)
    )
    action_ds = root.create_dataset(
        "action", shape=(0, action_dim), maxshape=(None, action_dim), dtype="float64", chunks=(1, action_dim)
    )
    return {
        "root": root,
        "path": save_path,
        "len": 0,
        "qpos": qpos_ds,
        "qvel": qvel_ds,
        "action": action_ds,
        "images": image_ds,
    }


def _append_episode_frame(
    writer: Dict[str, object],
    qpos_8: np.ndarray,
    qvel_8: np.ndarray,
    action_8: np.ndarray,
    image_map: Dict[str, np.ndarray],
) -> None:
    idx = int(writer["len"])
    qpos_ds: h5py.Dataset = writer["qpos"]  # type: ignore[assignment]
    qvel_ds: h5py.Dataset = writer["qvel"]  # type: ignore[assignment]
    action_ds: h5py.Dataset = writer["action"]  # type: ignore[assignment]
    image_ds: Dict[str, h5py.Dataset] = writer["images"]  # type: ignore[assignment]

    qpos_ds.resize((idx + 1, qpos_ds.shape[1]))
    qvel_ds.resize((idx + 1, qvel_ds.shape[1]))
    action_ds.resize((idx + 1, action_ds.shape[1]))
    qpos_ds[idx] = qpos_8
    qvel_ds[idx] = qvel_8
    action_ds[idx] = action_8

    for cam_name, img in image_map.items():
        ds = image_ds[cam_name]
        ds.resize((idx + 1, ds.shape[1], ds.shape[2], ds.shape[3]))
        ds[idx] = img

    writer["len"] = idx + 1


def _finalize_episode_writer(writer: Dict[str, object]) -> int:
    root: h5py.File = writer["root"]  # type: ignore[assignment]
    path: Path = writer["path"]  # type: ignore[assignment]
    episode_len = int(writer["len"])
    root.flush()
    root.close()
    print(f"[save] 已保存: {path}")
    print(f"[save] episode_len={episode_len}")
    return episode_len


def _add_label_geom(
    scn: mujoco.MjvScene,
    pos: np.ndarray,
    text: str,
    rgba: np.ndarray,
) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.001, 0.001, 0.001], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    geom.label = text[:98]
    scn.ngeom += 1


def _update_overlay_labels(
    viewer: mujoco.viewer.Handle,
    mocap_pos: np.ndarray,
    mocap_quat: np.ndarray,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    pos_err: float,
) -> None:
    # 在 viewer 场景添加文字标签（MuJoCo 官方 API：MjvGeom.label）
    mocap_text = (
        f"mocap p[{mocap_pos[0]:.3f},{mocap_pos[1]:.3f},{mocap_pos[2]:.3f}] "
        f"q[{mocap_quat[0]:.3f},{mocap_quat[1]:.3f},{mocap_quat[2]:.3f},{mocap_quat[3]:.3f}]"
    )
    ee_text = (
        f"ee p[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}] "
        f"q[{ee_quat[0]:.3f},{ee_quat[1]:.3f},{ee_quat[2]:.3f},{ee_quat[3]:.3f}] "
        f"err={pos_err:.4f}"
    )
    with viewer.lock():
        _add_label_geom(viewer.scn, mocap_pos + np.array([0.0, 0.0, 0.07]), mocap_text, np.array([1.0, 0.0, 1.0, 0.001]))
        _add_label_geom(viewer.scn, ee_pos + np.array([0.0, 0.0, 0.05]), ee_text, np.array([0.0, 1.0, 1.0, 0.001]))


def main() -> None:
    parser = argparse.ArgumentParser(description="鼠标拖动 mocap -> EE 跟随 + HDF5 数据采集（纯 MuJoCo API）")
    parser.add_argument(
        "--model",
        type=str,
        default="assets/panda_ee_pick_cube.xml",
        help="MuJoCo XML 路径",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="datasets/manual_mocap_pick_cube",
        help="数据保存目录",
    )
    parser.add_argument(
        "--camera-names",
        type=str,
        default="top",
        help="逗号分隔相机名，例如 top,angle",
    )
    parser.add_argument("--width", type=int, default=640, help="图像宽度")
    parser.add_argument("--height", type=int, default=480, help="图像高度")
    parser.add_argument("--ee-site", type=str, default="", help="可选：显式指定末端 site 名称")
    parser.add_argument("--robot-gravcomp", type=float, default=1.0, help="机器人重力补偿系数（0~1）")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    episode_idx = _next_episode_index(save_dir)

    camera_names = [c.strip() for c in args.camera_names.split(",") if c.strip()]
    if not camera_names:
        camera_names = ["top"]

    model = mujoco.MjModel.from_xml_path(str(model_path))
    _enable_robot_gravcomp(model, root_body_name="link0", value=float(np.clip(args.robot_gravcomp, 0.0, 1.0)))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    _randomize_red_box_pose(model, data)
    mujoco.mj_forward(model, data)

    mocap_id = _find_mocap_id(model)
    mocap_body_id = _find_mocap_body_id(model)
    ee_kind, ee_id, ee_name = _find_ee_target(model, args.ee_site)
    # 1) 初始位置严格使用 XML 中 mocap body 的 pos（避免 keyframe 后变成 0,0,0）
    if mocap_body_id >= 0:
        xml_mocap_pos = model.body_pos[mocap_body_id].copy()
        data.mocap_pos[mocap_id] = xml_mocap_pos
    else:
        xml_mocap_pos = data.mocap_pos[mocap_id].copy()
    # 2) 姿态固定为“末端朝下”的初始 EE 四元数（你之前验证正确）
    _, ee_init_quat = _get_ee_pose(data, ee_kind, ee_id)
    fixed_mocap_quat = ee_init_quat / max(np.linalg.norm(ee_init_quat), 1e-12)
    data.mocap_quat[mocap_id] = fixed_mocap_quat
    mujoco.mj_forward(model, data)
    print(f"[info] EE 跟踪对象: {ee_kind} '{ee_name}'")
    print("[info] 鼠标拖动: 先选中 mocap，按住 Ctrl + 鼠标右键拖拽平移")
    print(f"[info] mocap 姿态已锁定为固定四元数: {np.array2string(fixed_mocap_quat, precision=6)}")
    print("[info] Space: 开始/暂停记录 | Enter: 保存当前 episode | E: 张开夹爪 | R: 闭合夹爪 | Esc: 退出")

    flags = {"recording": False, "save": False, "exit": False}
    grip_low, grip_high = _gripper_range(model)
    # 当前夹爪值 & 目标值，用于平滑开合
    gripper_value = float(np.clip(data.qpos[7], grip_low, grip_high))
    gripper_target = gripper_value

    writer: Optional[Dict[str, object]] = None

    def key_callback(keycode: int) -> None:
        nonlocal gripper_target
        if keycode in SPACE_KEYS:
            flags["recording"] = not flags["recording"]
            print(f"\n[key] Space -> {'开始记录' if flags['recording'] else '暂停记录'}")
        elif keycode in ENTER_KEYS:
            flags["save"] = True
            print("\n[key] Enter -> 保存当前 episode")
        elif keycode in ESC_KEYS:
            flags["exit"] = True
            print("\n[key] Esc -> 退出")
        elif keycode in GRIPPER_OPEN_KEYS:
            gripper_target = grip_high
            print(f"\n[key] E -> 夹爪张开目标 ({gripper_target:.4f})")
        elif keycode in GRIPPER_CLOSE_KEYS:
            gripper_target = grip_low
            print(f"\n[key] R -> 夹爪闭合目标 ({gripper_target:.4f})")

    last_print_t = 0.0
    mocap_initialized_after_viewer_sync = False
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        if mocap_body_id >= 0:
            with viewer.lock():
                viewer.perturb.select = mocap_body_id
        while viewer.is_running() and not flags["exit"]:
            t0 = time.time()

            # 3) 每帧 sync + step，先同步 GUI 鼠标交互，再积分
            viewer.sync()
            # 某些平台下首帧 sync 会覆盖 mocap 状态，这里强制一次恢复 XML 初值
            if not mocap_initialized_after_viewer_sync:
                data.mocap_pos[mocap_id] = xml_mocap_pos
                data.mocap_quat[mocap_id] = fixed_mocap_quat
                mujoco.mj_forward(model, data)
                mocap_initialized_after_viewer_sync = True
            if viewer.perturb.select < 0 and mocap_body_id >= 0:
                with viewer.lock():
                    viewer.perturb.select = mocap_body_id

            # 固定末端朝向：只允许通过 mocap 平移控制，四元数每帧锁定
            data.mocap_quat[mocap_id] = fixed_mocap_quat

            # 夹爪开合控制（两个 position actuator 共用目标，带平滑）
            # 先把当前值向目标值缓慢推进，防止一下子撞飞 cube
            delta = np.clip(gripper_target - gripper_value, -MAX_GRIPPER_STEP, MAX_GRIPPER_STEP)
            gripper_value = float(np.clip(gripper_value + delta, grip_low, grip_high))
            if model.nu >= 2:
                data.ctrl[0] = gripper_value
                data.ctrl[1] = gripper_value
            mujoco.mj_step(model, data)

            mocap_pos = data.mocap_pos[mocap_id].copy()
            mocap_quat = data.mocap_quat[mocap_id].copy()
            ee_pos, ee_quat = _get_ee_pose(data, ee_kind, ee_id)
            pos_err = float(np.linalg.norm(mocap_pos - ee_pos))

            # 4) 仅做 GUI 叠加（终端不再每帧打印，避免刷屏）
            _update_overlay_labels(viewer, mocap_pos, mocap_quat, ee_pos, ee_quat, pos_err)

            if flags["recording"]:
                if writer is None:
                    save_path = save_dir / f"episode_{episode_idx}.hdf5"
                    writer = _init_episode_writer(
                        save_path=save_path,
                        camera_names=camera_names,
                        height=args.height,
                        width=args.width,
                        qpos_dim=8,
                        qvel_dim=8,
                        action_dim=8,
                    )

                # 严格对齐 ACT baseline：qpos/qvel/action 均使用物理量（m / rad / rad·s⁻¹）
                # qpos[:7] 7 个臂关节角度（rad），qpos[7] 为 finger_joint1 位置（m，范围 0~0.04）
                qpos_8 = data.qpos[:8].copy()
                qvel_8 = data.qvel[:8].copy()

                # action = 当前时刻期望的关节目标（与 record_sim_episodes.py 保存格式一致）
                action_8 = qpos_8.copy()
                action_8[7] = gripper_value  # 使用平滑后的夹爪目标值（单位 m）

                image_map: Dict[str, np.ndarray] = {}
                for cam_name in camera_names:
                    renderer.update_scene(data, camera=cam_name)
                    img = renderer.render()
                    image_map[cam_name] = img.copy()

                _append_episode_frame(
                    writer=writer,
                    qpos_8=qpos_8,
                    qvel_8=qvel_8,
                    action_8=action_8,
                    image_map=image_map,
                )

            if flags["save"]:
                if writer is None or int(writer["len"]) == 0:
                    print(f"[save] 跳过空 episode: episode_{episode_idx}.hdf5")
                else:
                    _finalize_episode_writer(writer)
                    writer = None
                    episode_idx += 1
                flags["save"] = False
                # 按一次 Enter 只采集一条，保存后直接退出
                flags["exit"] = True

            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    if writer is not None:
        # 用户异常退出时，确保文件句柄被关闭，避免损坏
        root: h5py.File = writer["root"]  # type: ignore[assignment]
        root.close()
    print("\n[info] 退出采集。")


if __name__ == "__main__":
    main()