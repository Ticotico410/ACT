#!/usr/bin/env python3
import argparse
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import mujoco
import mujoco.viewer
import numpy as np


SPACE_KEYS = {32}
ENTER_KEYS = {13, 257}
ESC_KEYS = {27, 256}
GRIPPER_OPEN_KEYS = {81, 113}   # Q / q 张开
GRIPPER_CLOSE_KEYS = {69, 101}  # E / e 闭合


def _ensure_mocap_selectable(xml_path: Path) -> Tuple[Path, Optional[Path]]:
    """If mocap body has no geom, create a temporary XML with a visible pick helper geom."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    mocap_body = root.find(".//body[@mocap='true']")
    if mocap_body is None:
        return xml_path, None

    geoms = mocap_body.findall("geom")
    changed = False
    if not geoms:
        ET.SubElement(
            mocap_body,
            "geom",
            {
                "name": "mocap_pick_helper",
                "type": "sphere",
                "size": "0.02",
                "rgba": "1 0 1 0.45",
                "contype": "0",
                "conaffinity": "0",
                "group": "0",
            },
        )
        changed = True

    if not changed:
        return xml_path, None

    fd, temp_name = tempfile.mkstemp(
        prefix=f"{xml_path.stem}_mocap_pickable_",
        suffix=".xml",
        dir=str(xml_path.parent),
    )
    os.close(fd)
    Path(temp_name).write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return Path(temp_name), Path(temp_name)


def _find_mocap_id(model: mujoco.MjModel) -> int:
    for body_id in range(model.nbody):
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id >= 0:
            return mocap_id
    raise RuntimeError("模型中未找到 mocap body。")


def _find_mocap_body_id(model: mujoco.MjModel) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap")
    if bid >= 0:
        return bid
    for body_id in range(model.nbody):
        if int(model.body_mocapid[body_id]) >= 0:
            return body_id
    return -1


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


def _normalize_gripper_position(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray) -> Tuple[float, float]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    if jid < 0:
        return 0.0, 0.0
    low, high = model.jnt_range[jid]
    span = max(high - low, 1e-8)
    qpos_norm = float((qpos[7] - low) / span)
    qvel_norm = float(qvel[7] / span)
    return qpos_norm, qvel_norm


def _gripper_range(model: mujoco.MjModel) -> Tuple[float, float]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    if jid < 0:
        return 0.0, 1.0
    low, high = model.jnt_range[jid]
    return float(low), float(high)


def _find_finger_joint_info(model: mujoco.MjModel) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    if j1 < 0 or j2 < 0:
        return None, None, None, None
    q1 = int(model.jnt_qposadr[j1])
    q2 = int(model.jnt_qposadr[j2])
    v1 = int(model.jnt_dofadr[j1])
    v2 = int(model.jnt_dofadr[j2])
    return q1, q2, v1, v2


def _save_episode_hdf5(
    save_path: Path,
    camera_names: List[str],
    data_buf: Dict[str, List[np.ndarray]],
    height: int,
    width: int,
) -> None:
    max_timesteps = len(data_buf["action"])
    if max_timesteps == 0:
        print(f"[save] 跳过空 episode: {save_path.name}")
        return

    with h5py.File(str(save_path), "w", rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs["sim"] = True
        root.create_dataset("timestamp", data=np.asarray(data_buf["timestamp"], dtype=np.float64))

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

        obs.create_dataset("qpos", data=np.asarray(data_buf["qpos"], dtype=np.float64))
        obs.create_dataset("qvel", data=np.asarray(data_buf["qvel"], dtype=np.float64))
        root.create_dataset("action", data=np.asarray(data_buf["action"], dtype=np.float64))

        # 额外保存调试/对齐信息
        obs.create_dataset("mocap_pose", data=np.asarray(data_buf["mocap_pose"], dtype=np.float64))
        obs.create_dataset("ee_pose", data=np.asarray(data_buf["ee_pose"], dtype=np.float64))
        root.create_dataset("ctrl", data=np.asarray(data_buf["ctrl"], dtype=np.float64))
        root.create_dataset("raw_qpos", data=np.asarray(data_buf["raw_qpos"], dtype=np.float64))
        root.create_dataset("raw_qvel", data=np.asarray(data_buf["raw_qvel"], dtype=np.float64))

    print(f"[save] 已保存: {save_path}")


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

    load_path, temp_xml = _ensure_mocap_selectable(model_path)
    if temp_xml is not None:
        print(f"[info] 已自动为 mocap 增加可选取 geom: {temp_xml}")

    model = mujoco.MjModel.from_xml_path(str(load_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    mocap_id = _find_mocap_id(model)
    ee_kind, ee_id, ee_name = _find_ee_target(model, args.ee_site)
    # 启动时把 mocap 对齐到当前 keyframe 下的 EE，避免一进场就被 weld 拉到奇怪姿态
    ee_init_pos, ee_init_quat = _get_ee_pose(data, ee_kind, ee_id)
    data.mocap_pos[mocap_id] = ee_init_pos
    data.mocap_quat[mocap_id] = ee_init_quat
    mujoco.mj_forward(model, data)
    print(f"[info] EE 跟踪对象: {ee_kind} '{ee_name}'")
    print("[info] 鼠标拖动: 先选中 mocap，按住 Ctrl + 鼠标右键拖拽平移（已禁用旋转，保持末端朝向稳定）")
    print("[info] Space: 开始/暂停记录 | Enter: 保存当前 episode | Q: 张开夹爪 | E: 闭合夹爪 | Esc: 退出")

    flags = {"recording": False, "save": False, "exit": False}
    grip_low, grip_high = _gripper_range(model)
    gripper_target = float(np.clip(data.qpos[7], grip_low, grip_high))
    q1_adr, q2_adr, v1_adr, v2_adr = _find_finger_joint_info(model)

    data_buf: Dict[str, List[np.ndarray]] = {
        "timestamp": [],
        "qpos": [],
        "qvel": [],
        "action": [],
        "mocap_pose": [],
        "ee_pose": [],
        "ctrl": [],
        "raw_qpos": [],
        "raw_qvel": [],
    }
    for cam_name in camera_names:
        data_buf[f"images/{cam_name}"] = []

    def _clear_buffer() -> None:
        for k in data_buf:
            data_buf[k].clear()

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
            print(f"\n[key] Q -> 夹爪张开 ({gripper_target:.4f})")
        elif keycode in GRIPPER_CLOSE_KEYS:
            gripper_target = grip_low
            print(f"\n[key] E -> 夹爪闭合 ({gripper_target:.4f})")

    last_print_t = 0.0
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        mocap_body_id = _find_mocap_body_id(model)
        if mocap_body_id >= 0:
            with viewer.lock():
                viewer.perturb.select = mocap_body_id
                viewer.perturb.active = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
                viewer.perturb.active2 = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
        while viewer.is_running() and not flags["exit"]:
            t0 = time.time()

            # 3) 每帧 sync + step，先同步 GUI 鼠标交互，再积分
            viewer.sync()
            if viewer.perturb.select < 0 and mocap_body_id >= 0:
                with viewer.lock():
                    viewer.perturb.select = mocap_body_id
                    viewer.perturb.active = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
                    viewer.perturb.active2 = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)

            # 夹爪开合控制（两个 position actuator 共用目标）
            if model.nu >= 2:
                data.ctrl[0] = gripper_target
                data.ctrl[1] = gripper_target
            # 保险做法：直接钳位 finger qpos，确保 Q/E 有稳定可见反馈
            if q1_adr is not None and q2_adr is not None:
                data.qpos[q1_adr] = gripper_target
                data.qpos[q2_adr] = gripper_target
                if v1_adr is not None and v2_adr is not None:
                    data.qvel[v1_adr] = 0.0
                    data.qvel[v2_adr] = 0.0
                mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)

            mocap_pos = data.mocap_pos[mocap_id].copy()
            mocap_quat = data.mocap_quat[mocap_id].copy()
            ee_pos, ee_quat = _get_ee_pose(data, ee_kind, ee_id)
            pos_err = float(np.linalg.norm(mocap_pos - ee_pos))

            # 4) GUI 叠加 + 控制台实时输出（便于复制日志）
            _update_overlay_labels(viewer, mocap_pos, mocap_quat, ee_pos, ee_quat, pos_err)
            now = time.time()
            if now - last_print_t > 0.1:
                print(
                    f"\rrec={int(flags['recording'])} t={data.time:8.3f} "
                    f"mocap_pos={np.array2string(mocap_pos, precision=4)} "
                    f"mocap_quat={np.array2string(mocap_quat, precision=4)} "
                    f"ee_pos={np.array2string(ee_pos, precision=4)} "
                    f"ee_quat={np.array2string(ee_quat, precision=4)} "
                    f"pos_err={pos_err:.5f} "
                    f"grip={gripper_target:.4f}",
                    end="",
                    flush=True,
                )
                last_print_t = now

            if flags["recording"]:
                qpos_norm_grip, qvel_norm_grip = _normalize_gripper_position(model, data.qpos, data.qvel)
                qpos_8 = np.concatenate([data.qpos[:7].copy(), np.array([qpos_norm_grip])])
                qvel_8 = np.concatenate([data.qvel[:7].copy(), np.array([qvel_norm_grip])])

                # 手动拖拽采集没有策略动作，这里按 ACT 数据格式写 8 维 action，占位为当前状态指令
                action_8 = qpos_8.copy()
                action_8[7] = (gripper_target - grip_low) / max(grip_high - grip_low, 1e-8)

                data_buf["timestamp"].append(np.array(data.time, dtype=np.float64))
                data_buf["qpos"].append(qpos_8)
                data_buf["qvel"].append(qvel_8)
                data_buf["action"].append(action_8)
                data_buf["mocap_pose"].append(np.concatenate([mocap_pos, mocap_quat]))
                data_buf["ee_pose"].append(np.concatenate([ee_pos, ee_quat]))
                data_buf["ctrl"].append(data.ctrl.copy())
                data_buf["raw_qpos"].append(data.qpos.copy())
                data_buf["raw_qvel"].append(data.qvel.copy())

                for cam_name in camera_names:
                    renderer.update_scene(data, camera=cam_name)
                    img = renderer.render()
                    data_buf[f"images/{cam_name}"].append(img.copy())

            if flags["save"]:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = save_dir / f"episode_{stamp}_{episode_idx:04d}.hdf5"
                _save_episode_hdf5(save_path, camera_names, data_buf, args.height, args.width)
                _clear_buffer()
                episode_idx += 1
                flags["save"] = False

            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    print("\n[info] 退出采集。")
    if temp_xml is not None and temp_xml.exists():
        temp_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()