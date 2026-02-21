#!/usr/bin/env python3
import argparse
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np


SPACE_KEYS = {32}
ENTER_KEYS = {13, 257}
ESC_KEYS = {27, 256}
PAUSE_KEYS = {80, 112}  # P / p


def _ensure_visible_mocap_geom(xml_path: Path) -> Tuple[Path, Optional[Path]]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    mocap_body = root.find(".//body[@mocap='true']")
    if mocap_body is None:
        return xml_path, None

    changed = False
    geoms = mocap_body.findall("geom")
    if not geoms:
        ET.SubElement(
            mocap_body,
            "geom",
            {
                "name": "mocap_pick_helper",
                "type": "sphere",
                "size": "0.025",
                "rgba": "1 0 0 0.35",
                "contype": "0",
                "conaffinity": "0",
                "group": "0",
            },
        )
        changed = True
    else:
        g0 = geoms[0]
        rgba = g0.get("rgba")
        if rgba is None:
            g0.set("rgba", "1 0 0 0.35")
            changed = True
        else:
            vals = rgba.strip().split()
            if len(vals) == 4:
                try:
                    alpha = float(vals[3])
                except ValueError:
                    alpha = 1.0
                if alpha <= 0.0:
                    g0.set("rgba", f"{vals[0]} {vals[1]} {vals[2]} 0.35")
                    changed = True
        if g0.get("group") != "0":
            g0.set("group", "0")
            changed = True

    if not changed:
        return xml_path, None

    fd, temp_name = tempfile.mkstemp(
        prefix=f"{xml_path.stem}_with_mocap_geom_",
        suffix=".xml",
        dir=str(xml_path.parent),
    )
    os.close(fd)
    Path(temp_name).write_text(
        ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    return Path(temp_name), Path(temp_name)


def _find_mocap_id(model: mujoco.MjModel) -> int:
    for body_id in range(model.nbody):
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id >= 0:
            return mocap_id
    raise RuntimeError("模型中未找到 mocap body。")


def _find_ee_target(model: mujoco.MjModel, preferred_site: str = "") -> Tuple[str, int, str]:
    if preferred_site:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, preferred_site)
        if sid >= 0:
            return "site", sid, preferred_site

    site_candidates = ["ee", "grasp", "tool_site", "end_effector_site", "tcp"]
    for name in site_candidates:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            return "site", sid, name

    for sid in range(model.nsite):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        low = sname.lower()
        if any(k in low for k in ("ee", "grasp", "tool", "end", "tcp")):
            return "site", sid, sname

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base_link")
    if bid >= 0:
        return "body", bid, "gripper_base_link"

    raise RuntimeError("未找到末端 site，且未找到 body='gripper_base_link'。")


def _site_quat(data: mujoco.MjData, site_id: int) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return quat


def _get_ee_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_kind: str,
    ee_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if ee_kind == "site":
        return data.site_xpos[ee_id].copy(), _site_quat(data, ee_id)
    if ee_kind == "body":
        return data.xpos[ee_id].copy(), data.xquat[ee_id].copy()
    raise ValueError(f"不支持的 ee_kind: {ee_kind}")


def _next_episode_index(save_dir: Path, date_tag: str) -> int:
    max_idx = -1
    prefix = f"episode_{date_tag}_ep"
    for p in save_dir.glob(f"{prefix}*.npz"):
        stem = p.stem
        if not stem.startswith(prefix):
            continue
        tail = stem[len(prefix) :]
        try:
            max_idx = max(max_idx, int(tail))
        except ValueError:
            continue
    return max_idx + 1


def _save_episode(save_path: Path, buf: Dict[str, List[np.ndarray]]) -> None:
    if len(buf["timestamp"]) == 0:
        print(f"[save] 跳过空 episode: {save_path.name}")
        return

    np.savez_compressed(
        str(save_path),
        timestamp=np.asarray(buf["timestamp"], dtype=np.float64),
        mocap_pos=np.asarray(buf["mocap_pos"], dtype=np.float64),
        mocap_quat=np.asarray(buf["mocap_quat"], dtype=np.float64),
        ee_pos=np.asarray(buf["ee_pos"], dtype=np.float64),
        ee_quat=np.asarray(buf["ee_quat"], dtype=np.float64),
        qpos=np.asarray(buf["qpos"], dtype=np.float64),
        qvel=np.asarray(buf["qvel"], dtype=np.float64),
        ctrl=np.asarray(buf["ctrl"], dtype=np.float64),
    )
    print(f"[save] 已保存: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo mocap 拖动 + EE 跟踪可视化 + 交互式数据采集"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path(__file__).resolve().parent / "assets" / "xarm6_ee_pick_cube.xml"),
        help="MuJoCo XML 路径",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./collected_episodes",
        help="episode 保存目录",
    )
    parser.add_argument(
        "--ee-site",
        type=str,
        default="",
        help="可选：指定末端 site 名称（优先使用）",
    )
    args = parser.parse_args()

    model_xml = Path(args.model).expanduser().resolve()
    if not model_xml.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_xml}")

    load_xml, temp_xml = _ensure_visible_mocap_geom(model_xml)
    if temp_xml is not None:
        print(f"[info] 已为 mocap body 添加/修正可见 geom（临时文件）: {temp_xml}")

    model = mujoco.MjModel.from_xml_path(str(load_xml))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    mocap_id = _find_mocap_id(model)
    ee_kind, ee_id, ee_name = _find_ee_target(model, preferred_site=args.ee_site)
    print(f"[info] EE 源: {ee_kind} '{ee_name}'")

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    date_tag = datetime.now().strftime("%Y%m%d")
    episode_idx = _next_episode_index(save_dir, date_tag)

    flags = {"recording": False, "save": False, "exit": False, "paused": False}
    buf: Dict[str, List[np.ndarray]] = {
        "timestamp": [],
        "mocap_pos": [],
        "mocap_quat": [],
        "ee_pos": [],
        "ee_quat": [],
        "qpos": [],
        "qvel": [],
        "ctrl": [],
    }

    def key_callback(keycode: int) -> None:
        if keycode in SPACE_KEYS:
            flags["recording"] = not flags["recording"]
            state = "开始记录" if flags["recording"] else "暂停记录"
            print(f"[key] Space -> {state}")
        elif keycode in ENTER_KEYS:
            flags["save"] = True
            print("[key] Enter -> 保存当前 episode")
        elif keycode in ESC_KEYS:
            flags["exit"] = True
            print("[key] Esc -> 退出")
        elif keycode in PAUSE_KEYS:
            flags["paused"] = not flags["paused"]
            state = "已暂停（可 Ctrl+左键 拖动 mocap）" if flags["paused"] else "继续仿真"
            print(f"[key] P -> {state}")

    print("控制说明: P 暂停/继续 | 暂停时 Ctrl+左键 点选并拖动 mocap 红球 | Space 开始/暂停记录 | Enter 保存 episode | Esc 退出")

    last_text_time = 0.0
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and not flags["exit"]:
            t0 = time.time()
            if not flags["paused"]:
                mujoco.mj_step(model, data)

            mocap_pos = data.mocap_pos[mocap_id].copy()
            mocap_quat = data.mocap_quat[mocap_id].copy()
            ee_pos, ee_quat = _get_ee_pose(model, data, ee_kind, ee_id)

            if flags["recording"] and not flags["paused"]:
                buf["timestamp"].append(np.array(data.time))
                buf["mocap_pos"].append(mocap_pos)
                buf["mocap_quat"].append(mocap_quat)
                buf["ee_pos"].append(ee_pos)
                buf["ee_quat"].append(ee_quat)
                buf["qpos"].append(data.qpos.copy())
                buf["qvel"].append(data.qvel.copy())
                buf["ctrl"].append(data.ctrl.copy())

            if flags["save"]:
                save_path = save_dir / f"episode_{date_tag}_ep{episode_idx:04d}.npz"
                _save_episode(save_path, buf)
                for k in buf:
                    buf[k].clear()
                episode_idx += 1
                flags["save"] = False

            now = time.time()
            if now - last_text_time > 0.10:
                err = np.linalg.norm(mocap_pos - ee_pos)
                print(
                    f"\rrec={int(flags['recording'])} "
                    f"t={data.time:8.3f} "
                    f"mocap_pos={np.array2string(mocap_pos, precision=4)} "
                    f"mocap_quat={np.array2string(mocap_quat, precision=4)} "
                    f"ee_pos={np.array2string(ee_pos, precision=4)} "
                    f"ee_quat={np.array2string(ee_quat, precision=4)} "
                    f"pos_err={err:.5f}",
                    end="",
                    flush=True,
                )
                last_text_time = now

            viewer.sync()
            if not flags["paused"]:
                dt = model.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
            else:
                time.sleep(0.02)

    print("\n[info] 已退出。")
    if temp_xml is not None and temp_xml.exists():
        temp_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
