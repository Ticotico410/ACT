import os
import cv2
import time
import h5py
import argparse
import numpy as np
import mujoco
import mujoco.viewer

from pathlib import Path
from utils import sample_box_pose
from typing import Dict, List, Optional, Tuple
from constants import DT, PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN


SPACE_KEYS = {32}
ENTER_KEYS = {13, 257}
ESC_KEYS = {27, 256}
GRIPPER_OPEN_KEYS = {69, 101}   # E: Open 
GRIPPER_CLOSE_KEYS = {82, 114}  # R: Close
MAX_GRIPPER_STEP = 0.0005       # Maximum gripper step size

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

    for bid in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if "gripper" in bname or "ee" in bname or "tool" in bname:
            return "body", bid, bname

    raise RuntimeError("No end-effector site/body found in the model.")


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
    max_timesteps = len(data_buf["action"])
    if max_timesteps == 0:
        print(f"Skipping empty episode: {save_path.name}")
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

    print(f"Saved: {save_path}")
    print(f"episode_len={max_timesteps}")


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
    print(f"Saved: {path}")
    print(f"episode_len={episode_len}")
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
    parser = argparse.ArgumentParser(description="Mouse drag mocap -> EE follow + HDF5 data collection")
    parser.add_argument("--model", type=str, default="assets/panda_ee_pick_cube.xml", help="MuJoCo XML path")
    parser.add_argument("--save-dir", type=str, default="datasets/manual_mocap_pick_cube", help="Data save directory")
    parser.add_argument("--camera-names", type=str, default="top", help="top,angle")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--ee-site", type=str, default="", help="end-effector site name")
    parser.add_argument("--robot-gravcomp", type=float, default=1.0, help="Robot gravity compensation coefficient")
    parser.add_argument("--save-dt", type=float, default=DT, help="Save data step")
    parser.add_argument("--guide-cameras", type=str, default="angle,left_pillar,right_pillar,front_close", help="Fixed guide window")
    parser.add_argument("--guide-width", type=int, default=400, help="Guide window width")
    parser.add_argument("--guide-height", type=int, default=300, help="Guide window height")
    parser.add_argument("--max-timesteps", type=int, default=0, help="Maximum number of frames to save per path")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

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
    guide_camera_names = [c.strip() for c in args.guide_cameras.split(",") if c.strip()]
    if not guide_camera_names:
        guide_camera_names = ["top", "angle"]
    guide_renderers = {
        cam_name: mujoco.Renderer(model, height=args.guide_height, width=args.guide_width)
        for cam_name in guide_camera_names
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name) >= 0
    }
    save_stride = max(1, int(round(float(args.save_dt) / float(model.opt.timestep))))

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    _randomize_red_box_pose(model, data)
    mujoco.mj_forward(model, data)

    mocap_id = _find_mocap_id(model)
    mocap_body_id = _find_mocap_body_id(model)
    ee_kind, ee_id, ee_name = _find_ee_target(model, args.ee_site)
    # Initial position strictly uses the pos of the mocap body
    if mocap_body_id >= 0:
        xml_mocap_pos = model.body_pos[mocap_body_id].copy()
        data.mocap_pos[mocap_id] = xml_mocap_pos
    else:
        xml_mocap_pos = data.mocap_pos[mocap_id].copy()
    # Initial EE quaternion fixed to downward
    _, ee_init_quat = _get_ee_pose(data, ee_kind, ee_id)
    fixed_mocap_quat = ee_init_quat / max(np.linalg.norm(ee_init_quat), 1e-12)
    data.mocap_quat[mocap_id] = fixed_mocap_quat
    mujoco.mj_forward(model, data)
    print(f"[info] EE tracking object: {ee_kind} '{ee_name}'")
    print("[info] Mouse drag: first select mocap, then hold Ctrl + mouse right click to drag")
    print(f"[info] mocap pose locked to fixed quaternion: {np.array2string(fixed_mocap_quat, precision=6)}")
    max_steps_info = "unlimited" if int(args.max_timesteps) <= 0 else str(int(args.max_timesteps))
    print(
        f"[info] Save sampling: save 1 frame every {save_stride} simulation steps "
        f"(sim dt={model.opt.timestep:.4f}s -> save dt≈{save_stride * model.opt.timestep:.4f}s), "
        f"max_timesteps={max_steps_info}"
    )
    guide_enabled = (cv2 is not None) and bool(guide_renderers)

    print("[info] Space: start/pause recording | Enter: save current episode | E: open gripper | R: close gripper | Esc: exit")

    flags = {"recording": False, "save": False, "exit": False}
    saved_any_episode = False
    record_step_counter = 0
    grip_low, grip_high = _gripper_range(model)
    gripper_value = float(np.clip(data.qpos[7], grip_low, grip_high))
    gripper_target = gripper_value

    writer: Optional[Dict[str, object]] = None

    def key_callback(keycode: int) -> None:
        nonlocal gripper_target, record_step_counter
        if keycode in SPACE_KEYS:
            flags["recording"] = not flags["recording"]
            if flags["recording"] and writer is None:
                record_step_counter = 0
            print(f"\n[key] Space -> {'start recording' if flags['recording'] else 'pause recording'}")
        elif keycode in ENTER_KEYS:
            flags["save"] = True
            print("\n[key] Enter -> save current episode")
        elif keycode in ESC_KEYS:
            flags["exit"] = True
            print("\n[key] Esc -> exit")
        elif keycode in GRIPPER_OPEN_KEYS:
            gripper_target = grip_high
            norm = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(gripper_target)
            print(f"\n[key] E -> gripper open target physical={gripper_target:.4f}m normalized={norm:.4f}")
        elif keycode in GRIPPER_CLOSE_KEYS:
            gripper_target = grip_low
            norm = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(gripper_target)
            print(f"\n[key] R -> gripper close target physical={gripper_target:.4f}m normalized={norm:.4f}")

    last_print_t = 0.0
    mocap_initialized_after_viewer_sync = False
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        if mocap_body_id >= 0:
            with viewer.lock():
                viewer.perturb.select = mocap_body_id
        while viewer.is_running() and not flags["exit"]:
            t0 = time.time()
            viewer.sync()
            if not mocap_initialized_after_viewer_sync:
                data.mocap_pos[mocap_id] = xml_mocap_pos
                data.mocap_quat[mocap_id] = fixed_mocap_quat
                mujoco.mj_forward(model, data)
                mocap_initialized_after_viewer_sync = True
            if viewer.perturb.select < 0 and mocap_body_id >= 0:
                with viewer.lock():
                    viewer.perturb.select = mocap_body_id

            # Fixed end-effector orientation
            data.mocap_quat[mocap_id] = fixed_mocap_quat

            # Gripper open/close control
            delta = np.clip(gripper_target - gripper_value, -MAX_GRIPPER_STEP, MAX_GRIPPER_STEP)
            gripper_value = float(np.clip(gripper_value + delta, grip_low, grip_high))
            if model.nu >= 2:
                data.ctrl[0] = gripper_value
                data.ctrl[1] = gripper_value
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

                should_save_this_step = (record_step_counter % save_stride == 0)
                max_steps = int(args.max_timesteps)
                reached_limit = (max_steps > 0 and int(writer["len"]) >= max_steps)
                if should_save_this_step and not reached_limit:
                    qpos_8 = data.qpos[:8].copy()
                    qvel_8 = data.qvel[:8].copy()
                    qpos_8[7] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos_8[7])
                    qvel_8[7] = PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(qvel_8[7])
                    action_8 = qpos_8.copy()
                    action_8[7] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(gripper_value)

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

                    if max_steps > 0 and int(writer["len"]) >= max_steps:
                        flags["recording"] = False
                        print(f"\n[info] Reached max_timesteps={max_steps}, automatically pause recording")

                record_step_counter += 1

            mujoco.mj_step(model, data)

            mocap_pos = data.mocap_pos[mocap_id].copy()
            mocap_quat = data.mocap_quat[mocap_id].copy()
            ee_pos, ee_quat = _get_ee_pose(data, ee_kind, ee_id)
            pos_err = float(np.linalg.norm(mocap_pos - ee_pos))

            # GUI overlay
            _update_overlay_labels(viewer, mocap_pos, mocap_quat, ee_pos, ee_quat, pos_err)
            if guide_enabled:
                status = "REC" if flags["recording"] else "PAUSE"
                tiles: List[np.ndarray] = []
                for cam_name, guide_renderer in guide_renderers.items():
                    guide_renderer.update_scene(data, camera=cam_name)
                    guide_img = guide_renderer.render()
                    guide_bgr = np.ascontiguousarray(guide_img[..., ::-1])
                    cv2.putText(
                        guide_bgr,
                        f"{cam_name} | {status} | err={pos_err:.3f}m",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (30, 220, 30),
                        1,
                        cv2.LINE_AA,
                    )
                    tiles.append(guide_bgr)

                n_cols = 2
                blank = np.zeros_like(tiles[0])
                rows: List[np.ndarray] = []
                for i in range(0, len(tiles), n_cols):
                    row_tiles = tiles[i : i + n_cols]
                    if len(row_tiles) < n_cols:
                        row_tiles.append(blank)
                    rows.append(cv2.hconcat(row_tiles))
                guide_canvas = cv2.vconcat(rows)
                cv2.imshow("ACT Guide MultiView", guide_canvas)
                cv2.waitKey(1)

            if flags["save"]:
                if writer is None or int(writer["len"]) == 0:
                    print(f"[save] Skipped empty episode: episode_{episode_idx}.hdf5")
                else:
                    _finalize_episode_writer(writer)
                    writer = None
                    episode_idx += 1
                    saved_any_episode = True
                flags["save"] = False
                # Only collect one episode per Enter press, exit after saving
                flags["exit"] = True

            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    if writer is not None:
        # Delete incomplete file if Enter is not pressed successfully
        root: h5py.File = writer["root"]  # type: ignore[assignment]
        path: Path = writer["path"]       # type: ignore[assignment]
        root.close()
        if not saved_any_episode and path.exists():
            path.unlink()
    if cv2 is not None:
        cv2.destroyAllWindows()
    print("\n[info] Exit collection.")


if __name__ == "__main__":
    main()