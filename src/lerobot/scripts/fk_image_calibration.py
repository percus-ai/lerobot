#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# NumPy 2.x 互換パッチ（urdf系が np.float を使うので）
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

# urchin を優先して使う（なければ urdfpy）
try:
    from urchin import URDF  # type: ignore
    _USE_URCHIN = True
except ImportError:
    from urdfpy import URDF  # type: ignore
    _USE_URCHIN = False


# ログに出てくる SO-101 の関節キー（LeRobot の action dict）
ACTION_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# 双腕用（左）
ACTION_KEYS_LEFT = [f"left_{k}" for k in ACTION_KEYS]
# 双腕用（右）
ACTION_KEYS_RIGHT = [f"right_{k}" for k in ACTION_KEYS]


def detect_arm_mode(action: Dict[str, Any]) -> str:
    """
    アクションデータから単腕/双腕モードを検出する。
    Returns: "single", "bimanual", or "unknown"
    """
    if isinstance(action, dict):
        keys = set(action.keys())
        has_left = any(k.startswith("left_") for k in keys)
        has_right = any(k.startswith("right_") for k in keys)
        if has_left and has_right:
            return "bimanual"
        elif not has_left and not has_right:
            return "single"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate affine transform from SO-101 FK (x,y) to image (u,v) "
            "using manual clicks, and save fk_image.conf under scripts/SO101."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help=(
            "Root directory of the recorded dataset, i.e. the parent of 'attn_videos'. "
            "Example: ./dataset/eval_test など。"
        ),
    )
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode index used in episode_values_{episode}.json (0-based).",
    )

    script_dir = Path(__file__).resolve().parent
    default_urdf_path = script_dir / "SO101" / "so101_new_calib.urdf"

    parser.add_argument(
        "--urdf-path",
        type=str,
        default=str(default_urdf_path),
        help="Path to SO-101 URDF file (default: SO101/so101_new_calib.urdf).",
    )
    parser.add_argument(
        "--ee-link",
        type=str,
        default="gripper_frame_link",
        help="End-effector link name in the URDF (default: gripper_frame_link).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=20,
        help="Step size when scanning frames for calibration candidates (default: 20).",
    )
    return parser.parse_args()


# ====== visualize_inference_trajectory 側と同じヘルパ類 ======

def load_episode_log(attn_dir: Path, episode: int) -> Dict[str, Any]:
    json_path = attn_dir / f"episode_values_{episode}.json"
    print(f"[STEP 1] Loading JSON log: {json_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON log file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "frames" not in data or not isinstance(data["frames"], list):
        raise ValueError(f"Invalid log format: 'frames' missing or not a list in {json_path}")

    total_actions = 0
    for rec in data["frames"]:
        actions = rec.get("actions", [])
        if isinstance(actions, list):
            total_actions += len(actions)

    print(
        f"[STEP 1] OK: loaded {len(data['frames'])} chunks, "
        f"total {total_actions} actions in episode {episode}"
    )
    return data


def resolve_video_path(attn_dir: Path, episode: int, repo_id: str | None) -> Path:
    candidates: List[Path] = []
    # ポリシーありのアテンション動画（旧仕様）
    candidates.append(attn_dir / f"attn_all_cameras_episode_{episode}.mp4")
    # ポリシーなしの簡易結合動画（SimpleRecordingManager）
    candidates.append(attn_dir / f"all_cameras_episode_{episode}.mp4")

    if repo_id is not None:
        repo_sanitized = repo_id.replace("/", "_")
        candidates.append(attn_dir / f"{repo_sanitized}_ep{episode:06d}.mp4")

    print("[STEP 2] Resolving attention video path.")
    for p in candidates:
        print(f"[STEP 2]  Trying candidate: {p}")
        if p.exists():
            print(f"[STEP 2]  -> FOUND: {p}")
            return p

    print("[STEP 2] ERROR: none of the candidate videos exist.")
    if attn_dir.exists():
        print("[STEP 2] Existing files in attn_videos:")
        for x in sorted(attn_dir.iterdir()):
            print(f"  - {x.name}")
    else:
        print(f"[STEP 2] attn_videos directory does not exist at: {attn_dir}")

    raise FileNotFoundError(
        f"No attention video found for episode {episode} in {attn_dir}. "
        f"Expected one of: attn_all_cameras_episode_{episode}.mp4, "
        f"all_cameras_episode_{episode}.mp4, or repo-specific names."
    )


def open_video(attn_dir: Path, episode: int, repo_id: str | None) -> cv2.VideoCapture:
    video_path = resolve_video_path(attn_dir, episode, repo_id)
    print(f"[STEP 2] Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"[STEP 2] OK: video opened successfully "
        f"(frames={frame_count}, fps={fps:.2f}, size={width}x{height})"
    )
    return cap


def build_joint_config_from_action(
    action: Dict[str, Any],
    urdf_joint_names: List[str],
    action_keys: List[str] | None = None,
) -> Dict[str, float]:
    """
    URDF の関節名と ACTION_KEYS の順番が異なる場合があるので、名前で対応付ける。
    action_keys: 使用するキーのリスト。Noneの場合はACTION_KEYSを使用。
                 双腕の場合はACTION_KEYS_LEFT or ACTION_KEYS_RIGHTを指定。
    """
    if action_keys is None:
        action_keys = ACTION_KEYS

    if len(action_keys) > len(urdf_joint_names):
        raise ValueError(
            f"Number of action_keys ({len(action_keys)}) is greater than number of URDF joints "
            f"({len(urdf_joint_names)}). Cannot build joint config."
        )

    # "shoulder_pan.pos" -> "shoulder_pan"
    # "left_shoulder_pan.pos" -> "shoulder_pan" (プレフィックスを除去)
    base_vals_deg: Dict[str, float] = {}
    for key in action_keys:
        if key not in action:
            raise KeyError(f"Action key '{key}' not found in action dict: {list(action.keys())}")
        # プレフィックスを除去してベース名を取得
        base_key = key
        if base_key.startswith("left_"):
            base_key = base_key[5:]  # "left_" を除去
        elif base_key.startswith("right_"):
            base_key = base_key[6:]  # "right_" を除去
        base_name = base_key.split(".")[0]
        base_vals_deg[base_name] = float(action[key])

    joint_cfg: Dict[str, float] = {}
    for joint_name in urdf_joint_names:
        if joint_name not in base_vals_deg:
            continue
        joint_cfg[joint_name] = float(np.deg2rad(base_vals_deg[joint_name]))

    return joint_cfg


def compute_fk_trajectory(
    urdf_path: Path,
    ee_link_name: str,
    frames: List[Dict[str, Any]],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray] | None, str]:
    """
    FK軌跡を計算する。双腕の場合は左右別々の軌跡を返す。
    Returns:
        frame_to_pos: 左腕/単腕の軌跡
        frame_to_pos_right: 右腕の軌跡（双腕の場合）、単腕の場合はNone
        arm_mode: "single" or "bimanual"
    """
    print(f"[STEP 3] Loading URDF: {urdf_path}")
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    if _USE_URCHIN:
        robot = URDF.load(str(urdf_path), lazy_load_meshes=True)
    else:
        robot = URDF.load(str(urdf_path))

    print(f"[STEP 3] URDF loaded. Links: {len(robot.link_map)}, Joints: {len(robot.joint_map)}")

    movable_joints = [j for j in robot.joints if getattr(j, "joint_type", None) != "fixed"]
    urdf_joint_names = [j.name for j in movable_joints]
    print(f"[STEP 3] Movable joints ({len(urdf_joint_names)}): {urdf_joint_names}")

    if len(urdf_joint_names) < len(ACTION_KEYS):
        raise ValueError(
            f"URDF has fewer movable joints ({len(urdf_joint_names)}) than ACTION_KEYS ({len(ACTION_KEYS)})."
        )

    if ee_link_name not in robot.link_map:
        raise ValueError(
            f"EE link '{ee_link_name}' not found in URDF. "
            f"Available links example: {list(robot.link_map.keys())[:10]}"
        )
    ee_link = robot.link_map[ee_link_name]

    # アームモードを検出
    arm_mode = "single"
    for rec in frames:
        actions = rec.get("actions", [])
        if actions and isinstance(actions[0], dict):
            arm_mode = detect_arm_mode(actions[0])
            break

    print(f"[STEP 3] Detected arm mode: {arm_mode}")

    frame_to_pos: Dict[int, np.ndarray] = {}
    frame_to_pos_right: Dict[int, np.ndarray] | None = {} if arm_mode == "bimanual" else None
    num_actions = 0

    for rec in frames:
        base_idx = int(rec["frame_idx"])
        actions = rec.get("actions", [])
        if not isinstance(actions, list):
            raise ValueError("Each frame record must have 'actions' as a list.")

        for local_idx, action in enumerate(actions):
            if not isinstance(action, dict):
                raise ValueError(f"Action must be a dict, got {type(action)}")
            frame_idx = base_idx + local_idx

            if arm_mode == "bimanual":
                # 左腕
                try:
                    joint_cfg_left = build_joint_config_from_action(action, urdf_joint_names, ACTION_KEYS_LEFT)
                    fk_all_left = robot.link_fk(joint_cfg_left)
                    T_left = fk_all_left[ee_link]
                    pos_left = np.asarray(T_left[:3, 3], dtype=float)
                    frame_to_pos[frame_idx] = pos_left
                except (KeyError, ValueError):
                    pass

                # 右腕
                try:
                    joint_cfg_right = build_joint_config_from_action(action, urdf_joint_names, ACTION_KEYS_RIGHT)
                    fk_all_right = robot.link_fk(joint_cfg_right)
                    T_right = fk_all_right[ee_link]
                    pos_right = np.asarray(T_right[:3, 3], dtype=float)
                    frame_to_pos_right[frame_idx] = pos_right
                except (KeyError, ValueError):
                    pass
            else:
                # 単腕
                joint_cfg = build_joint_config_from_action(action, urdf_joint_names, ACTION_KEYS)
                fk_all = robot.link_fk(joint_cfg)
                T = fk_all[ee_link]
                pos = np.asarray(T[:3, 3], dtype=float)
                frame_to_pos[frame_idx] = pos

            num_actions += 1

    print(
        f"[STEP 3] OK: computed FK for {len(frame_to_pos)} control frames "
        f"(total actions processed={num_actions}, arm_mode={arm_mode})."
    )

    return frame_to_pos, frame_to_pos_right, arm_mode


# ====== キャリブレーション固有の処理 ======

def compute_affine(points_xy: np.ndarray, points_uv: np.ndarray) -> np.ndarray:
    assert points_xy.shape[0] == points_uv.shape[0]
    n = points_xy.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 points to estimate affine transform, got {n}.")

    X = points_xy
    U = points_uv

    A = np.zeros((2 * n, 6), dtype=float)
    b = np.zeros((2 * n,), dtype=float)

    for i in range(n):
        x, y = X[i]
        u, v = U[i]
        A[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0]
        A[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0]
        b[2 * i] = u
        b[2 * i + 1] = v

    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    aff = params.reshape(2, 3)
    return aff


def interactive_collect_points(
    cap: cv2.VideoCapture,
    frame_to_pos: Dict[int, np.ndarray],
    frame_step: int,
    arm_label: str = "",
) -> Tuple[List[int], List[np.ndarray], List[Tuple[int, int]]]:
    """
    インタラクティブにキャリブレーション点を収集する。
    arm_label: 表示用のラベル（"LEFT", "RIGHT", ""など）
    
    操作方法：
    - 左クリック: その点を登録して自動的に次のフレームへ
    - n: 何も登録せずに次のフレームへ
    - b: 一つ前のフレームに戻る
    - q: 終了（3点以上必要）
    """
    frame_indices = sorted(frame_to_pos.keys())
    if not frame_indices:
        raise ValueError("frame_to_pos is empty; nothing to calibrate.")

    label_str = f" [{arm_label}]" if arm_label else ""
    print(
        f"\n--- Calibration controls{label_str} ---\n"
        "  左クリック: その点を登録して自動的に次のフレームへ\n"
        "  n: 何も登録せずに次のフレームへ\n"
        "  b: 一つ前のフレームに戻る\n"
        "  q: 終了（登録した点でアフィン推定、最低3点必要）\n"
        "  ※ウィンドウをクリックしてフォーカスしてからキーを押してください\n"
        "----------------------------\n"
    )

    window_name = f"fk_image_calibration{label_str}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # クリックで登録＆次フレームに進むためのフラグ
    clicked_and_registered = [False]

    used_frames: List[int] = []
    world_pts: List[np.ndarray] = []
    image_pts: List[Tuple[int, int]] = []

    current_frame_idx = [0]  # クロージャ用
    current_pos_3d: List[np.ndarray] = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # クリックしたら即登録
            frame_idx = frame_indices[current_frame_idx[0]]
            pos_3d = current_pos_3d[0]
            if pos_3d is not None:
                used_frames.append(frame_idx)
                world_pts.append(pos_3d.copy())
                image_pts.append((x, y))
                print(f"[登録{label_str}] frame {frame_idx}: world={pos_3d}, image=({x},{y}) - 合計{len(used_frames)}点")
                clicked_and_registered[0] = True

    cv2.setMouseCallback(window_name, on_mouse)

    i = 0
    quit_requested = False
    while not quit_requested:
        # フレーム範囲外になったら最初に戻る
        if i >= len(frame_indices):
            print(f"\n[INFO] フレーム終端に到達。最初に戻ります。現在{len(used_frames)}点登録済み。")
            i = 0
        if i < 0:
            i = 0

        frame_idx = frame_indices[i]
        pos_3d = frame_to_pos[frame_idx]
        current_frame_idx[0] = i
        current_pos_3d[0] = pos_3d

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] Failed to read frame {frame_idx}, skipping.")
            i += frame_step
            continue

        display = frame.copy()
        h, w = display.shape[:2]
        
        # 情報表示
        info1 = f"{arm_label} frame {frame_idx}/{frame_indices[-1]}  EE=({pos_3d[0]:.3f},{pos_3d[1]:.3f},{pos_3d[2]:.3f})"
        info2 = f"登録済み: {len(used_frames)}点  (3点以上でq終了可)"
        cv2.putText(display, info1, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, info2, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow(window_name, display)

        clicked_and_registered[0] = False
        while True:
            key = cv2.waitKeyEx(50)  # waitKeyExを使用してより確実にキーを取得
            
            # クリックで登録された場合は自動で次のフレームへ
            if clicked_and_registered[0]:
                i += frame_step
                break
            
            # キーが押されていない場合は-1が返る
            if key == -1:
                continue
            
            # macOSではキーコードが異なる場合があるので両方チェック
            key_lower = key & 0xFF
            
            if key_lower == ord("n") or key == ord("n"):
                print("[KEY] n pressed - 次のフレームへ")
                i += frame_step
                break
            elif key_lower == ord("b") or key == ord("b"):
                print("[KEY] b pressed - 前のフレームへ")
                i = max(0, i - frame_step)
                break
            elif key_lower == ord("q") or key == ord("q") or key_lower == 27 or key == 27:  # q or ESC
                print(f"[KEY] q/ESC pressed - 終了リクエスト (現在{len(used_frames)}点)")
                if len(used_frames) >= 3:
                    quit_requested = True
                    break
                else:
                    print(f"[WARN] まだ{len(used_frames)}点しか登録されていません。最低3点必要です。")

    cv2.destroyWindow(window_name)
    return used_frames, world_pts, image_pts


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    episode = args.episode
    urdf_path = Path(args.urdf_path)
    ee_link_name = args.ee_link
    frame_step = args.frame_step
    script_dir = Path(__file__).resolve().parent

    print("=== fk_image_calibration: start ===")
    print(f"  dataset-root = {dataset_root}")
    print(f"  episode      = {episode}")
    print(f"  urdf-path    = {urdf_path}")
    print(f"  ee-link      = {ee_link_name}")
    print(f"  frame-step   = {frame_step}")
    print(f"  using urchin = {_USE_URCHIN}")

    attn_dir = dataset_root / "attn_videos"
    if not attn_dir.exists():
        raise FileNotFoundError(f"Attention directory not found: {attn_dir}")

    # 1. JSON ログ読み込み
    log = load_episode_log(attn_dir, episode)
    frames = log["frames"]
    repo_id = log.get("repo_id", None)
    if repo_id is not None:
        print(f"[INFO] repo_id from JSON = {repo_id}")

    # 2. 動画オープン
    cap = open_video(attn_dir, episode, repo_id)

    # 3. FK で EE 軌跡 (3D) を計算（双腕対応）
    frame_to_pos, frame_to_pos_right, arm_mode = compute_fk_trajectory(
        urdf_path=urdf_path,
        ee_link_name=ee_link_name,
        frames=frames,
    )

    payload: Dict[str, Any] = {"arm_mode": arm_mode}

    if arm_mode == "bimanual":
        # 双腕モード: 左右別々にキャリブレーション
        print("\n" + "=" * 50)
        print("BIMANUAL MODE: Calibrating LEFT arm first")
        print("=" * 50)

        # 左腕のキャリブレーション
        used_frames_left, world_pts_left, image_pts_left = interactive_collect_points(
            cap=cap,
            frame_to_pos=frame_to_pos,
            frame_step=frame_step,
            arm_label="LEFT",
        )

        if len(used_frames_left) < 3:
            raise RuntimeError(
                f"Not enough calibration points for LEFT arm ({len(used_frames_left)}). "
                f"Collect at least 3 (click and press 'c')."
            )

        world_xy_left = np.stack([p[:2] for p in world_pts_left], axis=0)
        img_uv_left = np.array(image_pts_left, dtype=float)
        affine_left = compute_affine(world_xy_left, img_uv_left)
        print("[RESULT] Estimated affine matrix for LEFT arm (2x3):")
        print(affine_left)
        payload["affine_matrix_left"] = affine_left.tolist()

        print("\n" + "=" * 50)
        print("BIMANUAL MODE: Now calibrating RIGHT arm")
        print("=" * 50)

        # 右腕のキャリブレーション
        used_frames_right, world_pts_right, image_pts_right = interactive_collect_points(
            cap=cap,
            frame_to_pos=frame_to_pos_right,
            frame_step=frame_step,
            arm_label="RIGHT",
        )

        if len(used_frames_right) < 3:
            raise RuntimeError(
                f"Not enough calibration points for RIGHT arm ({len(used_frames_right)}). "
                f"Collect at least 3 (click and press 'c')."
            )

        world_xy_right = np.stack([p[:2] for p in world_pts_right], axis=0)
        img_uv_right = np.array(image_pts_right, dtype=float)
        affine_right = compute_affine(world_xy_right, img_uv_right)
        print("[RESULT] Estimated affine matrix for RIGHT arm (2x3):")
        print(affine_right)
        payload["affine_matrix_right"] = affine_right.tolist()

        # 後方互換のためにaffine_matrixも左腕のものを保存
        payload["affine_matrix"] = affine_left.tolist()

    else:
        # 単腕モード: 従来通り
        used_frames, world_pts_3d, image_pts = interactive_collect_points(
            cap=cap,
            frame_to_pos=frame_to_pos,
            frame_step=frame_step,
        )

        if len(used_frames) < 3:
            raise RuntimeError(
                f"Not enough calibration points collected ({len(used_frames)}). "
                f"Collect at least 3 (click and press 'c')."
            )

        world_xy = np.stack([p[:2] for p in world_pts_3d], axis=0)
        img_uv = np.array(image_pts, dtype=float)

        affine = compute_affine(world_xy, img_uv)
        print("[RESULT] Estimated affine matrix (2x3):")
        print(affine)
        payload["affine_matrix"] = affine.tolist()

    conf_dir = script_dir / "SO101"
    conf_dir.mkdir(parents=True, exist_ok=True)
    fk_conf_path = conf_dir / "fk_image.conf"
    fk_conf_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[SAVE] Calibration saved to: {fk_conf_path}")
    print("=== fk_image_calibration: done ===")


if __name__ == "__main__":
    main()
