#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# --- NumPy 2.x 互換パッチ: urdfpy / urchin が np.float を使うのでエイリアスを貼る ---
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

# まず urchin を優先的に使う（あればメッシュ lazy ロード可能）
try:
    from urchin import URDF  # type: ignore
    _USE_URCHIN = True
except ImportError:
    from urdfpy import URDF  # type: ignore
    _USE_URCHIN = False


# ログに出てくる SO-101 の関節キー（LeRobot の action dict）
# 単腕用
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


def detect_arm_mode(action: Dict[str, Any] | List | Tuple) -> str:
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
    elif isinstance(action, (list, tuple)):
        # リスト形式の場合、長さで判断
        if len(action) == len(ACTION_KEYS):
            return "single"
        elif len(action) == len(ACTION_KEYS) * 2:
            return "bimanual"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize VLA planned trajectory (future horizon) on attention video. "
            "For each chunk (frame_idx), actions[] are treated as the future plan."
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

    # URDF はこのスクリプトと同じ scripts フォルダに置く前提
    script_dir = Path(__file__).resolve().parent
    default_urdf_path = script_dir / "SO101" / "so101_new_calib.urdf"

    parser.add_argument(
        "--urdf-path",
        type=str,
        default=str(default_urdf_path),
        help=(
            "Path to SO-101 URDF file "
            "(default: SO101/so101_new_calib.urdf next to this script)."
        ),
    )
    parser.add_argument(
        "--ee-link",
        type=str,
        default="gripper_frame_link",
        help="End-effector link name in the URDF (default: gripper_frame_link).",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="plan_overlay",
        help=(
            "Suffix for output video file. "
            "Output path will be attn_videos/{suffix}_episode_{episode}.mp4 "
            "(default: plan_overlay)."
        ),
    )
    return parser.parse_args()


# ====== STEP 1: JSON ログ読み込み ======

def load_episode_log(attn_dir: Path, episode: int) -> Dict[str, Any]:
    """
    {dataset_root}/attn_videos/episode_values_{episode}.json を読む。
    """
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


# ====== STEP 2: 動画を開く ======

def resolve_video_path(attn_dir: Path, episode: int, repo_id: str | None) -> Path:
    """
    実際のファイル名が不明なので、複数パターンを試す。
    優先順位:
      1) attn_all_cameras_episode_{episode}.mp4
      2) {repo_id.replace('/', '_')}_ep{episode:06d}.mp4
    どれも無かったら attn_videos の中身を全部列挙してからエラー。
    """
    candidates: List[Path] = []
    candidates.append(attn_dir / f"attn_all_cameras_episode_{episode}.mp4")

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
        f"Check the actual file names listed above."
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


# ====== STEP 3: 各 chunk の「未来軌跡 (plan)」を FK で計算 ======

def build_joint_config_from_action(
    action: Dict[str, Any],
    urdf_joint_names: List[str],
    action_keys: List[str] | None = None,
) -> Dict[str, float]:
    """
    支援するフォーマット:
      - dict: {"shoulder_pan.pos": deg, ...} or {"left_shoulder_pan.pos": deg, ...}
      - list/tuple: [deg0, deg1, ...] （action_keys の順で並んでいる想定）

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

    if isinstance(action, (list, tuple)):
        if len(action) < len(action_keys):
            raise ValueError(
                f"Action list length {len(action)} is smaller than required keys {len(action_keys)}"
            )
        action = {k: action[i] for i, k in enumerate(action_keys)}
    elif not isinstance(action, dict):
        raise ValueError(f"Action must be dict or list, got {type(action)}")

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

def compute_fk_plans(
    urdf_path: Path,
    ee_link_name: str,
    frames: List[Dict[str, Any]],
) -> Tuple[List[int], List[List[np.ndarray]], List[List[np.ndarray]] | None, str]:
    """
    episode_values_*.json の frames から、
      - chunk_starts: 各 chunk の frame_idx のリスト
      - plans_3d: 各 chunk ごとの「未来軌跡」の 3D 位置リスト (単腕 or 左腕)
      - plans_3d_right: 双腕の場合の右腕軌跡、単腕の場合はNone
      - arm_mode: "single" or "bimanual"
    を返す。

    ここで actions は「n_action_steps step horizon の joint 指令列」とみなす。
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

    # 念のため frame_idx でソート
    records = sorted(frames, key=lambda r: int(r["frame_idx"]))

    # 最初のアクションからアームモードを検出
    arm_mode = "single"
    for rec in records:
        actions = rec.get("actions", [])
        if actions:
            first_action = actions[0]
            if isinstance(first_action, (list, tuple)) and first_action and isinstance(first_action[0], (list, tuple, dict)):
                first_action = first_action[0]
            arm_mode = detect_arm_mode(first_action)
            break

    print(f"[STEP 3] Detected arm mode: {arm_mode}")

    chunk_starts: List[int] = []
    plans_3d: List[List[np.ndarray]] = []  # 単腕 or 左腕
    plans_3d_right: List[List[np.ndarray]] | None = [] if arm_mode == "bimanual" else None

    total_actions = 0

    for rec in records:
        base_idx = int(rec["frame_idx"])
        actions = rec.get("actions", [])
        if not isinstance(actions, list) or len(actions) == 0:
            continue

        plan_points: List[np.ndarray] = []
        plan_points_right: List[np.ndarray] = []

        for action in actions:
            # action は以下どれか:
            #  - dict                  -> 1 ステップ
            #  - list/tuple of scalars -> 1 ステップ (ACTION_KEYS 順)
            #  - list/tuple of list    -> 複数ステップ (各要素が上記1ステップ)
            if isinstance(action, (list, tuple)) and action and isinstance(action[0], (list, tuple, dict)):
                seq = action  # 多段リストを展開
            else:
                seq = [action]

            for step in seq:
                if arm_mode == "bimanual":
                    # 左腕
                    try:
                        joint_cfg_left = build_joint_config_from_action(step, urdf_joint_names, ACTION_KEYS_LEFT)
                        fk_all_left = robot.link_fk(joint_cfg_left)
                        T_left = fk_all_left[ee_link]
                        pos_left = np.asarray(T_left[:3, 3], dtype=float)
                        plan_points.append(pos_left)
                    except (KeyError, ValueError) as e:
                        print(f"[STEP 3] Warning: Left arm FK failed: {e}")

                    # 右腕
                    try:
                        joint_cfg_right = build_joint_config_from_action(step, urdf_joint_names, ACTION_KEYS_RIGHT)
                        fk_all_right = robot.link_fk(joint_cfg_right)
                        T_right = fk_all_right[ee_link]
                        pos_right = np.asarray(T_right[:3, 3], dtype=float)
                        plan_points_right.append(pos_right)
                    except (KeyError, ValueError) as e:
                        print(f"[STEP 3] Warning: Right arm FK failed: {e}")
                else:
                    # 単腕
                    joint_cfg = build_joint_config_from_action(step, urdf_joint_names, ACTION_KEYS)
                    fk_all = robot.link_fk(joint_cfg)
                    T = fk_all[ee_link]
                    pos = np.asarray(T[:3, 3], dtype=float)
                    plan_points.append(pos)

                total_actions += 1

        chunk_starts.append(base_idx)
        plans_3d.append(plan_points)
        if plans_3d_right is not None:
            plans_3d_right.append(plan_points_right)

    print(
        f"[STEP 3] OK: computed plans for {len(chunk_starts)} chunks "
        f"(total actions processed={total_actions}, arm_mode={arm_mode})."
    )
    if chunk_starts and plans_3d[0]:
        print("[STEP 3] Sample plan (first chunk, first 5 points - left/single arm):")
        for i, pos in enumerate(plans_3d[0][:5]):
            print(f"  step {i}: EE pos = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        if plans_3d_right and plans_3d_right[0]:
            print("[STEP 3] Sample plan (first chunk, first 5 points - right arm):")
            for i, pos in enumerate(plans_3d_right[0][:5]):
                print(f"  step {i}: EE pos = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    return chunk_starts, plans_3d, plans_3d_right, arm_mode

# ====== STEP 4: fk_image.conf の読み込み ======

def load_fk_image_conf(attn_dir: Path | None = None) -> Tuple[np.ndarray, np.ndarray | None, str]:
    """
    scripts/SO101/fk_image.conf から affine_matrix を読む。
    双腕対応: affine_matrix_left, affine_matrix_right を読み込む。

    Returns:
        affine: 単腕または左腕用のaffine行列
        affine_right: 右腕用のaffine行列（双腕の場合）、単腕の場合はNone
        config_arm_mode: 設定ファイル内のarm_mode ("single" or "bimanual")
    """
    script_dir = Path(__file__).resolve().parent
    conf_path = script_dir / "SO101" / "fk_image.conf"
    print("[STEP 4] Loading calibration config (SO101).")
    print(f"[STEP 4]  Path: {conf_path}")
    if not conf_path.exists():
        raise FileNotFoundError(
            "Calibration file fk_image.conf not found.\n"
            "Run fk_image_calibration.py to generate it."
        )
    print(f"[STEP 4]  -> FOUND: {conf_path}")

    with conf_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    config_arm_mode = data.get("arm_mode", "single")

    if config_arm_mode == "bimanual":
        # 双腕モード
        if "affine_matrix_left" not in data or "affine_matrix_right" not in data:
            raise ValueError(
                "fk_image.conf is in bimanual mode but missing 'affine_matrix_left' or 'affine_matrix_right'."
            )

        A_left = np.asarray(data["affine_matrix_left"], dtype=float)
        A_right = np.asarray(data["affine_matrix_right"], dtype=float)

        if A_left.shape != (2, 3):
            raise ValueError(f"affine_matrix_left must be shape (2, 3), got {A_left.shape}.")
        if A_right.shape != (2, 3):
            raise ValueError(f"affine_matrix_right must be shape (2, 3), got {A_right.shape}.")

        print("[STEP 4] OK: loaded bimanual affine matrices from fk_image.conf:")
        print(f"  LEFT:\n{A_left}")
        print(f"  RIGHT:\n{A_right}")
        return A_left, A_right, config_arm_mode

    else:
        # 単腕モード（後方互換）
        if "affine_matrix" not in data:
            raise ValueError("fk_image.conf must contain key 'affine_matrix'.")

        A = np.asarray(data["affine_matrix"], dtype=float)
        if A.shape != (2, 3):
            raise ValueError(
                f"fk_image.conf['affine_matrix'] must be shape (2, 3), got {A.shape} instead."
            )

        print("[STEP 4] OK: loaded affine_matrix from fk_image.conf:")
        print(A)
        return A, None, config_arm_mode

def project_world_to_image(affine: np.ndarray, pos_3d: np.ndarray) -> tuple[int, int]:
    x, y = float(pos_3d[0]), float(pos_3d[1])
    vec = np.array([x, y, 1.0], dtype=float)
    uv = affine @ vec  # shape (2,)
    u, v = int(round(uv[0])), int(round(uv[1]))
    return u, v


# ====== STEP 5: 「計画軌跡」をオーバーレイして動画書き出し ======

def overlay_plan_trajectory_video(
    cap: cv2.VideoCapture,
    chunk_starts: List[int],
    plans_3d: List[List[np.ndarray]],
    affine: np.ndarray,
    output_path: Path,
    plans_3d_right: List[List[np.ndarray]] | None = None,
    arm_mode: str = "single",
    affine_right: np.ndarray | None = None,
) -> None:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Discord/Notion で再生されやすいコーデックを優先
    writer = None
    for fourcc_tag in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            break
    if writer is None or not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")

    print(f"[STEP 5] Writing overlay video to: {output_path}")
    print(f"[STEP 5]  frames={frame_count}, fps={fps:.2f}, size={width}x{height}, arm_mode={arm_mode}")

    # あらかじめ各 plan の 2D 投影を計算しておく（左腕/単腕）
    plans_uv: List[List[Tuple[int, int]]] = []
    for plan in plans_3d:
        uv_list = [project_world_to_image(affine, p) for p in plan]
        plans_uv.append(uv_list)

    # 右腕の2D投影（双腕の場合のみ）- 右腕用affine行列を使用
    plans_uv_right: List[List[Tuple[int, int]]] | None = None
    if plans_3d_right is not None:
        plans_uv_right = []
        # 右腕用のaffine行列がなければ左腕用を使う（後方互換）
        affine_for_right = affine_right if affine_right is not None else affine
        for plan in plans_3d_right:
            uv_list = [project_world_to_image(affine_for_right, p) for p in plan]
            plans_uv_right.append(uv_list)

    # 色の定義: 左腕=青, 右腕=赤, 単腕=赤
    COLOR_LEFT = (255, 0, 0)    # BGR: 青
    COLOR_RIGHT = (0, 0, 255)   # BGR: 赤
    COLOR_SINGLE = (0, 0, 255)  # BGR: 赤
    MARKER_LEFT = (0, 255, 255)   # BGR: 黄色（左腕現在位置）
    MARKER_RIGHT = (0, 255, 0)    # BGR: 緑（右腕現在位置）
    MARKER_SINGLE = (0, 255, 0)   # BGR: 緑（単腕現在位置）

    # 各チャンクの計画軌跡をオーバーレイ画像として事前生成
    plan_overlays: List[np.ndarray] = []
    for idx, plan_uv in enumerate(plans_uv):
        overlay = np.zeros((height, width, 3), dtype=np.float32)
        color = COLOR_LEFT if arm_mode == "bimanual" else COLOR_SINGLE
        if len(plan_uv) >= 2:
            for i in range(1, len(plan_uv)):
                cv2.line(overlay, plan_uv[i - 1], plan_uv[i], color, 2)
        # 右腕の軌跡も追加（双腕の場合）
        if plans_uv_right is not None and idx < len(plans_uv_right):
            plan_uv_r = plans_uv_right[idx]
            if len(plan_uv_r) >= 2:
                for i in range(1, len(plan_uv_r)):
                    cv2.line(overlay, plan_uv_r[i - 1], plan_uv_r[i], COLOR_RIGHT, 2)
        plan_overlays.append(overlay)

    # chunk_starts は昇順前提
    num_chunks = len(chunk_starts)
    if num_chunks == 0:
        print("[STEP 5] WARNING: no chunks to visualize (no plans).")
        # 単に元動画をコピーして終了
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
        return

    # 古い軌跡の透明度が動画末尾でもうっすら残るように減衰係数を設定
    base_alpha = 0.9
    min_alpha = 0.12  # 最古チャンクの目標透明度
    if num_chunks > 1:
        decay = (min_alpha / base_alpha) ** (1 / max(num_chunks - 1, 1))
    else:
        decay = 1.0
    print(f"[STEP 5] alpha decay per chunk: base={base_alpha}, decay={decay:.4f}, min~{min_alpha}")

    current_chunk_idx = 0

    for frame_idx in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            print(f"[STEP 5] WARNING: failed to read frame {frame_idx}, stopping.")
            break

        # 現在 frame に対応する chunk を更新
        while (
            current_chunk_idx + 1 < num_chunks
            and frame_idx >= chunk_starts[current_chunk_idx + 1]
        ):
            current_chunk_idx += 1

        # frame が最初の chunk 以前なら何も描かない
        if frame_idx < chunk_starts[0]:
            writer.write(frame)
            continue

        plan_uv = plans_uv[current_chunk_idx]
        plan_uv_r = plans_uv_right[current_chunk_idx] if plans_uv_right is not None else None

        if len(plan_uv) < 2 and (plan_uv_r is None or len(plan_uv_r) < 2):
            writer.write(frame)
            continue

        # 過去チャンクの軌跡も含めて重ね描き（古いほど薄く）
        overlay_acc = np.zeros_like(frame, dtype=np.float32)
        for past_idx in range(current_chunk_idx + 1):
            age = current_chunk_idx - past_idx
            alpha = base_alpha * (decay ** age)
            overlay_acc += plan_overlays[past_idx] * alpha
        overlay_uint8 = np.clip(overlay_acc, 0, 255).astype(np.uint8)
        frame = cv2.addWeighted(frame, 1.0, overlay_uint8, 1.0, 0.0)

        # 「今どこまで来ているか」を計画上で示す
        t0 = chunk_starts[current_chunk_idx]
        step_idx = frame_idx - t0
        if step_idx < 0:
            step_idx = 0

        # 左腕/単腕の現在位置マーカー
        if len(plan_uv) > 0:
            step_idx_left = min(step_idx, len(plan_uv) - 1)
            u_cur, v_cur = plan_uv[step_idx_left]
            marker_color = MARKER_LEFT if arm_mode == "bimanual" else MARKER_SINGLE
            cv2.circle(frame, (u_cur, v_cur), 5, marker_color, -1)

        # 右腕の現在位置マーカー（双腕の場合）
        if plan_uv_r is not None and len(plan_uv_r) > 0:
            step_idx_right = min(step_idx, len(plan_uv_r) - 1)
            u_cur_r, v_cur_r = plan_uv_r[step_idx_right]
            cv2.circle(frame, (u_cur_r, v_cur_r), 5, MARKER_RIGHT, -1)

        # デバッグ用に現在の chunk / step 情報を表示
        max_steps = max(len(plan_uv) - 1, 0)
        if plan_uv_r:
            max_steps = max(max_steps, len(plan_uv_r) - 1)
        info = f"chunk {current_chunk_idx}  frame {frame_idx}  plan_step {step_idx}/{max_steps}"
        if arm_mode == "bimanual":
            info += " [bimanual: blue=L, red=R]"
        cv2.putText(
            frame,
            info,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    print("[STEP 5] Done writing overlay video.")


# ====== main ======

def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    episode = args.episode
    urdf_path = Path(args.urdf_path)
    ee_link_name = args.ee_link
    output_suffix = args.output_suffix

    print("=== visualize_inference_trajectory (plan horizon): start ===")
    print(f"  dataset-root = {dataset_root}")
    print(f"  episode      = {episode}")
    print(f"  urdf-path    = {urdf_path}")
    print(f"  ee-link      = {ee_link_name}")
    print(f"  using urchin = {_USE_URCHIN}")
    print(f"  output-suffix= {output_suffix}")

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

    # 3. 各 chunk の未来軌跡 (plan) を FK で計算（双腕対応）
    chunk_starts, plans_3d, plans_3d_right, arm_mode = compute_fk_plans(
        urdf_path=urdf_path,
        ee_link_name=ee_link_name,
        frames=frames,
    )

    # 4. キャリブレーション fk_image.conf 読み込み（双腕対応）
    affine, affine_right, config_arm_mode = load_fk_image_conf(attn_dir)

    # 5. 計画軌跡オーバーレイ動画の書き出し（双腕対応）
    output_path = attn_dir / f"{output_suffix}_episode_{episode}.mp4"
    overlay_plan_trajectory_video(
        cap=cap,
        chunk_starts=chunk_starts,
        plans_3d=plans_3d,
        affine=affine,
        output_path=output_path,
        plans_3d_right=plans_3d_right,
        arm_mode=arm_mode,
        affine_right=affine_right,
    )

    print("=== visualize_inference_trajectory (plan horizon): done (step1-5 OK) ===")


if __name__ == "__main__":
    main()
