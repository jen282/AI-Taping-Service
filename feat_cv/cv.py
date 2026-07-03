import io
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import trimesh
from PIL import Image, ImageOps, UnidentifiedImageError
from rembg import remove
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import pillow_avif  # noqa: F401
except Exception:
    try:
        import pillow_avif_plugin  # noqa: F401
    except Exception:
        pass


MEDIAPIPE_MODEL_PATH = r"pose_landmarker_full.task"
BODY_JSON_DIR = r"body_jsons_final"
OBJ_DIR = r"body_models_final"
WIDTH_FEATURE_JSON_PATH = r"body_width_features.json"

TOP_K = 5
PREFILTER_K = 30

APP_TMP_DIR = Path("service_outputs")
APP_TMP_DIR.mkdir(parents=True, exist_ok=True)


MP_IDX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def normalize_sex(sex_value: Optional[str]) -> Optional[str]:
    if sex_value is None:
        return None
    s = str(sex_value).strip().lower()
    if s in ["male", "m", "man", "남", "남자"]:
        return "male"
    if s in ["female", "f", "woman", "여", "여자"]:
        return "female"
    return None


def normalize_tape_type(tape_type: Optional[str]) -> Optional[str]:
    if tape_type is None:
        return None

    raw = str(tape_type).strip()
    lower = raw.lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "y-strip": "Y-strip",
        "i-strip": "I-strip",
        "x-strip": "X-strip",
        "v-strip": "V-strip",
        "big-daddy": "Big-Daddy",
        "bigdaddy": "Big-Daddy",
    }
    return aliases.get(lower, raw)


def dist2(a: List[float], b: List[float]) -> float:
    a_np = np.array(a, dtype=np.float64)
    b_np = np.array(b, dtype=np.float64)
    return float(np.linalg.norm(a_np - b_np))


def safe_div(a: float, b: float, eps: float = 1e-8) -> float:
    return float(a / (b + eps))


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_uploaded_image_to_jpg(input_path: str, output_jpg_path: str) -> str:
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {input_path}")

    try:
        with Image.open(input_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img.save(output_jpg_path, "JPEG", quality=95)
    except UnidentifiedImageError as e:
        raise RuntimeError(
            f"Pillow가 이미지를 인식하지 못했습니다: {input_path}\n"
            f"AVIF면 pillow-avif-plugin 설치가 필요할 수 있습니다."
        ) from e
    except Exception as e:
        raise RuntimeError(f"이미지 JPG 변환에 실패했습니다: {input_path}") from e

    return output_jpg_path


def get_selected_rag_option(
    rag_result: Optional[Dict[str, Any]],
    selected_option_rank: int = 1,
) -> Optional[Dict[str, Any]]:
    if not rag_result:
        return None

    options = rag_result.get("options")
    if not isinstance(options, list) or not options:
        return None

    for option in options:
        if option.get("option_rank") == selected_option_rank:
            return option

    return options[0]


def extract_tape_type_from_rag_result(
    rag_result: Optional[Dict[str, Any]],
    selected_option_rank: int = 1,
) -> Optional[str]:
    selected_option = get_selected_rag_option(rag_result, selected_option_rank)
    if not selected_option:
        return None
    return normalize_tape_type(selected_option.get("tape_type"))


def load_taping_registry(registry_path: str) -> List[Dict[str, Any]]:
    path = Path(registry_path)
    if not path.exists():
        raise FileNotFoundError(f"Taping registry 파일을 찾을 수 없습니다: {registry_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Taping registry JSON은 list 형태여야 합니다.")

    return data


def normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip().lower()


def get_body_model_key_from_path(body_obj_path: str) -> str:
    """
    body obj 파일명에서 registry asset_id prefix로 사용할 body model key를 추출한다.

    예:
      body_models_final/3148M.obj      -> 3148M
      body_models_final/3148M_BD_B.obj -> 3148M
    """
    stem = Path(body_obj_path).stem.strip()
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def build_asset_id(body_obj_path: str, technique_code: str) -> str:
    body_model_key = get_body_model_key_from_path(body_obj_path)
    return f"{body_model_key}_{technique_code}"


def find_taping_asset_by_asset_id(
    registry: List[Dict[str, Any]],
    asset_id: str,
) -> Dict[str, Any]:
    target = normalize_text(asset_id)

    for row in registry:
        row_asset_id = normalize_text(row.get("asset_id"))
        is_active = bool(row.get("active", True))
        if is_active and row_asset_id == target:
            result = dict(row)
            result["match_level"] = "asset_id_exact"
            return result

    raise LookupError(f"asset_id로 일치하는 taping registry 항목을 찾지 못했습니다: {asset_id}")


def find_taping_asset_for_body(
    registry_path: str,
    body_obj_path: str,
    technique_code: str,
) -> Dict[str, Any]:
    registry = load_taping_registry(registry_path)
    asset_id = build_asset_id(body_obj_path, technique_code)
    result = find_taping_asset_by_asset_id(registry, asset_id)
    result["resolved_asset_id"] = asset_id
    return result


def detect_pose_from_photo(photo_path: str, model_path: str) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    image_bgr = cv2.imread(photo_path)
    if image_bgr is None:
        raise RuntimeError(f"OpenCV가 이미지를 읽지 못했습니다: {photo_path}")

    h, w = image_bgr.shape[:2]
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        output_segmentation_masks=False,
    )
    mp_image = mp.Image.create_from_file(photo_path)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        raise RuntimeError("전신 포즈를 찾지 못했습니다. 전신이 모두 보이는 정면 사진을 사용하세요.")

    lms = result.pose_landmarks[0]
    pts = {
        "Left shoulder": [lms[MP_IDX["left_shoulder"]].x * w, lms[MP_IDX["left_shoulder"]].y * h],
        "Right shoulder": [lms[MP_IDX["right_shoulder"]].x * w, lms[MP_IDX["right_shoulder"]].y * h],
        "Left elbow": [lms[MP_IDX["left_elbow"]].x * w, lms[MP_IDX["left_elbow"]].y * h],
        "Right elbow": [lms[MP_IDX["right_elbow"]].x * w, lms[MP_IDX["right_elbow"]].y * h],
        "Left wrist": [lms[MP_IDX["left_wrist"]].x * w, lms[MP_IDX["left_wrist"]].y * h],
        "Right wrist": [lms[MP_IDX["right_wrist"]].x * w, lms[MP_IDX["right_wrist"]].y * h],
        "Left hip": [lms[MP_IDX["left_hip"]].x * w, lms[MP_IDX["left_hip"]].y * h],
        "Right hip": [lms[MP_IDX["right_hip"]].x * w, lms[MP_IDX["right_hip"]].y * h],
        "Left knee": [lms[MP_IDX["left_knee"]].x * w, lms[MP_IDX["left_knee"]].y * h],
        "Right knee": [lms[MP_IDX["right_knee"]].x * w, lms[MP_IDX["right_knee"]].y * h],
        "Left ankle": [lms[MP_IDX["left_ankle"]].x * w, lms[MP_IDX["left_ankle"]].y * h],
        "Right ankle": [lms[MP_IDX["right_ankle"]].x * w, lms[MP_IDX["right_ankle"]].y * h],
        "Nose": [lms[MP_IDX["nose"]].x * w, lms[MP_IDX["nose"]].y * h],
    }
    return image_bgr, pts


def compute_body_features_from_points(pts: Dict[str, List[float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    ls = pts["Left shoulder"]; rs = pts["Right shoulder"]
    le = pts["Left elbow"]; re = pts["Right elbow"]
    lw = pts["Left wrist"]; rw = pts["Right wrist"]
    lh = pts["Left hip"]; rh = pts["Right hip"]
    lk = pts["Left knee"]; rk = pts["Right knee"]
    la = pts["Left ankle"]; ra = pts["Right ankle"]

    shoulder_width = dist2(ls, rs)
    hip_width = dist2(lh, rh)

    torso_left = dist2(ls, lh)
    torso_right = dist2(rs, rh)
    torso_length = (torso_left + torso_right) / 2.0

    upper_leg = (dist2(lh, lk) + dist2(rh, rk)) / 2.0
    lower_leg = (dist2(lk, la) + dist2(rk, ra)) / 2.0
    leg_length = upper_leg + lower_leg

    upper_arm = (dist2(ls, le) + dist2(rs, re)) / 2.0
    lower_arm = (dist2(le, lw) + dist2(re, rw)) / 2.0
    arm_length = upper_arm + lower_arm

    ratios = {
        "shoulder_to_hip": safe_div(shoulder_width, hip_width),
        "shoulder_to_torso": safe_div(shoulder_width, torso_length),
        "hip_to_torso": safe_div(hip_width, torso_length),
        "leg_to_torso": safe_div(leg_length, torso_length),
        "arm_to_torso": safe_div(arm_length, torso_length),
        "upper_to_lower_leg": safe_div(upper_leg, lower_leg),
        "upper_to_lower_arm": safe_div(upper_arm, lower_arm),
    }
    raw = {
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "torso_length": torso_length,
        "upper_leg": upper_leg,
        "lower_leg": lower_leg,
        "leg_length": leg_length,
        "upper_arm": upper_arm,
        "lower_arm": lower_arm,
        "arm_length": arm_length,
    }
    return raw, ratios


def segment_person_mask(input_path: str, save_mask_path: Optional[str] = None) -> np.ndarray:
    with open(input_path, "rb") as f:
        input_bytes = f.read()
    output_bytes = remove(input_bytes)
    out_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    out_np = np.array(out_img)
    alpha = out_np[:, :, 3]
    mask = (alpha > 127).astype(np.uint8)
    if save_mask_path is not None:
        Image.fromarray((mask * 255).astype(np.uint8)).save(save_mask_path)
    return mask


def find_segments_in_row(binary_row: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    inside = False
    start = 0
    for i, v in enumerate(binary_row):
        if v and not inside:
            inside = True
            start = i
        elif not v and inside:
            inside = False
            segments.append((start, i - 1))
    if inside:
        segments.append((start, len(binary_row) - 1))
    return segments


def select_segment_near_x(segments: List[Tuple[int, int]], target_x: float) -> Optional[Tuple[int, int]]:
    if not segments:
        return None
    centers = [((a + b) / 2.0) for a, b in segments]
    idx = int(np.argmin([abs(c - target_x) for c in centers]))
    return segments[idx]


def width_at_y_band(mask: np.ndarray, target_y: float, band_half: int, target_x: float) -> Tuple[Optional[float], List[Tuple[int, int, int]]]:
    h, _ = mask.shape
    y0 = max(0, int(target_y - band_half))
    y1 = min(h - 1, int(target_y + band_half))
    widths: List[int] = []
    chosen_segments: List[Tuple[int, int, int]] = []
    for y in range(y0, y1 + 1):
        row = mask[y]
        segs = find_segments_in_row(row)
        seg = select_segment_near_x(segs, target_x)
        if seg is None:
            continue
        x1, x2 = seg
        widths.append(x2 - x1 + 1)
        chosen_segments.append((y, x1, x2))
    if not widths:
        return None, []
    return float(np.median(widths)), chosen_segments


def compute_width_features_from_photo(mask: np.ndarray, pts: Dict[str, List[float]]) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    ys = np.where(mask > 0)[0]
    if len(ys) == 0:
        raise RuntimeError("segmentation mask가 비어 있습니다.")

    body_top_y = int(ys.min())
    body_bottom_y = int(ys.max())
    body_height = float(body_bottom_y - body_top_y)

    ls = np.array(pts["Left shoulder"], dtype=float)
    rs = np.array(pts["Right shoulder"], dtype=float)
    lh = np.array(pts["Left hip"], dtype=float)
    rh = np.array(pts["Right hip"], dtype=float)
    lk = np.array(pts["Left knee"], dtype=float)
    rk = np.array(pts["Right knee"], dtype=float)
    la = np.array(pts["Left ankle"], dtype=float)
    ra = np.array(pts["Right ankle"], dtype=float)

    shoulder_center = (ls + rs) / 2.0
    hip_center = (lh + rh) / 2.0
    shoulder_y = shoulder_center[1]
    hip_y = hip_center[1]
    torso_len = hip_y - shoulder_y

    chest_y = shoulder_y + 0.28 * torso_len
    waist_y = shoulder_y + 0.65 * torso_len
    pelvis_y = hip_y

    left_thigh_center = (lh + lk) / 2.0
    right_thigh_center = (rh + rk) / 2.0
    left_thigh_y = left_thigh_center[1]
    right_thigh_y = right_thigh_center[1]

    body_center_x = hip_center[0]
    band_half = max(2, int(body_height * 0.012))

    chest_width, chest_segments = width_at_y_band(mask, chest_y, band_half, body_center_x)
    waist_width, waist_segments = width_at_y_band(mask, waist_y, band_half, body_center_x)
    hip_width, hip_segments = width_at_y_band(mask, pelvis_y, band_half, body_center_x)
    left_thigh_width, left_thigh_segments = width_at_y_band(mask, left_thigh_y, band_half, left_thigh_center[0])
    right_thigh_width, right_thigh_segments = width_at_y_band(mask, right_thigh_y, band_half, right_thigh_center[0])

    thigh_vals = [v for v in [left_thigh_width, right_thigh_width] if v is not None]
    thigh_width_avg = float(np.mean(thigh_vals)) if thigh_vals else None

    shoulder_width = dist2(ls.tolist(), rs.tolist())
    left_leg = dist2(lh.tolist(), lk.tolist()) + dist2(lk.tolist(), la.tolist())
    right_leg = dist2(rh.tolist(), rk.tolist()) + dist2(rk.tolist(), ra.tolist())
    leg_length = (left_leg + right_leg) / 2.0

    features: Dict[str, Optional[float]] = {
        "body_height": body_height,
        "shoulder_width": shoulder_width,
        "leg_length": leg_length,
        "chest_width": chest_width,
        "waist_width": waist_width,
        "hip_width": hip_width,
        "left_thigh_width": left_thigh_width,
        "right_thigh_width": right_thigh_width,
        "thigh_width_avg": thigh_width_avg,
        "chest_width_to_height": None if chest_width is None else safe_div(chest_width, body_height),
        "waist_width_to_height": None if waist_width is None else safe_div(waist_width, body_height),
        "hip_width_to_height": None if hip_width is None else safe_div(hip_width, body_height),
        "thigh_width_avg_to_height": None if thigh_width_avg is None else safe_div(thigh_width_avg, body_height),
        "chest_width_to_shoulder": None if chest_width is None else safe_div(chest_width, shoulder_width),
        "waist_width_to_shoulder": None if waist_width is None else safe_div(waist_width, shoulder_width),
        "hip_width_to_shoulder": None if hip_width is None else safe_div(hip_width, shoulder_width),
        "thigh_width_avg_to_leg": None if thigh_width_avg is None else safe_div(thigh_width_avg, leg_length),
    }
    debug_info = {
        "body_top_y": body_top_y,
        "body_bottom_y": body_bottom_y,
        "chest_y": chest_y,
        "waist_y": waist_y,
        "hip_y": pelvis_y,
        "left_thigh_y": left_thigh_y,
        "right_thigh_y": right_thigh_y,
        "segments": {
            "chest": chest_segments,
            "waist": waist_segments,
            "hip": hip_segments,
            "left_thigh": left_thigh_segments,
            "right_thigh": right_thigh_segments,
        }
    }
    return features, debug_info


def load_body_json(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    keypoint_dict = {kp["name"]: [kp["x"], kp["y"], kp["z"]] for kp in data["keypoints"]}
    needed = [
        "Left shoulder", "Right shoulder", "Left elbow", "Right elbow", "Left wrist", "Right wrist",
        "Left hip", "Right hip", "Left knee", "Right knee", "Left ankle", "Right ankle",
    ]
    for name in needed:
        if name not in keypoint_dict:
            raise ValueError(f"{json_path.name} 에 필요한 keypoint가 없습니다: {name}")

    pts_xy = {k: [v[0], v[1]] for k, v in keypoint_dict.items()}
    raw, ratios = compute_body_features_from_points(pts_xy)

    actor = data.get("actor", {})
    mesh = data.get("mesh", {})
    pose = data.get("pose", {})
    annotation = data.get("annotation", {})

    return {
        "json_path": str(json_path),
        "annotation_id": annotation.get("id"),
        "model_id": mesh.get("mesh_id", json_path.stem),
        "actor_id": actor.get("id"),
        "pose_name": pose.get("name"),
        "sex": actor.get("sex"),
        "actor_height_cm": actor.get("height"),
        "actor_weight_kg": actor.get("weight"),
        "mesh_obj_file_name": mesh.get("obj_file_name"),
        "mesh_png_file_name": mesh.get("png_file_name"),
        "raw_features": raw,
        "ratio_features": ratios,
    }


def load_width_feature_index(width_feature_json_path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(width_feature_json_path):
        return {}
    with open(width_feature_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    index: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        for row in data:
            ann_id = row.get("annotation_id")
            if ann_id:
                index[ann_id] = row
    elif isinstance(data, dict):
        index = data
    return index


def skeleton_distance(user_ratios: Dict[str, float], model_ratios: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    weights = {
        "shoulder_to_hip": 2.0,
        "shoulder_to_torso": 1.5,
        "hip_to_torso": 1.2,
        "leg_to_torso": 2.2,
        "arm_to_torso": 1.2,
        "upper_to_lower_leg": 2.0,
        "upper_to_lower_arm": 0.8,
    }
    s = 0.0
    detail: Dict[str, float] = {}
    for k, w in weights.items():
        d = user_ratios[k] - model_ratios[k]
        detail[k] = float(d)
        s += w * (d ** 2)
    return math.sqrt(s), detail


def width_distance(user_width_features: Dict[str, Optional[float]], model_width_features: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, float]]:
    weights = {
        "chest_width_to_height": 1.2,
        "waist_width_to_height": 2.2,
        "hip_width_to_height": 1.8,
        "thigh_width_avg_to_height": 1.7,
        "waist_width_to_shoulder": 2.0,
        "hip_width_to_shoulder": 1.5,
        "thigh_width_avg_to_leg": 1.6,
    }
    s = 0.0
    detail: Dict[str, float] = {}
    used_weight_sum = 0.0
    for k, w in weights.items():
        u = user_width_features.get(k)
        m = model_width_features.get(k)
        if u is None or m is None:
            continue
        d = float(u) - float(m)
        detail[k] = d
        s += w * (d ** 2)
        used_weight_sum += w
    if used_weight_sum == 0:
        return None, detail
    return math.sqrt(s / used_weight_sum), detail


def combined_body_score(
    user_skeleton_ratios: Dict[str, float],
    model_skeleton_ratios: Dict[str, float],
    user_width_features: Dict[str, Optional[float]],
    model_width_features: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    skeleton_score, skeleton_detail = skeleton_distance(user_skeleton_ratios, model_skeleton_ratios)
    width_score, width_detail = width_distance(user_width_features, model_width_features)
    if width_score is None:
        final = skeleton_score
    else:
        final = 0.55 * skeleton_score + 0.45 * width_score
    return final, {
        "skeleton_score": skeleton_score,
        "width_score": width_score,
        "skeleton_detail": skeleton_detail,
        "width_detail": width_detail,
    }


def rerank_with_actor_info(
    base_score: float,
    model_info: Dict[str, Any],
    user_height_cm: Optional[float] = None,
    user_weight_kg: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    score = base_score
    bonus: Dict[str, float] = {}
    if user_height_cm is not None and model_info["actor_height_cm"] is not None:
        height_penalty = abs(float(user_height_cm) - float(model_info["actor_height_cm"])) / 15.0
        score += 0.18 * height_penalty
        bonus["height_penalty"] = 0.18 * height_penalty
    if user_weight_kg is not None and model_info["actor_weight_kg"] is not None:
        weight_penalty = abs(float(user_weight_kg) - float(model_info["actor_weight_kg"])) / 12.0
        score += 0.12 * weight_penalty
        bonus["weight_penalty"] = 0.12 * weight_penalty
    return score, bonus


def rank_all_models_integrated(
    user_ratio_features: Dict[str, float],
    user_width_features: Dict[str, Optional[float]],
    json_dir: str,
    width_feature_json_path: str,
    user_sex: Optional[str] = None,
    user_height_cm: Optional[float] = None,
    user_weight_kg: Optional[float] = None,
    top_k: int = TOP_K,
    prefilter_k: int = PREFILTER_K,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    user_sex_norm = normalize_sex(user_sex)
    width_index = load_width_feature_index(width_feature_json_path)
    stage1_results: List[Dict[str, Any]] = []

    for json_path in Path(json_dir).rglob("*.json"):
        try:
            info = load_body_json(json_path)
            json_sex_norm = normalize_sex(info.get("sex"))
            if user_sex_norm is not None and json_sex_norm is not None:
                if user_sex_norm != json_sex_norm:
                    continue

            ann_id = info.get("annotation_id")
            model_width_features = width_index.get(ann_id)

            shape_score, score_pack = combined_body_score(
                user_skeleton_ratios=user_ratio_features,
                model_skeleton_ratios=info["ratio_features"],
                user_width_features=user_width_features,
                model_width_features=model_width_features or {},
            )

            info["shape_score"] = shape_score
            info["skeleton_score"] = score_pack["skeleton_score"]
            info["width_score"] = score_pack["width_score"]
            info["score_detail"] = {
                "skeleton_detail": score_pack["skeleton_detail"],
                "width_detail": score_pack["width_detail"],
            }
            stage1_results.append(info)
        except Exception as e:
            print(f"[SKIP] {json_path.name}: {e}")

    if not stage1_results:
        return [], []

    stage1_results.sort(key=lambda x: x["shape_score"])
    stage2_candidates = stage1_results[:prefilter_k]

    reranked_results: List[Dict[str, Any]] = []
    for info in stage2_candidates:
        final_score, bonus = rerank_with_actor_info(
            info["shape_score"],
            info,
            user_height_cm=user_height_cm,
            user_weight_kg=user_weight_kg,
        )
        info["final_score"] = final_score
        info["bonus_detail"] = bonus
        reranked_results.append(info)

    reranked_results.sort(key=lambda x: x["final_score"])
    return reranked_results[:top_k], reranked_results


def resolve_obj_path(obj_dir: str, obj_file_name: Optional[str]) -> Optional[str]:
    if not obj_file_name:
        return None
    direct = Path(obj_dir) / obj_file_name
    if direct.exists():
        return str(direct)
    matches = list(Path(obj_dir).rglob(obj_file_name))
    if matches:
        return str(matches[0])
    return None


def load_trimesh_safe(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if mesh is None:
        raise RuntimeError(f"trimesh가 파일을 읽지 못했습니다: {path}")
    if isinstance(mesh, trimesh.Scene):
        geometries = []
        for g in mesh.geometry.values():
            if isinstance(g, trimesh.Trimesh):
                geometries.append(g)
        if not geometries:
            raise RuntimeError(f"Scene 안에 mesh geometry가 없습니다: {path}")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Trimesh 객체가 아닙니다: {path}")
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def apply_solid_color(mesh: trimesh.Trimesh, rgba: List[int]) -> None:
    rgba_np = np.array(rgba, dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=np.tile(rgba_np, (len(mesh.faces), 1)))


def export_body_only_glb(body_path: str, output_glb_path: str) -> str:
    body = load_trimesh_safe(body_path)
    apply_solid_color(body, [235, 235, 235, 255])
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name="body")
    scene.export(output_glb_path)
    if not os.path.exists(output_glb_path):
        raise RuntimeError(f"GLB 저장 실패: {output_glb_path}")
    return output_glb_path


def merge_body_and_mesh_to_glb(body_path: str, tape_path: str, output_glb_path: str) -> str:
    body = load_trimesh_safe(body_path)
    tape = load_trimesh_safe(tape_path)
    apply_solid_color(body, [235, 235, 235, 255])
    apply_solid_color(tape, [255, 120, 40, 255])
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name="body")
    scene.add_geometry(tape, node_name="tape")
    scene.export(output_glb_path)
    if not os.path.exists(output_glb_path):
        raise RuntimeError(f"GLB 저장 실패: {output_glb_path}")
    return output_glb_path


def draw_debug_preview(image_bgr: np.ndarray, pts: Dict[str, List[float]], debug_info: Dict[str, Any], save_path: str) -> str:
    vis = image_bgr.copy()
    skeleton_lines = [
        ("Left shoulder", "Right shoulder"),
        ("Left shoulder", "Left hip"),
        ("Right shoulder", "Right hip"),
        ("Left hip", "Right hip"),
        ("Left hip", "Left knee"),
        ("Left knee", "Left ankle"),
        ("Right hip", "Right knee"),
        ("Right knee", "Right ankle"),
        ("Left shoulder", "Left elbow"),
        ("Left elbow", "Left wrist"),
        ("Right shoulder", "Right elbow"),
        ("Right elbow", "Right wrist"),
    ]
    for a, b in skeleton_lines:
        p1 = tuple(np.int32(pts[a]))
        p2 = tuple(np.int32(pts[b]))
        cv2.line(vis, p1, p2, (0, 255, 255), 2)
    for _, p in pts.items():
        x, y = int(p[0]), int(p[1])
        cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)

    line_specs = [
        ("chest_y", (255, 0, 0), "chest"),
        ("waist_y", (0, 255, 255), "waist"),
        ("hip_y", (255, 0, 255), "hip"),
        ("left_thigh_y", (0, 165, 255), "left thigh"),
        ("right_thigh_y", (0, 165, 255), "right thigh"),
    ]
    h, w = vis.shape[:2]
    for key, color, label in line_specs:
        y = int(debug_info[key])
        cv2.line(vis, (0, y), (w - 1, y), color, 1)
        cv2.putText(vis, label, (10, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    seg_colors = {
        "chest": (255, 0, 0),
        "waist": (0, 255, 255),
        "hip": (255, 0, 255),
        "left_thigh": (0, 165, 255),
        "right_thigh": (0, 165, 255),
    }
    for name, segs in debug_info["segments"].items():
        color = seg_colors[name]
        for y, x1, x2 in segs:
            cv2.line(vis, (x1, y), (x2, y), color, 3)

    cv2.imwrite(save_path, vis)
    return save_path


def extract_user_features(
    image_path: str,
    height_cm: float,
    weight_kg: float,
    sex: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if output_dir is None:
        run_id = uuid.uuid4().hex[:12]
        output_dir = str(APP_TMP_DIR / run_id)

    ensure_dir(output_dir)

    converted_path = str(Path(output_dir) / "input_converted.jpg")
    mask_path = str(Path(output_dir) / "mask.png")
    debug_image_path = str(Path(output_dir) / "debug.jpg")

    converted_path = save_uploaded_image_to_jpg(image_path, converted_path)
    image_bgr, user_pts = detect_pose_from_photo(converted_path, MEDIAPIPE_MODEL_PATH)

    user_raw_features, user_ratio_features = compute_body_features_from_points(user_pts)
    mask = segment_person_mask(converted_path, mask_path)
    user_width_features, debug_info = compute_width_features_from_photo(mask, user_pts)
    draw_debug_preview(image_bgr, user_pts, debug_info, debug_image_path)

    return {
        "paths": {
            "input_image_path": image_path,
            "converted_image_path": converted_path,
            "mask_path": mask_path,
            "debug_image_path": debug_image_path,
            "output_dir": output_dir,
        },
        "user_info": {
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "sex": normalize_sex(sex),
        },
        "raw_features": user_raw_features,
        "ratio_features": user_ratio_features,
        "width_features": user_width_features,
        "pose_points": user_pts,
        "debug_info": debug_info,
    }


def rank_body_candidates(
    user_ratio_features: Dict[str, float],
    user_width_features: Dict[str, Optional[float]],
    user_height_cm: float,
    user_weight_kg: float,
    user_sex: str,
    top_k: int = TOP_K,
    prefilter_k: int = PREFILTER_K,
) -> Dict[str, Any]:
    top_matches, _ = rank_all_models_integrated(
        user_ratio_features=user_ratio_features,
        user_width_features=user_width_features,
        json_dir=BODY_JSON_DIR,
        width_feature_json_path=WIDTH_FEATURE_JSON_PATH,
        user_sex=user_sex,
        user_height_cm=user_height_cm,
        user_weight_kg=user_weight_kg,
        top_k=top_k,
        prefilter_k=prefilter_k,
    )

    if not top_matches:
        raise RuntimeError("조건에 맞는 후보를 찾지 못했습니다.")

    best_match = top_matches[0]
    best_obj_path = resolve_obj_path(OBJ_DIR, best_match.get("mesh_obj_file_name"))
    if best_obj_path is None:
        raise FileNotFoundError(f"best match의 OBJ 파일을 찾지 못했습니다: {best_match.get('mesh_obj_file_name')}")
    best_match["body_obj_path"] = best_obj_path

    normalized_top_matches: List[Dict[str, Any]] = []
    for idx, item in enumerate(top_matches, start=1):
        obj_path = resolve_obj_path(OBJ_DIR, item.get("mesh_obj_file_name"))
        normalized_top_matches.append({
            "rank": idx,
            "annotation_id": item.get("annotation_id"),
            "model_id": item.get("model_id"),
            "body_obj_path": obj_path,
            "json_path": item.get("json_path"),
            "shape_score": item.get("shape_score"),
            "final_score": item.get("final_score"),
        })

    return {
        "best_match": best_match,
        "top_matches": normalized_top_matches,
        "prefilter_k": prefilter_k,
    }


def render_body_glb(body_obj_path: str, output_dir: str, file_name: str = "result_body.glb") -> str:
    ensure_dir(output_dir)
    output_glb_path = str(Path(output_dir) / file_name)
    return export_body_only_glb(body_obj_path, output_glb_path)


def render_body_with_tape_glb(body_obj_path: str, tape_mesh_path: str, output_dir: str, file_name: str = "result_with_tape.glb") -> str:
    ensure_dir(output_dir)
    output_glb_path = str(Path(output_dir) / file_name)
    return merge_body_and_mesh_to_glb(body_obj_path, tape_mesh_path, output_glb_path)



def run_body_search(
    image_path: Optional[str],
    height_cm: Optional[float],
    weight_kg: Optional[float],
    sex: Optional[str],
    output_dir: Optional[str] = None,
    top_k: int = TOP_K,
    prefilter_k: int = PREFILTER_K,
    rag_result: Optional[Dict[str, Any]] = None,
    tape_type: Optional[str] = None,
    selected_option_rank: int = 1,
    registry_path: Optional[str] = None,
    privacy_opt_out: bool = False,
    default_body_obj_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    개인정보 입력이 없는 경우:
    - MediaPipe / body search 생략
    - default_body_obj_path 사용
    - RAG technique_code + body model name으로 asset_id 생성
    - registry에서 tape mesh 찾기
    - body 경로 / tape mesh 경로 / guide video 경로만 반환

    개인정보 입력이 있는 경우:
    - 기존 body search 알고리즘 수행
    - best body model 선택
    - RAG technique_code + best body model name으로 asset_id 생성
    - registry에서 tape mesh 찾기
    - body 경로 / tape mesh 경로 / guide video 경로만 반환

    주의:
    - 이 함수는 body+tape를 합성한 새 모델을 만들지 않는다.
    - 프론트/뷰어가 반환된 경로의 body와 tape mesh를 그대로 표시하면 된다.
    """
    if output_dir is None:
        request_id = uuid.uuid4().hex[:12]
        output_dir = str(APP_TMP_DIR / request_id)
    else:
        request_id = Path(output_dir).name

    ensure_dir(output_dir)

    selected_rag_option = get_selected_rag_option(rag_result, selected_option_rank)
    resolved_tape_type = normalize_tape_type(
        tape_type if tape_type is not None else extract_tape_type_from_rag_result(rag_result, selected_option_rank)
    )

    selected_tape_asset: Optional[Dict[str, Any]] = None
    body_obj_path: Optional[str] = None
    best_match_payload: Dict[str, Any]
    user_features_payload: Optional[Dict[str, Any]] = None
    top_matches_payload: List[Dict[str, Any]] = []
    converted_image_path: Optional[str] = None
    mask_path: Optional[str] = None
    debug_image_path: Optional[str] = None

    if privacy_opt_out:
        if not default_body_obj_path:
            raise ValueError("privacy_opt_out=True 인 경우 default_body_obj_path가 필요합니다.")
        body_obj_path = default_body_obj_path

        if selected_rag_option and registry_path:
            technique_code = selected_rag_option.get("technique_code")
            if not technique_code:
                raise ValueError("RAG option에 technique_code가 없습니다.")
            selected_tape_asset = find_taping_asset_for_body(
                registry_path=registry_path,
                body_obj_path=body_obj_path,
                technique_code=technique_code,
            )

        best_match_payload = {
            "annotation_id": None,
            "model_id": get_body_model_key_from_path(body_obj_path),
            "body_obj_path": body_obj_path,
            "json_path": None,
            "shape_score": None,
            "final_score": None,
        }

    else:
        if image_path is None or height_cm is None or weight_kg is None or sex is None:
            raise ValueError("개인정보 입력 사용 시 image_path, height_cm, weight_kg, sex가 모두 필요합니다.")

        user_features = extract_user_features(
            image_path=image_path,
            height_cm=height_cm,
            weight_kg=weight_kg,
            sex=sex,
            output_dir=output_dir,
        )

        ranked = rank_body_candidates(
            user_ratio_features=user_features["ratio_features"],
            user_width_features=user_features["width_features"],
            user_height_cm=height_cm,
            user_weight_kg=weight_kg,
            user_sex=sex,
            top_k=top_k,
            prefilter_k=prefilter_k,
        )

        best_match = ranked["best_match"]
        body_obj_path = best_match["body_obj_path"]

        if selected_rag_option and registry_path:
            technique_code = selected_rag_option.get("technique_code")
            if not technique_code:
                raise ValueError("RAG option에 technique_code가 없습니다.")
            selected_tape_asset = find_taping_asset_for_body(
                registry_path=registry_path,
                body_obj_path=body_obj_path,
                technique_code=technique_code,
            )

        user_features_payload = {
            "raw_features": user_features["raw_features"],
            "ratio_features": user_features["ratio_features"],
            "width_features": user_features["width_features"],
        }
        top_matches_payload = ranked["top_matches"]
        converted_image_path = user_features["paths"]["converted_image_path"]
        mask_path = user_features["paths"]["mask_path"]
        debug_image_path = user_features["paths"]["debug_image_path"]

        best_match_payload = {
            "annotation_id": best_match.get("annotation_id"),
            "model_id": best_match.get("model_id"),
            "body_obj_path": best_match.get("body_obj_path"),
            "json_path": best_match.get("json_path"),
            "shape_score": best_match.get("shape_score"),
            "final_score": best_match.get("final_score"),
        }

    report_path = str(Path(output_dir) / "body_search_report.json")
    result = {
        "request_id": request_id,
        "status": "success",
        "inputs": {
            "image_path": image_path,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "sex": normalize_sex(sex),
            "top_k": top_k,
            "prefilter_k": prefilter_k,
            "selected_option_rank": selected_option_rank,
            "privacy_opt_out": privacy_opt_out,
        },
        "selected_rag_option": selected_rag_option,
        "guide": {
            "tape_type": resolved_tape_type,
            "guide_video_url": selected_tape_asset.get("guide_video_url") if selected_tape_asset else None,
        },
        "selected_tape_asset": selected_tape_asset,
        "user_features": user_features_payload,
        "best_match": best_match_payload,
        "top_matches": top_matches_payload,
        "display_assets": {
            "body_model_path": body_obj_path,
            "tape_mesh_path": selected_tape_asset.get("mesh_file") if selected_tape_asset else None,
            "guide_video_url": selected_tape_asset.get("guide_video_url") if selected_tape_asset else None,
        },
        "artifacts": {
            "output_dir": output_dir,
            "converted_image_path": converted_image_path,
            "mask_path": mask_path,
            "debug_image_path": debug_image_path,
            "report_path": report_path,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":

    sample_image = "user_full_body.jpg"
    sample_rag_result = {
        "options": [
            {
                "option_rank": 1,
                "technique_name": "IT band 이완 테이핑",
                "tape_type": "Y-strip",
            }
        ]
    }

    if os.path.exists(sample_image):
        output = run_body_search(
            image_path=sample_image,
            height_cm=175,
            weight_kg=68,
            sex="male",
            rag_result=sample_rag_result,
        )
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print("샘플 실행을 원하면 user_full_body.jpg 파일을 현재 경로에 두세요.")
