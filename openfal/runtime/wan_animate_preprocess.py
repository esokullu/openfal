"""Real, local-only driving-video preprocessing for Wan 2.2 Animate.

The YOLO/ViTPose input normalization, DARK heatmap decoding, whole-body mapping,
pose colors and face crop follow Alibaba's Apache-2.0 Wan2.2 preprocessing:
https://github.com/Wan-Video/Wan2.2/tree/main/wan/modules/animate/preprocess
Replacement uses Transformers' SAM2 video propagation with real pose prompts.
Missing people, face landmarks or masks fail explicitly; no fabricated inputs.
"""

from __future__ import annotations

import base64
import colorsys
from dataclasses import dataclass, field
import gc
import io
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps


@dataclass
class WanAnimateInputs:
    pose_video: list[Image.Image]
    face_video: list[Image.Image]
    background_video: list[Image.Image] | None
    mask_video: list[Image.Image] | None
    fps: float
    source_frame_indices: list[int]
    face_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)


def _read_video_frames(source: str | Path, width: int, height: int, fps: float, limit: int):
    """Timestamp-based nearest-frame resampling, bounded by the real source duration."""
    import av

    if min(width, height, limit) <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Driving-video size, frame rate and frame limit must be positive")
    source = str(source)
    video_source = source
    if source.startswith("data:"):
        header, separator, body = source.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("Driving-video data URI must contain base64 video data")
        try:
            video_source = io.BytesIO(base64.b64decode(body, validate=True))
        except ValueError as exc:
            raise ValueError("Invalid driving-video base64 data") from exc
    frames, indices = [], []
    try:
        with av.open(video_source, timeout=(15.0, 30.0)) as container:
            if not container.streams.video:
                raise ValueError("Driving media has no video stream")
            stream = container.streams.video[0]
            source_fps = float(stream.average_rate or stream.guessed_rate or fps)
            previous = None
            origin = None
            last_time = 0.0
            for index, decoded in enumerate(container.decode(stream)):
                timestamp = float(decoded.time) if decoded.time is not None else index / source_fps
                if origin is None:
                    origin = timestamp
                timestamp -= origin
                if previous is not None and timestamp < last_time:
                    raise ValueError("Driving-video timestamps are not monotonic")
                current = ImageOps.pad(decoded.to_image().convert("RGB"), (width, height),
                                       method=Image.Resampling.LANCZOS, color=(0, 0, 0))
                while len(frames) < limit and len(frames) / fps <= timestamp + 1e-7:
                    target = len(frames) / fps
                    if previous is not None and target - last_time < timestamp - target:
                        frames.append(previous)
                        indices.append(index - 1)
                    else:
                        frames.append(current)
                        indices.append(index)
                previous, last_time = current, timestamp
                if len(frames) >= limit:
                    break
            # The last encoded frame remains visible for its real frame interval.
            # This allows frame-rate conversion, without looping beyond EOF.
            while previous is not None and len(frames) < limit and len(frames) / fps < last_time + 1 / source_fps - 1e-7:
                frames.append(previous)
                indices.append(index)
    except (OSError, av.FFmpegError) as exc:
        raise ValueError(f"Could not decode driving video: {exc}") from exc
    if len(frames) < 2:
        raise ValueError("Wan Animate requires at least two decoded driving-video frames")
    return frames, indices


def _select_person(detections: np.ndarray, frame_size: tuple[int, int], model_size: tuple[int, int], previous_bbox=None):
    """Select the largest initial subject, then retain that person's spatial track."""
    import cv2

    rows = np.asarray(detections).reshape(-1, 6)
    rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= 0.3) & (rows[:, 5] == 0)].copy()
    if not len(rows):
        raise ValueError("No person detected in driving frame")
    width, height = frame_size
    model_width, model_height = model_size
    rows[:, [0, 2]] *= width / model_width
    rows[:, [1, 3]] *= height / model_height
    rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, width)
    rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, height)
    rows = rows[(rows[:, 2] - rows[:, 0] >= 10) & (rows[:, 3] - rows[:, 1] >= 10)]
    if not len(rows):
        raise ValueError("Detected person is too small for pose estimation")
    xywh = rows[:, :4].copy()
    xywh[:, 2:] -= xywh[:, :2]
    keep = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), rows[:, 4].tolist(), 0.3, 0.5)).reshape(-1)
    rows = rows[keep]
    areas = (rows[:, 2] - rows[:, 0]) * (rows[:, 3] - rows[:, 1])
    largest = int(areas.argmax())
    if previous_bbox is not None:
        intersection_size = np.maximum(0, np.minimum(rows[:,2:4], previous_bbox[2:]) - np.maximum(rows[:,:2], previous_bbox[:2]))
        intersections = intersection_size.prod(axis=1)
        previous_area = np.prod(previous_bbox[2:] - previous_bbox[:2])
        overlaps = intersections / np.maximum(areas + previous_area - intersections, 1e-6)
        largest = int(overlaps.argmax())
        if overlaps[largest] < 0.1:
            raise ValueError("Lost the driving subject's track; use a continuous clip with a visible foreground person")
    return rows[largest, :4]


def _pose_input(rgb: np.ndarray, bbox: np.ndarray, resolution=(256, 192)):
    """Match the official ViTPose-H BGR crop and ImageNet normalization."""
    import cv2

    input_height, input_width = resolution
    center = (bbox[:2] + bbox[2:]) / 2
    crop_height = max((bbox[2] - bbox[0]) * input_height / input_width, bbox[3] - bbox[1]) * 1.25
    extent = np.array([crop_height * input_width / input_height, crop_height], dtype=np.float32)
    upper = np.round(center - extent / 2).astype(int)
    lower = np.round(center + extent / 2).astype(int)
    crop = np.zeros((lower[1] - upper[1], lower[0] - upper[0], 3), dtype=np.float32)
    x0, y0 = np.maximum(upper, 0)
    x1, y1 = np.minimum(lower, [rgb.shape[1], rgb.shape[0]])
    crop[y0-upper[1]:y1-upper[1], x0-upper[0]:x1-upper[0]] = rgb[y0:y1, x0:x1, ::-1]
    crop = cv2.resize(crop, (input_width, input_height))
    crop = (crop / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return crop.transpose(2, 0, 1)[None].astype(np.float32), center, extent


def _decode_pose_heatmaps(heatmaps: np.ndarray, center: np.ndarray, extent: np.ndarray):
    """DARK unbiased decoding used by the official whole-body ViTPose checkpoint."""
    import cv2

    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    if heatmaps.ndim != 4 or heatmaps.shape[:2] != (1, 133) or not np.isfinite(heatmaps).all():
        raise RuntimeError(f"Expected finite ViTPose whole-body heatmaps [1,133,H,W], got {heatmaps.shape}")
    maps = heatmaps[0]
    count, height, width = maps.shape
    flat = maps.reshape(count, -1)
    maxima = flat.argmax(axis=1)
    confidence = flat[np.arange(count), maxima]
    points = np.stack([maxima % width, maxima // width], axis=-1).astype(np.float32)
    for joint, original in enumerate(maps):
        padded = np.pad(original, 5)
        blurred = cv2.GaussianBlur(padded, (11, 11), 0)[5:-5, 5:-5]
        peak = float(blurred.max())
        if peak <= 0 or confidence[joint] <= 0:
            continue
        logmap = np.log(np.maximum(blurred * (confidence[joint] / peak), 1e-10))
        x, y = map(int, points[joint])
        if 1 < x < width - 2 and 1 < y < height - 2:
            dx = (logmap[y, x+1] - logmap[y, x-1]) * 0.5
            dy = (logmap[y+1, x] - logmap[y-1, x]) * 0.5
            dxx = (logmap[y, x+2] - 2*logmap[y, x] + logmap[y, x-2]) * 0.25
            dyy = (logmap[y+2, x] - 2*logmap[y, x] + logmap[y-2, x]) * 0.25
            dxy = (logmap[y+1, x+1] - logmap[y-1, x+1] - logmap[y+1, x-1] + logmap[y-1, x-1]) * 0.25
            hessian = np.array([[dxx, dxy], [dxy, dyy]])
            if abs(np.linalg.det(hessian)) > 1e-12:
                points[joint] -= np.linalg.solve(hessian, [dx, dy])
    points *= extent / np.array([width, height])
    points += center - extent / 2
    return np.column_stack([points, confidence])


class _WholeBodyEstimator:
    def __init__(self, detector_path: Path, pose_path: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = min(8, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        # Official ONNX CPU kernels work on Windows and Apple Silicon without a
        # second CUDA ABI dependency or implicit provider fallback.
        self.detector = ort.InferenceSession(str(detector_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.pose = ort.InferenceSession(str(pose_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.previous_bbox = None

    def __call__(self, image: Image.Image):
        import cv2

        rgb = np.asarray(image)
        det_input = self.detector.get_inputs()[0]
        det_height, det_width = map(int, det_input.shape[-2:])
        data = cv2.resize(rgb, (det_width, det_height)).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        detections = self.detector.run(None, {det_input.name: data})[0]
        bbox = _select_person(detections, image.size, (det_width, det_height), self.previous_bbox)
        pose_input = self.pose.get_inputs()[0]
        resolution = tuple(map(int, pose_input.shape[-2:]))
        data, center, extent = _pose_input(rgb, bbox, resolution)
        heatmaps = self.pose.run(None, {pose_input.name: data})[0]
        landmarks = _decode_pose_heatmaps(heatmaps, center, extent)
        if np.count_nonzero(landmarks[:17, 2] >= 0.5) < 4:
            raise ValueError("Too few reliable body landmarks in driving frame")
        self.previous_bbox = bbox
        return landmarks


_BODY_A = [0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 20]
_BODY_B = [0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 21]
_LIMBS = [(1,2), (1,5), (2,3), (3,4), (5,6), (6,7), (1,8), (8,9), (9,10),
          (1,11), (11,12), (12,13), (1,0), (0,14), (14,16), (0,15), (15,17), (13,18), (10,19)]
_COLORS = [(255,0,0), (255,85,0), (255,170,0), (255,255,0), (170,255,0), (85,255,0),
           (0,255,0), (0,255,85), (0,255,170), (0,255,255), (0,170,255), (0,85,255),
           (0,0,255), (85,0,255), (170,0,255), (255,0,255), (255,0,170), (255,0,85),
           (200,200,0), (100,100,0)]


def _render_pose(landmarks: np.ndarray, size: tuple[int, int]):
    import cv2

    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    body = (landmarks[_BODY_A] + landmarks[_BODY_B]) / 2
    thickness = max(min(width, height) // 200 - 1, 1)
    for (a, b), color in zip(_LIMBS, _COLORS):
        first, second = body[a], body[b]
        if min(first[2], second[2]) < 0.5:
            continue
        midpoint = (first[:2] + second[:2]) / 2
        delta = second[:2] - first[:2]
        polygon = cv2.ellipse2Poly(tuple(midpoint.astype(int)), (int(np.linalg.norm(delta)/2), thickness),
                                  int(math.degrees(math.atan2(delta[1], delta[0]))), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, tuple(int(channel*0.6) for channel in color))
    for point, color in zip(body, _COLORS):
        if point[2] >= 0.5:
            cv2.circle(canvas, tuple(point[:2].astype(int)), thickness, color, -1)
    edges = [(0, base) if part == 0 else (base+part-1, base+part)
             for base in (1,5,9,13,17) for part in range(4)]
    for hand in (landmarks[91:112], landmarks[112:133]):
        for index, (a, b) in enumerate(edges):
            if min(hand[a,2], hand[b,2]) >= 0.5 and (hand[[a,b],:2] > 0).all():
                color = tuple(int(v*255) for v in colorsys.hsv_to_rgb(index / 20, 1, 1))
                cv2.line(canvas, tuple(hand[a,:2].astype(int)), tuple(hand[b,:2].astype(int)), color, max(thickness//2,1))
        for point in hand:
            if point[2] >= 0.5 and (point[:2] > 0).all():
                cv2.circle(canvas, tuple(point[:2].astype(int)), max(thickness//2,1), (0,0,255), -1)
    return Image.fromarray(canvas)


def _crop_face(image: Image.Image, landmarks: np.ndarray):
    # WholeBody indices23:91 are the actual68 face landmarks. Require measured
    # confidence so an obscured face cannot become an arbitrary full-frame crop.
    face = landmarks[23:91]
    face = face[face[:,2] >= 0.25, :2]
    if len(face) < 8:
        raise ValueError("Face is not sufficiently visible in driving frame (fewer than eight reliable landmarks)")
    minimum, maximum = face.min(axis=0), face.max(axis=0)
    extent = maximum - minimum
    if min(extent) < 4:
        raise ValueError("Detected face is too small for expression conditioning")
    delta = extent * (math.sqrt(1.3) - 1)
    left, top = np.maximum(minimum - delta * [0.5,0.75], 0).astype(int)
    right, bottom = np.minimum(maximum + delta * [0.5,0.25], image.size).astype(int)
    if right <= left or bottom <= top:
        raise ValueError("Detected face lies outside driving frame")
    bounds = (int(left), int(top), int(right), int(bottom))
    crop = image.crop(bounds).resize((512,512), Image.Resampling.BICUBIC)
    crop.info["source_bbox"] = bounds
    return crop


def _augment_replacement_mask(mask: np.ndarray) -> np.ndarray:
    """Match upstream ``get_aug_mask(w_len=1, h_len=1)`` after dilation.

    A tight silhouette exposes the old outline in the preserved background and
    constrains a replacement's hair/body shape. Wan's default fills the subject's
    bounding rectangle. Its slices exclude the maximum row/column, retaining the
    original foreground pixels there; preserve that detail for exact parity.
    """
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError("Replacement subject mask must be two-dimensional")
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("Replacement subject mask is empty")
    left, right = int(columns.min()), int(columns.max())
    top, bottom = int(rows.min()), int(rows.max())
    # Native range(..., step=0) fails for a one-pixel row/column. Preserve its
    # existing foreground safely in that degenerate case.
    if right > left and bottom > top:
        mask[top:bottom, left:right] = 1
    return mask


def _segment_video(frames, landmarks, sam_path: Path, device, on_log):
    import cv2
    import torch
    from openfal.runtime.common import check_inference_dependencies

    check_inference_dependencies(("transformers", "torchvision"))
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    device = torch.device(device)
    if device.type == "mps":
        device = torch.device("cpu")
        if on_log:
            on_log("[Wan Animate] SAM2 segmentation uses CPU on Apple Silicon; generation remains on MPS")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = processor = session = None
    try:
        processor = Sam2VideoProcessor.from_pretrained(str(sam_path), local_files_only=True)
        model = Sam2VideoModel.from_pretrained(str(sam_path), local_files_only=True, torch_dtype=dtype).to(device).eval()
        session = processor.init_video_session(video=frames, inference_device=device,
                                               inference_state_device="cpu", processing_device="cpu",
                                               video_storage_device="cpu", dtype=dtype)
        # The official preprocessing anchors SAM2 with pose points on four frames.
        for index in sorted(set(np.linspace(0, len(frames)-1, min(4,len(frames)), dtype=int).tolist())):
            body = (landmarks[index][_BODY_A] + landmarks[index][_BODY_B]) / 2
            points = body[[0,1,2,5,8,11,10,13]]
            points = points[points[:,2] >= 0.5, :2]
            width, height = frames[index].size
            points = points[(points[:,0]>=0) & (points[:,0]<width) & (points[:,1]>=0) & (points[:,1]<height)]
            if not len(points):
                raise ValueError(f"No reliable person landmarks for segmentation frame {index+1}")
            processor.add_inputs_to_inference_session(session, frame_idx=index, obj_ids=1,
                input_points=[[points.tolist()]], input_labels=[[[1]*len(points)]])
        masks = [None]*len(frames)
        with torch.inference_mode():
            for result in model.propagate_in_video_iterator(session, start_frame_idx=0):
                index = int(result.frame_idx)
                width, height = frames[index].size
                mask = processor.post_process_masks([result.pred_masks], original_sizes=[(height,width)])[0][0,0]
                mask = mask.detach().cpu().numpy().astype(np.uint8)
                if not mask.any():
                    raise ValueError(f"SAM2 lost the subject in frame {index+1}")
                # Match upstream replacement dilation; white means regenerate.
                mask = cv2.dilate(mask, np.ones((7,7),dtype=np.uint8), iterations=3)
                mask = _augment_replacement_mask(mask)
                masks[index] = Image.fromarray(mask*255, mode="L")
        if any(mask is None for mask in masks):
            raise RuntimeError("SAM2 did not return a mask for every driving frame")
        backgrounds = [Image.fromarray(np.asarray(frame) * (1-(np.asarray(mask)>0))[:,:,None].astype(np.uint8))
                       for frame,mask in zip(frames,masks)]
        return backgrounds, masks
    finally:
        del session, model, processor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def preprocess_wan_animate(
    driving_video: str | Path, *, weights_dir: Path, width: int, height: int,
    fps: float, num_frames: int, mode: str = "animate", device="cpu",
    on_log: Callable[[str], None] | None = None,
) -> WanAnimateInputs:
    """Track one foreground person without downloads or fabricated frames.

    ``num_frames`` caps the sampled source length; shorter clips retain their real
    duration. Pose retargeting is disabled, as in upstream's basic animation mode.
    The largest detected person in the first frame is followed throughout the clip.
    All preprocessing models are released before returning CPU/PIL conditioning.
    """
    from openfal.model_assets import require_animate_assets

    if mode not in ("animate", "replace"):
        raise ValueError("Wan Animate mode must be 'animate' or 'replace'")
    if mode == "replace":
        from openfal.runtime.common import check_inference_dependencies

        check_inference_dependencies(("transformers", "torchvision"))
    assets = require_animate_assets(weights_dir)
    frames, indices = _read_video_frames(driving_video, width, height, fps, num_frames)
    if on_log:
        on_log(f"[Wan Animate] Extracting real whole-body pose and face motion from {len(frames)} frames (ONNX CPU)")
    estimator = _WholeBodyEstimator(assets["detector"], assets["pose"])
    landmarks, poses, faces = [], [], []
    try:
        for index, frame in enumerate(frames):
            try:
                detected = estimator(frame)
                poses.append(_render_pose(detected, frame.size))
                faces.append(_crop_face(frame, detected))
                landmarks.append(detected)
            except (ValueError, RuntimeError) as exc:
                raise ValueError(f"Driving frame {index+1}: {exc}") from exc
            if on_log and (index == 0 or (index+1)%10 == 0 or index+1 == len(frames)):
                on_log(f"[Wan Animate] Pose/face preprocessing {index+1}/{len(frames)}")
    finally:
        del estimator
        gc.collect()
    backgrounds = masks = None
    if mode == "replace":
        if on_log:
            on_log("[Wan Animate] Tracking the replacement subject with SAM2 video segmentation")
        backgrounds, masks = _segment_video(frames, landmarks, assets["sam2"], device, on_log)
    return WanAnimateInputs(poses, faces, backgrounds, masks, float(fps), indices,
                            [face.info["source_bbox"] for face in faces])
