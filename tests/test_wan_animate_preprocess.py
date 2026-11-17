import numpy as np
from PIL import Image
import pytest

from openfal.runtime.wan_animate_preprocess import (
    _augment_replacement_mask,
    _crop_face,
    _decode_pose_heatmaps,
    _pose_input,
    _read_video_frames,
    _render_pose,
    _segment_video,
    _select_person,
)


def test_video_resampling_preserves_cadence_and_stops_at_real_eof(tmp_path):
    import av

    path = tmp_path / "driving.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=3)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(6):
            array = np.full((32, 32, 3), 30*index, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    frames, indices = _read_video_frames(path, 64, 32, fps=2, limit=16)
    assert indices == [0, 2, 3, 5]
    assert len(frames) == 4  # 2 seconds, not a fabricated 8-second requested clip.
    assert all(frame.size == (64,32) for frame in frames)
    assert [round(float(np.asarray(frame)[8:24,24:40].mean()) / 30) for frame in frames] == indices
    assert not np.asarray(frames[-1])[:, :16].any()  # Aspect ratio is padded, not stretched.
    capped, cap_indices = _read_video_frames(path, 32, 32, fps=2, limit=3)
    assert len(capped) == 3 and cap_indices == indices[:3]


def test_person_selection_tracks_foreground_subject_without_switching_to_background():
    # Official end-to-end YOLO rows: x1, y1, x2, y2, confidence, zero-based class.
    person = [100,80,300,600,0.9,0]
    box = _select_person(np.array([[person, [0,0,640,640,0.99,16]]]), (320,320), (640,640))
    np.testing.assert_allclose(box, [50,40,150,300])
    with pytest.raises(ValueError, match="No person"):
        _select_person(np.array([[[0,0,640,640,0.99,16]]]), (320,320), (640,640))
    # A different person becomes larger; the existing track must remain selected.
    tracked = _select_person(np.array([[person, [350,20,639,639,0.95,0]]]), (320,320), (640,640), box)
    np.testing.assert_allclose(tracked, box)
    with pytest.raises(ValueError, match="Lost the driving subject"):
        _select_person(np.array([[[350,20,639,639,0.95,0]]]), (320,320), (640,640), box)


def test_pose_preprocessing_matches_bgr_normalization_and_geometry():
    rgb = np.zeros((80,60,3),dtype=np.uint8)
    rgb[:,:,0] = 255
    data, center, extent = _pose_input(rgb, np.array([10,10,50,70]))
    assert data.shape == (1,3,256,192)
    np.testing.assert_allclose(center, [30,40])
    np.testing.assert_allclose(extent, [56.25,75])
    # The official checkpoint was exported with BGR input, so red lands in channel2.
    np.testing.assert_allclose(data[0,:,128,96], (np.array([0,0,1])-[0.485,0.456,0.406])/[0.229,0.224,0.225], rtol=1e-6)


def test_dark_wholebody_heatmap_decoding_keeps_subpixel_coordinates():
    yy, xx = np.mgrid[:64,:48]
    target = np.array([22.3,30.7])
    gaussian = np.exp(-((xx-target[0])**2+(yy-target[1])**2)/(2*2**2)).astype(np.float32)
    maps = np.broadcast_to(gaussian, (1,133,64,48)).copy()
    landmarks = _decode_pose_heatmaps(maps, np.array([96,128]), np.array([192,256]))
    np.testing.assert_allclose(landmarks[:,:2], np.broadcast_to(target*4, (133,2)), atol=0.03)
    assert (landmarks[:,2] > 0.95).all()
    with pytest.raises(RuntimeError, match="whole-body"):
        _decode_pose_heatmaps(maps[:,:17], np.array([96,128]), np.array([192,256]))


def test_face_crops_require_real_confident_landmarks_and_pose_uses_hands():
    landmarks = np.zeros((133,3),dtype=np.float32)
    angles = np.linspace(0,2*np.pi,68,endpoint=False)
    landmarks[23:91] = np.column_stack([50+10*np.cos(angles),40+15*np.sin(angles),np.full(68,0.9)])
    image = Image.fromarray(np.tile(np.arange(100,dtype=np.uint8)[None,:,None], (100,1,3)))
    crop = np.asarray(_crop_face(image,landmarks))
    assert crop.shape == (512,512,3)
    assert 35 <= crop.min() < crop.max() <= 65
    landmarks[91:112] = np.column_stack([np.linspace(20,80,21),np.full(21,80),np.full(21,0.9)])
    assert np.asarray(_render_pose(landmarks,(100,100)))[78:83].any()
    landmarks[23:91,2] = 0.1
    with pytest.raises(ValueError, match="reliable landmarks"):
        _crop_face(image,landmarks)


def test_replacement_mask_frees_space_for_a_different_hair_and_body_outline():
    import cv2

    silhouette = np.zeros((100,100),dtype=np.uint8)
    cv2.ellipse(silhouette,(50,40),(15,25),0,0,360,1,-1)
    cv2.rectangle(silhouette,(40,65),(60,95),1,-1)
    dilated = cv2.dilate(silhouette,np.ones((7,7),dtype=np.uint8),iterations=3)
    original = dilated.copy()
    augmented = _augment_replacement_mask(dilated)
    rows, columns = np.nonzero(dilated)
    left,right,top,bottom = columns.min(),columns.max(),rows.min(),rows.max()
    # The new hair/shoulder shape can occupy corners outside the source silhouette.
    assert not dilated[top+1,left+1]
    assert augmented[top+1,left+1]
    assert augmented[top:bottom,left:right].all()
    assert not augmented[:top].any() and not augmented[:,:left].any()
    # Keep native's half-open maximum bounds, including original edge pixels.
    np.testing.assert_array_equal(augmented[bottom],dilated[bottom])
    np.testing.assert_array_equal(augmented[:,right],dilated[:,right])
    np.testing.assert_array_equal(dilated,original)


def test_replacement_mask_handles_empty_and_degenerate_bounds():
    with pytest.raises(ValueError,match="empty"):
        _augment_replacement_mask(np.zeros((4,4),dtype=np.uint8))
    for shape in ((1,5),(5,1),(1,1)):
        mask = np.ones(shape,dtype=np.uint8)
        np.testing.assert_array_equal(_augment_replacement_mask(mask),mask)
    with pytest.raises(ValueError,match="two-dimensional"):
        _augment_replacement_mask(np.ones((4,4,3),dtype=np.uint8))


@pytest.mark.parametrize("requested_device", ["cpu", "mps"])
def test_sam2_native_processor_masks_regenerate_person_and_preserve_background(monkeypatch, tmp_path, requested_device):
    from types import SimpleNamespace
    import torch
    from transformers import Sam2ImageProcessor, Sam2VideoModel, Sam2VideoProcessor, Sam2VideoVideoProcessor

    processor = Sam2VideoProcessor(
        image_processor=Sam2ImageProcessor(size={"height":64,"width":64}),
        video_processor=Sam2VideoVideoProcessor(size={"height":64,"width":64}),
    )
    # Exercise the real session/point and mask processors without pretrained
    # network inference in a CPU unit test; live fixtures validate SAM2 itself.
    class ModelOutputStub:
        def to(self, device):
            assert device.type == "cpu"
            return self

        def eval(self):
            return self

        def propagate_in_video_iterator(self, session, start_frame_idx):
            for index in range(2):
                logits = torch.full((1,1,16,16), -10.0)
                yy,xx = torch.meshgrid(torch.arange(16),torch.arange(16),indexing="ij")
                logits[:,:,(xx-8)**2+(yy-8)**2 < 16] = 10
                yield SimpleNamespace(frame_idx=index, pred_masks=logits)

    monkeypatch.setattr(Sam2VideoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(Sam2VideoModel, "from_pretrained", lambda *args, **kwargs: ModelOutputStub())
    frames = [Image.new("RGB", (96,64), (90,140,190)) for _ in range(2)]
    landmarks = [np.tile([48,32,0.9], (133,1)) for _ in frames]
    logs = []
    backgrounds, masks = _segment_video(frames, landmarks, tmp_path, requested_device, logs.append)
    if requested_device == "mps":
        assert any("CPU on Apple Silicon" in message for message in logs)
    for background, mask in zip(backgrounds, masks):
        array = np.asarray(mask)
        assert mask.mode == "L" and mask.size == (96,64)
        assert array[32,48] == 255 and array[0,0] == 0
        rows,columns = np.nonzero(array)
        assert (array[rows.min():rows.max(),columns.min():columns.max()] == 255).all()
        np.testing.assert_array_equal(np.asarray(background)[array==0], np.asarray(frames[0])[array==0])
        assert not np.asarray(background)[array>0].any()
