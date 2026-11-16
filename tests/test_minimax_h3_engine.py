from pathlib import Path
from PIL import Image
from openfal.config import EngineConfig
from openfal.engine.runner import OpenFalRunner


def test_qwen3_vl_cache_and_eviction(tmp_path: Path) -> None:
    cfg = EngineConfig(
        profile_id="dual_5060ti_32gb",
        output_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
    )
    runner = OpenFalRunner(cfg)

    emb1 = runner.qwen_encoder.encode("A cinematic shot of an astronaut walking on Mars")
    assert emb1.shape == (128, 64)
    assert runner.text_cache.misses == 1
    assert runner.memory.qwen3_vl_vram_mb == 0.0  # Evicted immediately after stage 1

    emb2 = runner.qwen_encoder.encode("A cinematic shot of an astronaut walking on Mars")
    assert runner.text_cache.hits == 1
    assert (emb1 == emb2).all()


def test_runner_t2i_and_video_face_swap(tmp_path: Path) -> None:
    cfg = EngineConfig(
        profile_id="single_5090_32gb",
        output_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
    )
    runner = OpenFalRunner(cfg)

    # 1. Generate a reference portrait image using H3-Image
    img_res = runner.execute(
        endpoint_id="minimax/h3/text-to-image",
        payload={
            "prompt": "Portrait of a smiling cybernetic explorer",
            "resolution": "480P",
            "aspect_ratio": "1:1",
            "num_inference_steps": 3,
            "seed": 42,
        },
    )
    assert len(img_res["images"]) == 1
    img_file = cfg.output_dir / img_res["images"][0]["file_name"]
    assert img_file.exists()
    with Image.open(img_file) as im:
        assert im.size == (512, 512)

    # 2. Run Video Face Swap (`minimax/h3-max/recast`) conditioned on the reference face image
    swap_res = runner.execute(
        endpoint_id="minimax/h3-max/recast",
        payload={
            "prompt": "Explorer speaking on bridge of starship",
            "face_image_url": str(img_file),
            "duration": 1.0,
            "fps": 12,
            "resolution": "480P",
            "num_inference_steps": 3,
            "seed": 99,
        },
    )
    assert "video" in swap_res
    assert swap_res["strata_engine"]["partition"] == "Ref2VA"
    vid_file = cfg.output_dir / swap_res["video"]["file_name"]
    assert vid_file.exists()
    assert vid_file.stat().st_size > 1000
