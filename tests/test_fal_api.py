import io
import time
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image

from openfal.config import EngineConfig
from openfal.server.app import create_app


def _make_client(tmp_path: Path) -> TestClient:
    cfg = EngineConfig(
        profile_id="dual_5060ti_32gb",
        weights_dir=tmp_path / "weights",
        output_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
    )
    app = create_app(cfg)
    return TestClient(app)


def test_model_downloader_and_strata_shard_packer(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    hw_resp = client.get("/v1/models/hardware-detect")
    assert hw_resp.status_code == 200
    assert "recommended_profile_id" in hw_resp.json()

    st0 = client.get("/v1/models/status").json()
    assert len(st0["components"]) == 5

    # Pack 50-block Strata safetensors shards for Ref2VA (Video Face Swap partition)
    dl_resp = client.post(
        "/v1/models/download",
        json={
            "component_id": "ref2va",
            "quantization": "int4_awq",
            "mode": "strata_repack",
            "run_async": False,
        },
    )
    assert dl_resp.status_code == 200
    assert dl_resp.json()["status"] == "COMPLETED"

    st1 = client.get("/v1/models/status").json()
    ref2va_info = next(c for c in st1["components"] if c["component_id"] == "ref2va")
    assert ref2va_info["installed"] is True
    assert ref2va_info["strata_50_blocks_ready"] is True
    assert ref2va_info["strata_blocks_count"] == 50

    # Delete component
    del_resp = client.delete("/v1/models/ref2va")
    assert del_resp.status_code == 200
    assert del_resp.json()["component"]["installed"] is False


def test_web_ui_and_engine_endpoints(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    ui_resp = client.get("/")
    assert ui_resp.status_code == 200
    assert "openfal" in ui_resp.text
    assert "MiniMax-H3" in ui_resp.text

    st_resp = client.get("/v1/engine/status")
    assert st_resp.status_code == 200
    st = st_resp.json()
    assert st["profile"]["profile_id"] == "dual_5060ti_32gb"
    assert len(st["blocks"]) == 50

    # Switch to 2x 5060 Ti 24GB profile
    sw_resp = client.post("/v1/engine/profile", json={"profile_id": "dual_5060ti_24gb"})
    assert sw_resp.status_code == 200
    assert sw_resp.json()["profile"]["total_vram_gb"] == 24.0

    # Sleep and Wake
    sl_resp = client.post("/v1/engine/sleep")
    assert sl_resp.status_code == 200
    assert sl_resp.json()["telemetry"]["sleeping"] is True

    wk_resp = client.post("/v1/engine/wake")
    assert wk_resp.status_code == 200
    assert wk_resp.json()["telemetry"]["sleeping"] is False


def test_fal_compatible_sync_and_queue_endpoints(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # 1. Upload a reference face image to `/v1/storage/upload`
    buf = io.BytesIO()
    Image.new("RGB", (128, 128), color=(210, 150, 120)).save(buf, format="PNG")
    up_resp = client.post(
        "/v1/storage/upload",
        content=buf.getvalue(),
        headers={"Content-Type": "image/png"},
    )
    assert up_resp.status_code == 200
    face_url = up_resp.json()["url"]

    # 2. Synchronous Text-to-Image (`fal-ai/minimax/h3/text-to-image`)
    t2i_resp = client.post(
        "/fal-ai/minimax/h3/text-to-image",
        json={
            "prompt": "Futuristic architectural pavilion at dusk",
            "resolution": "480P",
            "aspect_ratio": "16:9",
            "num_inference_steps": 2,
        },
    )
    assert t2i_resp.status_code == 200
    t2i_data = t2i_resp.json()
    assert len(t2i_data["images"]) == 1

    # Fetch generated image from `/files/{file_name}`
    fname = t2i_data["images"][0]["file_name"]
    file_resp = client.get(f"/files/{fname}")
    assert file_resp.status_code == 200
    assert len(file_resp.content) > 500

    # 3. Queued Video Face Swap (`POST /queue/minimax/h3-max/recast`)
    q_resp = client.post(
        "/queue/minimax/h3-max/recast",
        json={
            "input": {
                "prompt": "Speaker in neon-lit studio",
                "face_image_url": face_url,
                "duration": 1.0,
                "fps": 12,
                "resolution": "480P",
                "num_inference_steps": 2,
            }
        },
    )
    assert q_resp.status_code == 200
    req_id = q_resp.json()["request_id"]

    # Poll until completed
    for _ in range(50):
        status_resp = client.get(f"/minimax/h3-max/recast/requests/{req_id}/status?logs=1")
        assert status_resp.status_code == 200
        if status_resp.json()["status"] == "COMPLETED":
            break
        time.sleep(0.05)

    res_resp = client.get(f"/minimax/h3-max/recast/requests/{req_id}")
    assert res_resp.status_code == 200
    vid_data = res_resp.json()
    assert vid_data["video"]["file_name"].endswith(".mp4")
    assert vid_data["strata_engine"]["partition"] == "Ref2VA"
