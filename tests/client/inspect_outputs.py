"""Decode client artifacts and compare their actual media properties to API metadata.

Usage: python tests/client/inspect_outputs.py tests/client/api/outputs/run-...
Requires the project's inference dependencies. Visual quality still needs review.
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_directory", type=Path)
args = parser.parse_args()
root = args.run_directory.resolve(strict=True)
names = {
    "fal-ai/qwen-image-2.1/turbo": "qwen_image21_t2i.png",
    "fal-ai/qwen-image-2.1/edit": "qwen_image21_edit.png",
    "fal-ai/minimax/h3/text-to-image": "minimax_h3_t2i.png",
    "fal-ai/minimax/h3/image-to-image": "minimax_h3_i2i.png",
    "fal-ai/wan/v2.2-a14b/text-to-video": "wan22_t2v.mp4",
    "fal-ai/wan/v2.2-a14b/image-to-video": "wan22_i2v.mp4",
    "fal-ai/minimax/h3/text-to-video": "minimax_h3_t2v.mp4",
    "fal-ai/minimax/h3/image-to-video": "minimax_h3_i2v.mp4",
    "fal-ai/minimax/h3/reference-to-video": "minimax_h3_ref2va.mp4",
    "fal-ai/minimax/h3-max/recast": "minimax_h3_recast.mp4",
}
review = root / "review"
review.mkdir(exist_ok=True)
checks = []
for manifest in root.glob("fal-ai_*.json"):
    record = json.loads(manifest.read_text())
    endpoint, result = record["endpoint"], record["result"]
    path = root / names[endpoint]
    assert path.is_file(), f"Missing artifact: {path}"
    assert result["strata_engine"]["mode"] == "real_pytorch", manifest
    metadata = result["images"][0] if "images" in result else result["video"]
    assert path.stat().st_size == metadata["file_size"], path
    item = {"endpoint": endpoint, "artifact": str(path), "seed": result["seed"]}
    if path.suffix == ".png":
        with Image.open(path) as image:
            image.load()
            assert image.size == (metadata["width"], metadata["height"]), path
            item.update(width=image.width, height=image.height, decodable=True)
    else:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            targets = {0, stream.frames // 2, stream.frames - 1}
            count = 0
            for index, frame in enumerate(container.decode(video=0)):
                count += 1
                if index in targets:
                    frame.to_image().save(review / f"{path.stem}_{index}.png")
            assert (stream.width, stream.height) == (metadata["width"], metadata["height"]), path
            assert count == metadata["num_frames"], (path, count, metadata)
            assert float(stream.average_rate) == metadata["fps"], path
            duration = float(stream.duration * stream.time_base)
            assert abs(duration - metadata["duration"]) < 1 / float(stream.average_rate), path
            item.update(frames=count, width=stream.width, height=stream.height,
                        duration=duration, fps=float(stream.average_rate), decodable=True)
        with av.open(str(path)) as container:
            has_audio = bool(container.streams.audio)
            assert has_audio == metadata["stereo_audio"], path
            if has_audio:
                chunks = [frame.to_ndarray() for frame in container.decode(audio=0)]
                wave = np.concatenate(chunks, axis=1)
                assert wave.shape[0] == 2 and np.isfinite(wave).all(), path
                rms = float(np.sqrt(np.mean(wave ** 2)))
                assert rms > 1e-6, f"Silent audio: {path}"
                item.update(audio_channels=wave.shape[0], audio_rms=rms)
    checks.append(item)
assert checks, f"No successful inference manifests in {root}"
(root / "media_checks.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
