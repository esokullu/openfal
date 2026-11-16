"""
Example: Using the openfal Python SDK (fal_client drop-in compatible) starting with the
Primary Alibaba Stack (`Qwen-Image-2.1-Turbo` + `Wan 2.2 MoE`), followed by Optional `MiniMax-H3`:

1. Text-to-Image (`qwen/image-2.1/text-to-image` — 7B 8-Step Turbo, 100% in VRAM)
2. Fast MoE Text-to-Video (`wan/v2.2/text-to-video` — 27B-A14B Two-Expert MoE)
3. Fast Video Face Swap (`wan/v2.2/animate` — 14B Character & Face Swap)
4. Optional Omni Stereo-Audio Video (`minimax/h3-max/text-to-video` — 33B FL2VA)
"""

import openfal


def main() -> None:
    base_url = "http://localhost:8000"

    print("=== 1. [Alibaba Qwen-Image-2.1-Turbo] 7B Fast Text-to-Image ===")
    img_res = openfal.run(
        "qwen/image-2.1/text-to-image",
        arguments={
            "prompt": 'Studio portrait of an astronaut helmet reflecting a neon sign "OPENFAL"',
            "resolution": "2K",
            "aspect_ratio": "1:1",
            "num_inference_steps": 6,
        },
        base_url=base_url,
    )
    first_img_url = img_res["images"][0]["url"]
    print("Generated Image:", first_img_url)

    print("\n=== 2. [Alibaba Wan 2.2 MoE] 27B-A14B Fast Text-to-Video ===")
    vid_res = openfal.subscribe(
        "wan/v2.2/text-to-video",
        arguments={
            "prompt": "Cyberpunk street market in rain, neon reflections, sweeping camera motion",
            "duration": 2.0,
            "resolution": "768P",
            "aspect_ratio": "16:9",
            "num_inference_steps": 4,
        },
        on_queue_update=lambda st: [print(" ", l["message"]) for l in st.get("logs", [])],
        base_url=base_url,
    )
    print("Generated Wan 2.2 Video:", vid_res["video"]["url"])

    print("\n=== 3. [Alibaba Wan 2.2 Animate 14B] Fast Video Face Swap ===")
    swap_res = openfal.run(
        "wan/v2.2/animate",
        arguments={
            "prompt": "Cinematic close-up interview shot in studio lighting",
            "face_image_url": first_img_url,
            "target_video_url": vid_res["video"]["url"],
            "duration": 2.0,
            "resolution": "768P",
            "num_inference_steps": 4,
        },
        base_url=base_url,
    )
    print("Face-Swapped Video:", swap_res["video"]["url"])

    print("\n=== 4. [Optional MiniMax-H3] 33B Video + Native Stereo Audio ===")
    mm_res = openfal.run(
        "minimax/h3-max/text-to-video",
        arguments={
            "prompt": "Ocean waves crashing against volcanic cliffs at dusk, stereo ocean thunder",
            "duration": 2.0,
            "resolution": "768P",
            "num_inference_steps": 4,
        },
        base_url=base_url,
    )
    print("MiniMax-H3 Stereo Video:", mm_res["video"]["url"])


if __name__ == "__main__":
    main()
