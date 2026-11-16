"""
Example: Using the openfal Python SDK (fal_client drop-in compatible) for:
1. Text-to-Video (`minimax/h3-max/text-to-video`)
2. Image-to-Video (`minimax/h3-max/image-to-video`)
3. Video Face Swap / Recast (`minimax/h3-max/recast`)
4. Text-to-Image (`minimax/h3/text-to-image`)
"""

import openfal


def main() -> None:
    base_url = "http://localhost:8000"

    print("=== 1. Text-to-Video (FL2VA + Stereo Audio) ===")
    vid_res = openfal.subscribe(
        "minimax/h3-max/text-to-video",
        arguments={
            "prompt": "Cyberpunk street market in rain, neon reflections, ambient rain stereo sound",
            "duration": 2.0,
            "resolution": "768P",
            "aspect_ratio": "16:9",
            "num_inference_steps": 4,
        },
        on_queue_update=lambda st: [print(" ", l["message"]) for l in st.get("logs", [])],
        base_url=base_url,
    )
    print("Generated Video:", vid_res["video"]["url"])

    print("\n=== 2. Text-to-Image (H3-Image) ===")
    img_res = openfal.run(
        "minimax/h3/text-to-image",
        arguments={
            "prompt": "Studio portrait of an astronaut helmet reflecting a binary sunset",
            "resolution": "2K",
            "aspect_ratio": "1:1",
            "num_inference_steps": 4,
        },
        base_url=base_url,
    )
    first_img_url = img_res["images"][0]["url"]
    print("Generated Image:", first_img_url)

    print("\n=== 3. Video Face Swap / Recast (Ref2VA) ===")
    swap_res = openfal.run(
        "minimax/h3-max/recast",
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


if __name__ == "__main__":
    main()
