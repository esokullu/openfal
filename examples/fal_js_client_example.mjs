/**
 * Example: Using official `@fal-ai/client` against a local `openfal` server.
 *
 * Primary Fast Stack: Alibaba Qwen-Image-2.1-Turbo (7B) + Wan 2.2 MoE (27B-A14B)
 * Optional Omni Stack: MiniMax-H3 (33B)
 *
 * Run:
 *   npm install @fal-ai/client
 *   node examples/fal_js_client_example.mjs
 */

import { fal } from "@fal-ai/client";

// Point @fal-ai/client at your local openfal server
fal.config({
  proxyUrl: process.env.OPENFAL_URL || "http://localhost:8000",
});

async function main() {
  console.log("1. [Alibaba Qwen-Image-2.1-Turbo] 7B 8-Step Text-to-Image...");
  const t2i = await fal.subscribe("qwen/image-2.1/text-to-image", {
    input: {
      prompt: 'Neon sign reading "OPENFAL 2.1" in a rainy Tokyo alley, 8k photography',
      resolution: "2K",
      aspect_ratio: "16:9",
      num_inference_steps: 8,
    },
  });
  console.log("Image URL:", t2i.data?.images?.[0]?.url || t2i.images?.[0]?.url);

  console.log("\n2. [Alibaba Wan 2.2 MoE] 27B-A14B Fast Text-to-Video...");
  const t2v = await fal.subscribe("wan/v2.2/text-to-video", {
    input: {
      prompt: "Cinematic drone shot flying through a bioluminescent canyon at dawn, 4k.",
      duration: 3,
      resolution: "768P",
      aspect_ratio: "16:9",
      num_inference_steps: 8,
    },
    logs: true,
    onQueueUpdate: (update) => {
      if (update.status === "IN_PROGRESS" && update.logs) {
        update.logs.forEach((l) => console.log("  [Wan2.2-MoE]", l.message));
      }
    },
  });
  console.log("Video URL:", t2v.data?.video?.url || t2v.video?.url);

  console.log("\n3. [Alibaba Wan 2.2 Animate 14B] Fast Video Face Swap / Recast...");
  const recast = await fal.subscribe("wan/v2.2/animate", {
    input: {
      prompt: "Portrait shot of a speaker in warm studio lighting, natural facial expressions.",
      face_image_url: "https://example.com/reference_face.png",
      duration: 3,
      resolution: "768P",
    },
  });
  console.log("Face-Swapped Video URL:", recast.data?.video?.url || recast.video?.url);
}

main().catch(console.error);
