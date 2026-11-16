/**
 * Example: Using official `@fal-ai/client` against a local `openfal` server.
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
  console.log("1. Running Text-to-Video (minimax/h3-max/text-to-video)...");
  const t2v = await fal.subscribe("minimax/h3-max/text-to-video", {
    input: {
      prompt: "Cinematic drone shot rolling over a misty mountain ridge at dawn, 4k, wind howling.",
      duration: 3,
      resolution: "768P",
      aspect_ratio: "16:9",
      prompt_expansion_mode: "balanced",
    },
    logs: true,
    onQueueUpdate: (update) => {
      if (update.status === "IN_PROGRESS" && update.logs) {
        update.logs.forEach((l) => console.log("  [log]", l.message));
      }
    },
  });
  console.log("Video URL:", t2v.data?.video?.url || t2v.video?.url);

  console.log("\n2. Running Video Face Swap / Recast (minimax/h3-max/recast)...");
  const recast = await fal.subscribe("minimax/h3-max/recast", {
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
