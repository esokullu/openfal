/**
 * Text-to-Image client test (`fal-ai/qwen-image-2.1/turbo` & `fal-ai/minimax/h3/text-to-image`).
 *
 * Usage:
 *   node text_to_image.mjs                # runs both Qwen-Image-2.1-Turbo and MiniMax-H3-Image
 *   node text_to_image.mjs --model=qwen
 *   node text_to_image.mjs --model=minimax
 */
import { downloadOutput, fal, OPENFAL_URL, parseModelFlag } from "./config.mjs";

const modelFilter = parseModelFlag("all");

async function runQwenT2I() {
  console.log(`\n=== [Text-to-Image] fal-ai/qwen-image-2.1/turbo @ ${OPENFAL_URL} ===`);
  const result = await fal.subscribe("fal-ai/qwen-image-2.1/turbo", {
    input: {
      prompt:
        "A realistic photo of a tabby cat running after a fluffy apricot poodle in sunlit Istanbul cobblestone streets, Galata Tower in the distance",
      resolution: "768P",
      aspect_ratio: "16:9",
    },
    logs: true,
    onQueueUpdate(update) {
      for (const log of update.logs || []) {
        console.log(`  [Qwen-Image-2.1] ${log.message}`);
      }
    },
  });

  console.log("Response metadata:", {
    seed: result.data.seed,
    timings: result.data.timings,
    strata_engine: result.data.strata_engine,
    image: result.data.images?.[0],
  });

  const saved = await downloadOutput(result.data.images[0].url, "qwen_image21_t2i.png");
  console.log(`Saved -> ${saved}`);
}

async function runMiniMaxH3T2I() {
  console.log(`\n=== [Text-to-Image] fal-ai/minimax/h3/text-to-image @ ${OPENFAL_URL} ===`);
  const result = await fal.subscribe("fal-ai/minimax/h3/text-to-image", {
    input: {
      prompt:
        "A photorealistic wide shot of an orange tabby cat chasing a white poodle across an Istanbul cobblestone alley at sunset, Bosphorus in background",
      resolution: "768P",
      aspect_ratio: "16:9",
    },
    logs: true,
    onQueueUpdate(update) {
      for (const log of update.logs || []) {
        console.log(`  [MiniMax-H3-Image] ${log.message}`);
      }
    },
  });

  console.log("Response metadata:", {
    seed: result.data.seed,
    timings: result.data.timings,
    strata_engine: result.data.strata_engine,
    image: result.data.images?.[0],
  });

  const saved = await downloadOutput(result.data.images[0].url, "minimax_h3_t2i.png");
  console.log(`Saved -> ${saved}`);
}

if (modelFilter === "all" || modelFilter === "qwen") {
  await runQwenT2I();
}
if (modelFilter === "all" || modelFilter === "minimax") {
  await runMiniMaxH3T2I();
}
