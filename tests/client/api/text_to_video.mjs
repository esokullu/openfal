/**
 * Text-to-Video client test:
 *   - `fal-ai/wan/v2.2-a14b/text-to-video` (Alibaba Wan 2.2 27B-A14B 2-Expert MoE)
 *   - `fal-ai/minimax/h3/text-to-video` (MiniMax-H3 33B FL2VA Joint Video + 32kHz Stereo Audio)
 *
 * Usage:
 *   node text_to_video.mjs                # runs both
 *   node text_to_video.mjs --model=wan
 *   node text_to_video.mjs --model=minimax
 */
import { downloadOutput, fal, OPENFAL_URL, parseModelFlag } from "./config.mjs";

const modelFilter = parseModelFlag("all");

async function runWan22T2V() {
  console.log(`\n=== [Text-to-Video] fal-ai/wan/v2.2-a14b/text-to-video @ ${OPENFAL_URL} ===`);
  const result = await fal.subscribe("fal-ai/wan/v2.2-a14b/text-to-video", {
    input: {
      prompt:
        "A realistic cinematic video of an orange tabby cat running fast behind a fluffy apricot poodle down a sunlit cobblestone street in Istanbul, warm golden hour lighting",
      resolution: "480P",
      aspect_ratio: "16:9",
      duration: 2.0,
      fps: 16,
    },
    logs: true,
    onQueueUpdate(update) {
      for (const log of update.logs || []) {
        console.log(`  [Wan2.2-T2V] ${log.message}`);
      }
    },
  });

  console.log("Response metadata:", {
    seed: result.data.seed,
    timings: result.data.timings,
    strata_engine: result.data.strata_engine,
    video: result.data.video,
  });

  const saved = await downloadOutput(result.data.video.url, "wan22_t2v.mp4");
  console.log(`Saved -> ${saved}`);
}

async function runMiniMaxH3T2V() {
  console.log(`\n=== [Text-to-Video+Audio] fal-ai/minimax/h3/text-to-video @ ${OPENFAL_URL} ===`);
  const result = await fal.subscribe("fal-ai/minimax/h3/text-to-video", {
    input: {
      prompt:
        "A realistic cinematic video of a tabby cat running fast behind a fluffy apricot poodle down a sunlit cobblestone street in Istanbul, historic Ottoman wooden houses and Bosphorus ferry horn in the background, footsteps and playful barking",
      resolution: "480P",
      aspect_ratio: "16:9",
      duration: 3.0,
      enable_audio: true,
    },
    logs: true,
    onQueueUpdate(update) {
      for (const log of update.logs || []) {
        console.log(`  [MiniMax-H3-T2V] ${log.message}`);
      }
    },
  });

  console.log("Response metadata:", {
    seed: result.data.seed,
    timings: result.data.timings,
    strata_engine: result.data.strata_engine,
    video: result.data.video,
  });

  const saved = await downloadOutput(result.data.video.url, "minimax_h3_t2v.mp4");
  console.log(`Saved -> ${saved}`);
}

if (modelFilter === "all" || modelFilter === "wan") {
  await runWan22T2V();
}
if (modelFilter === "all" || modelFilter === "minimax") {
  await runMiniMaxH3T2V();
}
