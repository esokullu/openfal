/**
 * Image-to-Video client test:
 *   - `fal-ai/wan/v2.2-a14b/image-to-video` (Alibaba Wan 2.2 27B-A14B 2-Expert MoE I2V)
 *   - `fal-ai/minimax/h3/image-to-video` (MiniMax-H3 33B FL2VA Keyframe I2V + Stereo Audio)
 *
 * Usage:
 *   node image_to_video.mjs                # runs both
 *   node image_to_video.mjs --model=wan
 *   node image_to_video.mjs --model=minimax
 */
import { downloadOutput, fal, OPENFAL_URL, STREET_SAMPLE, parseModelFlag, uploadSampleFile } from "./config.mjs";

const modelFilter = parseModelFlag("all");
const startImageUrl = await uploadSampleFile(STREET_SAMPLE);
console.log("Uploaded start keyframe:", startImageUrl);

async function runWan22I2V() {
  console.log(`\n=== [Image-to-Video] fal-ai/wan/v2.2-a14b/image-to-video @ ${OPENFAL_URL} ===`);
  const result = await fal.subscribe("fal-ai/wan/v2.2-a14b/image-to-video", {
    input: {
      prompt:
        "Camera glides forward down the sunlit cobblestone street as a cat chases a playful poodle toward the golden sunset",
      start_image_url: startImageUrl,
      resolution: "480P",
      aspect_ratio: "16:9",
      duration: 2.0,
      fps: 16,
    },
    logs: true,
    onQueueUpdate(update) {
      for (const log of update.logs || []) {
        console.log(`  [Wan2.2-I2V] ${log.message}`);
      }
    },
  });

  console.log("Response metadata:", {
    seed: result.data.seed,
    timings: result.data.timings,
    strata_engine: result.data.strata_engine,
    video: result.data.video,
  });

  const saved = await downloadOutput(result.data.video.url, "wan22_i2v.mp4");
  console.log(`Saved -> ${saved}`);
}

async function runMiniMaxH3I2V() {
  console.log(`\n=== [Image-to-Video+Audio] fal-ai/minimax/h3/image-to-video @ ${OPENFAL_URL} ===`);
  const result = await fal.subscribe("fal-ai/minimax/h3/image-to-video", {
    input: {
      prompt:
        "A tabby cat sprints after a playful apricot poodle along the cobblestone street at sunset, lively street ambiance and paw steps",
      start_image_url: startImageUrl,
      resolution: "480P",
      aspect_ratio: "16:9",
      duration: 3.0,
      enable_audio: true,
    },
    logs: true,
    onQueueUpdate(update) {
      for (const log of update.logs || []) {
        console.log(`  [MiniMax-H3-I2V] ${log.message}`);
      }
    },
  });

  console.log("Response metadata:", {
    seed: result.data.seed,
    timings: result.data.timings,
    strata_engine: result.data.strata_engine,
    video: result.data.video,
  });

  const saved = await downloadOutput(result.data.video.url, "minimax_h3_i2v.mp4");
  console.log(`Saved -> ${saved}`);
}

if (modelFilter === "all" || modelFilter === "wan") {
  await runWan22I2V();
}
if (modelFilter === "all" || modelFilter === "minimax") {
  await runMiniMaxH3I2V();
}
