/**
 * Video Character Animation, Face Recast, Relight & Insert-Video client test:
 *   - `fal-ai/wan/v2.2/recast` (`/wan/v2.2/animate` — Wan2.2-Animate-14B)
 *   - `fal-ai/minimax/h3-max/recast` (`Ref2VA` partition)
 *   - `fal-ai/minimax/h3-max/relight` (`Ref2VA` partition)
 *
 * Usage:
 *   node video_recast_animate.mjs                # runs both Wan 2.2 Animate and MiniMax-H3 Recast
 *   node video_recast_animate.mjs --model=wan
 *   node video_recast_animate.mjs --model=minimax
 */
import fs from "node:fs";
import path from "node:path";
import { downloadOutput, fal, OPENFAL_URL, OUTPUTS_DIR, PORTRAIT_SAMPLE, STYLE_SAMPLE, parseModelFlag, uploadSampleFile } from "./config.mjs";

const modelFilter = parseModelFlag("all");
const portraitUrl = await uploadSampleFile(PORTRAIT_SAMPLE);
const styleUrl = await uploadSampleFile(STYLE_SAMPLE);

async function runWanAnimate() {
  console.log(`\n=== [Wan 2.2 Animate / Recast] fal-ai/wan/v2.2/recast @ ${OPENFAL_URL} ===`);
  const result = await fal.subscribe("fal-ai/wan/v2.2/recast", {
    input: {
      prompt:
        "Natural expressive portrait animation, subtle head turn, warm studio lighting, high detail",
      face_image_url: portraitUrl,
      start_image_url: portraitUrl,
      resolution: "480P",
      aspect_ratio: "1:1",
      duration: 2.0,
      fps: 16,
    },
    logs: true,
    onQueueUpdate(update) {
      for (const log of update.logs || []) {
        console.log(`  [Wan2.2-Animate] ${log.message}`);
      }
    },
  });

  console.log("Response metadata:", {
    seed: result.data.seed,
    timings: result.data.timings,
    strata_engine: result.data.strata_engine,
    video: result.data.video,
  });

  const saved = await downloadOutput(result.data.video.url, "wan22_animate_recast.mp4");
  console.log(`Saved -> ${saved}`);
}

async function runMiniMaxRecastAndRelight() {
  const drivingVideoPath = path.join(OUTPUTS_DIR, "minimax_h3_t2v.mp4");
  if (!fs.existsSync(drivingVideoPath)) {
    throw new Error("MiniMax recast needs a source video. Run node text_to_video.mjs --model=minimax first.");
  }
  const drivingVideoUrl = await uploadSampleFile(drivingVideoPath);
  console.log(`\n=== [MiniMax-H3 Recast] fal-ai/minimax/h3-max/recast @ ${OPENFAL_URL} ===`);
  const recastRes = await fal.subscribe("fal-ai/minimax/h3-max/recast", {
    input: {
      prompt:
        "The person in <Picture 1> walks through the cobblestone street shown in <Video 1>, with warm cinematic lighting inspired by <Picture 2>, natural footsteps and ambient street sounds",
      face_image_url: portraitUrl,
      target_video_url: drivingVideoUrl,
      reference_image_urls: [portraitUrl, styleUrl],
      resolution: "480P",
      aspect_ratio: "16:9",
      duration: 3.0,
      enable_audio: true,
      seed: 182904,
    },
    logs: true,
    onQueueUpdate(update) {
      for (const log of update.logs || []) {
        console.log(`  [MiniMax-Recast] ${log.message}`);
      }
    },
  });

  const savedRecast = await downloadOutput(recastRes.data.video.url, "minimax_h3_recast.mp4");
  console.log(`Saved -> ${savedRecast}`);
}

if (modelFilter === "all" || modelFilter === "wan") {
  await runWanAnimate();
}
if (modelFilter === "all" || modelFilter === "minimax") {
  await runMiniMaxRecastAndRelight();
}
