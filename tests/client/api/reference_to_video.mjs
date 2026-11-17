/**
 * Multi-Reference Omni Video-to-Video / Reference-to-Video client test:
 *   - `fal-ai/minimax/h3/reference-to-video` (`Ref2VA` 33B partition with `<Picture 1>`, `<Picture 2>`)
 *
 * Usage:
 *   node reference_to_video.mjs
 */
import { downloadOutput, fal, OPENFAL_URL, uploadSampleFile } from "./config.mjs";

const portraitUrl = await uploadSampleFile("sample_portrait.png");
const streetUrl = await uploadSampleFile("sample_street_scene.png");
console.log("Uploaded references:", { portraitUrl, streetUrl });

console.log(`\n=== [Reference-to-Video] fal-ai/minimax/h3/reference-to-video @ ${OPENFAL_URL} ===`);
const result = await fal.subscribe("fal-ai/minimax/h3/reference-to-video", {
  input: {
    prompt:
      "The character in <Picture 1> walks through the sunlit cobblestone street of <Picture 2> while a tabby cat chases a poodle in the background, warm ambient sound",
    reference_image_urls: [portraitUrl, streetUrl],
    resolution: "480P",
    aspect_ratio: "16:9",
    duration: 3.0,
    enable_audio: true,
  },
  logs: true,
  onQueueUpdate(update) {
    for (const log of update.logs || []) {
      console.log(`  [MiniMax-H3-Ref2VA] ${log.message}`);
    }
  },
});

console.log("Response metadata:", {
  seed: result.data.seed,
  timings: result.data.timings,
  strata_engine: result.data.strata_engine,
  video: result.data.video,
});

const saved = await downloadOutput(result.data.video.url, "minimax_h3_ref2va.mp4");
console.log(`Saved -> ${saved}`);
