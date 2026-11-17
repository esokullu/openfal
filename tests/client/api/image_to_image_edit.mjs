/**
 * Image-to-Image & Multi-Reference Portrait Edit client test
 * (`fal-ai/qwen-image-2.1/edit` & `fal-ai/minimax/h3/image-to-image`).
 *
 * Usage:
 *   node image_to_image_edit.mjs                # runs both
 *   node image_to_image_edit.mjs --model=qwen
 *   node image_to_image_edit.mjs --model=minimax
 */
import { downloadOutput, fal, OPENFAL_URL, PORTRAIT_SAMPLE, STYLE_SAMPLE, parseModelFlag, uploadSampleFile } from "./config.mjs";

const modelFilter = parseModelFlag("all");

const portraitUrl = await uploadSampleFile(PORTRAIT_SAMPLE);
const styleUrl = await uploadSampleFile(STYLE_SAMPLE);
console.log("Uploaded sample references:", { portraitUrl, styleUrl });

async function runQwenEdit() {
  console.log(`\n=== [Image Edit] fal-ai/qwen-image-2.1/edit @ ${OPENFAL_URL} ===`);
  const result = await fal.subscribe("fal-ai/qwen-image-2.1/edit", {
    input: {
      prompt:
        "Transform the person in the first image into a photorealistic neon cyberpunk detective on a rainy Istanbul street, using only the cyan and magenta lighting palette of the second reference image. Preserve the person's facial identity, hair, freckles, age and natural skin texture. Dress them in a dark detective coat.",
      reference_image_urls: [portraitUrl, styleUrl],
      resolution: "768P",
      aspect_ratio: "1:1",
    },
    logs: true,
    onQueueUpdate(update) {
      for (const log of update.logs || []) {
        console.log(`  [Qwen-Edit] ${log.message}`);
      }
    },
  });

  console.log("Response metadata:", {
    seed: result.data.seed,
    timings: result.data.timings,
    strata_engine: result.data.strata_engine,
    image: result.data.images?.[0],
  });

  const saved = await downloadOutput(result.data.images[0].url, "qwen_image21_edit.png");
  console.log(`Saved -> ${saved}`);
}

async function runMiniMaxH3Edit() {
  console.log(`\n=== [Image-to-Image] fal-ai/minimax/h3/image-to-image @ ${OPENFAL_URL} ===`);
  const result = await fal.subscribe("fal-ai/minimax/h3/image-to-image", {
    input: {
      prompt:
        "Restyle the person in <Picture 1> as a cinematic oil painting with dramatic golden rim lighting. Preserve their facial identity, hairstyle, age and expression, with visible painted brushwork.",
      reference_image_urls: [portraitUrl],
      resolution: "768P",
      aspect_ratio: "1:1",
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

  const saved = await downloadOutput(result.data.images[0].url, "minimax_h3_i2i.png");
  console.log(`Saved -> ${saved}`);
}

if (modelFilter === "all" || modelFilter === "qwen") {
  await runQwenEdit();
}
if (modelFilter === "all" || modelFilter === "minimax") {
  await runMiniMaxH3Edit();
}
