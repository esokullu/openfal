/**
 * Master runner executing the full openfal Node.js client suite.
 *
 * Usage:
 *   OPENFAL_URL=http://192.168.1.47:8008 node run_all.mjs
 */
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const scripts = [
  "engine_control.mjs",
  "text_to_image.mjs",
  "image_to_image_edit.mjs",
  "text_to_video.mjs",
  "image_to_video.mjs",
  "reference_to_video.mjs",
  "video_recast_animate.mjs",
];

for (const script of scripts) {
  console.log(`\n======================================================================`);
  console.log(`Running ${script}`);
  console.log(`======================================================================`);
  const res = spawnSync(process.execPath, [path.join(__dirname, script)], {
    stdio: "inherit",
    env: process.env,
  });
  if (res.status !== 0) {
    process.exit(res.status ?? 1);
  }
}
