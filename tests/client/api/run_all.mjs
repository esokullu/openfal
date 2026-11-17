/**
 * Master runner executing the full openfal Node.js client suite.
 *
 * Usage:
 *   OPENFAL_URL=http://127.0.0.1:8000 node run_all.mjs
 */
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const serverUrl = (process.env.OPENFAL_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const startupDeadline = Date.now() + 60000;
while (true) {
  try {
    const health = await fetch(`${serverUrl}/health`, { signal: AbortSignal.timeout(3000) });
    if (health.ok) break;
  } catch { /* The freshly launched server may still be importing dependencies. */ }
  if (Date.now() >= startupDeadline) throw new Error(`Server not ready at ${serverUrl} after 60s`);
  await new Promise((resolve) => setTimeout(resolve, 1000));
}

const scripts = [
  ["engine_control.mjs"],
  ["text_to_image.mjs", "--model=qwen"],
  ["text_to_image.mjs", "--model=minimax"],
  ["image_to_image_edit.mjs", "--model=qwen"],
  ["image_to_image_edit.mjs", "--model=minimax"],
  ["text_to_video.mjs", "--model=wan"],
  ["text_to_video.mjs", "--model=minimax"],
  ["image_to_video.mjs", "--model=wan"],
  ["image_to_video.mjs", "--model=minimax"],
  ["reference_to_video.mjs"],
  ["video_recast_animate.mjs", "--model=wan"],
  ["video_recast_animate.mjs", "--model=minimax"],
];

const results = [];
const outputsRoot = path.join(__dirname, "outputs");
const runDir = process.env.OPENFAL_OUTPUTS_DIR || path.join(outputsRoot, `run-${new Date().toISOString().replaceAll(":", "-")}`);
fs.mkdirSync(runDir, { recursive: true });
console.log(`Artifacts for this run: ${runDir}`);
for (const [script, ...args] of scripts) {
  console.log(`\n======================================================================`);
  console.log(`Running ${[script, ...args].join(" ")}`);
  console.log(`======================================================================`);
  const started = Date.now();
  const res = spawnSync(process.execPath, [path.join(__dirname, script), ...args], {
    stdio: "inherit",
    env: { ...process.env, OPENFAL_OUTPUTS_DIR: runDir },
  });
  results.push({ script, args, passed: res.status === 0, exitCode: res.status,
    seconds: (Date.now() - started) / 1000, error: res.error?.message });
}
fs.mkdirSync(outputsRoot, { recursive: true });
const report = JSON.stringify({ artifacts: runDir, results }, null, 2);
fs.writeFileSync(path.join(runDir, "suite_results.json"), report);
fs.writeFileSync(path.join(outputsRoot, "suite_results.json"), report);
console.table(results.map(({ script, args, passed, seconds }) => ({ test: [script, ...args].join(" "), passed, seconds })));
process.exitCode = results.every((result) => result.passed) ? 0 : 1;
