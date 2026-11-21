import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const runtime = require("../../desktop/runtime.js");

test("desktop package.json defines Electron main, builder targets, and bundled python source", () => {
  const pkg = JSON.parse(fs.readFileSync(new URL("../../desktop/package.json", import.meta.url), "utf8"));
  assert.equal(pkg.main, "main.js");
  assert.ok(pkg.scripts.dist);
  assert.ok(pkg.build.win.target.some(t => t.target === "nsis"));
  assert.ok(pkg.build.mac.target.includes("dmg"));
  assert.ok(pkg.build.linux.target.includes("AppImage"));
  assert.ok(pkg.build.extraResources.some(r => r.to === "openfal-src/openfal"));
});

test("resolveSourceDir returns packaged openfal-src when packaged and repo root in dev", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "openfal-src-"));
  const resPath = path.join(tmp, "resources");
  fs.mkdirSync(path.join(resPath, "openfal-src"), { recursive: true });
  assert.equal(runtime.resolveSourceDir(true, resPath, "/app/desktop"), path.join(resPath, "openfal-src"));
  assert.equal(runtime.resolveSourceDir(false, resPath, path.resolve("desktop")), path.resolve("."));
});

test("loadConfig and saveConfig persist weightsDir and mode cleanly", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "openfal-cfg-"));
  const cfgPath = path.join(tmp, "desktop-config.json");
  const initial = runtime.loadConfig(cfgPath, tmp, path.resolve("."));
  assert.equal(initial.mode, "local");
  assert.ok(initial.weightsDir);

  initial.mode = "remote";
  initial.remoteUrl = "http://127.0.0.1:9000";
  runtime.saveConfig(cfgPath, initial);

  const reloaded = runtime.loadConfig(cfgPath, tmp, path.resolve("."));
  assert.equal(reloaded.mode, "remote");
  assert.equal(reloaded.remoteUrl, "http://127.0.0.1:9000");
});

test("findFreePort returns a usable numeric port", async () => {
  const port = await runtime.findFreePort(0, "127.0.0.1");
  assert.equal(typeof port, "number");
  assert.ok(port > 0 && port < 65536);
});

test("buildServeSpec passes weightsDir, outputDir, cacheDir, and PYTHONPATH when needed", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "openfal-serve-"));
  const config = runtime.getDefaultConfig(tmp, path.resolve("."));
  config.syntheticFallback = true;
  const spec = runtime.buildServeSpec({
    runtime: { executable: "python", argsPrefix: [], needsPythonPath: true },
    config,
    port: 8123,
    sourceDir: path.resolve("."),
  });
  assert.equal(spec.command, "python");
  assert.ok(spec.args.includes("openfal.cli"));
  assert.ok(spec.args.includes("8123"));
  assert.equal(spec.env.OPENFAL_WEIGHTS_DIR, config.weightsDir);
  assert.equal(spec.env.OPENFAL_OUTPUT_DIR, config.outputDir);
  assert.equal(spec.env.OPENFAL_CACHE_DIR, config.cacheDir);
  assert.equal(spec.env.OPENFAL_SYNTHETIC_FALLBACK, "1");
  assert.ok(spec.env.PYTHONPATH.includes(path.resolve(".")));
});

test("detectEnvironment finds Python 3.10+ and openfal in this repo", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "openfal-env-"));
  const cfg = runtime.getDefaultConfig(tmp, path.resolve("."));
  const env = runtime.detectEnvironment(tmp, cfg, path.resolve("."));
  assert.ok(env.availablePythons.length >= 1);
  assert.ok(env.readyRuntime, "Expected readyRuntime to detect openfal + fastapi + uvicorn");
  assert.equal(env.readyRuntime.canServe, true);
});
