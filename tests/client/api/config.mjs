/**
 * Shared fal.ai client & helper configuration for openfal.
 *
 * Works both with `@fal-ai/client` (when `npm install` has been run) AND zero-dependency
 * out-of-the-box with Node 18+ built-in `fetch` + SSE streaming.
 *
 * Set `OPENFAL_URL` env var (defaults to `http://127.0.0.1:8000`).
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const OPENFAL_URL = (process.env.OPENFAL_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
export const SAMPLES_DIR = path.resolve(__dirname, "../samples");
export const OUTPUTS_DIR = path.resolve(__dirname, "outputs");
fs.mkdirSync(OUTPUTS_DIR, { recursive: true });

/**
 * Rewrite `https://fal.run/...` or `https://queue.fal.run/...` URLs from `@fal-ai/client`
 * to point directly at `OPENFAL_URL`.
 */
export function openfalFetch(input, init) {
  let urlStr = typeof input === "string" ? input : input.url || String(input);
  urlStr = urlStr
    .replace(/^https?:\/\/queue\.fal\.run\//, `${OPENFAL_URL}/queue/`)
    .replace(/^https?:\/\/fal\.run\//, `${OPENFAL_URL}/`);
  return fetch(urlStr, init);
}

/**
 * Upload a local sample image or video to `${OPENFAL_URL}/v1/storage/upload`
 * (compatible with `fal.storage.upload`).
 */
export async function uploadSampleFile(fileNameOrPath) {
  const fullPath = path.isAbsolute(fileNameOrPath)
    ? fileNameOrPath
    : path.join(SAMPLES_DIR, fileNameOrPath);
  const bytes = fs.readFileSync(fullPath);
  const ext = path.extname(fullPath).toLowerCase();
  const contentType =
    ext === ".mp4"
      ? "video/mp4"
      : ext === ".jpg" || ext === ".jpeg"
        ? "image/jpeg"
        : "image/png";

  const res = await fetch(`${OPENFAL_URL}/v1/storage/upload`, {
    method: "POST",
    headers: { "Content-Type": contentType },
    body: bytes,
  });
  if (!res.ok) {
    throw new Error(`Upload failed (${res.status}): ${await res.text()}`);
  }
  const data = await res.json();
  return data.url;
}

/**
 * Download a generated image or video URL to `tests/client/api/outputs/<filename>`.
 */
export async function downloadOutput(url, outFileName) {
  const dest = path.join(OUTPUTS_DIR, outFileName);
  if (url.startsWith("data:")) {
    const b64 = url.split(",")[1];
    fs.writeFileSync(dest, Buffer.from(b64, "base64"));
    return dest;
  }
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to download ${url}: HTTP ${res.status}`);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  fs.writeFileSync(dest, buf);
  return dest;
}

/**
 * Drop-in `fal` client supporting `fal.run`, `fal.subscribe`, `fal.queue.*`, and `fal.storage.upload`.
 * Automatically binds `@fal-ai/client` custom fetch if installed.
 */
async function createFalClient() {
  try {
    const mod = await import("@fal-ai/client");
    if (mod?.fal?.config) {
      mod.fal.config({
        credentials: process.env.FAL_KEY || "openfal-local-key",
        fetch: openfalFetch,
      });
    }
  } catch {
    // @fal-ai/client not installed in node_modules; built-in fal client below handles all calls
  }

  return {
    async run(endpointId, { input = {} } = {}) {
      const cleanEp = endpointId.replace(/^\/+/, "");
      const res = await fetch(`${OPENFAL_URL}/${cleanEp}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input }),
      });
      if (!res.ok) {
        throw new Error(`fal.run(${endpointId}) failed (${res.status}): ${await res.text()}`);
      }
      const data = await res.json();
      return { data, requestId: data.seed ? String(data.seed) : "sync" };
    },

    async subscribe(endpointId, { input = {}, logs = true, onQueueUpdate } = {}) {
      const cleanEp = endpointId.replace(/^\/+/, "");
      const submitRes = await fetch(`${OPENFAL_URL}/queue/${cleanEp}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input }),
      });
      if (!submitRes.ok) {
        throw new Error(
          `fal.subscribe(${endpointId}) submit failed (${submitRes.status}): ${await submitRes.text()}`
        );
      }
      const initial = await submitRes.json();
      const requestId = initial.request_id;
      if (onQueueUpdate) onQueueUpdate(initial);

      let seenLogs = (initial.logs || []).length;
      while (true) {
        const statusRes = await fetch(
          `${OPENFAL_URL}/requests/${requestId}/status?logs=${logs ? 1 : 0}`
        );
        if (!statusRes.ok) {
          throw new Error(`Status poll failed (${statusRes.status}): ${await statusRes.text()}`);
        }
        const status = await statusRes.json();
        const allLogs = status.logs || [];
        const newLogs = allLogs.slice(seenLogs);
        seenLogs = allLogs.length;
        if (onQueueUpdate && (newLogs.length > 0 || status.status !== "IN_PROGRESS")) {
          onQueueUpdate({ ...status, logs: newLogs });
        }
        if (status.status === "COMPLETED") {
          break;
        }
        if (status.status === "FAILED" || status.status === "CANCELLED") {
          throw new Error(`Job ${requestId} ${status.status}: ${status.error || "Unknown error"}`);
        }
        await new Promise((r) => setTimeout(r, 500));
      }

      const resultRes = await fetch(`${OPENFAL_URL}/requests/${requestId}`);
      if (!resultRes.ok) {
        throw new Error(`Result fetch failed (${resultRes.status}): ${await resultRes.text()}`);
      }
      const data = await resultRes.json();
      return { data, requestId };
    },

    storage: {
      upload: uploadSampleFile,
    },
  };
}

export const fal = await createFalClient();

export function parseModelFlag(defaultModel = "all") {
  const arg = process.argv.find((a) => a.startsWith("--model="));
  return arg ? arg.split("=")[1].toLowerCase() : defaultModel;
}
