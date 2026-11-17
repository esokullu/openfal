import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("../../openfal/ui/index.html", import.meta.url), "utf8");
const script = html.split("<script>")[1].split("</script>")[0];

function studio(responses) {
  const state = { refreshes: 0, calls: 0, requests: [] };
  const elements = { hfTokenInput: { value: "" }, logBox: { textContent: "", scrollHeight: 0 } };
  const context = vm.createContext({
    document: {
      getElementById: id => elements[id] ||= { value: "", textContent: "", style: {} },
      querySelectorAll: () => [],
    },
    window: { addEventListener() {}, location: { origin: "http://localhost:8008" } },
    setTimeout: callback => callback(),
    fetch: async (url, options) => {
      state.calls++;
      state.requests.push({ url, options });
      const response = responses.shift();
      assert.ok(response, "Unexpected request after installation reached a terminal state");
      if (response instanceof Error) throw response;
      return { ok: true, json: async () => response };
    },
  });
  vm.runInContext(script, context);
  context.refreshModelHub = async () => { state.refreshes++; };
  return { context, state, log: elements.logBox, elements };
}

test("Studio reports a background metadata failure and refreshes readiness", async () => {
  const task = { task_id: "install-1", status: "RUNNING", logs: ["starting"] };
  const { context, state, log } = studio([
    { task },
    { active_tasks: [{ ...task, logs: ["starting", "fetching metadata"] }] },
    { active_tasks: [{ ...task, status: "FAILED", error: "Missing tokenizer", logs: ["starting", "fetching metadata"] }] },
  ]);
  await context.triggerModelInstall("qwen_image_21", "runtime_assets");
  assert.match(log.textContent, /\[ERROR\] Missing tokenizer/);
  assert.doesNotMatch(log.textContent, /Completed 'qwen_image_21'/);
  assert.equal(log.textContent.match(/fetching metadata/g).length, 1);
  assert.equal(state.calls, 3);
  assert.equal(state.refreshes, 1);
});

test("Studio waits for successful completion before reporting it", async () => {
  const task = { task_id: "install-2", status: "RUNNING", logs: [] };
  const { context, state, log } = studio([
    { task }, { active_tasks: [{ ...task, status: "COMPLETED", logs: ["assets verified"] }] },
  ]);
  await context.triggerModelInstall("qwen_image_21", "runtime_assets");
  assert.match(log.textContent, /assets verified/);
  assert.match(log.textContent, /Completed 'qwen_image_21'/);
  assert.equal(state.calls, 2);
  assert.equal(state.refreshes, 1);
});

test("Studio surfaces failed status polling instead of reporting success", async () => {
  const { context, state, log } = studio([
    { task: { task_id: "install-3", status: "RUNNING", logs: [] } }, new Error("Connection lost"),
  ]);
  await context.triggerModelInstall("qwen_image_21", "runtime_assets");
  assert.match(log.textContent, /\[ERROR\] Connection lost/);
  assert.doesNotMatch(log.textContent, /Completed 'qwen_image_21'/);
  assert.equal(state.refreshes, 1);
});

test("Recast requires a driving video before submitting generation", async () => {
  const { context, state, log } = studio([]);
  vm.runInContext('currentEndpoint = "minimax/h3-max/recast";', context);
  context.refreshTelemetry = async () => {};
  await context.runInference();
  assert.match(log.textContent, /Recast requires a driving video URL/);
  assert.equal(state.calls, 0);
});

for (const endpoint of ["wan/v2.2/animate", "wan/v2.2/recast"]) {
  test(`${endpoint} is selectable and sends one identity plus its driving video`, () => {
    const { context } = studio([]);
    const tag = html.match(new RegExp(`<button[^>]*data-ep="${endpoint}"[^>]*>`))[0];
    assert.doesNotMatch(tag, /disabled/);
    context.selectTask({
      classList: { add() {} },
      getAttribute: attr => attr === "data-ep" ? endpoint : "Wan2.2-Animate-14B",
    });
    const element = id => context.document.getElementById(id);
    assert.equal(element("refImageUrl").required, true);
    assert.equal(element("targetVideoUrl").required, true);
    assert.equal(element("targetVideoGroup").style.display, "block");
    element("refImageUrl").value = "/files/identity.png";
    element("targetVideoUrl").value = "/files/driver.mp4";
    const payload = context.buildPayload();
    assert.equal(payload.face_image_url, "/files/identity.png");
    assert.equal(payload.driving_video_url, "/files/driver.mp4");
    assert.equal(payload.reference_image_urls, undefined);
    assert.equal(payload.start_image_url, undefined);
    assert.equal(payload.target_video_url, undefined);
    assert.equal(payload.num_inference_steps, undefined);
    assert.equal(payload.guidance_scale, undefined);
  });

  for (const missing of ["refImageUrl", "targetVideoUrl"]) {
    test(`${endpoint} rejects a missing ${missing} before network submission`, async () => {
      const { context, state, log } = studio([]);
      vm.runInContext(`currentEndpoint = "${endpoint}";`, context);
      context.document.getElementById("refImageUrl").value = "/files/identity.png";
      context.document.getElementById("targetVideoUrl").value = "/files/driver.mp4";
      context.document.getElementById(missing).value = "";
      context.refreshTelemetry = async () => {};
      await context.runInference();
      assert.match(log.textContent, /requires an identity image and a driving video URL/);
      assert.equal(state.calls, 0);
    });
  }
}

test("Video upload populates the driving URL without replacing the identity image", async () => {
  const { context, state } = studio([{ url: "/files/uploaded-driver.mp4" }]);
  context.document.getElementById("refImageUrl").value = "/files/identity.png";
  const file = { type: "video/mp4" };
  context.document.getElementById("videoUploadInput").files = [file];
  await context.uploadLocalFile("videoUploadInput", "targetVideoUrl");
  assert.equal(context.document.getElementById("targetVideoUrl").value, "/files/uploaded-driver.mp4");
  assert.equal(context.document.getElementById("refImageUrl").value, "/files/identity.png");
  assert.equal(state.requests[0].options.body, file);
  assert.equal(state.requests[0].options.headers["Content-Type"], "video/mp4");
});

test("Video upload failures remain visible and do not install an undefined URL", async () => {
  const { context, log } = studio([new Error("Upload connection lost")]);
  context.document.getElementById("videoUploadInput").files = [{ type: "video/mp4" }];
  await context.uploadLocalFile("videoUploadInput", "targetVideoUrl");
  assert.equal(context.document.getElementById("targetVideoUrl").value, "");
  assert.match(log.textContent, /Upload connection lost/);
});

test("Rejected generation submission reports the server error without polling an undefined request", async () => {
  const { context, state, log } = studio([{ detail: "Driving video has no detected person" }]);
  vm.runInContext('currentEndpoint = "wan/v2.2/animate";', context);
  context.document.getElementById("refImageUrl").value = "/files/identity.png";
  context.document.getElementById("targetVideoUrl").value = "/files/driver.mp4";
  context.refreshTelemetry = async () => {};
  await context.runInference();
  assert.match(log.textContent, /Driving video has no detected person/);
  assert.equal(state.calls, 1);
});
