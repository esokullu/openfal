import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const script = readFileSync(new URL("../../openfal/ui/index.html", import.meta.url), "utf8")
  .split("<script>")[1].split("</script>")[0];

function studio(responses) {
  const state = { refreshes: 0, calls: 0 };
  const elements = { hfTokenInput: { value: "" }, logBox: { textContent: "", scrollHeight: 0 } };
  const context = vm.createContext({
    document: { getElementById: id => elements[id] ||= { value: "", textContent: "", style: {} } },
    window: { addEventListener() {} },
    setTimeout: callback => callback(),
    fetch: async () => {
      state.calls++;
      const response = responses.shift();
      assert.ok(response, "Unexpected request after installation reached a terminal state");
      if (response instanceof Error) throw response;
      return { ok: true, json: async () => response };
    },
  });
  vm.runInContext(script, context);
  context.refreshModelHub = async () => { state.refreshes++; };
  return { context, state, log: elements.logBox };
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
