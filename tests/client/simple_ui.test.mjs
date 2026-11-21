import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("../../openfal/ui/index.html", import.meta.url), "utf8");
const script = html.split("<script>")[1].split("</script>")[0];

function makeElement(id) {
  const attrs = {};
  const classes = new Set();
  const children = [];
  return {
    id,
    value: "",
    textContent: "",
    innerHTML: "",
    checked: true,
    disabled: false,
    src: "",
    href: "",
    download: "",
    style: {},
    options: [],
    files: [],
    children,
    classList: {
      add: c => classes.add(c),
      remove: c => classes.delete(c),
      toggle: (c, force) => {
        if (force === undefined) {
          if (classes.has(c)) classes.delete(c);
          else classes.add(c);
        } else if (force) classes.add(c);
        else classes.delete(c);
      },
      contains: c => classes.has(c),
    },
    setAttribute: (k, v) => { attrs[k] = String(v); },
    getAttribute: k => attrs[k] ?? null,
    appendChild: node => {
      children.push(node);
      if (id === "simpleModelSelect") {
        children[children.length - 1].selected = children.length === 1;
        // Keep options array synchronized
      }
      return node;
    },
    append: (...nodes) => { children.push(...nodes); },
    querySelectorAll: () => [],
  };
}

function createSimpleContext(responses = []) {
  const elements = {};
  const requests = [];
  const getEl = id => {
    if (!elements[id]) elements[id] = makeElement(id);
    return elements[id];
  };

  // Override appendChild on simpleModelSelect to keep options & value updated
  const modelSelect = getEl("simpleModelSelect");
  modelSelect.appendChild = node => {
    modelSelect.options.push(node);
    if (modelSelect.options.length === 1) modelSelect.value = node.value;
    return node;
  };
  Object.defineProperty(modelSelect, "innerHTML", {
    get: () => "",
    set: () => { modelSelect.options = []; modelSelect.value = ""; },
  });

  const context = vm.createContext({
    document: {
      getElementById: getEl,
      querySelectorAll: () => [],
      createElement: tag => makeElement(tag),
    },
    window: { addEventListener() {}, location: { origin: "http://localhost:8008" } },
    setTimeout: cb => cb(),
    setInterval: () => 1,
    clearInterval: () => {},
    fetch: async (url, options) => {
      requests.push({ url, options });
      const response = responses.shift();
      assert.ok(response, `Unexpected fetch call to ${url}`);
      if (response instanceof Error) throw response;
      return { ok: response._ok !== false, json: async () => response };
    },
  });
  vm.runInContext(script, context);
  return { context, elements, requests, getEl };
}

test("HTML contains Simple and Studio view tabs and Simple tab sections", () => {
  assert.match(html, /id="tabBtnSimple"/);
  assert.match(html, /id="tabBtnStudio"/);
  assert.match(html, /id="simpleView"/);
  assert.match(html, /id="studioView"/);
  for (const task of ["t2i", "t2v", "i2v", "i2i", "animate"]) {
    assert.match(html, new RegExp(`data-simple-task="${task}"`));
  }
});

test("switchMainView toggles visibility between Simple and Studio views", () => {
  const { context, getEl } = createSimpleContext();
  context.switchMainView("simple");
  assert.equal(getEl("simpleView").style.display, "block");
  assert.equal(getEl("studioView").style.display, "none");
  assert.equal(getEl("simpleHeaderControls").style.display, "flex");
  assert.equal(getEl("studioHeaderControls").style.display, "none");

  context.switchMainView("studio");
  assert.equal(getEl("simpleView").style.display, "none");
  assert.equal(getEl("studioView").style.display, "grid");
  assert.equal(getEl("simpleHeaderControls").style.display, "none");
  assert.equal(getEl("studioHeaderControls").style.display, "flex");
});

test("Simple Text-to-Image builds clean payload without video duration", () => {
  const { context, getEl } = createSimpleContext();
  context.selectSimpleTask("t2i");
  getEl("simplePrompt").value = "A sleek ceramic lamp on oak";
  getEl("simpleResolution").value = "1080P";
  context.selectSimpleRatio("1:1");
  context.selectSimpleNumImages(2);

  const { endpoint, payload } = context.buildSimplePayload();
  assert.equal(endpoint, "qwen/image-2.1/text-to-image");
  assert.equal(payload.prompt, "A sleek ceramic lamp on oak");
  assert.equal(payload.resolution, "1080P");
  assert.equal(payload.aspect_ratio, "1:1");
  assert.equal(payload.num_images, 2);
  assert.equal(payload.duration, undefined);
});

test("Simple Text-to-Video switches between Wan 2.2 and MiniMax H3 with audio", () => {
  const { context, getEl } = createSimpleContext();
  context.selectSimpleTask("t2v");
  assert.equal(getEl("simpleDurationGroup").style.display, "block");
  assert.equal(getEl("simpleNumImagesGroup").style.display, "none");
  assert.equal(getEl("simpleAudioGroup").style.display, "none");

  let built = context.buildSimplePayload();
  assert.equal(built.endpoint, "wan/v2.2/text-to-video");
  assert.equal(built.payload.enable_audio, undefined);

  getEl("simpleModelSelect").value = "minimax/h3-max/text-to-video";
  context.onSimpleModelChange();
  assert.equal(getEl("simpleAudioGroup").style.display, "block");
  getEl("simpleEnableAudio").checked = true;
  context.selectSimpleDuration(8);

  built = context.buildSimplePayload();
  assert.equal(built.endpoint, "minimax/h3-max/text-to-video");
  assert.equal(built.payload.duration, 8);
  assert.equal(built.payload.enable_audio, true);
});

test("Simple Image-to-Video validates missing image and submits valid payload", async () => {
  const { context, getEl, requests } = createSimpleContext([
    { request_id: "req-i2v-1" },
    { status: "COMPLETED", logs: [{ message: "Done" }] },
    { video: { url: "/files/out.mp4", file_name: "out.mp4" }, seed: 42, timings: { inference: 1.5 } },
  ]);
  context.selectSimpleTask("i2v");
  getEl("simplePrompt").value = "Gentle camera zoom";

  await context.runSimpleInference();
  assert.equal(requests.length, 0);
  assert.match(getEl("simpleLogBox").textContent, /input image/i);

  getEl("simpleImageUrl").value = "/files/start.png";
  await context.runSimpleInference();
  assert.equal(requests.length, 3);
  assert.equal(requests[0].url, "/queue/wan/v2.2/image-to-video");
  const sentBody = JSON.parse(requests[0].options.body);
  assert.equal(sentBody.start_image_url, "/files/start.png");
  assert.equal(getEl("simpleDownloadLink").href, "/files/out.mp4");
});

test("reuseOutputAsInput loads generated image into Image-to-Video", async () => {
  const { context, getEl } = createSimpleContext([
    { request_id: "req-t2i-1" },
    { status: "COMPLETED", logs: [{ message: "Image ready" }] },
    { images: [{ url: "/files/gen_0.png", file_name: "gen_0.png" }], seed: 777, timings: { inference: 0.9 } },
  ]);
  context.selectSimpleTask("t2i");
  getEl("simplePrompt").value = "Mountain sunset";
  await context.runSimpleInference();

  assert.equal(getEl("simpleToI2vBtn").style.display, "inline-block");
  context.reuseOutputAsInput("i2v");
  assert.equal(getEl("simpleImageUrl").value, "/files/gen_0.png");
  const built = context.buildSimplePayload();
  assert.equal(built.endpoint, "wan/v2.2/image-to-video");
  assert.equal(built.payload.start_image_url, "/files/gen_0.png");
});

test("Simple Animate validates driving video and identity image", async () => {
  const { context, getEl, requests } = createSimpleContext();
  context.selectSimpleTask("animate");
  getEl("simpleImageUrl").value = "/files/face.png";
  await context.runSimpleInference();
  assert.equal(requests.length, 0);
  assert.match(getEl("simpleLogBox").textContent, /identity image and a driving video/i);

  getEl("simpleVideoUrl").value = "/files/driver.mp4";
  const built = context.buildSimplePayload();
  assert.equal(built.endpoint, "wan/v2.2/animate");
  assert.equal(built.payload.face_image_url, "/files/face.png");
  assert.equal(built.payload.driving_video_url, "/files/driver.mp4");
});
