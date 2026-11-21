"use strict";

const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const net = require("node:net");
const path = require("node:path");

function getVenvPythonPath(userDataDir, platform = process.platform) {
  return platform === "win32"
    ? path.join(userDataDir, "venv", "Scripts", "python.exe")
    : path.join(userDataDir, "venv", "bin", "python");
}

function resolveSourceDir(isPackaged, resourcesPath, appDir = __dirname) {
  if (isPackaged) {
    const packagedSrc = path.join(resourcesPath, "openfal-src");
    if (fs.existsSync(packagedSrc)) return packagedSrc;
  }
  return path.resolve(appDir, "..");
}

function getDefaultConfig(userDataDir, sourceDir) {
  const repoWeights = path.join(sourceDir, "weights", "MiniMax-H3");
  const weightsDir = fs.existsSync(repoWeights)
    ? repoWeights
    : path.join(userDataDir, "weights", "MiniMax-H3");
  return {
    mode: "local", // "local" | "remote"
    remoteUrl: "http://127.0.0.1:8000",
    pythonCommand: null,
    pythonArgsPrefix: [],
    weightsDir,
    outputDir: path.join(userDataDir, "outputs"),
    cacheDir: path.join(userDataDir, "cache"),
    profile: "auto",
    syntheticFallback: false,
  };
}

function loadConfig(configPath, userDataDir, sourceDir) {
  const defaults = getDefaultConfig(userDataDir, sourceDir);
  try {
    if (fs.existsSync(configPath)) {
      const raw = JSON.parse(fs.readFileSync(configPath, "utf8"));
      return { ...defaults, ...raw };
    }
  } catch (err) {
    // Fall back to defaults on corrupt config
  }
  return defaults;
}

function saveConfig(configPath, config) {
  fs.mkdirSync(path.dirname(configPath), { recursive: true });
  fs.writeFileSync(configPath, JSON.stringify(config, null, 2), "utf8");
}

function getPythonCandidates(userDataDir, customPython = null, platform = process.platform) {
  const candidates = [];
  if (customPython) {
    candidates.push({ command: customPython, argsPrefix: [], label: "Configured Python" });
  }
  const venvPy = getVenvPythonPath(userDataDir, platform);
  if (fs.existsSync(venvPy)) {
    candidates.push({ command: venvPy, argsPrefix: [], label: "Managed App Venv", isManagedVenv: true });
  }
  if (platform === "win32") {
    candidates.push({ command: "py", argsPrefix: ["-3.11"], label: "Windows py -3.11" });
    candidates.push({ command: "py", argsPrefix: ["-3"], label: "Windows py -3" });
    candidates.push({ command: "python", argsPrefix: [], label: "System python" });
    candidates.push({ command: "python3", argsPrefix: [], label: "System python3" });
  } else {
    candidates.push({ command: "python3", argsPrefix: [], label: "System python3" });
    candidates.push({ command: "python", argsPrefix: [], label: "System python" });
  }
  return candidates;
}

function probePythonCandidate(candidate, sourceDir = null) {
  const probeScript = [
    "import importlib.util, json, sys",
    "ver = list(sys.version_info[:3])",
    "deps = {name: bool(importlib.util.find_spec(name)) for name in ('fastapi', 'uvicorn', 'openfal', 'torch', 'diffusers')}",
    "print(json.dumps({'version': ver, 'executable': sys.executable, 'deps': deps}))",
  ].join("; ");

  const env = { ...process.env };
  if (sourceDir && fs.existsSync(path.join(sourceDir, "openfal"))) {
    env.PYTHONPATH = env.PYTHONPATH ? `${sourceDir}${path.delimiter}${env.PYTHONPATH}` : sourceDir;
  }

  try {
    const res = spawnSync(
      candidate.command,
      [...(candidate.argsPrefix || []), "-c", probeScript],
      { encoding: "utf8", timeout: 8000, windowsHide: true, env }
    );
    if (res.status !== 0 || !res.stdout) return null;
    const info = JSON.parse(res.stdout.trim().split(/\r?\n/).pop());
    const [major, minor] = info.version || [0, 0];
    if (major < 3 || (major === 3 && minor < 10)) return null;
    const deps = info.deps || {};
    return {
      ...candidate,
      executable: info.executable || candidate.command,
      version: info.version.join("."),
      deps,
      canServe: Boolean(deps.fastapi && deps.uvicorn && deps.openfal),
      hasInferenceStack: Boolean(deps.torch && deps.diffusers),
    };
  } catch (err) {
    return null;
  }
}

function detectEnvironment(userDataDir, config, sourceDir) {
  const candidates = getPythonCandidates(userDataDir, config.pythonCommand);
  const probed = [];
  const seenExes = new Set();

  for (const c of candidates) {
    const res = probePythonCandidate(c, null);
    if (res && !seenExes.has(res.executable)) {
      seenExes.add(res.executable);
      probed.push(res);
    }
  }

  // Prefer managed venv that can serve, then any Python that can serve, then any valid Python 3.10+
  let readyRuntime = probed.find(p => p.isManagedVenv && p.canServe) || probed.find(p => p.canServe) || null;

  // When running directly from the git repo without pip-installed openfal, check if deps exist with PYTHONPATH=sourceDir
  if (!readyRuntime && sourceDir && fs.existsSync(path.join(sourceDir, "openfal"))) {
    for (const c of candidates) {
      const res = probePythonCandidate(c, sourceDir);
      if (res && res.canServe) {
        readyRuntime = { ...res, needsPythonPath: true };
        break;
      }
    }
  }

  return {
    readyRuntime,
    availablePythons: probed,
    venvPythonPath: getVenvPythonPath(userDataDir),
    sourceDir,
  };
}

function findFreePort(preferredPort = 8000, host = "127.0.0.1") {
  const tryPort = port => new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(port, host, () => {
      const address = srv.address();
      const chosen = address && typeof address === "object" ? address.port : port;
      srv.close(() => resolve(chosen));
    });
  });

  return tryPort(preferredPort).catch(() => tryPort(0));
}

function buildServeSpec({ runtime, config, port, host = "127.0.0.1", sourceDir }) {
  fs.mkdirSync(config.weightsDir, { recursive: true });
  fs.mkdirSync(config.outputDir, { recursive: true });
  fs.mkdirSync(config.cacheDir, { recursive: true });

  const args = [
    ...(runtime.argsPrefix || []),
    "-m",
    "openfal.cli",
    "serve",
    "--host",
    host,
    "--port",
    String(port),
    "--profile",
    config.profile || "auto",
    "--weights-dir",
    config.weightsDir,
  ];

  const env = {
    ...process.env,
    PYTHONUNBUFFERED: "1",
    OPENFAL_WEIGHTS_DIR: config.weightsDir,
    OPENFAL_OUTPUT_DIR: config.outputDir,
    OPENFAL_CACHE_DIR: config.cacheDir,
  };
  if (config.syntheticFallback) {
    env.OPENFAL_SYNTHETIC_FALLBACK = "1";
  }
  if (runtime.needsPythonPath && sourceDir) {
    env.PYTHONPATH = env.PYTHONPATH ? `${sourceDir}${path.delimiter}${env.PYTHONPATH}` : sourceDir;
  }

  return {
    command: runtime.executable || runtime.command,
    args,
    env,
    cwd: sourceDir,
  };
}

function checkServerHealth(baseUrl, timeoutMs = 1500) {
  return new Promise(resolve => {
    try {
      const url = new URL("/health", baseUrl);
      const req = http.get(url, { timeout: timeoutMs }, res => {
        let body = "";
        res.on("data", chunk => { body += chunk; });
        res.on("end", () => {
          if (res.statusCode === 200) {
            try {
              resolve({ ok: true, data: JSON.parse(body) });
            } catch (e) {
              resolve({ ok: true, data: null });
            }
          } else {
            resolve({ ok: false });
          }
        });
      });
      req.on("error", () => resolve({ ok: false }));
      req.on("timeout", () => {
        req.destroy();
        resolve({ ok: false });
      });
    } catch (e) {
      resolve({ ok: false });
    }
  });
}

module.exports = {
  buildServeSpec,
  checkServerHealth,
  detectEnvironment,
  findFreePort,
  getDefaultConfig,
  getPythonCandidates,
  getVenvPythonPath,
  loadConfig,
  probePythonCandidate,
  resolveSourceDir,
  saveConfig,
};
