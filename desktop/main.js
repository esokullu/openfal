"use strict";

const { app, BrowserWindow, dialog, ipcMain, Menu, shell } = require("electron");
const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const {
  buildServeSpec,
  checkServerHealth,
  detectEnvironment,
  findFreePort,
  getVenvPythonPath,
  loadConfig,
  resolveSourceDir,
  saveConfig,
} = require("./runtime");

let mainWindow = null;
let backendProc = null;
let activeServerUrl = null;
let isQuitting = false;

function getPaths() {
  const userDataDir = app.getPath("userData");
  const sourceDir = resolveSourceDir(app.isPackaged, process.resourcesPath, __dirname);
  const configPath = path.join(userDataDir, "desktop-config.json");
  return { userDataDir, sourceDir, configPath };
}

function sendToWindow(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

function stopBackendProcess() {
  if (!backendProc || backendProc.killed) {
    backendProc = null;
    activeServerUrl = null;
    return;
  }
  const proc = backendProc;
  backendProc = null;
  activeServerUrl = null;
  try {
    if (process.platform === "win32" && proc.pid) {
      spawnSync("taskkill", ["/pid", String(proc.pid), "/t", "/f"], { windowsHide: true });
    } else {
      proc.kill("SIGTERM");
    }
  } catch (err) {
    // Ignore errors during teardown
  }
}

async function waitForServerReady(baseUrl, timeoutMs = 45000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!backendProc) {
      throw new Error("Local openfal server exited before becoming ready.");
    }
    const health = await checkServerHealth(baseUrl, 1200);
    if (health.ok) return health.data;
    await new Promise(r => setTimeout(r, 350));
  }
  throw new Error("Timed out waiting for local openfal server to respond on /health.");
}

async function startLocalBackendAndLoad() {
  const { userDataDir, sourceDir, configPath } = getPaths();
  const config = loadConfig(configPath, userDataDir, sourceDir);
  const envInfo = detectEnvironment(userDataDir, config, sourceDir);

  if (!envInfo.readyRuntime) {
    throw new Error("No Python runtime with openfal installed was found.");
  }

  stopBackendProcess();
  sendToWindow("desktop:bootStatus", { state: "starting", message: "Starting local openfal engine..." });

  const port = await findFreePort(8000, "127.0.0.1");
  const spec = buildServeSpec({
    runtime: envInfo.readyRuntime,
    config,
    port,
    host: "127.0.0.1",
    sourceDir,
  });

  const serverUrl = `http://127.0.0.1:${port}`;
  let startupLogs = "";

  backendProc = spawn(spec.command, spec.args, {
    cwd: spec.cwd,
    env: spec.env,
    windowsHide: true,
  });

  backendProc.stdout.on("data", chunk => {
    const text = chunk.toString();
    startupLogs += text;
    if (startupLogs.length > 8000) startupLogs = startupLogs.slice(-8000);
    sendToWindow("desktop:bootStatus", { state: "starting", log: text });
  });

  backendProc.stderr.on("data", chunk => {
    const text = chunk.toString();
    startupLogs += text;
    if (startupLogs.length > 8000) startupLogs = startupLogs.slice(-8000);
    sendToWindow("desktop:bootStatus", { state: "starting", log: text });
  });

  backendProc.on("exit", code => {
    const wasActive = Boolean(backendProc);
    backendProc = null;
    if (!isQuitting && wasActive && activeServerUrl === null) {
      sendToWindow("desktop:bootStatus", {
        state: "error",
        message: `Local server exited with code ${code}.\n${startupLogs.slice(-1500)}`,
      });
    }
  });

  await waitForServerReady(serverUrl, 60000);
  activeServerUrl = serverUrl;
  if (mainWindow && !mainWindow.isDestroyed()) {
    await mainWindow.loadURL(`${serverUrl}/`);
  }
  return { ok: true, serverUrl };
}

function runCommandStreaming(command, args, opts, onLine) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { ...opts, windowsHide: true });
    child.stdout.on("data", d => onLine(d.toString()));
    child.stderr.on("data", d => onLine(d.toString()));
    child.on("error", err => reject(err));
    child.on("close", code => {
      if (code === 0) resolve();
      else reject(new Error(`Command '${command} ${args.join(" ")}' failed with exit code ${code}`));
    });
  });
}

function buildAppMenu() {
  const template = [
    {
      label: "openfal",
      submenu: [
        {
          label: "Setup & Connection...",
          click: () => {
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.loadFile(path.join(__dirname, "setup.html"));
            }
          },
        },
        {
          label: "Open Outputs Folder",
          click: async () => {
            const { userDataDir, sourceDir, configPath } = getPaths();
            const cfg = loadConfig(configPath, userDataDir, sourceDir);
            fs.mkdirSync(cfg.outputDir, { recursive: true });
            await shell.openPath(cfg.outputDir);
          },
        },
        {
          label: "Open Weights Folder",
          click: async () => {
            const { userDataDir, sourceDir, configPath } = getPaths();
            const cfg = loadConfig(configPath, userDataDir, sourceDir);
            fs.mkdirSync(cfg.weightsDir, { recursive: true });
            await shell.openPath(cfg.weightsDir);
          },
        },
        { type: "separator" },
        { role: "quit" },
      ],
    },
    {
      label: "Engine",
      submenu: [
        {
          label: "Restart Local Engine",
          click: async () => {
            if (mainWindow && !mainWindow.isDestroyed()) {
              await mainWindow.loadFile(path.join(__dirname, "setup.html"), { query: { booting: "1" } });
              try {
                await startLocalBackendAndLoad();
              } catch (err) {
                sendToWindow("desktop:bootStatus", { state: "error", message: err.message });
              }
            }
          },
        },
        {
          label: "Reload UI",
          accelerator: "CmdOrCtrl+R",
          click: () => {
            if (mainWindow && !mainWindow.isDestroyed()) mainWindow.reload();
          },
        },
      ],
    },
    {
      label: "View",
      submenu: [
        { role: "toggledevtools" },
        { role: "resetzoom" },
        { role: "zoomin" },
        { role: "zoomout" },
        { role: "togglefullscreen" },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

async function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 960,
    minHeight: 680,
    backgroundColor: "#090b10",
    icon: path.join(__dirname, "assets", process.platform === "win32" ? "icon.ico" : "icon.png"),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    try {
      const parsed = new URL(url);
      if (parsed.hostname === "127.0.0.1" || parsed.hostname === "localhost") {
        return { action: "allow" };
      }
      shell.openExternal(url);
    } catch (err) {}
    return { action: "deny" };
  });

  const { userDataDir, sourceDir, configPath } = getPaths();
  const cfg = loadConfig(configPath, userDataDir, sourceDir);

  if (cfg.mode === "remote" && cfg.remoteUrl) {
    const health = await checkServerHealth(cfg.remoteUrl, 1500);
    if (health.ok) {
      activeServerUrl = cfg.remoteUrl;
      await mainWindow.loadURL(`${cfg.remoteUrl.replace(/\/$/, "")}/`);
      return;
    }
  }

  const envInfo = detectEnvironment(userDataDir, cfg, sourceDir);
  if (envInfo.readyRuntime) {
    await mainWindow.loadFile(path.join(__dirname, "setup.html"), { query: { booting: "1" } });
    try {
      await startLocalBackendAndLoad();
    } catch (err) {
      sendToWindow("desktop:bootStatus", { state: "error", message: err.message });
    }
  } else {
    await mainWindow.loadFile(path.join(__dirname, "setup.html"));
  }
}

function registerIpcHandlers() {
  ipcMain.handle("desktop:getInfo", () => {
    const { userDataDir, sourceDir, configPath } = getPaths();
    const config = loadConfig(configPath, userDataDir, sourceDir);
    return {
      ...config,
      serverUrl: activeServerUrl,
      userDataDir,
      sourceDir,
      isPackaged: app.isPackaged,
    };
  });

  ipcMain.handle("desktop:detectEnvironment", () => {
    const { userDataDir, sourceDir, configPath } = getPaths();
    const config = loadConfig(configPath, userDataDir, sourceDir);
    return {
      config,
      env: detectEnvironment(userDataDir, config, sourceDir),
    };
  });

  ipcMain.handle("desktop:chooseWeightsDir", async () => {
    const { userDataDir, sourceDir, configPath } = getPaths();
    const config = loadConfig(configPath, userDataDir, sourceDir);
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "Select Model Weights Folder",
      defaultPath: config.weightsDir,
      properties: ["openDirectory", "createDirectory"],
    });
    if (result.canceled || !result.filePaths.length) return { canceled: true };
    config.weightsDir = result.filePaths[0];
    saveConfig(configPath, config);
    return { canceled: false, weightsDir: config.weightsDir };
  });

  ipcMain.handle("desktop:choosePythonExe", async () => {
    const { userDataDir, sourceDir, configPath } = getPaths();
    const config = loadConfig(configPath, userDataDir, sourceDir);
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "Select Python 3.10+ Executable",
      properties: ["openFile"],
    });
    if (result.canceled || !result.filePaths.length) return { canceled: true };
    config.pythonCommand = result.filePaths[0];
    saveConfig(configPath, config);
    return { canceled: false, pythonCommand: config.pythonCommand };
  });

  ipcMain.handle("desktop:openOutputs", async () => {
    const { userDataDir, sourceDir, configPath } = getPaths();
    const config = loadConfig(configPath, userDataDir, sourceDir);
    fs.mkdirSync(config.outputDir, { recursive: true });
    await shell.openPath(config.outputDir);
    return { ok: true };
  });

  ipcMain.handle("desktop:openSetup", async () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      await mainWindow.loadFile(path.join(__dirname, "setup.html"));
    }
    return { ok: true };
  });

  ipcMain.handle("desktop:restartBackend", async () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      await mainWindow.loadFile(path.join(__dirname, "setup.html"), { query: { booting: "1" } });
    }
    await startLocalBackendAndLoad();
    return { ok: true };
  });

  ipcMain.handle("desktop:installRuntime", async (_evt, opts = {}) => {
    const { userDataDir, sourceDir, configPath } = getPaths();
    const config = loadConfig(configPath, userDataDir, sourceDir);
    if (opts.weightsDir) config.weightsDir = opts.weightsDir;
    if (opts.pythonCommand) config.pythonCommand = opts.pythonCommand;
    config.syntheticFallback = Boolean(opts.syntheticFallback);
    config.mode = "local";
    saveConfig(configPath, config);

    const envInfo = detectEnvironment(userDataDir, config, sourceDir);
    const basePython = envInfo.availablePythons.find(p => p.executable === opts.basePythonExecutable)
      || envInfo.availablePythons[0];

    if (!basePython) {
      throw new Error("Python 3.10+ was not found on this machine. Please install Python 3.11+ or select python.exe.");
    }

    const venvDir = path.join(userDataDir, "venv");
    const venvPy = getVenvPythonPath(userDataDir);
    const log = line => sendToWindow("desktop:installLog", { line });

    if (!fs.existsSync(venvPy)) {
      log(`[openfal-desktop] Creating managed virtual environment in ${venvDir}...\n`);
      await runCommandStreaming(
        basePython.command,
        [...(basePython.argsPrefix || []), "-m", "venv", venvDir],
        { cwd: userDataDir },
        log
      );
    }

    log("[openfal-desktop] Upgrading pip...\n");
    await runCommandStreaming(venvPy, ["-m", "pip", "install", "--upgrade", "pip"], { cwd: sourceDir }, log);

    const targetSpec = opts.includeInference ? `${sourceDir}[inference]` : sourceDir;
    log(`[openfal-desktop] Installing openfal from ${targetSpec}...\n`);
    await runCommandStreaming(venvPy, ["-m", "pip", "install", targetSpec], { cwd: sourceDir }, log);

    log("[openfal-desktop] Installation complete! Launching local server...\n");
    await startLocalBackendAndLoad();
    return { ok: true };
  });

  ipcMain.handle("desktop:launchLocalEngine", async (_evt, opts = {}) => {
    const { userDataDir, sourceDir, configPath } = getPaths();
    const config = loadConfig(configPath, userDataDir, sourceDir);
    config.mode = "local";
    if (opts.weightsDir) config.weightsDir = opts.weightsDir;
    if (opts.syntheticFallback !== undefined) config.syntheticFallback = Boolean(opts.syntheticFallback);
    saveConfig(configPath, config);
    await startLocalBackendAndLoad();
    return { ok: true };
  });

  ipcMain.handle("desktop:connectRemoteServer", async (_evt, rawUrl) => {
    const { userDataDir, sourceDir, configPath } = getPaths();
    const config = loadConfig(configPath, userDataDir, sourceDir);
    const url = String(rawUrl || "http://127.0.0.1:8000").trim().replace(/\/$/, "");
    const health = await checkServerHealth(url, 2500);
    if (!health.ok) {
      throw new Error(`Could not reach openfal server at ${url}/health`);
    }
    stopBackendProcess();
    config.mode = "remote";
    config.remoteUrl = url;
    saveConfig(configPath, config);
    activeServerUrl = url;
    if (mainWindow && !mainWindow.isDestroyed()) {
      await mainWindow.loadURL(`${url}/`);
    }
    return { ok: true };
  });
}

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(async () => {
    buildAppMenu();
    registerIpcHandlers();
    await createMainWindow();
  });

  app.on("before-quit", () => {
    isQuitting = true;
    stopBackendProcess();
  });

  app.on("window-all-closed", () => {
    stopBackendProcess();
    if (process.platform !== "darwin") app.quit();
  });
}
