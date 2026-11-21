"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("openfalDesktop", {
  isDesktop: true,
  getInfo: () => ipcRenderer.invoke("desktop:getInfo"),
  detectEnvironment: () => ipcRenderer.invoke("desktop:detectEnvironment"),
  chooseWeightsDir: () => ipcRenderer.invoke("desktop:chooseWeightsDir"),
  choosePythonExe: () => ipcRenderer.invoke("desktop:choosePythonExe"),
  openOutputs: () => ipcRenderer.invoke("desktop:openOutputs"),
  openSetup: () => ipcRenderer.invoke("desktop:openSetup"),
  restartBackend: () => ipcRenderer.invoke("desktop:restartBackend"),
  installRuntime: opts => ipcRenderer.invoke("desktop:installRuntime", opts),
  launchLocalEngine: opts => ipcRenderer.invoke("desktop:launchLocalEngine", opts),
  connectRemoteServer: url => ipcRenderer.invoke("desktop:connectRemoteServer", url),
  onInstallLog: callback => {
    const handler = (_event, payload) => callback(payload);
    ipcRenderer.on("desktop:installLog", handler);
    return () => ipcRenderer.removeListener("desktop:installLog", handler);
  },
  onBootStatus: callback => {
    const handler = (_event, payload) => callback(payload);
    ipcRenderer.on("desktop:bootStatus", handler);
    return () => ipcRenderer.removeListener("desktop:bootStatus", handler);
  },
});
