# openfal Desktop App & Installer

Install and run **openfal Studio** as a standalone desktop application with built-in model downloading and a managed local inference server (while still supporting connecting to an external `openfal serve` URL).

## Run in Development

```bash
cd desktop
npm install
npm start
```

## Build Desktop Installers

```bash
cd desktop
npm install
npm run dist        # Builds installer for current OS into desktop/dist/
```

Platform-specific targets:
- **Windows:** `npm run dist:win` (NSIS installer `.exe` + portable `.exe`)
- **macOS:** `npm run dist:mac` (`.dmg` + `.zip`)
- **Linux:** `npm run dist:linux` (`.AppImage` + `.deb`)

## First Launch Experience

1. On first launch, if `openfal` is not yet installed in a local Python environment, the desktop setup window lets you pick a weights folder and install the local runtime (plus optional PyTorch/Diffusers dependencies) with a single click—no CLI required.
2. You can also connect to an already-running `openfal serve` instance (for example `http://127.0.0.1:8000`).
3. Once inside the app, click **📦 Models** in the header (or click a model's status badge) to download/manage Hugging Face checkpoints directly from the GUI.
