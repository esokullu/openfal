"""Exercise the shipped Bash client against a local HTTP stub, without inference."""

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/client/curl/05_animate_recast_wan22.sh"
if os.name == "nt":
    BASH = next((str(path) for path in (
        Path("C:/Program Files/Git/bin/bash.exe"), Path("C:/Program Files/Git/usr/bin/bash.exe"),
    ) if path.is_file()), None)
else:
    BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH or not shutil.which("curl"), reason="Bash and curl required")


@pytest.fixture
def server():
    state = {"requests": [], "fail": None, "uploads": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["requests"].append((self.path, body))
            if state["fail"] == self.path:
                self.send_response(422)
                result = {"detail": "Rejected test input"}
            elif self.path == "/v1/storage/upload":
                self.send_response(200)
                state["uploads"] += 1
                result = {"url": f"http://127.0.0.1:{self.server.server_port}/input/{state['uploads']}"}
            else:
                self.send_response(200)
                result = {"video": {"url": f"http://127.0.0.1:{self.server.server_port}/output.mp4"}}
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"test-video-response")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd, state
    finally:
        httpd.shutdown()
        thread.join()
        httpd.server_close()


def run_script(tmp_path, server, mode):
    httpd, _ = server
    identity, driver = tmp_path / "identity photo.png", tmp_path / "driving clip.mp4"
    identity.write_bytes(b"identity")
    driver.write_bytes(b"driver")
    # Git's Windows checkout can use CRLF; validate the LF form shipped to Unix.
    script = tmp_path / "animate.sh"
    script.write_bytes(SCRIPT.read_bytes().replace(b"\r\n", b"\n"))
    env = {
        **os.environ,
        "OPENFAL_URL": f"http://127.0.0.1:{httpd.server_port}",
        "OPENFAL_ANIMATE_MODE": mode,
        "OPENFAL_PORTRAIT": identity.as_posix(),
        "OPENFAL_REPLACEMENT_PORTRAIT": identity.as_posix(),
        "OPENFAL_DRIVING_VIDEO": driver.as_posix(),
        "OPENFAL_OUTPUTS_DIR": (tmp_path / "out").as_posix(),
        "OPENFAL_TEST_SEED": "20261117",
        "PYTHON": Path(sys.executable).as_posix(),
    }
    return subprocess.run([BASH, script.as_posix()], env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("mode,endpoint", [("animate", "animate"), ("replace", "recast")])
def test_curl_sends_identity_driver_mode_and_fixed_seed(tmp_path, server, mode, endpoint):
    result = run_script(tmp_path, server, mode)
    assert result.returncode == 0, result.stdout + result.stderr
    _, state = server
    assert len(state["requests"]) == 3
    assert [body for _, body in state["requests"][:2]] == [b"identity", b"driver"]
    route, raw = state["requests"][-1]
    payload = json.loads(raw)
    assert route == f"/wan/v2.2/{endpoint}"
    assert payload["animate_mode"] == mode and payload["seed"] == 20261117
    assert payload["face_image_url"].endswith("/input/1")
    assert payload["driving_video_url"].endswith("/input/2")
    assert "start_image_url" not in payload
    output = tmp_path / "out" / f"05_wan22_{mode}.mp4"
    assert output.read_bytes() == b"test-video-response"
    assert output.with_suffix(".request.json").is_file()
    assert output.with_suffix(".response.json").is_file()


@pytest.mark.parametrize("failed_path", ["/v1/storage/upload", "/wan/v2.2/animate"])
def test_curl_http_errors_are_fatal_and_show_response_body(tmp_path, server, failed_path):
    server[1]["fail"] = failed_path
    result = run_script(tmp_path, server, "animate")
    assert result.returncode != 0
    assert "Rejected test input" in result.stderr
    assert not (tmp_path / "out/05_wan22_animate.mp4").exists()
