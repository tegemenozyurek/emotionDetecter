"""Tiny local server for tools/landmarks.html.

Serves the training images to the browser, where MediaPipe's face detector
(the same one the live webcam app uses) finds eyes, nose and mouth. The page
POSTs the keypoints back and they are written to data/aligned/keypoints.jsonl.

    python tools/landmark_server.py      ->  open http://127.0.0.1:8765/tools/landmarks.html
"""
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "aligned" / "keypoints.jsonl"
NEG_OUT = ROOT / "data" / "aligned" / "neg_keypoints.jsonl"  # non-human faces for the face gate


def manifest(kind="faces"):
    if kind == "negatives":
        neg = ROOT / "data" / "negatives"
        paths = sorted(Path(d, f) for d, _, files in os.walk(neg, followlinks=True)  # afhq / cartoon are links
                       for f in files if f.lower().endswith((".jpg", ".png")))
        return [f"data/negatives/{p.relative_to(neg)}" for p in paths]
    paths = sorted((ROOT / "data" / "fer2013").glob("*/*/*.jpg"))
    paths += sorted((ROOT / "data" / "rafdb" / "DATASET").glob("*/*/*.jpg"))
    return [str(p.relative_to(ROOT)) for p in paths]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path.startswith("/manifest.json"):
            body = json.dumps(manifest("negatives" if "negatives" in self.path else "faces")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if not (self.path.startswith("/tools/") or self.path.startswith("/data/fer2013/")
                or self.path.startswith("/data/rafdb/") or self.path.startswith("/data/negatives/")):
            self.send_error(404)  # only expose what the tool needs
            return
        super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/results"):
            self.send_error(404)
            return
        rows = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = NEG_OUT if "negatives" in self.path else OUT
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        pass  # keep the console quiet


if __name__ == "__main__":
    print(f"{len(manifest())} images -> http://127.0.0.1:8765/tools/landmarks.html")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
