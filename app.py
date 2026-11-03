# -*- coding: utf-8 -*-
"""EvidenceDesk web app (stdlib only). Serves the UI and /api/ask."""
import json, os, time
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import rag

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS = rag.load_docs(os.path.join(BASE, "docs"))
CHUNKS = rag.chunk_docs(DOCS)

PAGE = open(os.path.join(BASE, "static", "index.html"), encoding="utf-8").read()

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlparse(self.path).path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/docs":
            self._send(200, json.dumps(
                {"docs": [{"id": d["id"], "title": d["title"]} for d in DOCS],
                 "chunks": len(CHUNKS), "model": rag.MODEL_NAME}, ensure_ascii=False))
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        global CHUNKS
        if self.path == "/api/upload":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(n) or b"{}")
                title, text = data.get("title", "").strip(), data.get("text", "").strip()
            except Exception:
                self._send(400, '{"error":"bad request"}'); return
            if not title or len(text) < 30:
                self._send(400, '{"error":"need title and text (30+ chars)"}'); return
            docid = f"user-{len(DOCS)}"
            full = f"[사용자 등록 문서] {title}\n{text}"
            with open(os.path.join(BASE, "docs", docid + ".txt"), "w", encoding="utf-8") as f:
                f.write(full)
            d = {"id": docid, "title": f"[사용자 등록 문서] {title}", "body": text}
            DOCS.append(d)
            CHUNKS = rag.chunk_docs(DOCS)
            self._send(200, json.dumps({"ok": True, "id": docid, "chunks": len(CHUNKS)}, ensure_ascii=False))
            return
        if self.path != "/api/ask":
            self._send(404, '{"error":"not found"}'); return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            q = json.loads(self.rfile.read(n) or b"{}").get("q", "").strip()
        except Exception:
            self._send(400, '{"error":"bad request"}'); return
        if not q:
            self._send(400, '{"error":"empty question"}'); return
        t0 = time.time()
        res = rag.answer(q, CHUNKS)
        res["elapsed_ms"] = int((time.time() - t0) * 1000)
        self._send(200, json.dumps(res, ensure_ascii=False))

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    port = int(os.environ.get("ED_PORT", "8090"))
    print(f"EvidenceDesk on http://127.0.0.1:{port}  docs={len(DOCS)} chunks={len(CHUNKS)}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
