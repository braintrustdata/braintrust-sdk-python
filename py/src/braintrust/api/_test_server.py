"""Shared local HTTP server helper for API tests."""

import contextlib
import http.server
import socketserver
import threading
import time


@contextlib.contextmanager
def scripted_server(script):
    """Run a local server driven by sequential actions or a request callback."""

    class ScriptedHandler(http.server.BaseHTTPRequestHandler):
        request_count = 0
        requests = []

        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self._handle()

        def do_POST(self):
            self._handle()

        def do_PATCH(self):
            self._handle()

        def do_DELETE(self):
            self._handle()

        def _handle(self):
            request_number = type(self).request_count
            type(self).request_count += 1
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length) if content_length else b""
            type(self).requests.append((self.command, self.path, body, self.headers.get("Authorization")))
            action = (
                script(self.command, self.path, body, self.headers)
                if callable(script)
                else script[min(request_number, len(script) - 1)]
            )

            if action == "close":
                self.connection.close()
                return

            if action[0] == "sleep":
                _, delay, status, headers, response_body = action
                time.sleep(delay)
            else:
                status, headers, response_body = action

            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            try:
                self.wfile.write(response_body)
            except BrokenPipeError:
                pass

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), ScriptedHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()

    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", ScriptedHandler
    finally:
        server.shutdown()
        server.server_close()
