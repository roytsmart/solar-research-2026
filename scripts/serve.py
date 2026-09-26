"""
Serve the repository for a local preview of the slides.

.. code-block:: bash

    python scripts/serve.py

and open http://localhost:8000/docs/.

``python -m http.server`` would serve the same files, but it ignores HTTP
range requests, and a browser cannot seek in a video from a server which
does: it seems to, but only within whatever it happens to have cached, and
not at all in a movie which has just been rendered again. This server answers
them, as GitHub Pages does, so the preview behaves like the published page.

It also asks the browser to check each file again before using a cached copy,
since the figures change under the same names every time they are rendered.
"""

import functools
import http.server
import os
import pathlib
import re
import sys

__all__ = [
    "RangeRequestHandler",
]


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """A static file handler which answers single-range requests."""

    _range: None | tuple[int, int] = None

    def send_head(self):
        self._range = None
        header = self.headers.get("Range")
        path = self.translate_path(self.path)
        if header is None or os.path.isdir(path):
            return super().send_head()

        match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
        if match is None or match.groups() == ("", ""):
            return super().send_head()

        try:
            file = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        stat = os.fstat(file.fileno())
        size = stat.st_size
        first, last = match.groups()
        if first == "":
            # A suffix, the last so many bytes of the file.
            first, last = max(size - int(last), 0), size - 1
        else:
            first = int(first)
            last = min(int(last), size - 1) if last else size - 1

        if first >= size or first > last:
            file.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.send_header("Content-Length", str(last - first + 1))
        self.send_header("Last-Modified", self.date_time_string(int(stat.st_mtime)))
        self.end_headers()

        file.seek(first)
        self._range = (first, last)
        return file

    def copyfile(self, source, outputfile):
        if self._range is None:
            return super().copyfile(source, outputfile)
        first, last = self._range
        remaining = last - first + 1
        while remaining > 0:
            chunk = source.read(min(1 << 16, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def handle(self):
        # A browser abandons a range it no longer needs as soon as the user
        # seeks elsewhere, which is not an error worth a traceback.
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def main(port: int = 8000) -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    handler = functools.partial(RangeRequestHandler, directory=str(root))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Serving {root} at http://localhost:{port}/docs/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main(*(int(arg) for arg in sys.argv[1:2]))
