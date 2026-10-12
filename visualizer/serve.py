#!/usr/bin/env python3
"""Serves SkeVU on http://localhost:PORT without letting the browser cache anything, so that an edited script, style or example
is always loaded fresh (python3 -m http.server leaves the browser to guess how long a file stays valid).

    python3 serve.py [PORT]          (default 5005; run it from any directory: the files are those next to this script)
    python3 serve.py 5005 --all      (listen on all interfaces, not only on this machine)
"""
import functools
import http.server
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, max-age=0')
        super().end_headers()


def main(argv):
    args = [a for a in argv if not a.startswith('--')]
    port = int(args[0]) if args else 5005
    bind = '' if '--all' in argv else '127.0.0.1'
    handler = functools.partial(NoCacheHandler, directory=HERE)
    with http.server.ThreadingHTTPServer((bind, port), handler) as server:
        print('SkeVU on http://%s:%d (no caching), Ctrl+C to stop' % ('localhost' if bind else '0.0.0.0', port))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()


if __name__ == '__main__':
    main(sys.argv[1:])
