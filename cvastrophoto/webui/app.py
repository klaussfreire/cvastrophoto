import os.path
import threading
import six

import tornado.web

from .main import MainHandler, MainCanvasHandler


class WebUIApplication(tornado.web.Application):
    def __init__(self, **settings):
        settings.setdefault("compress_response", True)
        settings.setdefault("cookie_secret", os.environ.get("CV_COOKIE_SECRET", "cvapdefaultcookie"))
        super(WebUIApplication, self).__init__(
            [
                (r"/", tornado.web.RedirectHandler, {"url": "/index.html"}),
                (r"/index.html", MainHandler),
                (r"/socket", MainCanvasHandler),
                (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": os.path.dirname(__file__) + "/static"}),
            ],
            template_loader=tornado.template.Loader(os.path.dirname(__file__) + "/templates"),
            **settings)

    def run_in_thread(self, daemon=True, autostart=True):
        if six.PY3:
            import asyncio
            thread = threading.Thread(target=asyncio.run, args=(self._arun(),))
        else:
            thread = threading.Thread(target=self.run)
        if daemon:
            thread.daemon = True
        if autostart:
            thread.start()
        return thread

    if six.PY3:
        # Prevent a syntax error in py2
        exec("""
async def _arun(self):
    self.run()
    shutdown_event = asyncio.Event()
    await shutdown_event.wait()
""")

    def run(self):
        self.listen(8888)
        if six.PY2:
            tornado.ioloop.IOLoop.current().start()
