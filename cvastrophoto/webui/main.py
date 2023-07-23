import tornado.web

from .pantograph.handlers import PantographHandler


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class MainCanvasHandler(PantographHandler):

    def on_canvas_init(self, message):
        super(MainCanvasHandler, self).on_canvas_init(message)
