import tornado.web
import tornado.websocket
import tornado.template
from tornado.ioloop import IOLoop

import random
import json
import os
import datetime
from collections import namedtuple


DEFAULT_INTERVAL = 10

InputEvent = namedtuple(
    "InputEvent",
    [
        "type", "x", "y", "button",
        "alt_key", "ctrl_key", "meta_key",
        "shift_key", "key_code",
    ]
)


class PantographHandler(tornado.websocket.WebSocketHandler):

    interval = None

    def on_canvas_init(self, message):
        self.width = message["width"]
        self.height = message["height"]
        self.setup()
        self.do_operation("refresh")

        # randomize the first timeout so we don't get every timer
        # expiring at the same time
        if self.interval is not None:
            interval = (random.random() * 0.5 + 0.5) * self.interval
            delta = datetime.timedelta(milliseconds=int(interval * 1000))
            self.timeout = IOLoop.current().add_timeout(delta, self.timer_tick)
        else:
            self.timeout = None

    def on_close(self):
        if self.timeout is not None:
            IOLoop.current().remove_timeout(self.timeout)

    def on_message(self, raw_message):
        message = json.loads(raw_message)
        event_type = message.get("type")

        if event_type == "setbounds":
            self.on_canvas_init(message)
        else:
            event_callbacks = {
                "mousedown": self.on_mouse_down,
                "mouseup": self.on_mouse_up,
                "mousemove": self.on_mouse_move,
                "click": self.on_click,
                "dblclick": self.on_dbl_click,
                "keydown": self.on_key_down,
                "keyup": self.on_key_up,
                "keypress": self.on_key_press
            }
            event_callbacks[event_type](InputEvent(**message))

    def do_operation(self, operation, **kwargs):
        message = dict(kwargs, operation=operation)
        raw_message = json.dumps(message)
        self.write_message(raw_message)

    def draw(self, shape_type, **kwargs):
        shape = dict(kwargs, type=shape_type)
        self.do_operation("draw", shape=shape)

    def draw_rect(self, x, y, width, height, color = "#000", **extra):
        self.draw("rect", x=x, y=y, width=width, height=height,
                          lineColor=color, **extra)

    def fill_rect(self, x, y, width, height, color = "#000", **extra):
        self.draw("rect", x=x, y=y, width=width, height=height,
                          fillColor=color, **extra)

    def clear_rect(self, x, y, width, height, **extra):
        self.draw("clear", x=x, y=y, width=width, height=height, **extra)

    def draw_oval(self, x, y, width, height, color = "#000", **extra):
        self.draw("oval", x=x, y=y, width=width, height=height,
                          lineColor=color, **extra)

    def fill_oval(self, x, y, width, height, color = "#000", **extra):
        self.draw("oval", x=x, y=y, width=width, height=height,
                          fillColor=color, **extra)

    def draw_circle(self, x, y, radius, color = "#000", **extra):
        self.draw("circle", x=x, y=y, radius=radius,
                            lineColor=color, **extra)

    def fill_circle(self, x, y, radius, color = "#000", **extra):
        self.draw("circle", x=x, y=y, radius=radius,
                           fillColor=color, **extra)

    def draw_line(self, startX, startY, endX, endY, color = "#000", **extra):
        self.draw("line", startX=startX, startY=startY,
                          endX=endX, endY=endY, color=color, **extra)

    def fill_polygon(self, points, color = "#000", **extra):
        self.draw("polygon", points=points, fillColor=color, **extra)

    def draw_polygon(self, points, color = "#000", **extra):
        self.draw("polygon", points=points, lineColor=color, **extra)

    def draw_image(self, img_name, x, y, width=None, height=None, **extra):
        app_path = os.path.join("./images", img_name)
        handler_path = os.path.join("./images", self.name, img_name)

        if os.path.isfile(handler_path):
            img_src = os.path.join("/img", self.name, img_name)
        elif os.path.isfile(app_path):
            img_src = os.path.join("/img", img_name)
        else:
            raise FileNotFoundError("Could not find " + img_name)

        self.draw("image", src=img_src, x=x, y=y,
                           width=width, height=height, **extra)


    def timer_tick(self):
        self.update()
        self.do_operation("refresh")
        delta = datetime.timedelta(milliseconds = self.interval)
        self.timeout = IOLoop.current().add_timeout(delta, self.timer_tick)

    def setup(self):
        pass

    def update(self):
        pass

    def on_mouse_down(self, event):
        pass

    def on_mouse_up(self, event):
        pass

    def on_mouse_move(self, event):
        pass

    def on_click(self, event):
        pass

    def on_dbl_click(self, event):
        pass

    def on_key_down(self, event):
        pass

    def on_key_up(self, event):
        pass

    def on_key_press(self, event):
        pass
