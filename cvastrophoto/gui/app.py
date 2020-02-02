# -*- coding: utf-8 -*-
import tkinter as tk

import threading

class Application(tk.Frame):

    _new_snap = None

    def __init__(self, interactive_guider, master=None):
        super(Application, self).__init__(master)
        self.guider = interactive_guider
        self.master = master
        self.pack()
        self.create_widgets()

        self.guider.guider.add_snap_listener(self.update_snap)

    def create_widgets(self):
        self.current_snap = tk.Label(self)
        self.current_snap["geometry"] = "1280x1024"

    def _periodic(self):
        if self._new_snap is not None:
            new_snap = self._new_snap
            self._new_snap = None
            self._update_snap(new_snap)

        self.master.after(100, self._periodic)

    def _update_snap(self, image):
        img = image.get_img(
            bright=self.guider.guider.snap_bright,
            gamma=self.guider.guider.snap_gamma)
        self.current_snap["image"] = tk.PhotoImage(data=img.tobytes(encode="PGM"))

    def update_snap(self, image):
        self._new_snap = image


def launch_app(interactive_guider):
    root = tk.Tk()
    app = Application(interactive_guider, master=root)
    app.main_thread = threading.Thread(target=app.mainloop)
    app.main_thread.daemon = True
    app.main_thread.start()
    return app
