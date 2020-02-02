# -*- coding: utf-8 -*-
import tkinter as tk

import threading

class Application(tk.Frame):

    _new_snap = None

    def __init__(self, interactive_guider, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.current_snap = tk.Label(self)
        self.current_snap["geometry"] = "1280x1024"

    def _periodic(self):
        if self._new_snap is not None:
            self._update_snap(self._new_snap)

        self.master.after(100, self._periodic)

    def _update_snap(self, image):
        self.current_snap["image"] = tk.PhotoImage(data=image.tobytes(encode="PGM"))

    def update_snap(self, image):
        self._new_snap = image


def launch_app(interactive_guider):
    root = tk.Tk()
    app = Application(interactive_guider, master=root)
    app.main_thread = threading.Thread(target=app.mainloop)
    app.main_thread.daemon = True
    app.main_thread.start()
    return app
