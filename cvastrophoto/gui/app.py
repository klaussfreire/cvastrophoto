# -*- coding: utf-8 -*-
import tkinter as tk

from PIL import Image, ImageTk
import threading
import logging


logger = logging.getLogger(__name__)


class Application(tk.Frame):

    _new_snap = None

    def __init__(self, interactive_guider, master=None):
        tk.Frame.__init__(self, master)
        self.guider = interactive_guider
        self.master = master
        self.pack()
        self.create_widgets()

        self.guider.add_snap_listener(self.update_snap)

        self.master.after(100, self._periodic)

    def create_widgets(self):
        self.current_snap = tk.Label(self)

        self._set_snap_image(Image.new('L', (1280, 1024)))

    def _periodic(self):
        if self._new_snap is not None:
            new_snap = self._new_snap
            self._new_snap = None
            self._update_snap(new_snap)

        self.master.after(100, self._periodic)

    def _update_snap(self, image):
        self._set_snap_image(image.get_img(
            bright=self.guider.guider.snap_bright,
            gamma=self.guider.guider.snap_gamma))

    def _set_snap_image(self, img):
        self.current_snap["image"] = image = ImageTk.PhotoImage(img)
        self.current_snap.image = image
        self.current_snap.pack()

    def update_snap(self, image):
        self._new_snap = image

    @staticmethod
    def launch(ready, interactive_guider):
        root = tk.Tk()
        Application.instance = app = Application(interactive_guider, master=root)
        ready.set()
        app.mainloop()


def launch_app(interactive_guider):
    ready = threading.Event()
    Application.main_thread = main_thread = threading.Thread(
        target=Application.launch,
        args=(ready, interactive_guider))
    main_thread.daemon = True
    main_thread.start()
    ready.wait()
    return Application.instance
