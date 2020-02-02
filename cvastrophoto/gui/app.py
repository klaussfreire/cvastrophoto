# -*- coding: utf-8 -*-
import Tkinter as tk

from PIL import Image, ImageTk
import threading
import logging

from cvastrophoto.guiding.calibration import norm2


logger = logging.getLogger(__name__)


def _p(w, *p, **kw):
    w.pack(*p, **kw)
    return w


class Application(tk.Frame):

    _new_snap = None

    def __init__(self, interactive_guider, master=None):
        tk.Frame.__init__(self, master)

        self.master.title('cvastrophoto')

        self.guider = interactive_guider
        self.master = master
        self.pack()
        self.create_widgets()

        if self.guider is not None:
            self.guider.add_snap_listener(self.update_snap)

        self.master.after(100, self._periodic)

    def create_widgets(self):
        self.snap_box = tk.Frame(self)
        self.create_snap(self.snap_box)
        self.snap_box.pack()

        self.gamma_box = tk.Frame(self)
        self.create_gamma(self.gamma_box)
        self.gamma_box.pack(fill='x', padx=5)

        self.button_box = tk.Frame(self)
        self.create_buttons(self.button_box)
        self.button_box.pack(fill='x', padx=5)

        self.status_box = tk.Frame(self)
        self.create_status(self.status_box)
        self.status_box.pack(side='bottom', fill='x', padx=5)

    def create_buttons(self, box):
        self.guide_button = _p(tk.Button(box, text='Guide', command=self.guide_start), side='left')
        self.stop_button = _p(tk.Button(box, text='Stop', command=self.guide_stop), side='left')
        self.calibrate_button = _p(tk.Button(box, text='Calibrate', command=self.calibrate), side='left')
        self.ucalibrate_button = _p(tk.Button(box, text='Refine cal', command=self.update_calibration), side='left')
        self.dither_box = _p(tk.Frame(box), side='left')
        self.dither_var = tk.IntVar()
        self.dither_var.set(10)
        self.dither_bar = _p(tk.Scale(self.dither_box,
            length=100, showvalue=True, to=40.0, variable=self.dither_var,
            orient=tk.HORIZONTAL))
        self.dither_button = _p(tk.Button(self.dither_box, text='Dither', command=self.dither))
        self.dither_bar["from"] = 1.0

    def create_gamma(self, box):
        self.bright_box = bbox = _p(tk.Frame(box), fill='x')
        self.bright_label = _p(tk.Label(bbox, text='Brightness'), side='left')
        self.bright_var = tk.DoubleVar()
        self.bright_var.set(10.0)
        self.bright_bar = _p(tk.Scale(bbox,
            to=64.0, length=500, resolution=0.1,
            variable=self.bright_var, orient=tk.HORIZONTAL, showvalue=False),
            side='left', fill='x')
        self.bright_bar["from"] = 1.0

        self.bright_box = gbox = _p(tk.Frame(box), fill='x')
        self.gamma_label = _p(tk.Label(gbox, text='Gamma'), side='left')
        self.gamma_var = tk.DoubleVar()
        self.gamma_var.set(3.0)
        self.gamma_bar = _p(tk.Scale(gbox,
            to=6.0, length=500, resolution=0.1,
            variable=self.gamma_var, orient=tk.HORIZONTAL, showvalue=False),
            side='left', fill='x')
        self.gamma_bar["from"] = 1.0

    def guide_start(self):
        if self.guider is not None:
            self.guider.start_guiding()

    def guide_stop(self):
        if self.guider is not None:
            self.guider.stop_guiding()

    def calibrate(self):
        if self.guider is not None:
            self.guider.calibrate()

    def update_calibration(self):
        if self.guider is not None:
            self.guider.update_calibration()

    def dither(self):
        if self.guider is not None:
            self.guider.dither(self.dither_var.get())

    def create_snap(self, box):
        self.current_snap = tk.Label(box)
        self._set_snap_image(Image.new('L', (1280, 1024)))

    def create_status(self, box):
        self.status_label = tk.Label(box)
        self.status_label.text = tk.StringVar()
        self.status_label.text.set("Initializing...")
        self.status_label.config(font='Helvetica 18', textvariable=self.status_label.text)
        self.status_label.pack(side='left')

        self.rms_label = tk.Label(box)
        self.rms_label.text = tk.StringVar()
        self.rms_label.text.set('rms=N/A')
        self.rms_label.config(
            font='Helvetica 16 italic',
            textvariable=self.rms_label.text)
        self.rms_label.pack(side='right', anchor='e')

    def _periodic(self):
        if self._new_snap is not None:
            new_snap = self._new_snap
            self._new_snap = None
            self._update_snap(new_snap)

        if self.guider is not None:
            status = self.guider.guider.state
            if status != self.status_label.text.get():
                self.status_label.text.set(status)

            self.update_rms(self.guider.guider.offsets)

        self.master.after(100, self._periodic)

    def update_rms(self, offsets):
        mags = list(map(norm2, offsets))
        rms = sum(mags) / len(mags)
        self.rms_label.text.set('rms=%.3f' % rms)

    def _update_snap(self, image):
        img = image.get_img(
            bright=self.bright_var.get(),
            gamma=self.gamma_var.get())

        self._set_snap_image(img)

    def _set_snap_image(self, img):
        # Resize to something sensible
        w, h = img.size
        while h > 720 or w > 1280:
            w /= 2
            h /= 2
        if (w, h) != img.size:
            img = img.resize((w, h))

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
