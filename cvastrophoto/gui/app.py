# -*- coding: utf-8 -*-
import Tkinter as tk
import ttk

from PIL import Image, ImageTk
import threading
import logging
import math
import numpy
import subprocess

from cvastrophoto.guiding.calibration import norm2, sub
from cvastrophoto.image.rgb import RGB


logger = logging.getLogger(__name__)


def _p(w, *p, **kw):
    w.pack(*p, **kw)
    return w


class Application(tk.Frame):

    _new_snap = None

    DEFAULT_CAP_EXPOSURE = '60'
    CAP_EXPOSURE_VALUES = (
        '1',
        '2',
        '4',
        '6',
        '8',
        '10',
        '15',
        '20',
        '30',
        '45',
        '60',
        '90',
        '120',
        '180',
        '240',
        '300',
        '360',
        '480',
        '600',
        '900',
        '1200',
    )

    def __init__(self, interactive_guider, master=None):
        tk.Frame.__init__(self, master)

        self.master.title('cvastrophoto')

        self.guider = interactive_guider
        self.master = master
        self.pack()
        self.create_widgets()

        if self.guider is not None:
            self.guider.add_snap_listener(self.update_snap)

        self.zoom_point = (640, 512)
        black = numpy.zeros(dtype=numpy.uint16, shape=(1024, 1280))
        self._update_snap(RGB.from_gray(black, linear=True, autoscale=False))
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
        self.solve_button = _p(tk.Button(box, text='Platesolve', command=self.platesolve), side='left')

        self.dither_box = _p(tk.Frame(box), side='left')
        self.dither_var = tk.IntVar()
        self.dither_var.set(10)
        self.dither_bar = _p(tk.Scale(self.dither_box,
            length=100, showvalue=True, to=40.0, variable=self.dither_var,
            orient=tk.HORIZONTAL))
        self.dither_button = _p(tk.Button(self.dither_box, text='Dither', command=self.dither))
        self.dither_bar["from"] = 1.0

        fullsize_var = tk.BooleanVar()
        fullsize_var.set(False)
        self.fullsize_check = _p(tk.Checkbutton(box, text='Full-size', variable=fullsize_var), side='left')
        self.fullsize_check.value = fullsize_var

        self.capture_box = _p(tk.Frame(box), side='left')
        self.dither_n_var = tk.IntVar()
        self.dither_n_var.set(10)
        self.dither_n_bar = _p(tk.Scale(self.capture_box,
            length=100, showvalue=True, to=40.0, variable=self.dither_n_var,
            orient=tk.HORIZONTAL))
        self.dither_n_bar["from"] = 1.0
        self.cap_exposure_var = tk.StringVar()
        self.cap_exposure_var.set(self.DEFAULT_CAP_EXPOSURE)
        self.cap_exposure_combo = _p(ttk.Combobox(self.capture_box, width=5,
            textvariable=self.cap_exposure_var, values=self.CAP_EXPOSURE_VALUES))
        self.capture_button = _p(tk.Button(self.capture_box, text='Capture', command=self.capture), side='left')
        self.stop_capture_button = _p(tk.Button(self.capture_box, text='Stop', command=self.stop_capture), side='left')

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
        self.gamma_bar["from"] = 1.1

    def guide_start(self):
        if self.guider is not None:
            self.guider.cmd_start()

    def guide_stop(self):
        if self.guider is not None:
            self.guider.cmd_stop()

    def calibrate(self):
        if self.guider is not None:
            self.guider.cmd_calibrate()

    def update_calibration(self):
        if self.guider is not None:
            self.guider.cmd_update_calibration()

    def dither(self):
        if self.guider is not None:
            self.guider.cmd_dither(self.dither_var.get())

    def platesolve(self):
        if self.guider is not None:
            img = self.guider.cmd_annotate()
            if img is not None:
                subprocess.check_call(['xdg-open', img.name])

    def capture(self):
        if self.guider is not None:
            self.guider.cmd_capture(
                self.cap_exposure_var.get(),
                self.dither_n_var.get(),
                self.dither_var.get())

    def stop_capture(self):
        if self.guider is not None:
            self.guider.cmd_stop_capture()

    def create_snap(self, box):
        self.current_snap = _p(tk.Label(box), side='left')
        self.current_snap.bind("<1>", self.snap_click)
        self.current_zoom = _p(tk.Label(box), side='left')

        self.current_snap.current_gamma = None
        self.current_snap.current_bright = None
        self.current_snap.current_zoom = None

    def snap_click(self, ev):
        self.zoom_point = (
            ev.x * self.current_snap.full_size[0] / self.current_snap.view_size[0],
            ev.y * self.current_snap.full_size[1] / self.current_snap.view_size[1],
        )
        self._update_snap()

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

        self.cap_status_label = tk.Label(box)
        self.cap_status_label.text = tk.StringVar()
        self.cap_status_label.config(font='Helvetica 18', textvariable=self.cap_status_label.text)
        self.cap_status_label.pack()

    def _periodic(self):
        if self._new_snap is not None:
            new_snap = self._new_snap
            self._new_snap = None
        else:
            new_snap = None
        self._update_snap(new_snap)

        if self.guider is not None:
            status = self.guider.guider.state
            if status != self.status_label.text.get():
                self.status_label.text.set(status)

            if self.guider.capture_seq is not None:
                cap_status = self.guider.capture_seq.state
                if cap_status != self.cap_status_label.text.get():
                    self.cap_status_label.text.set(cap_status)

            self.update_rms(self.guider.guider.offsets)

        self.master.after(100, self._periodic)

    def update_rms(self, offsets):
        if offsets:
            off0 = offsets[0]
            mags = [norm2(sub(off, off0)) for off in offsets]
            rms = math.sqrt(sum(mags) / len(mags))
            self.rms_label.text.set('rms=%.3f' % rms)
        else:
            self.rms_label.text.set('rms=N/A')

    def _update_snap(self, image=None):
        if image is not None:
            self.snap_img = image
            needs_update = True
        else:
            image = self.snap_img
            needs_update = False

        new_bright = self.bright_var.get()
        new_gamma = self.gamma_var.get()
        new_zoom = self.zoom_point

        # Check parameter changes
        needs_update = (
            needs_update
            or new_gamma != self.current_snap.current_gamma
            or new_bright != self.current_snap.current_bright
            or new_zoom != self.current_snap.current_zoom
        )

        if not needs_update:
            return

        img = image.get_img(
            bright=new_bright,
            gamma=new_gamma,
            component=0)

        self._set_snap_image(img)

        self.current_snap.current_gamma = new_gamma
        self.current_snap.current_bright = new_bright
        self.current_snap.current_zoom = new_zoom

    def _set_snap_image(self, img):
        zx, zy = self.zoom_point

        crop_img = img.crop((zx - 128, zy - 128, zx + 128, zy + 128))
        self.current_zoom["image"] = image = ImageTk.PhotoImage(crop_img)
        self.current_zoom.image = image

        # Resize to something sensible
        self.current_snap.full_size = img.size
        if not self.fullsize_check.value.get():
            w, h = img.size
            while h > 720 or w > 1280:
                w /= 2
                h /= 2
            if (w, h) != img.size:
                img = img.resize((w, h), resample=Image.BOX)

        self.current_snap["image"] = image = ImageTk.PhotoImage(img)
        self.current_snap.view_size = img.size
        self.current_snap.image = image

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
