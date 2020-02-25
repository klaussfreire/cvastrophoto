# -*- coding: utf-8 -*-
import Tkinter as tk
import ttk

from PIL import Image, ImageTk
import threading
import logging
import math
import numpy
import subprocess
import functools

from cvastrophoto.guiding.calibration import norm2, sub
from cvastrophoto.image.rgb import RGB


logger = logging.getLogger(__name__)


def _p(w, *p, **kw):
    w.pack(*p, **kw)
    return w


def _g(w, *p, **kw):
    w.grid(*p, **kw)
    return w


def with_guider(f):
    @functools.wraps(f)
    def decor(self, *p, **kw):
        if self.guider is not None:
            return f(self, *p, **kw)
    return decor


class AsyncTasks(threading.Thread):

    def __init__(self):
        self.wake = threading.Event()
        self._stop = False
        self.busy = False
        threading.Thread.__init__(self)
        self.daemon = True
        self.requests = {}

    def run(self):
        while not self._stop:
            self.wake.wait(1)
            self.wake.clear()

            for key in self.requests.keys():
                task = self.requests.pop(key, None)
                if task is not None:
                    try:
                        self.busy = True
                        task()
                    except Exception:
                        logger.exception("Error performing task")
                        self.busy = False

    def add_request(self, key, fn, *p, **kw):
        self.requests[key] = functools.partial(fn, *p, **kw)
        self.wake.set()

    def stop(self):
        self._stop = True
        self.wake.set()


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

    GUIDE_SPEED_VALUES = (
        '0.5',
        '1.0',
        '2.0',
        '4.0',
        '8.0',
        '15.0',
        '16.0',
    )

    def __init__(self, interactive_guider, master=None):
        tk.Frame.__init__(self, master)

        self.async_executor = AsyncTasks()
        self.async_executor.start()

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
        self.tab_parent = ttk.Notebook(self)
        self.tab_parent.grid(row=0, sticky=tk.EW)

        self.guide_tab = tk.Frame(self.tab_parent)
        self.tab_parent.add(self.guide_tab, text='Guiding')
        self.create_guide_tab(self.guide_tab)

        self.capture_tab = tk.Frame(self.tab_parent)
        self.tab_parent.add(self.capture_tab, text='Capture')
        self.create_capture_tab(self.capture_tab)

        self.goto_tab = tk.Frame(self.tab_parent)
        self.tab_parent.add(self.goto_tab, text='Goto')
        self.create_goto_tab(self.goto_tab)

        self.status_box = tk.Frame(self)
        self.create_status(self.status_box)
        self.status_box.grid(padx=5, row=1, sticky=tk.EW)

    def create_capture_tab(self, box):
        pass

    def create_goto_tab(self, box):
        ra_text_var = tk.StringVar()
        self.goto_ra_label = _g(tk.Label(box, text='RA'), row=0, column=0)
        self.goto_ra = _g(tk.Entry(box, textvar=ra_text_var, width=30), row=0, column=1, sticky=tk.EW)
        self.goto_ra.text = ra_text_var

        dec_text_var = tk.StringVar()
        self.goto_dec_label = _g(tk.Label(box, text='DEC'), row=1, column=0)
        self.goto_dec = _g(tk.Entry(box, textvar=dec_text_var, width=30), row=1, column=1, sticky=tk.EW)
        self.goto_dec.text = dec_text_var

        epoch_text_var = tk.StringVar()
        self.epoch_label = _g(tk.Label(box, text='Speed'), row=2, column=0)
        self.epoch = _g(
            tk.Entry(box, textvar=epoch_text_var, width=30),
            row=2, column=1, sticky=tk.EW)
        self.epoch.text = epoch_text_var

        self.goto = _g(tk.Button(box, text='Goto', command=self.goto), row=3, sticky=tk.EW, columnspan=2)
        self.sync = _g(tk.Button(box, text='Sync', command=self.sync), row=4, sticky=tk.EW, columnspan=2)

        solve_var = tk.BooleanVar()
        solve_var.set(True)
        self.goto_solve = _g(
            tk.Checkbutton(box, text='Use plate solving', variable=solve_var),
            row=5, sticky=tk.EW, columnspan=2)
        self.goto_solve.value = solve_var

        speed_text_var = tk.StringVar()
        speed_text_var.set("0.5")
        self.goto_speed_label = _g(tk.Label(box, text='Speed'), row=6, column=0)
        self.goto_speed = _g(
            ttk.Combobox(
                box, width=5,
                textvariable=speed_text_var, values=self.GUIDE_SPEED_VALUES),
            row=6, column=1, sticky=tk.EW)
        self.goto_speed.text = speed_text_var

    def create_guide_tab(self, box):
        self.snap_box = tk.Frame(box)
        self.zoom_box = tk.Frame(box)
        self.create_snap(self.snap_box, self.zoom_box)
        self.snap_box.grid(row=0, column=0)
        self.zoom_box.grid(row=0, column=1)

        self.gamma_box = tk.Frame(box)
        self.create_gamma(self.gamma_box)
        self.gamma_box.grid(padx=5, row=1, column=0, sticky=tk.EW)

        self.button_box = tk.Frame(box)
        self.create_buttons(self.button_box)
        self.button_box.grid(padx=5, row=2, column=0, columnspan=2, sticky=tk.EW)

    def create_buttons(self, box):
        self.guide_button = _g(
            tk.Button(box, text='Guide', command=self.guide_start),
            column=0, row=0, sticky=tk.NSEW)
        self.stop_button = _g(
            tk.Button(box, text='Stop', command=self.guide_stop),
            column=0, row=1, sticky=tk.NSEW)
        self.calibrate_button = _g(
            tk.Button(box, text='Calibrate', command=self.calibrate),
            column=1, row=0, sticky=tk.NSEW)
        self.ucalibrate_button = _g(
            tk.Button(box, text='Refine cal', command=self.update_calibration),
            column=1, row=1, sticky=tk.NSEW)
        self.solve_button = _g(
            tk.Button(box, text='Platesolve', command=self.platesolve),
            column=3, row=0, sticky=tk.NSEW)

        fullsize_var = tk.BooleanVar()
        fullsize_var.set(False)
        self.fullsize_check = _g(tk.Checkbutton(box, text='Full-size', variable=fullsize_var), column=3, row=1)
        self.fullsize_check.value = fullsize_var

        self.dither_box = _g(
            tk.Frame(box, relief=tk.SUNKEN, borderwidth=1),
            column=4, row=0, rowspan=2, sticky=tk.NSEW, ipadx=3)
        self.dither_label = _g(tk.Label(self.dither_box, text='Dither'))
        self.dither_var = tk.IntVar()
        self.dither_var.set(10)
        self.dither_bar = _g(
            tk.Scale(
                self.dither_box,
                length=100, showvalue=True, to=40.0, variable=self.dither_var,
                orient=tk.HORIZONTAL),
            sticky=tk.NSEW)
        self.dither_bar["from"] = 1.0
        self.dither_button = _g(
            tk.Button(self.dither_box, text='Dither', command=self.dither),
            sticky=tk.NSEW)

        self.capture_box = _g(
            tk.Frame(box, relief=tk.SUNKEN, borderwidth=1),
            column=5, row=0, rowspan=2, sticky=tk.NSEW, ipadx=3)
        self.capture_label = _g(tk.Label(self.capture_box, text='Capture'), columnspan=2)
        self.dither_n_var = tk.IntVar()
        self.dither_n_var.set(10)
        self.dither_n_bar = _g(
            tk.Scale(
                self.capture_box,
                length=100, showvalue=True, to=40.0, variable=self.dither_n_var,
                orient=tk.HORIZONTAL),
            sticky=tk.NSEW, columnspan=2)
        self.dither_n_bar["from"] = 1.0

        self.cap_exposure_var = tk.StringVar()
        self.cap_exposure_var.set(self.DEFAULT_CAP_EXPOSURE)
        self.cap_exposure_combo = _g(
            ttk.Combobox(
                self.capture_box, width=5,
                textvariable=self.cap_exposure_var, values=self.CAP_EXPOSURE_VALUES),
            sticky=tk.NSEW, columnspan=2)
        self.capture_button = _g(
            tk.Button(self.capture_box, text='Capture', command=self.capture),
            row=3, column=0, sticky=tk.NSEW)
        self.stop_capture_button = _g(
            tk.Button(self.capture_box, text='Stop', command=self.stop_capture),
            row=3, column=1, sticky=tk.NSEW)

    def create_gamma(self, box):
        self.bright_label = _g(tk.Label(box, text='Brightness'), column=0, row=0)
        self.bright_var = tk.DoubleVar()
        self.bright_var.set(10.0)
        self.bright_bar = _g(tk.Scale(
            box,
            to=64.0, length=500, resolution=0.1,
            variable=self.bright_var, orient=tk.HORIZONTAL, showvalue=False
        ), column=1, row=0, sticky=tk.EW)
        self.bright_bar["from"] = 1.0

        self.gamma_label = _g(tk.Label(box, text='Gamma'), column=0, row=1)
        self.gamma_var = tk.DoubleVar()
        self.gamma_var.set(3.0)
        self.gamma_bar = _g(tk.Scale(
            box,
            to=6.0, length=500, resolution=0.1,
            variable=self.gamma_var, orient=tk.HORIZONTAL, showvalue=False
        ), column=1, row=1, sticky=tk.EW)
        self.gamma_bar["from"] = 1.1

    @with_guider
    def goto(self):
        ra = self.goto_ra.text.get().strip()
        dec = self.goto_dec.text.get().strip()
        epoch = self.epoch.text.get().strip()
        speed = self.goto_speed.text.get().strip()
        to_ = ','.join(filter(None, [ra, dec, epoch]))
        if self.goto_solve.value.get():
            logger.info("Executing go + platesolve to %s", to_)
            self.async_executor.add_request(
                "goto",
                self.guider.cmd_goto_solve,
                'guide', to_, speed,
            )
        else:
            logger.info("Executing go to %s", to_)
            self.async_executor.add_request(
                "goto",
                self.guider.cmd_goto,
                to_, speed=speed,
            )

    @with_guider
    def sync(self):
        pass

    @with_guider
    def guide_start(self):
        self.guider.cmd_start()

    @with_guider
    def guide_stop(self):
        self.guider.cmd_stop()

    @with_guider
    def calibrate(self):
        self.guider.cmd_calibrate()

    @with_guider
    def update_calibration(self):
        self.guider.cmd_update_calibration()

    @with_guider
    def dither(self):
        self.guider.cmd_dither(self.dither_var.get())

    @with_guider
    def platesolve(self):
        img = self.guider.cmd_annotate()
        if img is not None:
            subprocess.check_call(['xdg-open', img.name])

    @with_guider
    def capture(self):
        self.guider.cmd_capture(
            self.cap_exposure_var.get(),
            self.dither_n_var.get(),
            self.dither_var.get())

    @with_guider
    def stop_capture(self):
        self.guider.cmd_stop_capture()

    def create_snap(self, snapbox, zoombox):
        self.current_snap = _p(tk.Label(snapbox), side='left')
        self.current_snap.bind("<1>", self.snap_click)
        self.current_zoom = _p(tk.Label(zoombox), side='left')

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
        box.grid_columnconfigure(0, weight=1)
        box.grid_columnconfigure(1, weight=2)
        box.grid_columnconfigure(2, weight=1)

        self.status_label = tk.Label(box)
        self.status_label.text = tk.StringVar()
        self.status_label.text.set("Initializing...")
        self.status_label.config(font='Helvetica 18', textvariable=self.status_label.text, relief=tk.RIDGE)
        self.status_label.grid(column=0, row=0, sticky=tk.NSEW)

        self.cap_status_label = tk.Label(box)
        self.cap_status_label.text = tk.StringVar()
        self.cap_status_label.config(font='Helvetica 18', textvariable=self.cap_status_label.text, relief=tk.RIDGE)
        self.cap_status_label.grid(column=1, row=0, sticky=tk.NSEW)

        self.rms_label = tk.Label(box)
        self.rms_label.text = tk.StringVar()
        self.rms_label.text.set('rms=N/A')
        self.rms_label.config(
            font='Helvetica 16 italic',
            textvariable=self.rms_label.text,
            relief=tk.RIDGE)
        self.rms_label.grid(column=2, row=0, sticky=tk.NSEW)

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
