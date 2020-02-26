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
import multiprocessing.pool
import skimage.transform

from cvastrophoto.guiding.calibration import norm2, sub
from cvastrophoto.image.rgb import RGB
import cvastrophoto.image
from cvastrophoto.rops.bias import localgradient


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

        self.processing_pool = multiprocessing.pool.ThreadPool(3)

        self.master.title('cvastrophoto')

        self.guider = interactive_guider
        self.master = master
        self.pack()
        self.create_widgets()

        if self.guider is not None:
            self.guider.add_snap_listener(self.update_snap)

        self.zoom_point = (640, 512)
        self.cap_zoom_point = (640, 512)
        black = numpy.zeros(dtype=numpy.uint16, shape=(1024, 1280))
        black_rgb = RGB.from_gray(black, linear=True, autoscale=False)
        self._update_snap(black_rgb)
        self._set_cap_image(black_rgb.get_img())
        self.master.after(100, self._periodic)

        self.skyglow_rop = None
        self.skyglow_model = None

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
        self.cap_box = tk.Frame(box)
        self.cap_zoom_box = tk.Frame(box)
        self.cap_stats_box = tk.Frame(box)
        self.create_cap(self.cap_box, self.cap_zoom_box)
        self.cap_box.grid(row=0, column=0, rowspan=2)
        self.cap_zoom_box.grid(row=0, column=1)
        self.cap_stats_box.grid(row=1, column=1, sticky=tk.NSEW)

        self.cap_gamma_box = tk.Frame(box)
        self.create_gamma(self.cap_gamma_box, prefix='cap_', bright=1.0, gamma=1.8, show=True)
        self.cap_gamma_box.grid(padx=5, row=2, column=0, sticky=tk.EW)

        self.cap_button_box = tk.Frame(box)
        self.create_cap_buttons(self.cap_button_box)
        self.cap_button_box.grid(padx=5, row=3, column=0, columnspan=2, sticky=tk.EW)

        self.create_cap_stats(self.cap_stats_box)

    def create_cap_stats(self, box):
        self.cap_channel_stat_vars = svars = {}

        box.grid_columnconfigure(0, weight=1)
        box.grid_columnconfigure(1, weight=2)
        box.grid_columnconfigure(2, weight=2)
        box.grid_columnconfigure(3, weight=2)

        self.channel_labels = [
            _g(tk.Label(box, text='R', fg='red'), column=1, row=0),
            _g(tk.Label(box, text='G', fg='green'), column=2, row=0),
            _g(tk.Label(box, text='B', fg='blue'), column=3, row=0),
        ]

        self.channel_toggles = {
            'r': tk.BooleanVar(),
            'g': tk.BooleanVar(),
            'b': tk.BooleanVar(),
        }
        for v in self.channel_toggles.values():
            v.set(True)

        self.channel_toggle_checks = {
            'r': _g(tk.Checkbutton(box, variable=self.channel_toggles['r']), column=1, row=6),
            'g': _g(tk.Checkbutton(box, variable=self.channel_toggles['g']), column=2, row=6),
            'b': _g(tk.Checkbutton(box, variable=self.channel_toggles['b']), column=3, row=6),
        }

        var_specs = (
            (1, 'min'),
            (2, 'max'),
            (3, 'mean'),
            (4, 'median'),
            (5, 'std'),
        )
        self.channel_stats_labels = labels = {
            'titles': [
                _g(tk.Label(box, text=vname), column=0, row=row, sticky=tk.W)
                for row, vname in var_specs
            ],
        }

        for column, cname, color in ((1, 'r', 'red'), (2, 'g', 'green'), (3, 'b', 'blue')):
            self.create_channel_cap_stats(
                box,
                column,
                svars.setdefault(cname, {}),
                labels.setdefault(cname, []),
                var_specs,
                color)

    def create_channel_cap_stats(self, box, column, svars, labels, var_specs, color):
        for row, vname in var_specs:
            svars[vname] = v = tk.DoubleVar()
            v.set(0)
            labels.append(_g(tk.Label(box, textvar=v, fg=color), column=column, row=row))

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
        self.create_gamma(self.gamma_box, show=True)
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

    def create_cap_buttons(self, box):
        self.capture_box = _g(
            tk.Frame(box, relief=tk.SUNKEN, borderwidth=1),
            column=0, row=0, rowspan=2, sticky=tk.NSEW, ipadx=3)
        self.capture_label = _g(tk.Label(self.capture_box, text='Sequence'), columnspan=2)
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

        fullsize_var = tk.BooleanVar()
        fullsize_var.set(False)
        self.cap_fullsize_check = _g(
            tk.Checkbutton(box, text='Full-size', variable=fullsize_var),
            column=2, row=0)
        self.cap_fullsize_check.value = fullsize_var

        skyglow_var = tk.BooleanVar()
        skyglow_var.set(False)
        self.cap_skyglow_check = _g(
            tk.Checkbutton(box, text='Remove background', variable=skyglow_var),
            column=2, row=1)
        self.cap_skyglow_check.value = skyglow_var

        self.guide_button = _g(
            tk.Button(box, text='Refresh', command=self.cap_snap_update),
            column=3, row=0, sticky=tk.NSEW)
        self.stop_button = _g(
            tk.Button(box, text='Update bg\nmodel', command=self.cap_bg_update),
            column=3, row=1, sticky=tk.NSEW)

    def create_gamma(self, box, prefix='', bright=10.0, gamma=3.0, show=False):
        bright_label = _g(tk.Label(box, text='Brightness'), column=0, row=0)
        bright_var = tk.DoubleVar()
        bright_var.set(bright)
        bright_bar = _g(tk.Scale(
            box,
            to=64.0, length=500, resolution=0.1,
            variable=bright_var, orient=tk.HORIZONTAL, showvalue=False
        ), column=1, row=0, sticky=tk.EW)
        bright_bar["from"] = 1.0

        if show:
            bright_value_label = _g(tk.Label(box, textvar=bright_var), column=2, row=0)

        gamma_label = _g(tk.Label(box, text='Gamma'), column=0, row=1)
        gamma_var = tk.DoubleVar()
        gamma_var.set(gamma)
        gamma_bar = _g(tk.Scale(
            box,
            to=6.0, length=500, resolution=0.1,
            variable=gamma_var, orient=tk.HORIZONTAL, showvalue=False
        ), column=1, row=1, sticky=tk.EW)
        gamma_bar["from"] = 1.1

        if show:
            gamma_value_label = _g(tk.Label(box, textvar=gamma_var), column=2, row=1)

        setattr(self, prefix + 'bright_label', bright_label)
        setattr(self, prefix + 'bright_var', bright_var)
        setattr(self, prefix + 'bright_bar', bright_bar)
        setattr(self, prefix + 'gamma_label', gamma_label)
        setattr(self, prefix + 'gamma_var', gamma_var)
        setattr(self, prefix + 'gamma_bar', gamma_bar)

        if show:
            setattr(self, prefix + 'bright_value_label', bright_value_label)
            setattr(self, prefix + 'gamma_value_label', gamma_value_label)

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

    def cap_snap_update(self):
        self.async_executor.add_request("cap_snap", self.update_capture, True)

    @with_guider
    def cap_bg_update(self):
        self.skyglow_model = None
        self.cap_snap_update()

    def create_snap(self, snapbox, zoombox):
        self.current_snap = _p(tk.Label(snapbox), side='left')
        self.current_snap.bind("<1>", self.snap_click)
        self.current_zoom = _p(tk.Label(zoombox), side='left')

        self.current_snap.current_gamma = None
        self.current_snap.current_bright = None
        self.current_snap.current_zoom = None

    def create_cap(self, snapbox, zoombox):
        self.current_cap = _p(tk.Label(snapbox), side='left')
        self.current_cap.bind("<1>", self.cap_click)
        self.current_cap_zoom = _p(tk.Label(zoombox), side='left')

        self.current_cap.current_gamma = None
        self.current_cap.current_bright = None
        self.current_cap.current_zoom = None
        self.current_cap.current_skyglow = False
        self.current_cap.current_channels = (True, True, True)
        self.current_cap.raw_image = None
        self.current_cap.debiased_image = None
        self.current_cap.display_image = None

    def snap_click(self, ev):
        self.zoom_point = (
            ev.x * self.current_snap.full_size[0] / self.current_snap.view_size[0],
            ev.y * self.current_snap.full_size[1] / self.current_snap.view_size[1],
        )
        self._update_snap()

    def cap_click(self, ev):
        self.cap_zoom_point = (
            ev.x * self.current_cap.full_size[0] / self.current_cap.view_size[0],
            ev.y * self.current_cap.full_size[1] / self.current_cap.view_size[1],
        )
        self.update_cap_snap(zoom_only=True)

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

        if self.current_cap.debiased_image is not None:
            try:
                self.update_cap_snap()
            except Exception:
                logger.exception("Error updating capture snapshot")

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

    def _shrink_dims(self, dims, maxw=1280, maxh=720):
        w, h = dims
        factor = 1
        while h > maxh or w > maxw:
            w /= 2
            h /= 2
            factor *= 2
        return w, h, factor

    def __set_snap_image(self, img, zoom_point, current_zoom, current_snap, fullsize_check, zoom_only=False):
        zx, zy = zoom_point

        crop_img = img.crop((zx - 128, zy - 128, zx + 128, zy + 128))
        current_zoom["image"] = image = ImageTk.PhotoImage(crop_img)
        current_zoom.image = image

        if zoom_only:
            return

        # Resize to something sensible
        current_snap.full_size = img.size
        if not fullsize_check.value.get():
            w, h, factor = self._shrink_dims(img.size)
            if (w, h) != img.size:
                img = img.resize((w, h), resample=Image.BOX)

        current_snap["image"] = image = ImageTk.PhotoImage(img)
        current_snap.view_size = img.size
        current_snap.image = image

    def _set_snap_image(self, img, **kw):
        self.__set_snap_image(img, self.zoom_point, self.current_zoom, self.current_snap, self.fullsize_check, **kw)

    def update_snap(self, image):
        self._new_snap = image

    def _set_cap_image(self, img, **kw):
        self.__set_snap_image(
            img,
            self.cap_zoom_point, self.current_cap_zoom,
            self.current_cap, self.cap_fullsize_check,
            **kw)

    def update_raw_stats(self, img):
        raw_image = img.rimg.raw_image
        raw_colors = img.rimg.raw_colors
        black_level = img.rimg.black_level_per_channel or [0,0,0,0]
        self.update_channel_stats(raw_image[raw_colors == 0], 'r', black_level[0])
        self.update_channel_stats(raw_image[raw_colors == 1], 'g', black_level[1])
        self.update_channel_stats(raw_image[raw_colors == 2], 'b', black_level[2])

    def update_channel_stats(self, cdata, cname, black):
        stats = self.cap_channel_stat_vars[cname]
        if cdata.dtype.kind == 'u' and cdata.dtype.itemsize <= 2:
            histogram = numpy.bincount(cdata.reshape(cdata.size))
            hnz, = numpy.nonzero(histogram)
            if len(hnz) > 0:
                hvals = numpy.arange(len(histogram))
                chistogram = numpy.cumsum(histogram)
                hsum = chistogram[-1]
                cmin = hnz[0]
                cmax = hnz[-1]
                cmean = (histogram * hvals).sum() / max(1, float(hsum))
                cstd = numpy.sqrt((histogram * numpy.square(hvals - cmean)).sum() / max(1, float(hsum)))
                cmedian = numpy.searchsorted(chistogram, hsum / 2 + 1)
            else:
                cmin = cmax = cmean = cstd = cmedian = 0
        else:
            cmin = cdata.min()
            cmax = cdata.max()
            cmean = numpy.average(cdata)
            cmedian = numpy.median(cdata)
            cstd = numpy.std(cdata)

        stats['min'].set(max(0, cmin - black))
        stats['max'].set(cmax - black)
        stats['mean'].set(int(cmean - black))
        stats['median'].set(int(cmedian - black))
        stats['std'].set(int(cstd))

    def update_capture(self, force=False):
        last_capture = self.guider.last_capture
        if not force and self.current_cap.raw_image is not None and self.current_cap.raw_image.name == last_capture:
            return

        # Load and shrink image
        img = cvastrophoto.image.Image.open(last_capture)
        self.update_raw_stats(img)
        if img.postprocessing_params is not None:
            img.postprocessing_params.half_size = True
        imgpp = img.postprocessed
        del img

        reduce_factor = self._shrink_dims(imgpp.shape[:2], maxw=4000, maxh=3000)[2]

        if reduce_factor > 1:
            imgpp = skimage.transform.downscale_local_mean(
                imgpp,
                (reduce_factor,) * 2 + (1,) * (len(imgpp.shape) - 2))

        new_raw = RGB(
            last_capture, img=imgpp, linear=True, autoscale=False,
            default_pool=self.processing_pool)

        if self.current_cap.raw_image is None:
            self.current_cap.raw_image = new_raw
            self.current_cap.debiased_image = RGB(
                last_capture, img=imgpp.copy(), linear=True, autoscale=False,
                default_pool=self.processing_pool)
        else:
            # Update image in-place to avoid memory leaks if some components
            # already hold a reference to the image
            self.current_cap.raw_image.set_raw_image(new_raw.rimg.raw_image)
        del new_raw

        self.update_cap_snap(reprocess=True)

    def update_cap_snap(self, reprocess=False, zoom_only=False):
        new_bright = self.cap_bright_var.get()
        new_gamma = self.cap_gamma_var.get()
        new_zoom = self.cap_zoom_point
        new_skyglow = self.cap_skyglow_check.value.get()
        new_channels = (
            self.channel_toggles['r'].get(),
            self.channel_toggles['g'].get(),
            self.channel_toggles['b'].get(),
        )

        # Check parameter changes
        needs_reprocess = (
            new_channels != self.current_cap.current_channels
            or new_skyglow != self.current_cap.current_skyglow
        )

        needs_reimage = (
            new_gamma != self.current_cap.current_gamma
            or new_bright != self.current_cap.current_bright
        )

        needs_update = (
            reprocess
            or needs_reimage or needs_reprocess
            or new_zoom != self.current_cap.current_zoom
        )

        if not needs_update:
            return
        elif self.current_cap.debiased_image is None:
            return

        if reprocess or needs_reprocess:
            logger.info("Reprocessing capture snaphost")
            self.current_cap.debiased_image.set_raw_image(
                self.current_cap.raw_image.rimg.raw_image)
            if new_skyglow:
                self.update_skyglow_model()

                self.current_cap.debiased_image.set_raw_image(
                    self.skyglow_rop.correct(
                        self.current_cap.debiased_image.rimg.raw_image,
                        self.skyglow_model
                    )
                )

            rcheck = self.channel_toggles['r'].get()
            gcheck = self.channel_toggles['g'].get()
            bcheck = self.channel_toggles['b'].get()
            if not rcheck or not gcheck or not bcheck:
                raw_image = self.current_cap.debiased_image.rimg.raw_image
                raw_colors = self.current_cap.debiased_image.rimg.raw_colors
                if rcheck + gcheck + bcheck == 1:
                    # Show grayscale channel
                    if rcheck:
                        c = 0
                    elif gcheck:
                        c = 1
                    elif bcheck:
                        c = 2
                    cimage = raw_image[raw_colors == c]
                    raw_image[raw_colors == 0] = cimage
                    raw_image[raw_colors == 1] = cimage
                    raw_image[raw_colors == 2] = cimage
                else:
                    # Zero unselected channels
                    if not rcheck:
                        raw_image[raw_colors == 0] = 0
                    if not gcheck:
                        raw_image[raw_colors == 1] = 0
                    if not bcheck:
                        raw_image[raw_colors == 2] = 0

            self.current_cap.display_image = None
        if self.current_cap.display_image is None or needs_reimage:
            self.current_cap.display_image = self.current_cap.debiased_image.get_img(
                bright=new_bright,
                gamma=new_gamma)
        self._set_cap_image(self.current_cap.display_image, zoom_only=zoom_only)

        self.current_cap.current_gamma = new_gamma
        self.current_cap.current_bright = new_bright
        self.current_cap.current_zoom = new_zoom
        self.current_cap.current_skyglow = new_skyglow
        self.current_cap.current_channels = new_channels

    def update_skyglow_model(self):
        if self.skyglow_rop is None:
            self.skyglow_rop = localgradient.QuickGradientBiasRop(self.current_cap.raw_image, copy=False)
        if self.skyglow_model is None:
            logger.info("Updating skyglow model")
            self.skyglow_model = self.skyglow_rop.detect(self.current_cap.raw_image.rimg.raw_image)

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
