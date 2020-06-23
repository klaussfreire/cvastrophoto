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

    DEFAULT_FLAT_EXPOSURE = '1'
    FLAT_EXPOSURE_VALUES = (
        '0.1',
        '0.15',
        '0.2',
        '0.25',
        '0.4',
        '0.5',
        '0.8',
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

    PERIODIC_MS = 100
    SLOWPERIODIC_MS = 570

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
        self.master.after(self.PERIODIC_MS, self._periodic)
        self.master.after(self.SLOWPERIODIC_MS, self._slowperiodic)

        self.skyglow_rop = None
        self.skyglow_model = None

    def create_widgets(self):
        self.tab_parent = ttk.Notebook(self)
        self.tab_parent.grid(row=0, sticky=tk.EW)

        self.guide_tab_index = self.tab_parent.index('end')
        self.guide_tab = tk.Frame(self.tab_parent)
        self.tab_parent.add(self.guide_tab, text='Guiding')
        self.create_guide_tab(self.guide_tab)

        self.capture_tab_index = self.tab_parent.index('end')
        self.capture_tab = tk.Frame(self.tab_parent)
        self.tab_parent.add(self.capture_tab, text='Capture')
        self.create_capture_tab(self.capture_tab)

        self.goto_tab = tk.Frame(self.tab_parent)
        self.tab_parent.add(self.goto_tab, text='Goto')
        self.create_goto_tab(self.goto_tab)

        self.ccd_tab = tk.Frame(self.tab_parent)
        self.tab_parent.add(self.ccd_tab, text='CCD')
        self.create_ccd_tab(self.ccd_tab)

        self.equipment_tab = tk.Frame(self.tab_parent)
        self.tab_parent.add(self.equipment_tab, text='Equipment')
        self.create_equipment_tab(self.equipment_tab)

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

        var_specs = (
            (1, 'min'),
            (2, 'max'),
            (3, 'mean'),
            (4, 'median'),
            (5, 'std'),
            (6, '% sat'),
        )
        show_row = max(row for row, vname in var_specs) + 1

        self.channel_toggle_checks = {
            'r': _g(tk.Checkbutton(box, variable=self.channel_toggles['r']), column=1, row=show_row),
            'g': _g(tk.Checkbutton(box, variable=self.channel_toggles['g']), column=2, row=show_row),
            'b': _g(tk.Checkbutton(box, variable=self.channel_toggles['b']), column=3, row=show_row),
        }

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

        temp_var = tk.StringVar()
        self.temp_label = _g(tk.Label(box, text='Temp'), column=0, row=show_row + 1)
        self.temp_value = _g(tk.Label(box, textvar=temp_var), column=1, row=show_row + 1)
        self.temp_value.text = temp_var

    def create_channel_cap_stats(self, box, column, svars, labels, var_specs, color):
        for row, vname in var_specs:
            svars[vname] = v = tk.DoubleVar()
            v.set(0)
            labels.append(_g(tk.Label(box, textvar=v, fg=color), column=column, row=row))

    def create_goto_tab(self, box):
        self.gotocmdbox = _g(tk.Frame(box), row=0, column=0, sticky=tk.NW)
        self.gotoinfobox = _g(tk.Frame(box, relief=tk.SUNKEN, borderwidth=1), row=0, column=1, sticky=tk.NE)

        self.create_goto_cmd_box(self.gotocmdbox)
        self.create_goto_info_box(self.gotoinfobox)

    def create_goto_cmd_box(self, box):
        ra_text_var = tk.StringVar()
        self.goto_ra_label = _g(tk.Label(box, text='RA'), row=0, column=0)
        self.goto_ra = _g(tk.Entry(box, textvar=ra_text_var, width=30), row=0, column=1, sticky=tk.EW)
        self.goto_ra.text = ra_text_var

        dec_text_var = tk.StringVar()
        self.goto_dec_label = _g(tk.Label(box, text='DEC'), row=1, column=0)
        self.goto_dec = _g(tk.Entry(box, textvar=dec_text_var, width=30), row=1, column=1, sticky=tk.EW)
        self.goto_dec.text = dec_text_var

        epoch_text_var = tk.StringVar()
        self.epoch_label = _g(tk.Label(box, text='Epoch'), row=2, column=0)
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

    def create_goto_info_box(self, box):
        self.goto_info_title = _g(tk.Label(box, text='Current mount status'), row=0, column=0)

        ra_value = tk.StringVar()
        self.goto_info_ra_label = _g(tk.Label(box, text='RA'), row=1, column=0)
        self.goto_info_ra_value = _g(tk.Label(box, textvar=ra_value), row=1, column=1)
        self.goto_info_ra_value.text = ra_value

        dec_value = tk.StringVar()
        self.goto_info_dec_label = _g(tk.Label(box, text='DEC'), row=2, column=0)
        self.goto_info_dec_value = _g(tk.Label(box, textvar=dec_value), row=2, column=1)
        self.goto_info_dec_value.text = dec_value

    def update_goto_info_box(self):
        eff_telescope_coords = self.guider.guider.calibration.eff_telescope_coords
        if eff_telescope_coords is None:
            eff_telescope_coords = ['N/A', 'N/A']

        self.goto_info_ra_value.text.set(eff_telescope_coords[0])
        self.goto_info_dec_value.text.set(eff_telescope_coords[1])

    def create_ccd_tab(self, box):
        box.grid_columnconfigure(0, weight=1)
        self.gccd_info_box = _g(CCDInfoBox(box, 'Guider'), row=0, sticky=tk.EW, ipadx=3, ipady=3, pady=5)
        self.iccd_info_box = _g(CCDInfoBox(box, 'Imaging'), row=1, sticky=tk.EW, ipadx=3, ipady=3, pady=5)

    def create_equipment_tab(self, box):
        box.grid_columnconfigure(0, weight=1)
        self.devices = {}
        self.devices_nb = _g(EquipmentNotebook(box, self.guider), sticky=tk.NSEW)

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

    def create_cap_buttons(self, box):
        self.dither_box = _g(
            tk.Frame(box, relief=tk.SUNKEN, borderwidth=1),
            column=0, row=0, rowspan=2, sticky=tk.NSEW, ipadx=3)
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
            column=1, row=0, rowspan=2, sticky=tk.NSEW, ipadx=3)
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

        self.flat_box = _g(
            tk.Frame(box, relief=tk.SUNKEN, borderwidth=1),
            column=2, row=0, rowspan=2, sticky=tk.NSEW, ipadx=3)
        self.flat_capture_label = _g(tk.Label(self.flat_box, text='Flats'), columnspan=2)

        self.flat_n_label = _g(tk.Label(self.flat_box, text='N'), row=1, column=0)
        self.flat_n_var = tk.IntVar()
        self.flat_n_var.set(30)
        self.flat_n_spinner = _g(
            tk.Spinbox(self.flat_box, textvariable=self.flat_n_var, width=4, from_=1, to=100),
            sticky=tk.NSEW, row=1, column=1)

        self.flat_exposure_var = tk.StringVar()
        self.flat_exposure_var.set(self.DEFAULT_FLAT_EXPOSURE)
        self.flat_exposure_combo = _g(
            ttk.Combobox(
                self.flat_box, width=5,
                textvariable=self.flat_exposure_var, values=self.FLAT_EXPOSURE_VALUES),
            sticky=tk.NSEW, columnspan=2)
        self.flat_capture_button = _g(
            tk.Button(self.flat_box, text='Capture', command=self.capture_flats),
            row=3, column=0, sticky=tk.NSEW)
        self.stop_flat_capture_button = _g(
            tk.Button(self.flat_box, text='Stop', command=self.stop_capture),
            row=3, column=1, sticky=tk.NSEW)

        self.dark_box = _g(
            tk.Frame(box, relief=tk.SUNKEN, borderwidth=1),
            column=3, row=0, rowspan=2, sticky=tk.NSEW, ipadx=3)
        self.dark_capture_label = _g(tk.Label(self.dark_box, text='Darks'), columnspan=2)

        self.dark_n_label = _g(tk.Label(self.dark_box, text='N darks'), row=1, column=0)
        self.dark_n_var = tk.IntVar()
        self.dark_n_var.set(30)
        self.dark_n_spinner = _g(
            tk.Spinbox(self.dark_box, textvariable=self.dark_n_var, width=4, from_=1, to=100),
            sticky=tk.NSEW, row=1, column=1)
        self.dark_capture_button = _g(
            tk.Button(self.dark_box, text='Capture', command=self.capture_darks),
            row=1, column=3, sticky=tk.NSEW)

        self.dark_flat_n_label = _g(tk.Label(self.dark_box, text='N flats'), row=2, column=0)
        self.dark_flat_n_var = tk.IntVar()
        self.dark_flat_n_var.set(30)
        self.dark_flat_n_spinner = _g(
            tk.Spinbox(self.dark_box, textvariable=self.dark_flat_n_var, width=4, from_=1, to=100),
            sticky=tk.NSEW, row=2, column=1)
        self.dark_flat_capture_button = _g(
            tk.Button(self.dark_box, text='Capture', command=self.capture_dark_flats),
            row=2, column=3, sticky=tk.NSEW)

        self.stop_capture_button = _g(
            tk.Button(self.dark_box, text='Stop', command=self.stop_capture),
            row=3, column=0, columnspan=3, sticky=tk.NSEW)

        fullsize_var = tk.BooleanVar()
        fullsize_var.set(False)
        self.cap_fullsize_check = _g(
            tk.Checkbutton(box, text='Full-size', variable=fullsize_var),
            column=4, row=0)
        self.cap_fullsize_check.value = fullsize_var

        skyglow_var = tk.BooleanVar()
        skyglow_var.set(False)
        self.cap_skyglow_check = _g(
            tk.Checkbutton(box, text='Remove background', variable=skyglow_var),
            column=4, row=1)
        self.cap_skyglow_check.value = skyglow_var

        self.cap_update_button = _g(
            tk.Button(box, text='Refresh', command=self.cap_snap_update),
            column=5, row=0, sticky=tk.NSEW)
        self.bg_update_button = _g(
            tk.Button(box, text='Update bg\nmodel', command=self.cap_bg_update),
            column=5, row=1, sticky=tk.NSEW)

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

    @with_guider
    def capture_flats(self):
        self.guider.cmd_capture_flats(
            self.flat_exposure_var.get(),
            self.flat_n_var.get())

    @with_guider
    def capture_darks(self):
        self.guider.cmd_capture_darks(
            self.cap_exposure_var.get(),
            self.dark_n_var.get())

    @with_guider
    def capture_dark_flats(self):
        self.guider.cmd_capture_dark_flats(
            self.flat_exposure_var.get(),
            self.dark_flat_n_var.get())

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

    @with_guider
    def update_cap_info_box(self):
        if not self.guider.capture_seq:
            return

        ccd = self.guider.capture_seq.ccd
        self.temp_value.text.set(ccd.properties.get('CCD_TEMPERATURE', ['N/A'])[0])

    @with_guider
    def update_iccd_info_box(self):
        if not self.guider.capture_seq:
            return

        self.iccd_info_box.update(self.guider.capture_seq.ccd)

    @with_guider
    def update_gccd_info_box(self):
        self.gccd_info_box.update(self.guider.guider.ccd)

    def update_equipment_info_box(self):
        pass

    def _periodic(self):
        updates = [
            self.__update_snap,
            self.__update_cap,
            self.update_goto_info_box,
            self.update_cap_info_box,
        ]
        if self.guider is not None:
            updates += [
                self.__update_state,
            ]

        for updatefn in updates:
            try:
                updatefn()
            except Exception:
                logger.exception("Error in periodic update")

        self.master.after(self.PERIODIC_MS, self._periodic)

    def _slowperiodic(self):
        updates = []
        if self.guider is not None:
            updates += [
                self.update_iccd_info_box,
                self.update_gccd_info_box,
                self.update_equipment_info_box,
            ]

        for updatefn in updates:
            try:
                updatefn()
            except Exception:
                logger.exception("Error in periodic update")

        self.master.after(self.SLOWPERIODIC_MS, self._slowperiodic)

    def __update_snap(self):
        if self.tab_parent.index('current') != self.guide_tab_index:
            return
        if self._new_snap is not None:
            new_snap = self._new_snap
            self._new_snap = None
        else:
            new_snap = None
        self._update_snap(new_snap)

    def __update_state(self):
        status = self.guider.guider.state
        status_detail = self.guider.guider.state_detail
        if status_detail is not None:
            status = '%s (%s)' % (status, status_detail)
        if status != self.status_label.text.get():
            self.status_label.text.set(status)

        if self.guider.capture_seq is not None:
            cap_status = self.guider.capture_seq.state
            cap_detail = self.guider.capture_seq.state_detail
            if cap_detail:
                cap_status = '%s (%s)' % (cap_status, cap_detail)
            if cap_status != self.cap_status_label.text.get():
                self.cap_status_label.text.set(cap_status)

        self.update_rms(self.guider.guider.offsets)

    def __update_cap(self):
        if self.current_cap.debiased_image is not None:
            self.async_executor.add_request("cap_snap_upd", self.update_cap_snap)

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
        white = img.rimg.raw_image.max()
        self.update_channel_stats(raw_image, raw_image[raw_colors == 0], 'r', black_level[0], white)
        self.update_channel_stats(raw_image, raw_image[raw_colors == 1], 'g', black_level[1], white)
        self.update_channel_stats(raw_image, raw_image[raw_colors == 2], 'b', black_level[2], white)

    def update_channel_stats(self, data, cdata, cname, black, white):
        stats = self.cap_channel_stat_vars[cname]

        # Some margin
        white = white * 90 / 100

        if cdata.dtype.kind == 'u' and cdata.dtype.itemsize <= 2:
            histogram = numpy.bincount(cdata.reshape(cdata.size))
            hnz, = numpy.nonzero(histogram)
            if len(hnz) > 0:
                hvals = numpy.arange(len(histogram), dtype=numpy.float32)
                chistogram = numpy.cumsum(histogram)
                hsum = chistogram[-1]
                cmin = hnz[0]
                cmax = hnz[-1]
                cmean = (histogram * hvals).sum() / max(1, float(hsum))
                cstd = numpy.sqrt((histogram * numpy.square(hvals - cmean)).sum() / max(1, float(hsum)))
                cmedian = numpy.searchsorted(chistogram, hsum / 2 + 1)
                if white < len(histogram):
                    csat = (hsum - chistogram[white]) / max(1, float(hsum))
                else:
                    csat = 0
            else:
                cmin = cmax = cmean = cstd = cmedian = csat = 0
        else:
            cmin = cdata.min()
            cmax = cdata.max()
            cmean = numpy.average(cdata)
            cmedian = numpy.median(cdata)
            cstd = numpy.std(cdata)
            csat = numpy.count_nonzero(cdata >= white)

        stats['min'].set(max(0, cmin - black))
        stats['max'].set(cmax - black)
        stats['mean'].set(int(cmean - black))
        stats['median'].set(int(cmedian - black))
        stats['std'].set(int(cstd))
        stats['% sat'].set(int(csat * 10000) / 100.0)

    def update_capture(self, force=False):
        last_capture = self.guider.last_capture
        if last_capture is None:
            return
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

        rcheck = self.channel_toggles['r'].get()
        gcheck = self.channel_toggles['g'].get()
        bcheck = self.channel_toggles['b'].get()
        new_channels = (rcheck, gcheck, bcheck)

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

        # Avoid re-entry unless something changes
        self.current_cap.current_gamma = new_gamma
        self.current_cap.current_bright = new_bright
        self.current_cap.current_zoom = new_zoom
        self.current_cap.current_skyglow = new_skyglow
        self.current_cap.current_channels = new_channels

        if self.current_cap.debiased_image is None:
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


class CCDInfoBox(tk.Frame):

    COOLING_UPDATE_PERIOD_MS = 20000

    def __init__(self, box, title_prefix, *p, **kw):
        self.title_prefix = title_prefix
        self.ccd = None
        tk.Frame.__init__(self, box, *p, relief=tk.SUNKEN, borderwidth=1, **kw)

        var = tk.StringVar()
        var.set(title_prefix)
        self.boxlabel = _g(tk.Label(self, textvar=var, font='Helvetica 18 bold'), row=0, pady=5)
        self.boxlabel.value = var

        self.temp_box = _g(
            tk.Frame(self, relief=tk.SUNKEN, borderwidth=1),
            column=0, row=1, sticky=tk.NSEW, ipadx=3, ipady=3, padx=5)
        self.temp_boxlabel = _g(tk.Label(self.temp_box, text='Temperature'), row=0, columnspan=2, pady=3)

        curvalue = tk.StringVar()
        tgtvalue = tk.StringVar()
        self.temp_curlabel = _g(tk.Label(self.temp_box, text='Current'), row=1, column=0)
        self.temp_curvalue = _g(tk.Label(self.temp_box, textvar=curvalue, font='Helvetica 18'), row=2, column=0)
        self.temp_curvalue.value = curvalue
        self.temp_tgtlabel = _g(tk.Label(self.temp_box, text='Target'), row=1, column=1)
        self.temp_tgtvalue = _g(tk.Label(self.temp_box, textvar=tgtvalue, font='Helvetica 18'), row=2, column=1)
        self.temp_tgtvalue.value = tgtvalue

        self.cool_box = _g(
            tk.Frame(self, relief=tk.SUNKEN, borderwidth=1),
            column=1, row=1, sticky=tk.NSEW, ipadx=3, ipady=3, padx=5)
        self.cool_box.visible = True
        self.cool_boxlabel = _g(tk.Label(self.cool_box, text='Cooling'), row=0, columnspan=3, pady=3)

        self.cool_enable = _g(
            tk.Button(self.cool_box, text='Enable', command=self.switch_cooling),
            row=1, columnspan=3, pady=3, sticky=tk.NSEW)

        var = tk.DoubleVar()
        var.set(20)
        self.cool_setlabel = _g(tk.Label(self.cool_box, text='Target'), row=2, column=0, padx=5)
        self.cool_setvalue = _g(
            tk.Spinbox(self.cool_box, textvariable=var, width=3, from_=-20, to=20),
            sticky=tk.NSEW, row=2, column=1)
        self.cool_setvalue.value = var
        self.cool_set_target = None
        self.cool_setbtn = _g(tk.Button(self.cool_box, text='Set', command=self.advance_cooling), row=2, column=2)

    def switch_cooling(self):
        ccd = self.ccd
        if not ccd:
            return

        if ccd.cooling_enabled:
            # Turn off
            ccd.disable_cooling(quick=True, optional=True)
            self.cool_enable.configure(text='Disabling', state=tk.ACTIVE)
        else:
            # Turn on
            # Can't go directly to the set temperature, must let it drift there gradually
            # so just set the current temperature as target, and let the slow tick
            # gradually push that to our desired set temperature over time
            self.start_cooling()
            self.cool_enable.configure(text='Enabling', state=tk.NORMAL)

    def start_cooling(self):
        set_temp = self.cool_setvalue.value.get()
        try:
            self.cool_set_target = max(set_temp, float(self.temp_curvalue.value.get()) - 1)
        except (ValueError, TypeError):
            self.cool_set_target = set_temp
        logger.info("Setting cooling temperature of %r to %r (set %r)", self.ccd.name, self.cool_set_target, set_temp)
        self.ccd.enable_cooling(self.cool_set_target, quick=True, optional=True)
        self.master.after(self.COOLING_UPDATE_PERIOD_MS, self.advance_cooling)

    def advance_cooling(self):
        ccd = self.ccd
        if ccd is None or not ccd.cooling_enabled:
            return

        props = ccd.properties
        cur_temp = props.get('CCD_TEMPERATURE', (None,))[0]
        tgt_temp = self.cool_set_target
        set_temp = self.cool_setvalue.value.get()

        if tgt_temp is None or cur_temp is None:
            tgt_temp = cur_temp

        if tgt_temp == set_temp:
            logger.info("Set temperature reached on %r", self.ccd.name)
            return

        if abs(cur_temp - tgt_temp) < 0.5:
            if set_temp > tgt_temp:
                tgt_temp = min(set_temp, tgt_temp + 1)
            elif set_temp < tgt_temp:
                tgt_temp = max(set_temp, tgt_temp - 1)
            else:
                tgt_temp = set_temp
            logger.info("Setting cooling temperature of %r to %r (set %r)", self.ccd.name, tgt_temp, set_temp)
            ccd.set_cooling_temp(tgt_temp, quick=True, optional=True)
            self.cool_set_target = tgt_temp

        self.master.after(self.COOLING_UPDATE_PERIOD_MS, self.advance_cooling)

    def update(self, ccd):
        self.ccd = ccd

        if not ccd:
            return

        self.boxlabel.value.set('%s: %s' % (self.title_prefix, self.ccd.name))
        self.temp_curvalue.value.set(ccd.properties.get('CCD_TEMPERATURE', ['-'])[0])

        if ccd.supports_cooling:

            if not self.cool_box.visible:
                self.cool_box.grid()
                self.cool_box.visible = True
            if ccd.cooling_enabled:
                self.cool_enable.configure(text='Enabled', state=tk.ACTIVE)
                self.temp_tgtvalue.value.set(self.cool_set_target if self.cool_set_target is not None else 'N/S')
            else:
                self.cool_enable.configure(text='Disabled', state=tk.NORMAL)
                self.temp_tgtvalue.value.set('-')
        elif self.cool_box.visible:
            self.temp_tgtvalue.value.set('-')
            self.cool_box.grid_remove()
            self.cool_box.visible = False

    def cool_step(self):
        ccd = self.ccd
        if not ccd or not ccd.cooling_enabled:
            return


class EquipmentNotebook(ttk.Notebook):

    UPDATE_PERIOD_MS = 1770

    def __init__(self, box, guider, *p, **kw):
        self.guider = guider
        ttk.Notebook.__init__(self, box, *p, **kw)

        self.devices = {}

        self.master.after(self.UPDATE_PERIOD_MS, self.refresh)

    def refresh(self):
        # Gather devices - only those that are interesting
        devices = {}

        def add(root, *path):
            if root is None:
                return
            for k in path:
                root = getattr(root, k, None)
                if root is None:
                    return
            devices[root.name] = root

        add(self.guider, 'guider', 'ccd')
        add(self.guider, 'guider', 'telescope')
        add(self.guider, 'guider', 'controller', 'telescope')
        add(self.guider, 'guider', 'controller', 'st4')
        add(self.guider, 'capture_seq', 'ccd')

        for dname in self.devices:
            if dname not in devices:
                self.remove_device(dname)
        for dname in devices:
            if dname not in self.devices:
                self.add_device(dname, devices[dname])
            else:
                self.devices[dname].refresh()

    def remove_device(self, dname):
        dev = self.devices.get(dname)
        if dev is not None:
            dev.destroy()

    def add_device(self, dname, device):
        if dname not in self.devices:
            self.devices[dname] = DeviceControlSet(self, dname, device)


class DeviceControlSet(ttk.Notebook):

    def __init__(self, tab_parent, name, device, *p, **kw):
        self.device_name = name
        self.device = device
        self.parent_tab_index = tab_parent.index('end')
        ttk.Notebook.__init__(self, tab_parent, *p, **kw)
        tab_parent.add(self, text=name)

        self.property_group_map = {}
        self.property_groups = {}

        self.refresh()

    def refresh(self):
        device_props = self.device.properties
        property_group_map = self.property_group_map
        property_groups = self.property_groups
        changed_props = set(device_props).symmetric_difference(property_group_map.viewkeys())
        for prop in changed_props:
            if prop in property_group_map:
                if prop not in device_props:
                    property_groups[property_group_map.pop(prop)].remove_prop(prop)
            elif prop in device_props and prop not in property_group_map:
                self.add_prop(prop)
        for property_group in property_groups.itervalues():
            propetry_group.refresh()

    def add_prop(self, prop):
        device = self.device
        group = None

        avp = device.getAnyProperty(prop)
        if avp is not None:
            group = avp.group
            if group not in self.property_groups:
                group_widget = self.add_group(self, group)
            else:
                group_widget = self.property_groups[group]
            group_widget.add_prop(prop, avp)
        self.property_group_map.setdefault(prop, group)

    def add_group(self, group):
        self.property_groups[group] = PropertyGroup(self, self.device, group)


class PropertyGroup(tk.Frame):

    def __init__(self, tab_parent, device, group):
        self.group = group
        self.device = device
        self.properties = {}
        self.parent_tab_index = tab_parent.index('end')
        tk.Frame.__init__(self, tab_parent)
        tab_parent.add(self, text=group)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def add_prop(self, prop, avp):
        label = _g(tk.Label(self, text=avp.label), sticky=tk.EW)
        if hasattr(avp, 'sp'):
            self.properties[prop] = _g(SwitchProperty(self, self.device, prop, label), column=1, sticky=tk.EW)
        elif hasattr(avp, 'np'):
            self.properties[prop] = _g(NumberProperty(self, self.device, prop, label), column=1, sticky=tk.EW)
        elif hasattr(avp, 'tp'):
            self.properties[prop] = _g(TextProperty(self, self.device, prop, label), column=1, sticky=tk.EW)

    def refresh(self):
        for prop in self.properties.itervalues():
            prop.refresh()


class SwitchProperty(tk.Frame):

    def __init__(self, box, device, prop, label):
        self.label = label
        self.prop = prop
        self.device = device
        tk.Frame.__init__(self, box)


class NumberProperty(tk.Frame):

    def __init__(self, box, device, prop, label):
        self.label = label
        self.prop = prop
        self.device = device
        tk.Frame.__init__(self, box)


class TextProperty(tk.Frame):

    def __init__(self, box, device, prop, label):
        self.label = label
        self.prop = prop
        self.device = device
        tk.Frame.__init__(self, box)


def launch_app(interactive_guider):
    ready = threading.Event()
    Application.main_thread = main_thread = threading.Thread(
        target=Application.launch,
        args=(ready, interactive_guider))
    main_thread.daemon = True
    main_thread.start()
    ready.wait()
    return Application.instance
