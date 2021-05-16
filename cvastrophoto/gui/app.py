# -*- coding: utf-8 -*-
from __future__ import division

try:
    import ttk
    import Tkinter as tk
    import tkFileDialog as filedialog
    import ScrolledText as scrolledtext
except ImportError:
    import tkinter as tk
    from tkinter import ttk, filedialog, scrolledtext

from six import itervalues
from past.builtins import xrange
from PIL import Image, ImageTk
import threading
import logging
import math
import numpy
import subprocess
import functools
import collections
import os.path
import multiprocessing.pool
import skimage.transform

from astropy import wcs

from cvastrophoto.guiding.calibration import norm2, sub
from cvastrophoto.image.rgb import RGB, Templates
import cvastrophoto.image
from cvastrophoto.rops.bias import localgradient
from cvastrophoto.rops.measures import fwhm, focus
import cvastrophoto.constants.exposure

from .utils import _g, _p
from .equipment import EquipmentNotebook
from .ccdinfo import CCDInfoBox
from . import icons


logger = logging.getLogger(__name__)


def with_guider(f):
    @functools.wraps(f)
    def decor(self, *p, **kw):
        if self.guider is not None:
            return f(self, *p, **kw)
    return decor


class AsyncTasks(threading.Thread):

    def __init__(self, autostart=True):
        self.wake = threading.Event()
        self.__stop = False
        self.busy = False
        threading.Thread.__init__(self)
        self.daemon = True
        self.requests = {}
        if autostart:
            self.start()

    def run(self):
        while not self.__stop:
            self.wake.wait(1)
            self.wake.clear()

            for key in list(self.requests.keys()):
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


class AsyncTaskPool(object):

    def __init__(self):
        self.pools = collections.defaultdict(AsyncTasks)

    def add_request(self, pool, key, fn, *p, **kw):
        self.pools[pool].add_request(key, fn, *p, **kw)

    def stop(self):
        for tasks in itervalues(self.pools):
            tasks.stop()

    def join(self, *p):
        for tasks in itervalues(self.pools):
            tasks.join(*p)


class Application(tk.Frame):

    _new_snap = None

    FILE_TYPES = [
        ('Any image', '*.fits *.fit *.cr2 *.nef *.tiff *.tif *.png *.jpg *.jpeg'),
        ('FITS data', '*.fits *.fit *.fits.gz *.fit.gz'),
        ('Canon Raw File', '*.cr2'),
        ('Nikkon Raw File', '*.nef'),
        ('TIFF image', '*.tiff *.tif'),
        ('PNG image', '*.png'),
        ('JPEG image', '*.jpg *.jpeg'),
        ('Any file', '*'),
    ]

    DEFAULT_CAP_EXPOSURE = '60'
    CAP_EXPOSURE_VALUES = tuple(["%g" % exp for exp in cvastrophoto.constants.exposure.CAP_EXPOSURE_VALUES])

    DEFAULT_FLAT_EXPOSURE = '1'
    FLAT_EXPOSURE_VALUES = tuple(["%g" % exp for exp in cvastrophoto.constants.exposure.FLAT_EXPOSURE_VALUES])

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

        self._stop_updates = self._quit = False
        self.cap_shift_from = self.cap_shift_to = None
        self.snap_shift_from = self.snap_shift_to = None
        self.last_cap_solve_data = self.last_snap_solve_data = None

        self.init_icons()

        self.async_executor = AsyncTaskPool()

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

    def init_icons(self):
        self.green_crosshair = icons.get('CROSSHAIRS', foreground='green')
        self.red_crosshair = icons.get('CROSSHAIRS', foreground='red')

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

        namevar = tk.StringVar()
        self.current_cap.name_label = tk.Label(box, textvar=namevar)
        self.current_cap.name_label.value = namevar
        self.current_cap.name_label.grid(padx=5, row=2, column=0, sticky=tk.EW)

        self.cap_gamma_box = tk.Frame(box)
        self.create_gamma(self.cap_gamma_box, prefix='cap_', bright=1.0, gamma=1.8, show=True)
        self.cap_gamma_box.grid(padx=5, row=3, column=0, sticky=tk.EW)

        self.cap_button_box = tk.Frame(box)
        self.create_cap_buttons(self.cap_button_box)
        self.cap_button_box.grid(padx=5, row=4, column=0, columnspan=2, sticky=tk.EW)

        self.create_cap_stats(self.cap_stats_box)

    def create_cap_stats(self, box):
        box.grid_columnconfigure(0, weight=1)

        self.cap_stats_nb = nb = _g(ttk.Notebook(box), sticky=tk.NSEW)

        self.cap_adu_box = tk.Frame(nb)
        self.create_cap_adu_stats(self.cap_adu_box)
        nb.add(self.cap_adu_box, text='ADU')

        self.cap_fwhm_box = tk.Frame(nb)
        self.create_cap_fwhm(self.cap_fwhm_box)
        nb.add(self.cap_fwhm_box, text='FWHM')

        self.cap_tilt_box = tk.Frame(nb)
        self.create_cap_tilt(self.cap_tilt_box)
        nb.add(self.cap_tilt_box, text='Tilt')

        self.cap_focus_box = tk.Frame(nb)
        self.create_cap_focus(self.cap_focus_box)
        self.cap_focus_tab_index = nb.index('end')
        nb.add(self.cap_focus_box, text='Focus')

    def create_cap_adu_stats(self, box):
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

    def create_cap_fwhm(self, box):
        self.cap_fwhm_vars = svars = [[tk.StringVar() for _ in xrange(3)] for _ in xrange(3)]
        for row in svars:
            for var in row:
                var.set('-')

        box.grid_columnconfigure(0, weight=1)
        box.grid_columnconfigure(1, weight=1)
        box.grid_columnconfigure(2, weight=1)

        self.cap_fwhm_labels = [
            [
                _g(tk.Label(box, textvar=svars[row][column]), column=column, row=row, sticky=tk.W)
                for column in xrange(3)
            ]
            for row in xrange(3)
        ]

        self.cap_fwhm_btn = _g(tk.Button(box, text='Measure', command=self.on_measure_fwhm), columnspan=3)

    @with_guider
    def on_measure_fwhm(self):
        self.async_executor.add_request("cap_measure", "fwhm",
            self._measure_fwhm,
            self.guider.last_capture,
            half_size=False)

    def _measure_fwhm(self, path, half_size=False):
        img = cvastrophoto.image.Image.open(path)
        if img.postprocessing_params is not None:
            img.postprocessing_params.half_size = True
        imgpp = img.postprocessed

        if len(imgpp.shape) > 2:
            imgpp = numpy.average(imgpp, axis=2)
            img.close()
            img = Templates.LUMINANCE

        mrop = fwhm.FWHMMeasureRop(img)
        fwhm_values = mrop.measure_scalar(imgpp, quadrants=True)

        for row in xrange(3):
            for column in xrange(3):
                self.cap_fwhm_vars[row][column].set('%.2f' % (fwhm_values[row, column],))

    @with_guider
    def on_measure_focus(self, force=False):
        if not force and self.cap_stats_nb.index('current') != self.cap_focus_tab_index:
            return
        if not any(toggle.get() for toggle in self.focus_channel_toggles.values()):
            return

        self.async_executor.add_request("cap_measure", "focus",
            self._measure_focus,
            self.guider.last_capture,
            half_size=False)

    def _measure_focus(self, path, half_size=False):
        svars = self.cap_focus_vars
        toggles = self.focus_channel_toggles

        img = cvastrophoto.image.Image.open(path)
        if img.postprocessing_params is not None:
            img.postprocessing_params.half_size = True
        imgpp = img.postprocessed

        if len(imgpp.shape) > 2:
            imgpp = numpy.average(imgpp, axis=2)
            img.close()
            img = Templates.LUMINANCE

        if toggles['fwhm'].get():
            mrop = fwhm.FWHMMeasureRop(img, quick=self.focus_fast_check.value.get())
            fwhm_value = mrop.measure_scalar(imgpp)
        else:
            fwhm_value = 0

        if toggles['focus'].get():
            mrop = focus.FocusMeasureRop(img, quick=self.focus_fast_check.value.get())
            focus_value = mrop.measure_scalar(imgpp)
        else:
            focus_value = 0

        pos_value = 0
        if self.guider.capture_seq is not None:
            focuser = self.guider.capture_seq.focuser
            if focuser is not None:
                pos_value = focuser.absolute_position

        # Update variables
        max_hist = len(svars['focus'])
        for vname, newval in (('pos', pos_value),('focus', focus_value),('fwhm', fwhm_value),):
            for row in range(max_hist-1, 0, -1):
                svars[vname][row].set(svars[vname][row-1].get())
            svars[vname][0].set(newval)


    def create_cap_tilt(self, box):
        self.cap_tilt_vars = svars = [[tk.StringVar() for _ in xrange(3)] for _ in xrange(3)]
        for row in svars:
            for var in row:
                var.set('-')

        box.grid_columnconfigure(0, weight=1)
        box.grid_columnconfigure(1, weight=1)
        box.grid_columnconfigure(2, weight=1)

        self.cap_tilt_labels = [
            [
                _g(tk.Label(box, textvar=svars[row][column]), column=column, row=row, sticky=tk.W)
                for column in xrange(3)
            ]
            for row in xrange(3)
        ]

        self.cap_tilt_btn = _g(tk.Button(box, text='Measure', command=self.on_measure_tilt), columnspan=3)

    @with_guider
    def on_measure_tilt(self):
        self.async_executor.add_request("cap_measure", "tilt",
            self._measure_tilt,
            self.guider.last_capture,
            half_size=False)

    def _measure_tilt(self, path, half_size=False):
        img = cvastrophoto.image.Image.open(path)
        if img.postprocessing_params is not None:
            img.postprocessing_params.half_size = True
        imgpp = img.postprocessed

        if len(imgpp.shape) > 2:
            imgpp = numpy.average(imgpp, axis=2)
            img.close()
            img = Templates.LUMINANCE

        mrop = fwhm.TiltMeasureRop(img)
        fwhm_values = mrop.measure_scalar(imgpp, quadrants=True)

        for row in xrange(3):
            for column in xrange(3):
                self.cap_tilt_vars[row][column].set('%.2f' % (fwhm_values[row, column],))

    def create_cap_focus(self, box):
        max_hist = 6
        show_row = max_hist + 1
        control_col = 3

        self.cap_focus_vars = svars = {
            'pos': [],
            'focus': [],
            'fwhm': [],
        }

        self.focus_channel_labels = [
            _g(tk.Label(box, text='Pos'), column=0, row=0),
            _g(tk.Label(box, text='Focus', fg='red'), column=1, row=0),
            _g(tk.Label(box, text='FWHM', fg='green'), column=2, row=0),
        ]

        self.focus_channel_toggles = toggles = {
            'focus': tk.BooleanVar(),
            'fwhm': tk.BooleanVar(),
        }
        for v in self.focus_channel_toggles.values():
            v.set(True)

        self.focus_channel_toggle_checks = {
            'focus': _g(tk.Checkbutton(box, variable=toggles['focus']), column=1, row=show_row),
            'fwhm': _g(tk.Checkbutton(box, variable=toggles['fwhm']), column=2, row=show_row),
        }

        self.focus_channel_labels = labels = {
            'pos': [],
            'focus': [],
            'fwhm': [],
        }

        for row in range(max_hist):
            for vname, col in (('pos', 0),('focus', 1),('fwhm', 2),):
                v = tk.DoubleVar()
                v.set(0)
                svars[vname].append(v)
                labels[vname].append(_g(tk.Label(box, textvar=v), row=row+1, column=col))

        step_slow = tk.IntVar()
        step_fast = tk.IntVar()
        cur_pos = tk.IntVar()
        focus_fast_var = tk.BooleanVar()
        focus_fast_var.set(True)
        step_slow.set(500)
        step_fast.set(5000)
        self.focus_fast_check = _g(
            tk.Checkbutton(box, text='Fast', variable=focus_fast_var),
            row=1, column=control_col,
        )
        self.focus_fast_check.value = focus_fast_var
        self.focus_step_slow_spin = slow_spin = _g(
            tk.Spinbox(box, textvariable=step_slow, width=5, from_=1, to=20000),
            row=2, column=control_col,
        )
        self.focus_step_fast_spin = fast_spin = _g(
            tk.Spinbox(box, textvariable=step_fast, width=5, from_=1, to=20000),
            row=3, column=control_col,
        )
        self.focus_step_slow_spin.value = step_slow
        self.focus_step_fast_spin.value = step_fast
        self.focus_pos_label = _g(tk.Label(box, textvar=cur_pos), row=4, column=3)
        self.focus_pos_label.value = cur_pos

        self.focus_step_slow_out = _g(
            tk.Button(box, text='+', command=functools.partial(self.on_step, step_slow, 1)),
            row=2, column=control_col+1, sticky=tk.EW,
        )
        self.focus_step_slow_in = _g(
            tk.Button(box, text='-', command=functools.partial(self.on_step, step_slow, -1)),
            row=2, column=control_col+2, sticky=tk.EW,
        )
        self.focus_step_fast_out = _g(
            tk.Button(box, text='+', command=functools.partial(self.on_step, step_fast, 1)),
            row=3, column=control_col+1, sticky=tk.EW,
        )
        self.focus_step_fast_in = _g(
            tk.Button(box, text='-', command=functools.partial(self.on_step, step_fast, -1)),
            row=3, column=control_col+2, sticky=tk.EW,
        )

    @with_guider
    def on_step(self, step, mult):
        self.guider.capture_seq.focuser.moveRelative(step.get() * mult)

    @with_guider
    def update_focus_pos(self):
        if self.guider.capture_seq is None or self.guider.capture_seq.focuser is None:
            return
        self.focus_pos_label.value.set(self.guider.capture_seq.focuser.absolute_position)

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

        self.solve_info_box = _g(tk.Frame(box), row=1, column=0, columnspan=2, sticky=tk.NSEW)
        self.create_solve_info_box(self.solve_info_box)

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

        speed_text_var = tk.StringVar()
        speed_text_var.set("0.5")
        self.goto_speed_label = _g(tk.Label(box, text='Speed'), row=5, column=0)
        self.goto_speed = _g(
            ttk.Combobox(
                box, width=5,
                textvariable=speed_text_var, values=self.GUIDE_SPEED_VALUES),
            row=5, column=1, sticky=tk.EW)
        self.goto_speed.text = speed_text_var

        solve_var = tk.BooleanVar()
        solve_var.set(True)
        self.goto_solve = _g(
            tk.Checkbutton(box, text='Use plate solving', variable=solve_var),
            row=6, sticky=tk.EW, column=0)
        self.goto_solve.value = solve_var

        recalibrate_var = tk.BooleanVar()
        recalibrate_var.set(True)
        self.goto_recalibrate = _g(
            tk.Checkbutton(box, text='Recalibrate on target', variable=recalibrate_var),
            row=6, sticky=tk.EW, column=1)
        self.goto_recalibrate.value = recalibrate_var

        goto_ccd_var = tk.StringVar()
        goto_ccd_var.set('guide')
        self.goto_ccd_combo = _g(
            ttk.Combobox(
                box, width=5,
                textvariable=goto_ccd_var, values=('guide', 'main')),
            row=7, sticky=tk.NSEW, column=0)
        self.goto_ccd_combo.value = goto_ccd_var

        goto_exp_var = tk.StringVar()
        goto_exp_var.set(8)
        self.goto_exposure_combo = _g(
            ttk.Combobox(
                box, width=5,
                textvariable=goto_exp_var, values=self.CAP_EXPOSURE_VALUES),
            row=7, sticky=tk.NSEW, column=1)
        self.goto_exposure_combo.value = goto_exp_var

    def create_goto_info_box(self, box):
        (
            self.goto_info_title,
            self.goto_info_ra_label, self.goto_info_ra_value,
            self.goto_info_dec_label, self.goto_info_dec_value,
            _, _,
        ) = self._create_ra_dec_rot(box, 0, 0, 'Current mount status', False)

        (
            self.goto_info_cap_title,
            self.goto_info_cap_ra_label, self.goto_info_cap_ra_value,
            self.goto_info_cap_dec_label, self.goto_info_cap_dec_value,
            self.goto_info_cap_rot_label, self.goto_info_cap_rot_value,
        ) = self._create_ra_dec_rot(box, 3, 0, 'Last capture', True)

        (
            self.goto_info_ref_title,
            self.goto_info_ref_ra_label, self.goto_info_ref_ra_value,
            self.goto_info_ref_dec_label, self.goto_info_ref_dec_value,
            self.goto_info_ref_rot_label, self.goto_info_ref_rot_value,
        ) = self._create_ra_dec_rot(box, 3, 2, 'Reference sub', True)

    def _create_ra_dec_rot(self, box, rowbase, colbase, title, wrot=True):
        title = _g(tk.Label(box, text=title), row=rowbase + 0, column=colbase + 0, columnspan=2, pady=5, sticky=tk.W)

        var = tk.StringVar()
        ra_label = _g(tk.Label(box, text='RA'), row=rowbase + 1, column=colbase + 0, sticky=tk.E, padx=5)
        ra_value = _g(tk.Label(box, textvar=var), row=rowbase + 1, column=colbase + 1, sticky=tk.W, padx=5)
        ra_value.text = var

        var = tk.StringVar()
        dec_label = _g(tk.Label(box, text='DEC'), row=rowbase + 2, column=colbase + 0, sticky=tk.E, padx=5)
        dec_value = _g(tk.Label(box, textvar=var), row=rowbase + 2, column=colbase + 1, sticky=tk.W, padx=5)
        dec_value.text = var

        if wrot:
            var = tk.StringVar()
            rot_label = _g(tk.Label(box, text='ROT'), row=rowbase + 3, column=colbase + 0, sticky=tk.E, padx=5)
            rot_value = _g(tk.Label(box, textvar=var), row=rowbase + 3, column=colbase + 1, sticky=tk.W, padx=5)
            rot_value.text = var
        else:
            rot_label = rot_value = None

        return (
            title,
            ra_label, ra_value,
            dec_label, dec_value,
            rot_label, rot_value,
        )

    def create_solve_info_box(self, box):
        box.grid_columnconfigure(0, weight=1)

        self.ref_box = ref_box = _g(tk.Frame(box), row=0, sticky=tk.NSEW)

        self.ref_select_button = _g(
            tk.Button(ref_box, text='Select Reference', command=self.on_select_reference),
            row=0, column=0, sticky=tk.NSEW)
        var = tk.StringVar()
        var.set('-- Not Selected --')
        self.ref_label = _g(tk.Label(ref_box, textvar=var), row=0, column=1, sticky=tk.NSEW)
        self.ref_label.value = var

        self.solve_info_nb = _g(ttk.Notebook(box), row=1)
        self.guide_solve_box = self.create_solve_box(box, 'Guide')
        self.cap_solve_box = self.create_solve_box(box, 'Capture')
        self.ref_solve_box = self.create_solve_box(box, 'Reference')
        self.solve_info_nb.add(self.guide_solve_box, text='Guide')
        self.solve_info_nb.add(self.cap_solve_box, text='Capture')
        self.solve_info_nb.add(self.ref_solve_box, text='Reference')

    def create_solve_box(self, box, title):
        solve_box = tk.Frame(box, relief=tk.SUNKEN, borderwidth=1)
        solve_box.grid_columnconfigure(0, weight=1)
        solve_box.title_label = _g(
            tk.Label(solve_box, text=title, font='Helvetica 16'),
            row=0, columnspan=2, sticky=tk.NSEW)
        solve_box.solve_text = _g(
            scrolledtext.ScrolledText(solve_box, relief=tk.SUNKEN, state=tk.DISABLED),
            row=1, sticky=tk.NSEW)
        solve_box.solve_text.tag_config('key', foreground='blue')
        solve_box.solve_text.tag_config('error', foreground='red')
        solve_box.solve_text.tag_config('coord', foreground='green')
        solve_box.solve_text.insert(tk.END, 'Helo\n')
        return solve_box

    def set_solve_data(self, box, headers, coords, ravar=None, decvar=None, rotvar=None):
        text = box.solve_text
        END = tk.END
        text.config(state=tk.NORMAL)
        try:
            text.delete(1.0, END)
            if headers is not None:
                if coords is not None:
                    ra, dec = coords[2:4]
                    text.insert(END, 'RA:\t', 'key')
                    text.insert(END, '%s\n' % (ra,), 'coord')
                    text.insert(END, 'DEC:\t', 'key')
                    text.insert(END, '%s\n' % (dec,), 'coord')
                    if ravar is not None:
                        ravar.set(ra)
                    if decvar is not None:
                        decvar.set(dec)
                if 'CROTA1' in headers:
                    rot = headers['CROTA1']
                    text.insert(END, 'ROT:\t', 'key')
                    text.insert(END, '%s\n' % (rot,), 'coord')
                    if rotvar is not None:
                        rotvar.set(rot)
                text.insert(END, '\n')
                for card in headers.cards:
                    if card.is_blank:
                        continue
                    text.insert(END, '%s:\t' % (card.keyword), 'key')
                    text.insert(END, '%s\n' % (card.value,))
            else:
                text.insert(END, 'Plate solving failed', 'error')
        finally:
            text.config(state=tk.DISABLED)

    def on_select_reference(self):
        initial = self.ref_label.value.get()
        if not initial and self.guider is not None:
            initial = self.guider.last_capture
        if not initial and self.guider is not None and self.guider.capture_seq is not None:
            initialdir = self.guider.capture_seq.base_dir
            initialfile = None
        elif initial:
            initialdir = os.path.dirname(initial)
            initialfile = os.path.basename(initial)
        else:
            initialdir = initialfile = None

        newref = filedialog.askopenfilename(
            parent=self,
            title='Select reference image',
            initialdir=initialdir,
            initialfile=initialfile,
            filetypes=self.FILE_TYPES,
        )

        if not newref or newref == self.ref_label.value.get():
            return

        # Initiate platesolve on new reference
        self.ref_label.value.set(newref)
        if self.guider is not None:
            success, solver, path, coords, hdu, kw = self.guider.cmd_solve('main', path=newref, hint=self.solve_hint)
            self.set_solve_data(
                self.ref_solve_box, hdu, coords,
                self.goto_info_ref_ra_value.text, self.goto_info_ref_dec_value.text,
                self.goto_info_ref_rot_value.text)

    def update_goto_info_box(self):
        if self.guider is not None:
            eff_telescope_coords = self.guider.guider.calibration.eff_telescope_coords
        else:
            eff_telescope_coords = None
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

    def create_sequence_tab(self, box):
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
        self.capture_label = _g(tk.Label(self.capture_box, text='Lights'), columnspan=2)
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
        self.capture_button = _g(
            tk.Button(self.capture_box, text='Test', command=self.capture_test),
            row=4, column=0, sticky=tk.NSEW, columnspan=2)

        self.filters_box = _g(
            tk.Frame(box, relief=tk.SUNKEN, borderwidth=1),
            column=2, row=0, rowspan=2, sticky=tk.NSEW, ipadx=3)
        self.filters_label = _g(tk.Label(self.filters_box, text='Filters'), columnspan=2)

        seq_text_var = tk.StringVar()
        self.filters_seq_label = _g(tk.Label(self.filters_box, text='Sequence'), row=0, column=0)
        self.filters_seq = _g(tk.Entry(self.filters_box, textvar=seq_text_var, width=30), row=0, column=1, sticky=tk.EW)
        self.filters_seq.text = seq_text_var

        self.filters_exposures = {}

        cfw = None
        if self.guider is not None and self.guider.capture_seq is not None:
            cfw = self.guider.capture_seq.cfw
        if cfw and cfw.filter_names:
            CAP_EXPOSURE_VALUES = ("Default",) + self.CAP_EXPOSURE_VALUES
            MAX_FILTERS_PER_COLUMN = 5
            for fpos, fname in enumerate(cfw.filter_names):
                if not fname:
                    continue
                row = fpos % MAX_FILTERS_PER_COLUMN + 1
                col = fpos // MAX_FILTERS_PER_COLUMN
                fpos += 1
                exposure_var = tk.StringVar()
                exposure_var.set(CAP_EXPOSURE_VALUES[0])
                cap_label = _g(tk.Label(self.filters_box, text=fname), row=row, column=col*2)
                self.filters_exposures[fpos] = exposure_combo = _g(
                    ttk.Combobox(
                        self.filters_box, width=5,
                        textvariable=exposure_var, values=CAP_EXPOSURE_VALUES),
                    sticky=tk.NSEW, row=row, column=col*2+1)
                exposure_combo.text = exposure_var

    def create_calibration_tab(self, box):
        self.flat_box = _g(
            tk.Frame(box, relief=tk.SUNKEN, borderwidth=1),
            column=0, row=0, rowspan=2, sticky=tk.NSEW, ipadx=3)
        self.flat_capture_label = _g(tk.Label(self.flat_box, text='Flats'), columnspan=2)

        self.flat_n_label = _g(tk.Label(self.flat_box, text='N'), row=1, column=0)
        self.flat_n_var = tk.IntVar()
        self.flat_n_var.set(30)
        self.flat_n_spinner = _g(
            tk.Spinbox(self.flat_box, textvariable=self.flat_n_var, width=4, from_=1, to=100),
            sticky=tk.NSEW, row=1, column=1)

        self.flat_adu_label = _g(tk.Label(self.flat_box, text='ADU'), row=2, column=0)
        self.flat_adu_var = tk.IntVar()
        self.flat_adu_var.set(30000)
        self.flat_adu_spinner = _g(
            tk.Spinbox(self.flat_box, textvariable=self.flat_adu_var, width=6, from_=1, to=65000),
            sticky=tk.NSEW, row=2, column=1)

        self.flat_exposure_var = tk.StringVar()
        self.flat_exposure_var.set(self.DEFAULT_FLAT_EXPOSURE)
        self.flat_exposure_combo = _g(
            ttk.Combobox(
                self.flat_box, width=5,
                textvariable=self.flat_exposure_var, values=self.FLAT_EXPOSURE_VALUES),
            sticky=tk.NSEW, columnspan=2)

        self.flat_capture_button = _g(
            tk.Button(self.flat_box, text='Capture', command=self.capture_flats),
            row=4, column=0, sticky=tk.NSEW)
        self.stop_flat_capture_button = _g(
            tk.Button(self.flat_box, text='Stop', command=self.stop_capture),
            row=4, column=1, sticky=tk.NSEW)
        self.flat_capture_test_button = _g(
            tk.Button(self.flat_box, text='Test', command=self.capture_test_flats),
            row=5, column=0, sticky=tk.NSEW)
        self.flat_capture_test_button = _g(
            tk.Button(self.flat_box, text='Auto', command=self.capture_auto_flats),
            row=5, column=1, sticky=tk.NSEW)

        self.dark_box = _g(
            tk.Frame(box, relief=tk.SUNKEN, borderwidth=1),
            column=1, row=0, rowspan=2, sticky=tk.NSEW, ipadx=3)
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

    def create_cap_display_tab(self, box):
        fullsize_var = tk.BooleanVar()
        fullsize_var.set(False)
        self.cap_fullsize_check = _g(
            tk.Checkbutton(box, text='Full-size', variable=fullsize_var),
            column=0, row=0)
        self.cap_fullsize_check.value = fullsize_var

        skyglow_var = tk.BooleanVar()
        skyglow_var.set(False)
        self.cap_skyglow_check = _g(
            tk.Checkbutton(box, text='Remove background', variable=skyglow_var),
            column=0, row=1)
        self.cap_skyglow_check.value = skyglow_var

        autorefresh_var = tk.BooleanVar()
        autorefresh_var.set(False)
        self.cap_autorefresh_check = _g(
            tk.Checkbutton(box, text='Autorefresh', variable=autorefresh_var),
            column=1, row=0)
        self.cap_autorefresh_check.value = autorefresh_var

        self.cap_update_button = _g(
            tk.Button(box, text='Refresh', command=self.cap_snap_update),
            column=2, row=0, sticky=tk.NSEW)
        self.bg_update_button = _g(
            tk.Button(box, text='Update bg\nmodel', command=self.cap_bg_update),
            column=2, row=1, sticky=tk.NSEW)

        self.solve_button = _g(
            tk.Button(box, text='Platesolve', command=self.iplatesolve),
            column=3, row=0, sticky=tk.NSEW)
        self.solve_button = _g(
            tk.Button(box, text='ASTAP', command=self.cap_snap_to_astap),
            column=3, row=1, sticky=tk.NSEW)

    def create_cap_buttons(self, box):
        self.cap_tabs = _g(ttk.Notebook(box), sticky=tk.NSEW)

        self.cap_sequence_tab = tk.Frame(self.cap_tabs)
        self.cap_tabs.add(self.cap_sequence_tab, text='Sequence')
        self.create_sequence_tab(self.cap_sequence_tab)

        self.cap_calibration_tab = tk.Frame(self.cap_tabs)
        self.cap_tabs.add(self.cap_calibration_tab, text='Calibration')
        self.create_calibration_tab(self.cap_calibration_tab)

        self.cap_display_tab = tk.Frame(self.cap_tabs)
        self.cap_tabs.add(self.cap_display_tab, text='Display')
        self.create_cap_display_tab(self.cap_display_tab)

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

    @property
    def goto_destination(self):
        ra = self.goto_ra.text.get().strip()
        dec = self.goto_dec.text.get().strip()
        epoch = self.epoch.text.get().strip()
        return ','.join(filter(None, [ra, dec, epoch]))

    @with_guider
    def goto(self):
        speed = self.goto_speed.text.get().strip()
        to_ = self.goto_destination
        if self.goto_solve.value.get():
            logger.info("Executing go + platesolve to %s", to_)
            self.async_executor.add_request(
                "goto", "goto",
                self.guider.cmd_goto_solve,
                self.goto_ccd_combo.value.get(), to_, speed,
                recalibrate=self.goto_recalibrate.value.get(),
                exposure=float(self.goto_exposure_combo.value.get()),
            )
        else:
            logger.info("Executing go to %s", to_)
            self.async_executor.add_request(
                "goto", "goto",
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

    @property
    def solve_hint(self):
        if self.guider is None or self.guider.guider.telescope is None:
            return self.goto_destination or None

    @with_guider
    def platesolve(self):
        self.async_executor.add_request("ui", "platesolve", self._platesolve)

    def _platesolve(self):
        def on_solve_data(success, solver, path, coords, hdu, **kw):
            if success:
                self.last_snap_solve_data = (coords, hdu)
            self.set_solve_data(self.guide_solve_box, hdu, coords)
        img = self.guider.cmd_annotate(solve_callback=on_solve_data, hint=self.solve_hint)
        if img is not None:
            subprocess.check_call(['xdg-open', img.name])

    @with_guider
    def iplatesolve(self):
        self.async_executor.add_request("ui", "iplatesolve", self._iplatesolve)

    def _iplatesolve(self):
        def on_solve_data(success, solver, path, coords, hdu, **kw):
            if success:
                self.last_cap_solve_data = (coords, hdu)
            self.set_solve_data(
                self.cap_solve_box, hdu, coords,
                self.goto_info_cap_ra_value.text, self.goto_info_cap_dec_value.text,
                self.goto_info_cap_rot_value.text)

        img = self.guider.cmd_annotate(
            'main',
            path=self.guider.last_capture,
            solve_callback=on_solve_data,
            hint=self.solve_hint)
        if img is not None:
            subprocess.check_call(['xdg-open', img.name])

    @with_guider
    def cap_snap_to_astap(self):
        from cvastrophoto.platesolve import astap
        solver = astap.ASTAPSolver()
        self.async_executor.add_request("cap_snap", "astap",
            solver.open_interactive,
            self.guider.last_capture,
            half_size=False)

    @with_guider
    def capture(self):
        self.guider.cmd_capture(
            self.cap_exposure_var.get(),
            self.dither_n_var.get(),
            self.dither_var.get(),
            filter_sequence=self.filters_seq.text.get().strip() or None,
            filter_exposures={
                fpos: float(fcontrol.text.get())
                for fpos, fcontrol in self.filters_exposures.items()
                if fcontrol.text.get() != "Default"
            })

    @with_guider
    def capture_test(self):
        self.guider.cmd_capture(
            self.cap_exposure_var.get(),
            self.dither_n_var.get(),
            self.dither_var.get(),
            number=1)

    @with_guider
    def stop_capture(self):
        self.async_executor.add_request("ui", "stop_capture", self.guider.cmd_stop_capture)

    @with_guider
    def capture_flats(self):
        self.guider.cmd_capture_flats(
            self.flat_exposure_var.get(),
            self.flat_n_var.get())

    @with_guider
    def capture_test_flats(self):
        self.guider.cmd_capture_flats(self.flat_exposure_var.get(), 1)

    @with_guider
    def capture_auto_flats(self):
        self.guider.cmd_auto_flats(
            self.flat_n_var.get(),
            self.flat_adu_var.get(),
            filter_sequence=self.filters_seq.text.get().strip() or None,
        )

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

    def cap_snap_update(self, force=True):
        self.async_executor.add_request("cap_snap", "update", self.update_capture, force)

    @with_guider
    def cap_bg_update(self):
        self.skyglow_model = None
        self.cap_snap_update()

    def create_snap(self, snapbox, zoombox):
        self.current_snap = snap = _g(tk.Canvas(snapbox))
        snap.imgid = snap.create_image((0, 0), anchor=tk.NW)
        snap.bind("<1>", self.snap_click)
        snap.bind("<2>", self.snap_rclick)
        snap.bind("<3>", self.snap_rclick)
        snap.shift_from_id = snap.create_image((0, 0), image=self.green_crosshair, state=tk.HIDDEN)
        snap.shift_to_id = snap.create_image((0, 0), image=self.red_crosshair, state=tk.HIDDEN)
        snap.lock_y_id = snap.create_line(0, 0, 0, 0, fill="#0a0", dash=(2, 2), state=tk.HIDDEN)
        snap.lock_x_id = snap.create_line(0, 0, 0, 0, fill="#0a0", dash=(2, 2), state=tk.HIDDEN)
        snap.lock_r_id = snap.create_rectangle(0, 0, 0, 0, fill="", outline="#0b0", dash=(2, 2), state=tk.HIDDEN)
        self.snap_toolbar = _g(SnapToolBar(zoombox))
        self.current_zoom = zoom = _g(tk.Canvas(zoombox))
        zoom.imgid = self.current_zoom.create_image((0, 0), anchor=tk.NW)
        zoom.lock_y_id = zoom.create_line(0, 0, 0, 0, fill="#0a0", dash=(2, 2), state=tk.HIDDEN)
        zoom.lock_x_id = zoom.create_line(0, 0, 0, 0, fill="#0a0", dash=(2, 2), state=tk.HIDDEN)

        snap.current_gamma = None
        snap.current_bright = None
        snap.current_zoom = None

    def create_cap(self, snapbox, zoombox):
        self.current_cap = current_cap = _g(tk.Canvas(snapbox))
        current_cap.imgid = current_cap.create_image((0, 0), anchor=tk.NW)
        current_cap.bind("<1>", self.cap_click)
        current_cap.bind("<2>", self.cap_rclick)
        current_cap.bind("<3>", self.cap_rclick)
        current_cap.shift_from_id = current_cap.create_image((0, 0), image=self.green_crosshair, state=tk.HIDDEN)
        current_cap.shift_to_id = current_cap.create_image((0, 0), image=self.red_crosshair, state=tk.HIDDEN)
        self.cap_toolbar = _g(CapToolBar(zoombox))
        self.current_cap_zoom = _g(tk.Canvas(zoombox))
        self.current_cap_zoom.imgid = self.current_cap_zoom.create_image((0, 0), anchor=tk.NW)

        current_cap.current_gamma = None
        current_cap.current_bright = None
        current_cap.current_zoom = None
        current_cap.current_skyglow = False
        current_cap.current_channels = (True, True, True)
        current_cap.raw_image = None
        current_cap.debiased_image = None
        current_cap.display_image = None

    def snap_rclick(self, ev):
        tool = self.snap_toolbar.current_tool
        if tool == 'shift':
            self.snap_shift_from = self.snap_shift_to = None
            self.current_snap.itemconfig(self.current_snap.shift_from_id, state=tk.HIDDEN)
            self.current_snap.itemconfig(self.current_snap.shift_to_id, state=tk.HIDDEN)

    def snap_click(self, ev):
        tool = self.snap_toolbar.current_tool
        click_point = (
            ev.x * self.current_snap.full_size[0] // self.current_snap.view_size[0],
            ev.y * self.current_snap.full_size[1] // self.current_snap.view_size[1],
        )
        if tool == 'zoom':
            self.zoom_point = click_point
            self._update_snap()
        elif tool == 'setref':
            if self.guider is not None:
                self.guider.cmd_set_reference(*click_point)
        elif tool == 'shift':
            if self.snap_shift_from is None:
                self.snap_shift_from = click_point
                self.current_snap.coords(self.current_snap.shift_from_id, (ev.x, ev.y))
                self.current_snap.itemconfig(self.current_snap.shift_from_id, state=tk.NORMAL)
            elif self.snap_shift_to is None:
                self.snap_shift_to = click_point
                self.current_snap.coords(self.current_snap.shift_to_id, (ev.x, ev.y))
                self.current_snap.itemconfig(self.current_snap.shift_to_id, state=tk.NORMAL)
            else:
                try:
                    self.snap_shift_exec(self.snap_shift_from, self.snap_shift_to)
                finally:
                    self.snap_shift_from = self.snap_shift_to = None
                    self.current_snap.itemconfig(self.current_snap.shift_from_id, state=tk.HIDDEN)
                    self.current_snap.itemconfig(self.current_snap.shift_to_id, state=tk.HIDDEN)

    def cap_rclick(self, ev):
        tool = self.cap_toolbar.current_tool
        if tool == 'shift':
            self.cap_shift_from = self.cap_shift_to = None
            self.current_cap.itemconfig(self.current_cap.shift_from_id, state=tk.HIDDEN)
            self.current_cap.itemconfig(self.current_cap.shift_to_id, state=tk.HIDDEN)

    def cap_click(self, ev):
        tool = self.cap_toolbar.current_tool
        click_point = (
            ev.x * self.current_cap.full_size[0] // self.current_cap.view_size[0],
            ev.y * self.current_cap.full_size[1] // self.current_cap.view_size[1],
        )
        if tool == 'zoom':
            self.cap_zoom_point = click_point
            self.update_cap_snap(zoom_only=True)
        elif tool == 'shift':
            if self.cap_shift_from is None:
                self.cap_shift_from = click_point
                self.current_cap.coords(self.current_cap.shift_from_id, (ev.x, ev.y))
                self.current_cap.itemconfig(self.current_cap.shift_from_id, state=tk.NORMAL)
            elif self.cap_shift_to is None:
                self.cap_shift_to = click_point
                self.current_cap.coords(self.current_cap.shift_to_id, (ev.x, ev.y))
                self.current_cap.itemconfig(self.current_cap.shift_to_id, state=tk.NORMAL)
            else:
                try:
                    self.cap_shift_exec(self.cap_shift_from, self.cap_shift_to)
                finally:
                    self.current_cap.itemconfig(self.current_cap.shift_from_id, state=tk.HIDDEN)
                    self.current_cap.itemconfig(self.current_cap.shift_to_id, state=tk.HIDDEN)
                    self.cap_shift_from = self.cap_shift_to = None

    @with_guider
    def snap_shift_exec(self, snap_shift_from, snap_shift_to):
        logger.info("Executing guider shift from %r to %r", snap_shift_from, snap_shift_to)
        self.async_executor.add_request(
            "goto", "goto",
            self.guider.cmd_shift_pixels,
            snap_shift_from[0] - snap_shift_to[0],
            snap_shift_from[1] - snap_shift_to[1],
            None,
        )

    def _cap_shift_exec(self, cap_shift_from, cap_shift_to, solve_hint):
        # Get a guider and last capture solution to translate into guider coordinates
        logger.info("Plate-solving guider image")
        success, guide_solver, path, guide_coords, guide_hdu, kw = self.guider.cmd_solve(hint=solve_hint)
        if not success:
            if self.last_snap_solve_data is not None:
                logger.warning("Guider solve failed, but have previous solution")
                guide_coords, guide_hdu = self.last_snap_solve_data
            else:
                logger.error("Guider solve failed, have no workable solution")
                return
        else:
            logger.info("Guider solution successful")

        logger.info("Plate-solving capture image")
        success, cap_solver, path, cap_coords, cap_hdu, kw = self.guider.cmd_solve(
            'main',
            path=self.guider.last_capture,
            hint=solve_hint)
        if not success:
            if self.last_cap_solve_data is not None:
                logger.warning("Capture solve failed, but have previous solution")
                cap_coords, cap_hdu = self.last_cap_solve_data
            else:
                logger.error("Capture solve failed, have no workable solution")
                return
        else:
            logger.info("Capture solution successful")

        # Translate from, to and center coordinates into guider pixel locations
        guider_wcs = wcs.WCS(guide_hdu)
        cap_wcs = wcs.WCS(cap_hdu)

        guider_wcs.wcs.bounds_check(False, False)

        guider_from = guider_wcs.wcs.s2p(cap_wcs.wcs.p2s([cap_shift_from], 0)['world'], 0)['pixcrd'][0]
        guider_to = guider_wcs.wcs.s2p(cap_wcs.wcs.p2s([cap_shift_to], 0)['world'], 0)['pixcrd'][0]

        logger.info("Executing guider shift from %r to %r", guider_from, guider_to)
        self.guider.cmd_shift_pixels(
            guider_from[0] - guider_to[0],
            guider_from[1] - guider_to[1],
            None,
        )

    @with_guider
    def cap_shift_exec(self, cap_shift_from, cap_shift_to):
        self.async_executor.add_request(
            "goto", "goto",
            self._cap_shift_exec,
            cap_shift_from, cap_shift_to, self.solve_hint,
        )

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
        if self._quit:
            self.master.destroy()
            return
        if self._stop_updates:
            return

        updates = [
            self.__update_snap,
            self.__update_cap,
            self.update_goto_info_box,
            self.update_cap_info_box,
            self.update_focus_pos,
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
        if self._stop_updates:
            return

        updates = []
        if self.guider is not None:
            updates += [
                self.update_iccd_info_box,
                self.update_gccd_info_box,
                self.update_equipment_info_box,
            ]

            if (self.guider.capture_seq is not None and self.guider.capture_seq.new_capture
                    and self.cap_autorefresh_check.value.get()):
                updates.append(functools.partial(self.cap_snap_update, False))

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

        cap_status = cap_detail = None
        if self.guider.capture_seq is not None:
            cap_status = self.guider.capture_seq.state
            cap_detail = self.guider.capture_seq.state_detail
        if self.guider.goto_state is not None:
            cap_status = self.guider.goto_state
            cap_detail = self.guider.goto_state_detail

        if cap_detail:
            cap_status = '%s (%s)' % (cap_status, cap_detail)
        if cap_status != self.cap_status_label.text.get():
            self.cap_status_label.text.set(cap_status)

        self.update_rms(self.guider.guider.offsets)

    def __update_cap(self):
        if self.current_cap.debiased_image is not None:
            self.async_executor.add_request("cap_snap", "update", self.update_cap_snap)

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

        guider = self.guider
        if guider is not None:
            guider = guider.guider
        if guider is not None:
            lock_pos = guider.lock_pos
            lock_region = guider.lock_region
            snap = self.current_snap
            zoom = self.current_zoom
            vw, vh = snap.view_size
            fw, fh = snap.full_size
            zvw, zvh = zoom.view_size
            zox, zoy = zoom.view_origin

            if lock_pos is not None:
                lock_y, lock_x = lock_pos
                lock_vy = lock_y * vh // fh
                lock_vx = lock_x * vw // fw
                lock_zvy = lock_y - zoy
                lock_zvx = lock_x - zox
                snap.coords(snap.lock_y_id, 0, lock_vy, vw, lock_vy)
                snap.coords(snap.lock_x_id, lock_vx, 0, lock_vx, vh)
                if 0 <= lock_zvy < zvh:
                    zoom.coords(zoom.lock_y_id, 0, lock_zvy, zvw, lock_zvy)
                    zoom.itemconfig(zoom.lock_y_id, state=tk.NORMAL)
                else:
                    zoom.itemconfig(zoom.lock_y_id, state=tk.HIDDEN)
                if 0 <= lock_zvx < zvw:
                    zoom.coords(zoom.lock_x_id, lock_zvx, 0, lock_zvx, zvh)
                    zoom.itemconfig(zoom.lock_x_id, state=tk.NORMAL)
                else:
                    zoom.itemconfig(zoom.lock_x_id, state=tk.HIDDEN)
                snap.itemconfig(snap.lock_y_id, state=tk.NORMAL)
                snap.itemconfig(snap.lock_x_id, state=tk.NORMAL)
            else:
                snap.itemconfig(snap.lock_y_id, state=tk.HIDDEN)
                snap.itemconfig(snap.lock_x_id, state=tk.HIDDEN)
                zoom.itemconfig(zoom.lock_y_id, state=tk.HIDDEN)
                zoom.itemconfig(zoom.lock_x_id, state=tk.HIDDEN)

            if lock_region is not None:
                t, l, b, r = lock_region
                t = t * vh // fh
                b = b * vh // fh
                l = l * vw // fw
                r = r * vw // fw
                snap.coords(snap.lock_r_id, l, t, r, b)
                snap.itemconfig(snap.lock_r_id, state=tk.NORMAL)
            else:
                snap.itemconfig(snap.lock_r_id, state=tk.HIDDEN)

    def _shrink_dims(self, dims, maxw=1280, maxh=720):
        w, h = dims
        factor = 1
        while h > maxh or w > maxw:
            w //= 2
            h //= 2
            factor *= 2
        return w, h, factor

    def __set_snap_image(self, img, zoom_point, current_zoom, current_snap, fullsize_check, zoom_only=False):
        zx, zy = zoom_point

        crop_img = img.crop((zx - 128, zy - 128, zx + 128, zy + 128))
        image = ImageTk.PhotoImage(crop_img)
        current_zoom.itemconfig(current_zoom.imgid, image=image)
        if getattr(current_zoom, 'view_size', None) != img.size:
            _, _, w, h = current_zoom.bbox(tk.ALL)
            current_zoom.configure(width=w, height=h)
            current_zoom.view_size = img.size
        current_zoom.image = image
        current_zoom.view_origin = (zx - 128, zy - 128)

        if zoom_only:
            return

        # Resize to something sensible
        current_snap.full_size = img.size
        if not fullsize_check.value.get():
            w, h, factor = self._shrink_dims(img.size)
            if (w, h) != img.size:
                img = img.resize((w, h), resample=Image.BOX)

        image = ImageTk.PhotoImage(img)
        current_snap.itemconfig(current_snap.imgid, image=image)
        if getattr(current_snap, 'view_size', None) != img.size:
            _, _, w, h = current_snap.bbox(tk.ALL)
            current_snap.configure(width=w, height=h)
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
                    csat = (hsum - chistogram[int(white)]) / max(1, float(hsum))
                else:
                    csat = 0
            else:
                cmin = cmax = cmean = cstd = cmedian = csat = 0
        elif cdata.size:
            cmin = cdata.min()
            cmax = cdata.max()
            cmean = numpy.average(cdata)
            cmedian = numpy.median(cdata)
            cstd = numpy.std(cdata)
            csat = numpy.count_nonzero(cdata >= white)
        else:
            cmin = cmax = cmean = cstd = cmedian = csat = 0

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
        if not force and self.guider.capture_seq and not self.guider.capture_seq.new_capture:
            return
        if not force and self.current_cap.raw_image is not None and self.current_cap.raw_image.name == last_capture:
            return

        # Load and shrink image
        img = cvastrophoto.image.Image.open(last_capture)
        self.update_raw_stats(img)

        capture_seq = self.guider.capture_seq
        master_dark = capture_seq.master_dark
        if master_dark is None and capture_seq.dark_library is not None:
            dark_key = capture_seq.dark_library.classify_frame(img)
            if dark_key is not None:
                master_dark = capture_seq.dark_library.get_master(dark_key, raw=img)
        if master_dark is None and capture_seq.bias_library is not None:
            dark_key = capture_seq.bias_library.classify_frame(img)
            if dark_key is not None:
                master_dark = capture_seq.bias_library.get_master(dark_key, raw=img)
        if master_dark is not None:
            img.denoise([master_dark], entropy_weighted=False)

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
        elif self.current_cap.raw_image.rimg.raw_image.shape == new_raw.rimg.raw_image.shape:
            # Update image in-place to avoid memory leaks if some components
            # already hold a reference to the image
            self.current_cap.raw_image.set_raw_image(new_raw.rimg.raw_image)
            self.current_cap.raw_image.name = new_raw.name
        else:
            self.current_cap.raw_image = new_raw
            self.current_cap.debiased_image = RGB(
                last_capture, img=imgpp.copy(), linear=True, autoscale=False,
                default_pool=self.processing_pool)
        del new_raw

        self.update_cap_snap(reprocess=True)

        if self.guider.capture_seq and self.guider.capture_seq.new_capture:
            if (self.current_cap.raw_image is None
                    or (self.current_cap.raw_image.name == last_capture and self.guider.last_capture == last_capture)):
                self.guider.capture_seq.new_capture = False

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
        self.current_cap.name_label.value.set(self.current_cap.raw_image.name)

        self.on_measure_focus()

    def update_skyglow_model(self):
        if self.skyglow_rop is None:
            self.skyglow_rop = localgradient.QuickGradientBiasRop(self.current_cap.raw_image, copy=False)
        if self.skyglow_model is None:
            logger.info("Updating skyglow model")
            self.skyglow_model = self.skyglow_rop.detect(self.current_cap.raw_image.rimg.raw_image)

    @staticmethod
    def launch(ready, interactive_guider, logger=logger):
        root = tk.Tk()
        Application.instance = app = Application(interactive_guider, master=root)
        ready.set()
        try:
            app.mainloop()
        except Exception:
            logger.exception("Exception in main loop, exiting")
        finally:
            logger.info("GUI main loop exited")

    def shutdown(self):
        self.async_executor.stop()
        self.async_executor.join(5)
        self.processing_pool.close()

        self._stop_updates = True
        self._quit = True
        self.main_thread.join(5)

        logger.info("GUI shutdown")


class SnapToolBar(ttk.Notebook):

    TOOLS = [
        ('zoom', lambda:icons.ZOOM),
        ('setref', lambda:icons.SETREF),
        ('shift', lambda:icons.SHIFT),
    ]

    def __init__(self, box, **kw):
        ttk.Notebook.__init__(self, box, **kw)

        icons.init()
        self.state = 'zoom'
        self.states = {}

        for k, label in self.TOOLS:
            self.states[k] = f = tk.Frame(self)
            self.add(f, image=label())

    @property
    def current_tool(self):
        return self.TOOLS[self.index('current')][0]


class CapToolBar(SnapToolBar):

    TOOLS = [
        ('zoom', lambda:icons.ZOOM),
        ('shift', lambda:icons.SHIFT),
    ]


def launch_app(interactive_guider):
    ready = threading.Event()
    Application.main_thread = main_thread = threading.Thread(
        target=Application.launch,
        args=(ready, interactive_guider))
    main_thread.daemon = True
    main_thread.start()
    ready.wait()
    return Application.instance
