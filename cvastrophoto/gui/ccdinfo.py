# -*- coding: utf-8 -*-
try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

import logging

from .utils import _g


logger = logging.getLogger(__name__)


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
