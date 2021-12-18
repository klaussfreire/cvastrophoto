# -*- coding: utf-8 -*-
try:
    import Tkinter as tk
    import ttk
except ImportError:
    import tkinter as tk
    from tkinter import ttk

try:
    import PyIndi
except:
    # Only allows deviceless testing, but won't really work like this
    PyIndi = None

import logging
import functools
from six import itervalues, viewkeys

from .utils import _g, _p, _focus_get


logger = logging.getLogger(__name__)


class EquipmentNotebook(ttk.Notebook):

    UPDATE_PERIOD_MS = 1770

    def __init__(self, box, guider, extra_devices=None, *p, **kw):
        self.guider = guider
        self.extra_devices = extra_devices or []
        ttk.Notebook.__init__(self, box, *p, **kw)

        self.devices = {}

        self.master.after(self.UPDATE_PERIOD_MS, self.periodic_refresh)

    def periodic_refresh(self):
        self.master.after(self.UPDATE_PERIOD_MS, self.periodic_refresh)
        self.refresh()

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
        add(self.guider, 'capture_seq', 'cfw')
        add(self.guider, 'capture_seq', 'focuser')

        for dname in self.devices:
            if dname not in devices:
                self.remove_device(dname)
        for dname in devices:
            if dname not in self.devices:
                self.add_device(dname, devices[dname])
            else:
                self.devices[dname].refresh()

        for dname in self.extra_devices:
            if dname not in self.devices:
                device = self.guider.indi_client.waitDevice(dname)
                if device is not None:
                    devices[dname] = device
                    logger.info("Adding device %r", dname)
                    self.add_device(dname, device)
            else:
                self.devices[dname].refresh()

    def remove_device(self, dname):
        dev = self.devices.get(dname)
        if dev is not None:
            dev.destroy()

    def add_device(self, dname, device):
        if dname not in self.devices:
            self.devices[dname] = DeviceControlSet(self, dname, device, width=640, height=640)


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
        changed_props = set(device_props).symmetric_difference(viewkeys(property_group_map))
        for prop in changed_props:
            if prop in property_group_map:
                if prop not in device_props:
                    property_groups[property_group_map.pop(prop)].remove_prop(prop)
            elif prop in device_props and prop not in property_group_map:
                self.add_prop(prop)
        for property_group in itervalues(property_groups):
            property_group.refresh()

    def add_prop(self, prop):
        device = self.device
        group = None

        avp = device.getAnyProperty(prop)
        if avp is not None:
            group = avp.group
            if group not in self.property_groups:
                group_widget = self.add_group(group)
            else:
                group_widget = self.property_groups[group]
            group_widget.add_prop(prop, avp)
        self.property_group_map.setdefault(prop, group)

    def add_group(self, group):
        self.property_groups[group] = pg = PropertyGroup(self, self.device, group)
        return pg


class PropertyGroup(tk.Frame):

    def __init__(self, tab_parent, device, group):
        self.group = group
        self.device = device
        self.properties = {}
        self.parent_tab_index = tab_parent.index('end')
        tk.Frame.__init__(self, tab_parent)
        tab_parent.add(self, text=group)
        self.canvas = canvas = _p(tk.Canvas(self), side="left", fill="both", expand=True)
        self.scrollbar = _p(tk.Scrollbar(self, orient='vertical', command=canvas.yview), side="right", fill="y")
        self.propbox = tk.Frame(self.canvas)
        self.propbox.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self.propbox.grid_columnconfigure(1, weight=1)
        self.nextrow = 0

        canvas.create_window((0, 0), window=self.propbox, anchor="nw")
        canvas.configure(yscrollcommand=self.scrollbar.set)

    def add_prop(self, prop, avp):
        row = self.nextrow
        self.nextrow += 1

        props = self.properties
        device = self.device
        box = self.propbox
        label = _g(tk.Label(box, text=avp.label), sticky=tk.E)
        opts = dict(borderwidth=1, relief=tk.SUNKEN)

        if hasattr(avp, 'sp'):
            props[prop] = _g(SwitchProperty(box, device, prop, label, avp, **opts), row=row, column=1, sticky=tk.W)
        elif hasattr(avp, 'np'):
            props[prop] = _g(NumberProperty(box, device, prop, label, avp, **opts), row=row, column=1, sticky=tk.W)
        elif hasattr(avp, 'tp'):
            props[prop] = _g(TextProperty(box, device, prop, label, avp, **opts), row=row, column=1, sticky=tk.W)

    def remove_prop(self, prop):
        self.properties[prop].label.destroy()
        self.properties.pop(prop).destroy()

    def refresh(self):
        for prop in itervalues(self.properties):
            prop.refresh()


class SwitchProperty(tk.Frame):

    def __init__(self, box, device, prop, label, svp, **kw):
        self.label = label
        self.prop = prop
        self.device = device
        tk.Frame.__init__(self, box, **kw)

        self.values = values = []
        self.buttons = buttons = []

        writeable = svp.p != PyIndi.IP_RO
        state = tk.NORMAL if writeable else tk.DISABLED
        for i, sp in enumerate(svp):
            v = tk.BooleanVar()
            v.set(sp.s == PyIndi.ISS_ON)

            opts = {}
            if writeable:
                if svp.r == PyIndi.ISR_1OFMANY:
                    opts['command'] = functools.partial(self._clickNary, i)
                elif svp.r == PyIndi.ISR_ATMOST1:
                    opts['command'] = functools.partial(self._clickAtMost1, i)
                elif svp.r == PyIndi.ISR_NOFMANY:
                    opts['command'] = self._clickNofMany
            buttons.append(_p(tk.Checkbutton(self, text=sp.label, variable=v, state=state, **opts)))
            values.append(v)

    def _clickNary(self, i):
        self.device.setNarySwitch(self.prop, i, quick=True, optional=True)

    def _clickAtMost1(self, i):
        vals = [v.get() for v in self.values]
        if not any(vals):
            self.device.setSwitch(self.prop, vals, quick=True, optional=True)
        else:
            self.device.setNarySwitch(self.prop, i, quick=True, optional=True)

    def _clickNofMany(self):
        self.device.setSwitch(self.prop, [v.get() for v in self.values], quick=True, optional=True)

    def refresh(self):
        for var, value in zip(self.values, self.device.properties.get(self.prop, ())):
            var.set(value)


class NumberProperty(tk.Frame):

    def __init__(self, box, device, prop, label, nvp, **kw):
        self.label = label
        self.prop = prop
        self.device = device
        tk.Frame.__init__(self, box, **kw)

        self.values = values = []
        self.labels = labels = []
        self.controls = controls = []
        self.vinfo = vinfo = []

        self.writeable = writeable = nvp.p != PyIndi.IP_RO

        if writeable:
            widget = tk.Entry
        else:
            widget = tk.Label

        for i, np in enumerate(nvp):
            v = tk.StringVar()

            fmt = np.format
            if fmt.endswith('m'):
                fmt = '%f'

            v.set(fmt % np.value)

            values.append(v)
            vinfo.append(dict(format=fmt, min=np.min, max=np.max, step=np.step, name=np.label))
            labels.append(_g(tk.Label(self, text=np.label), row=i))
            controls.append(_g(widget(self, textvariable=v), row=i, column=1))

            if writeable:
                controls[-1].bind("<Return>", functools.partial(self._edit, i))

    def _edit(self, i, *p, **kw):
        self.device.setNumber(self.prop, {self.vinfo[i]["name"]: float(self.values[i].get())}, quick=True, optional=True)

    def refresh(self):
        writeable = self.writeable
        pval = self.device.properties.get(self.prop, ())
        for var, control, vinfo, value in zip(self.values, self.controls, self.vinfo, pval):
            sval = vinfo["format"] % value
            if not writeable or (sval != var.get() and control is not _focus_get(control)):
                var.set(sval)


class TextProperty(tk.Frame):

    def __init__(self, box, device, prop, label, tvp, **kw):
        self.label = label
        self.prop = prop
        self.device = device
        tk.Frame.__init__(self, box, **kw)

        self.values = values = []
        self.labels = labels = []
        self.controls = controls = []
        self.vinfo = vinfo = []

        self.writeable = writeable = tvp.p != PyIndi.IP_RO

        if writeable:
            widget = tk.Entry
        else:
            widget = tk.Label

        for i, tp in enumerate(tvp):
            v = tk.StringVar()
            v.set(tp.text)

            values.append(v)
            vinfo.append(dict(name=tp.label))
            labels.append(_g(tk.Label(self, text=tp.label), row=i))
            controls.append(_g(widget(self, textvariable=v), row=i, column=1))

            if writeable:
                controls[-1].bind("<Return>", functools.partial(self._edit, i))

    def _edit(self, i, *p, **kw):
        self.device.setText(self.prop, {self.vinfo[i]["name"]: self.values[i].get()}, quick=True, optional=True)

    def refresh(self):
        writeable = self.writeable
        for var, control, value in zip(self.values, self.controls, self.device.properties.get(self.prop, ())):
            if not writeable or (value != var.get() and control is not _focus_get(control)):
                var.set(value)
