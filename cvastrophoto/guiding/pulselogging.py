from __future__ import absolute_import

from past.builtins import basestring
import csv
import sys
import datetime
import time

import cvastrophoto


class PulseLogger(object):

    LOG_VERSION = '2.5'

    def __init__(self, path_or_fileobj):
        if isinstance(path_or_fileobj, basestring):
            self.fileobj = open(path_or_fileobj, "w")
        else:
            self.fileobj = path_or_fileobj

        self.csv = csv.writer(self.fileobj, lineterminator='\n')

    def start(self):
        self.fileobj.write("cvastrophoto version %s [%s], Log version %s. Log enabled at %s\n\n" % (
            cvastrophoto.__version__, sys.platform, self.LOG_VERSION,
            datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')))
        self.fileobj.flush()

    def info(self, fmt_string, *args):
        self.fileobj.write(("INFO: " + fmt_string + "\n") % args)
        self.fileobj.flush()

    def start_section(self, controller):
        self.section_start = time.time()
        if controller.telescope:
            telescope_pier_side = controller.telescope.properties.get('TELESCOPE_PIER_SIDE', (False, False))
            if telescope_pier_side[0]:
                pier_side = 'East'
            elif telescope_pier_side[1]:
                pier_side = 'West'
            else:
                pier_side = 'N/A'
        else:
            pier_side = 'N/A'
        header_info = dict(
            start_date=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            mount_name=controller.telescope.name if controller.telescope else 'N/A',
            pier_side=pier_side,
            ra_hist=controller.ra_switch_resistence,
            dec_hist=controller.dec_switch_resistence,
            ra_minpulse_ms=int(controller.min_pulse_ra * 1000),
            dec_minpulse_ms=int(controller.min_pulse_dec * 1000),
            ra_backlash_delay_ms=int(controller.ra_switch_resistence * 1000),
            dec_backlash_delay_ms=int(controller.dec_switch_resistence * 1000),
        )
        header_fmt = """
Controller Begins at %(start_date)s
Equipment Profile = default
Mount = %(mount_name)s, Pier side = %(pier_side)s
RA Minimum pulse = %(ra_minpulse_ms)d ms
DEC Minimum pulse = %(dec_minpulse_ms)d ms
RA Backlash comp, RA pulse = %(ra_backlash_delay_ms)d ms, DEC pulse %(dec_backlash_delay_ms)d ms
"""
        self.fileobj.write(header_fmt % header_info)
        self.csv.writerow([
            'Time', 'mount',
            'RADuration', 'RADirection', 'DECDuration', 'DECDirection',
            'RADuty', 'DECDuty', 'RADriftSpeed', 'DECDriftSpeed',
            'RAGearstate', 'DECGearState',
        ])
        self.fileobj.flush()

    def finish_section(self, controller):
        footer_info = dict(finish_date=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        footer_fmt = "Controller Ends at %(finish_date)s\n"
        self.fileobj.write(footer_fmt % footer_info)
        self.fileobj.flush()

    def pulse(self, controller, pulse_we, pulse_ns, we_duty, ns_duty, mount="Mount"):
        self.csv.writerow([
            time.time() - self.section_start, mount,
            abs(int(pulse_we * 1000)), 'W' if pulse_we > 0 else ('E' if pulse_we < 0 else ''),
            abs(int(pulse_ns * 1000)), 'N' if pulse_ns > 0 else ('S' if pulse_ns < 0 else ''),
            int(we_duty * 1000), int(ns_duty * 1000),
            controller.we_drift, controller.ns_drift,
            int(controller.unsync_gear_state_we * 1000), int(controller.unsync_gear_state_ns * 1000),
        ])
        self.fileobj.flush()

    def close(self):
        self.fileobj.close()
