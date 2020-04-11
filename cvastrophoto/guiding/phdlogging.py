from __future__ import absolute_import

import csv
import sys
import datetime
import time
import math

import cvastrophoto
from cvastrophoto.guiding.calibration import norm


def _fmt_or_na(fmt, x, unit=None):
    if not x:
        return 'N/A'
    if unit:
        fmt = fmt + ' %s'
        return fmt % (x, unit)
    else:
        return fmt % (x,)


class PHD2Logger(object):

    LOG_VERSION = '2.5'

    def __init__(self, path_or_fileobj):
        if isinstance(path_or_fileobj, basestring):
            self.fileobj = open(path_or_fileobj, "w")
        else:
            self.fileobj = path_or_fileobj

        self.csv = csv.writer(self.fileobj)

    def start(self):
        self.fileobj.write("cvastrophoto version %s [%s], Log version %s. Log enabled at %s\n\n" % (
            cvastrophoto.__version__, sys.platform, self.LOG_VERSION,
            datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')))
        self.fileobj.flush()

    def info(self, fmt_string, *args):
        self.fileobj.write("INFO:", fmt_string + "\n" % args)
        self.fileobj.flush()

    def start_calibration(self, calibration):
        eff_telescope_coords = calibration.eff_telescope_coords or (None, None)
        eff_telescope_hcoords = calibration.eff_telescope_hcoords or (None, None)
        if calibration.telescope:
            telescope_pier_side = calibration.telescope.properties.get('TELESCOPE_PIER_SIDE', (False, False))
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
            camera_name=calibration.ccd.name,
            exposure_ms=int(calibration.guide_exposure * 1000),
            pixel_scale=_fmt_or_na("%.2f", calibration.image_scale, "arc-sec/px"),
            binning=calibration.ccd.properties.get('CCD_BINNING', (1, 1))[0],
            guide_fl=_fmt_or_na("%d", calibration.eff_guider_fl, "mm"),
            mount_name=calibration.telescope.name if calibration.telescope else 'N/A',
            calibration_step_ms=max(
                calibration.eff_calibration_pulse_s_ra,
                calibration.eff_calibration_pulse_s_dec,
            ),
            calibration_distance=calibration.calibration_min_move_px,
            ra_hr=_fmt_or_na("%.3f", eff_telescope_coords[0], 'hr'),
            dec_deg=_fmt_or_na("%.3f", eff_telescope_coords[1], 'deg'),
            hour_angle='N/A',
            pier_side=pier_side,
            alt=_fmt_or_na('%.3f', eff_telescope_hcoords[0], 'deg'),
            az=_fmt_or_na('%.3f', eff_telescope_hcoords[1], 'deg'),
            hfd='N/A'
        )
        header_fmt = """
Calibration Begin at %(start_date)s
Equipment Profile = default
Camera = %(camera_name)s
Exposure = %(exposure_ms)d ms
Pixel scale = %(pixel_scale)s, Binning = %(binning)d, Focal length = %(guide_fl)s,
Mount = %(mount_name)s, Calibration Step = %(calibration_step_ms)d ms, Calibration Distance = %(calibration_distance)d px, Assume orthogonal axes = yes
RA = %(ra_hr)s, Dec = %(dec_deg)s, Hour angle = N/A, Pier side = %(pier_side)s, Rotator pos = N/A, Alt = %(alt)s, Az = %(az)s
"""
        self.fileobj.write(header_fmt % header_info)
        self.csv.writerow(['direction', 'step', 'dx', 'dy', 'x', 'y', 'dist'])
        self.fileobj.flush()

    def finish_calibration(self, calibration):
        footer_info = dict(
            mount_name=calibration.telescope.name if calibration.telescope else 'N/A',
        )
        image_scale = calibration.image_scale
        if image_scale:
            footer_info.update(dict(
                ra_speed=_fmt_or_na(".2f", norm(calibration.wstep) * image_scale, "a-s/s"),
                dec_speed=_fmt_or_na(".2f", norm(calibration.nstep) * image_scale, "a-s/s"),
            ))
        else:
            footer_info.update(dict(
                ra_speed=_fmt_or_na(".2f", norm(calibration.wstep), "px/s"),
                dec_speed=_fmt_or_na(".2f", norm(calibration.nstep), "px/s"),
            ))
        footer_fmt = """Calibration guide speeds: RA = %(ra_speed)s, Dec = %(dec_speed)s
Calibration complete, mount = %(mount_name).
"""
        self.fileobj.write(footer_fmt % footer_info)

    def calibration_step(self, direction, step, dx, dy):
        self.csv.writerow([direction, step, dx, dy, dx, dy, norm((dy, dx))])
        self.fileobj.flush()

    def start_guiding(self, guider):
        self.guide_start = time.time()
        eff_telescope_coords = guider.calibration.eff_telescope_coords or (None, None)
        eff_telescope_hcoords = guider.calibration.eff_telescope_hcoords or (None, None)
        image_scale = guider.calibration.image_scale
        if guider.telescope:
            telescope_pier_side = guider.telescope.properties.get('TELESCOPE_PIER_SIDE', (False, False))
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
            camera_name=guider.ccd.name,
            exposure_ms=int(guider.calibration.guide_exposure * 1000),
            pixel_scale=_fmt_or_na("%.2f", guider.calibration.image_scale, "arc-sec/px"),
            binning=guider.ccd.properties.get('CCD_BINNING', (1, 1))[0],
            guide_fl=_fmt_or_na("%d", guider.calibration.eff_guider_fl, "mm"),
            mount_name=guider.telescope.name if guider.telescope else 'N/A',
            ra_hr=_fmt_or_na("%.3f", eff_telescope_coords[0], 'hr'),
            dec_deg=_fmt_or_na("%.3f", eff_telescope_coords[1], 'deg'),
            hour_angle='N/A',
            pier_side=pier_side,
            alt=_fmt_or_na('%.3f', eff_telescope_hcoords[0], 'deg'),
            az=_fmt_or_na('%.3f', eff_telescope_hcoords[1], 'deg'),
            lock_x=guider.lock_pos[0], lock_y=guider.lock_pos[1],
            star_x=guider.lock_pos[0], star_y=guider.lock_pos[1],
            hfd='N/A',
            sleep_time_ms=int(guider.sleep_period * 1000),
            track_distance=_fmt_or_na('%d', getattr(guider.tracker_class, 'track_distance', None), 'px'),
            img_w=guider.ccd.properties.get('CCD_INFO', (None, None))[0] or 0,
            img_h=guider.ccd.properties.get('CCD_INFO', (None, None))[1] or 0,
            have_dark='have dark' if guider.master_dark is not None else 'no dark',
            pixel_size=guider.ccd.properties.get('CCD_INFO', (None, None, None))[2] or 0,
            ra_hist=guider.controller.ra_switch_resistence,
            dec_hist=guider.controller.dec_switch_resistence,
            ra_agg=guider.aggressiveness,
            dec_agg=guider.aggressiveness,
            ra_drift_agg=guider.drift_aggressiveness,
            dec_drift_agg=guider.drift_aggressiveness,
            ra_minmove=(image_scale or 1) * norm(guider.calibration.wstep) * guider.controller.min_pulse,
            dec_minmove=(image_scale or 1) * norm(guider.calibration.nstep) * guider.controller.min_pulse,
            backlash_delay_ms=int(guider.controller.dec_switch_resistence * 1000),
            ra_max_pulse_ms=int(guider.eff_max_pulse * 1000),
            dec_max_pulse_ms=int(guider.eff_max_pulse * 1000),
        )
        if image_scale:
            header_info.update(dict(
                ra_speed=_fmt_or_na(".2f", norm(guider.calibration.wstep) * image_scale, "a-s/s"),
                dec_speed=_fmt_or_na(".2f", norm(guider.calibration.nstep) * image_scale, "a-s/s"),
            ))
        else:
            header_info.update(dict(
                ra_speed=_fmt_or_na(".2f", norm(guider.calibration.wstep), "px/s"),
                dec_speed=_fmt_or_na(".2f", norm(guider.calibration.nstep), "px/s"),
            ))
        header_fmt = """
Guiding Begins at %(start_date)s
Dither = both axes, Image noise reduction = none, Guide-frame time lapse = %(sleep_time_ms)d ms, Server disabled
Pixel scale = %(pixel_scale)s, Binning = %(binning)d, Focal length = %(guide_fl)s
Search region = %(track_distance)s
Equipment Profile = default
Camera = %(camera_name)s, full size = %(img_w)d x %(img_h)d, %(have_dark)s, pixel size = %(pixel_size)s
Exposure = %(exposure_ms)d ms
Mount = %(mount_name)s,  connected, guiding enabled,
X guide algorithm = Drift, Hysteresis = %(ra_hist).3f, Aggression = %(ra_agg).3f, Minimum move = %(ra_minmove).3f, Drift Aggression = %(ra_drift_agg).3f
Y guide algorithm = Drift, Hysteresis = %(dec_hist).3f, Aggression = %(dec_agg).3f, Minimum move = %(dec_minmove).3f, Drift Aggression = %(dec_drift_agg).3f
Backlash comp = enabled, pulse = %(backlash_delay_ms)d ms
Max RA duration = %(ra_max_pulse_ms)d, Max DEC duration = %(dec_max_pulse_ms)d, DEC guide mode = Auto
RA Guide Speed = %(ra_speed)s, Dec Guide Speed = %(dec_speed)s
RA = %(ra_hr)s, Dec = %(dec_deg)s, Hour angle = %(hour_angle)s, Pier side = %(pier_side)s, Rotator pos = N/A, Alt = %(alt)s, Az = %(az)s
Lock position = %(lock_x).3f, %(lock_y).3f, Star position = %(star_x).3f, %(star_y).3f, HFD = %(hfd)s
"""
        self.fileobj.write(header_fmt % header_info)
        self.csv.writerow([
            'Frame', 'Time', 'mount',
            'dx', 'dy', 'RARawDistance', 'DECRawDistance', 'RAGuideDistance', 'DECGuideDistance',
            'RADuration', 'RADirection', 'DECDuration', 'DECDirection',
            'XStep', 'YStep',
            'StarMass', 'SNR', 'ErrorCode', 'ErrorDescription',
            'RADriftSpeed', 'DECDriftSpeed',
        ])
        self.fileobj.flush()

    def finish_guiding(self, guider):
        footer_info = dict(finish_date=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        footer_fmt = "Guiding Ends at %(finish_date)s\n"
        self.fileobj.write(footer_fmt % footer_info)

    def guide_step(self, guider, frame, dx, dy, dra, ddec, pulse_we, pulse_ns, mount="Mount", error_code='', error_str=''):
        image_scale = guider.calibration.image_scale
        guide_ra = pulse_we * norm(guider.calibration.wstep) * image_scale
        guide_dec = pulse_ns * norm(guider.calibration.nstep) * image_scale
        self.csv.writerow([
            frame, time.time() - self.guide_start, mount,
            dx, dy, dra, ddec, guide_ra, guide_dec,
            math.abs(int(pulse_we * 1000)), 'W' if pulse_we > 0 else ('E' if pulse_we < 0 else ''),
            math.abs(int(pulse_ns * 1000)), 'N' if pulse_ns > 0 else ('S' if pulse_ns < 0 else ''),
            '', '', '', '',
            error_code, error_str,
            guider.controller.we_drift, guider.controller.ns_drift,
        ])

    def close(self):
        self.fileobj.close()
