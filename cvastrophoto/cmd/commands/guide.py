# -*- coding: utf-8 -*-
from __future__ import print_function

import threading
import functools
import itertools
import datetime
import time
import logging
import os.path
import numpy
import re


logger = logging.getLogger(__name__)


def add_opts(subp):
    ap = subp.add_parser('guide', help="Start an interactive guider process")

    ap.add_argument('--darklib', help='Location of the main dark library', default=None)
    ap.add_argument('--biaslib', help='Location of the bias library', default=None)

    ap.add_argument('--phdlog', '-L', help='Write a PHD2-style log file in PATH', metavar='PATH')
    ap.add_argument('--exposure', '-x', help='Guiding exposure length', default=4.0, type=float)
    ap.add_argument('--gain', '-G', help='Guiding CCD gain', type=float)
    ap.add_argument('--offset', '-O', help='Guiding CCD offset', type=float)
    ap.add_argument('--autostart', '-A', default=False, action='store_true',
        help='Start guiding immediately upon startup')
    ap.add_argument('--pepa-sim', default=False, action='store_true',
        help='Simulate PE/PA')
    ap.add_argument('--debug-tracks', default=False, action='store_true',
        help='Save debug tracking images to ./Tracks/guide_*.jpg')
    ap.add_argument('--client-timeout', type=float, help='Default timeout waiting for server')

    ap.add_argument('--aggression', '-a', type=float,
        help='Defines how strongly it will apply immediate corrections')
    ap.add_argument('--drift-aggression', '-ad', type=float,
        help='Defines the learn rate of the drift model')
    ap.add_argument('--history-length', '-H', type=int,
        help='Defines how long a memory should be used for the drift model, in steps')

    ap.add_argument('--min-pulse', '-Pm', type=float,
        help='Defines the minimum pulse that ought to be sent to the mount, in seconds.')
    ap.add_argument('--max-pulse', '-PM', type=float,
        help='Defines the maximum pulse that ought to be sent to the mount, in seconds.')
    ap.add_argument('--target-pulse', '-Pt', type=float,
        help='Defines the optimal pulse that should be sent to the mount, in seconds.')

    ap.add_argument('--track-distance', '-d', type=int, default=192,
        help=(
            'The maximum search distance. The default should be fine. '
            'Lower values consume less resources'
        ))
    ap.add_argument('--track-resolution', '-r', type=int, default=64,
        help=(
            'The tracking correlation resolution. The default should be fine. '
            'Lower values consume slightly less resources but are less precise.'
        ))

    ap.add_argument('--guide-ccd', '-gccd', help='The name of the guide cam', metavar='GCCD', required=True)
    ap.add_argument('--guide-st4', '-gst4', help='The name of the guide interface', metavar='ST4')
    ap.add_argument('--guide-on-ccd', '-Gc', action='store_true', help='A shorthand to set ST4=GCCD')
    ap.add_argument('--guide-on-mount', '-Gm', action='store_true', help='A shorthand to set ST4=MOUNT')
    ap.add_argument('--mount', '-m', help='The name of the mount interface', metavar='MOUNT')
    ap.add_argument('--imaging-ccd', '-iccd', help='The name of the imaging cam', metavar='ICCD')
    ap.add_argument('--save-on-cam', action='store_true', default=False,
        help='Save light frames on-camera, when supported')
    ap.add_argument('--save-dir', metavar='DIR',
        help='Save light frames locally on DIR')
    ap.add_argument('--save-prefix', metavar='PREFIX',
        help='Save light frames with given prefix')
    ap.add_argument('--save-native', action='store_true', default=False,
        help='Save light frames in native format, rather than FITS')

    ap.add_argument('--sim-fl', help='When using the telescope simulator, set the FL',
        type=float, default=400)
    ap.add_argument('--sim-ap', help='When using the telescope simulator, set the apperture',
        type=float, default=70)
    ap.add_argument('--pepa-ra-speed', help='When using the PE/PA siimulator, the mount RA speed',
        type=float, default=1)
    ap.add_argument('--pepa-dec-speed', help='When using the PE/PA siimulator, the mount DEC speed',
        type=float, default=1)

    ap.add_argument('--guide-fl', help="The guide scope's FL",
        type=float, default=400)
    ap.add_argument('--guide-ap', help="The guide scope's apperture",
        type=float, default=70)
    ap.add_argument('--imaging-fl', help="The imaging scope's FL",
        type=float, default=750)
    ap.add_argument('--imaging-ap', help="The imaging scope's apperture",
        type=float, default=150)

    ap.add_argument('indi_addr', metavar='HOSTNAME:PORT', help='Indi server address',
        default='localhost:7624', nargs='?')

def main(opts, pool):
    import cvastrophoto.devices.indi
    from cvastrophoto.devices.indi import client
    from cvastrophoto.guiding import controller, guider, calibration, phdlogging
    import cvastrophoto.guiding.simulators.mount
    from cvastrophoto.rops.tracking import correlation, extraction
    from cvastrophoto.image import rgb

    if opts.guide_on_ccd:
        guide_st4 = opts.guide_ccd
    elif opts.guide_on_mount:
        guide_st4 = opts.mount
    elif opts.guide_st4:
        guide_st4 = opts.guide_st4
    else:
        logger.error("Either --guide-on-ccd, --guide-on-mount or --guide-st4 must be specified")
        return 1

    if opts.phdlog:
        phdlogger = phdlogging.PHD2Logger(os.path.join(
            opts.phdlog,
            'cvastrophoto_guidelog_%s.txt' % datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')))
        phdlogger.start()
    else:
        phdlogger = None

    indi_host, indi_port = opts.indi_addr.split(':')
    indi_port = int(indi_port)

    indi_client = client.IndiClient()

    if opts.client_timeout:
        indi_client.DEFAULT_TIMEOUT = opts.client_timeout

    indi_client.setServer(indi_host, indi_port)
    indi_client.connectServer()
    indi_client.startWatchdog()

    telescope = indi_client.waitTelescope(opts.mount) if opts.mount else None
    st4 = indi_client.waitST4(guide_st4)
    ccd = indi_client.waitCCD(opts.guide_ccd)
    ccd_name = 'CCD1'
    iccd_name = 'CCD1'
    imaging_ccd = indi_client.waitCCD(opts.imaging_ccd) if opts.imaging_ccd else None

    if telescope is not None:
        logger.info("Connecting telescope")
        telescope.connect()

    logger.info("Connecting ST4")
    st4.connect()

    logger.info("Connecting guiding CCD")
    ccd.connect()

    if imaging_ccd:
        logger.info("Connecting imaging CCD")
        imaging_ccd.connect()

    st4.waitConnect(False)
    ccd.waitConnect(False)
    ccd.subscribeBLOB(ccd_name)

    if telescope is not None:
        telescope.waitConnect(False)

    if imaging_ccd is not None:
        imaging_ccd.waitConnect(False)
        imaging_ccd.subscribeBLOB(iccd_name)

    if telescope is not None and opts.mount == 'Telescope Simulator':
        telescope.setNumber("TELESCOPE_INFO", [150, 750, opts.sim_ap, opts.sim_fl])

        ra = float(os.getenv('RA', repr((279.23473479 * 24.0)/360.0) ))
        dec = float(os.getenv('DEC', repr(+38.78368896) ))

        logger.info("Slew to target")
        telescope.trackTo(ra, dec)
        time.sleep(1)
        telescope.waitSlew()
        logger.info("Slewd to target")
    elif telescope is not None or st4 is not None:
        # Set telescope info if given
        if opts.imaging_fl or opts.imaging_ap or opts.guide_fl or opts.guide_ap:
            if telescope is not None:
                telescope.waitNumber("TELESCOPE_INFO")
                tel_info = telescope.properties.get("TELESCOPE_INFO", [0]*4)
            else:
                tel_info = [0]*4
            if opts.imaging_ap:
                tel_info[0] = opts.imaging_ap
            if opts.imaging_fl:
                tel_info[1] = opts.imaging_fl
            if opts.guide_ap:
                tel_info[2] = opts.guide_ap
            if opts.guide_fl:
                tel_info[3] = opts.guide_fl
            if telescope is not None:
                telescope.setNumber("TELESCOPE_INFO", tel_info)
            elif st4 is not None:
                # Inject locally only, for the benefit of the guider
                st4.properties["TELESCOPE_INFO"] = tel_info

    logger.info("Detecting CCD info")
    ccd.detectCCDInfo(ccd_name)
    if imaging_ccd is not None:
        imaging_ccd.detectCCDInfo(iccd_name)
    logger.info("Detected CCD info")

    # We'll need the guider CCD's blobs
    ccd.waitPropertiesReady()
    ccd.setNarySwitch("UPLOAD_MODE", "Client")
    ccd.setNarySwitch("TELESCOPE_TYPE", "Guide", quick=True, optional=True)

    tracker_class = functools.partial(correlation.CorrelationTrackingRop,
        track_distance=opts.track_distance,
        resolution=opts.track_resolution,
        luma_preprocessing_rop=extraction.ExtractStarsRop(
            rgb.Templates.LUMINANCE, copy=False, quick=True))

    if opts.pepa_sim:
        controller_class = cvastrophoto.guiding.simulators.mount.PEPASimGuiderController
        controller_class.we_speed = opts.pepa_ra_speed
        controller_class.ns_speed = opts.pepa_dec_speed
    else:
        controller_class = controller.GuiderController
    guider_controller = controller_class(telescope, st4)

    if opts.min_pulse:
        guider_controller.min_pulse = opts.min_pulse
    if opts.max_pulse:
        guider_controller.max_pulse = opts.max_pulse
    if opts.target_pulse:
        guider_controller.target_pulse = opts.target_pulse

    calibration_seq = calibration.CalibrationSequence(
        telescope, guider_controller, ccd, ccd_name, tracker_class,
        phdlogger=phdlogger)
    calibration_seq.guide_exposure = opts.exposure
    if opts.guide_fl:
        calibration_seq.guider_fl = opts.guide_fl
    if opts.imaging_fl:
        calibration_seq.imaging_fl = opts.imaging_fl
    guider_process = guider.GuiderProcess(
        telescope, calibration_seq, guider_controller, ccd, ccd_name, tracker_class,
        phdlogger=phdlogger)
    guider_process.save_tracks = opts.debug_tracks
    if opts.aggression:
        guider_process.aggressivenes = opts.aggression
    if opts.drift_aggression:
        guider_process.drift_aggressiveness = opts.drift_aggression
    if opts.history_length:
        guider_process.history_length = opts.history_length

    guider_controller.start()
    guider_process.start()

    if telescope is not None:
        indi_client.autoReconnect(telescope)
    indi_client.autoReconnect(st4)
    indi_client.autoReconnect(ccd)

    if opts.autostart:
        guider_process.start_guiding(wait=False)

    if imaging_ccd is not None:
        capture_seq = CaptureSequence(guider_process, imaging_ccd, iccd_name, phdlogger=phdlogger)
        capture_seq.save_on_client = False

        imaging_ccd.waitPropertiesReady()

        imaging_ccd.setNarySwitch("UPLOAD_MODE", "Local")
        if imaging_ccd is not ccd:
            imaging_ccd.setNarySwitch("TELESCOPE_TYPE", "Primary", quick=True, optional=True)
        if opts.save_dir or opts.save_prefix:
            if opts.save_dir:
                capture_seq.base_dir = opts.save_dir
            imaging_ccd.setUploadSettings(
                upload_dir=os.path.join(capture_seq.base_dir, capture_seq.target_dir),
                image_type='light')
        if "CCD_TRANSFER_FORMAT" in imaging_ccd.properties:
            imaging_ccd.setNarySwitch("CCD_TRANSFER_FORMAT", 1 if opts.save_native else 0)
        if "CCD_CAPTURE_TARGET" in imaging_ccd.properties and "CCD_SD_CARD_ACTION" in imaging_ccd.properties:
            if opts.save_on_cam:
                imaging_ccd.setNarySwitch("CCD_CAPTURE_TARGET", 1)
                imaging_ccd.setNarySwitch("CCD_SD_CARD_ACTION", 0)
            else:
                imaging_ccd.setNarySwitch("CCD_CAPTURE_TARGET", 0)
                imaging_ccd.setNarySwitch("CCD_SD_CARD_ACTION", 1)
    else:
        if opts.save_dir or opts.save_prefix:
            # Configure guide CCD as imaging CCD for live stacking
            ccd.setText("UPLOAD_SETTINGS", [
                opts.save_dir or ccd.properties["UPLOAD_SETTINGS"][0],
                opts.save_prefix or ccd.properties["UPLOAD_SETTINGS"][1],
            ])
        capture_seq = None

    iguider = InteractiveGuider(guider_process, guider_controller, ccd_name, capture_seq)

    if opts.gain:
        iguider.cmd_gain(opts.gain)
    if opts.offset:
        iguider.cmd_offset(opts.offset)

    iguider.run()

    logger.info("Shutting down")
    guider_process.stop()
    guider_controller.stop()
    indi_client.stopWatchdog()

    logger.info("Exit")


class CaptureSequence(object):

    dither_interval = 5
    dither_px = 20
    stabilization_s_min = 5
    stabilization_s = 10
    stabilization_s_max = 30
    stabilization_px = 4
    cooldown_s = 10
    flat_cooldown_s = 1

    save_on_client = False

    base_dir = '.'
    target_dir = 'Lights'
    flat_target_dir = 'Flats'
    dark_target_dir = 'Darks'
    dark_flat_target_dir = 'Dark Flats'
    pattern = flat_pattern = dark_pattern = dark_flat_pattern = '%04d.fits'

    start_seq = 1
    flat_seq = 1
    dark_seq = 1
    flat_dark_seq = 1

    def __init__(self, guider_process, ccd, ccd_name='CCD1', phdlogger=None):
        self.guider = guider_process
        self.ccd = ccd
        self.ccd_name = ccd_name
        self.state = 'idle'
        self.state_detail = None
        self.phdlogger = phdlogger
        self._stop = False

    @property
    def last_capture(self):
        img_prefix = self.ccd.properties['UPLOAD_SETTINGS'][1]
        basedir = self.ccd.properties['UPLOAD_SETTINGS'][0]
        nameprefix = os.path.basename(img_prefix).rstrip('X')
        try:
            lastimg = max(
                iter(p for p in os.listdir(basedir) if p.startswith(nameprefix)),
                key=lambda p: os.stat(os.path.join(basedir, p)).st_ctime
            )
        except ValueError:
            logger.exception("Can't find last capture")
            return None
        return os.path.join(basedir, lastimg)

    @property
    def all_captures(self):
        img_prefix = self.ccd.properties['UPLOAD_SETTINGS'][1]
        basedir = self.ccd.properties['UPLOAD_SETTINGS'][0]
        nameprefix = os.path.basename(img_prefix).rstrip('X')
        return [
            os.path.join(basedir, p)
            for p in os.listdir(basedir) if p.startswith(nameprefix)
        ]

    def new_captures(self, last_capture):
        return [
            p
            for p in self.all_captures
            if p > last_capture
        ]

    def wait_capture_ready(self, last_capture, sleep_time=1):
        cur_last_capture = self.last_capture
        deadline = time.time() + self.cooldown_s
        while cur_last_capture == last_capture and time.time() < deadline:
            time.sleep(sleep_time)
            cur_last_capture = self.last_capture
        return cur_last_capture

    def capture(self, exposure):
        next_dither = self.dither_interval
        last_capture = self.last_capture
        self.ccd.setLight()
        self.ccd.setUploadSettings(
            upload_dir=os.path.join(self.base_dir, self.target_dir),
            image_type='light')
        while not self._stop:
            try:
                logger.info("Starting sub exposure %d", self.start_seq)
                self.state = 'capturing'
                self.state_detail = 'sub %d' % self.start_seq

                if self.phdlogger is not None:
                    try:
                        self.phdlogger.info("Sub %d start", self.start_seq)
                    except Exception:
                        logger.exception("Error writing to PHD log")

                self.ccd.expose(exposure)
                time.sleep(exposure)
                if self.save_on_client:
                    blob = self.ccd.pullBLOB(self.ccd_name)
                    path = os.path.join(self.base_dir, self.target_dir, self.pattern % self.start_seq)
                    with open(path, 'wb') as f:
                        f.write(blob)
                logger.info("Finished sub exposure %d", self.start_seq)

                if self.phdlogger is not None:
                    try:
                        self.phdlogger.info("Sub %d finish", self.start_seq)
                    except Exception:
                        logger.exception("Error writing to PHD log")

                self.start_seq += 1
                next_dither -= 1

                if self._stop:
                    break

                self.state = 'cooldown'
                if next_dither > 0:
                    time.sleep(self.cooldown_s)
                else:
                    # Shorter sleep in case the main cam is still exposing
                    time.sleep(min(self.cooldown_s, 1))

                if not self.save_on_client:
                    last_capture = self.wait_capture_ready(last_capture, min(self.cooldown_s, 1))

                # Even if we don't stabilize in s_max time, it's worth waiting
                # half the exposure length. If stabiliztion delays a bit and we
                # start shooting, we'll waste "exposure" sub time, so we might
                # as well spend it waiting.
                # If we're forced to wait for longer, however, it's better to be
                # exposing, in case things do stabilize and the sub turns out
                # usable anyway.
                stabilization_s_max = max(self.stabilization_s_max, exposure / 2)

                if next_dither <= 0:
                    self.state = 'dither'
                    self.state_detail = 'start'
                    logger.info("Starting dither")
                    self.guider.dither(self.dither_px)

                    self.state_detail = 'wait stable'
                    self.guider.wait_stable(self.stabilization_px, self.stabilization_s, stabilization_s_max)
                    time.sleep(self.stabilization_s)
                    next_dither = self.dither_interval

                    self.guider.stop_dither()
                    if self.guider.state != 'guiding':
                        logger.info("Force-stop dither")
                        self.state_detail = 'force stop'
                        self.guider.wait_stable(self.stabilization_px, self.stabilization_s, stabilization_s_max)
                    if self.guider.state != 'guiding':
                        logger.info("Not stabilized, continuing anyway")
                    else:
                        logger.info("Stabilized, continuing")
                elif self.guider.state != 'guiding':
                    logger.info("Guiding unstable, waiting for stabilization before capture")
                    if self.guider.state != 'guiding':
                        self.state_detail = 'wait-stable'
                        self.guider.wait_stable(self.stabilization_px, self.stabilization_s_min, stabilization_s_max)
                    if self.guider.state != 'guiding':
                        logger.info("Not stabilized, continuing anyway")
                    else:
                        logger.info("Stabilized, continuing")
            except Exception:
                self.state = 'cooldown after error'
                self.state_detail = None
                logger.exception("Error capturing sub")
                time.sleep(self.cooldown_s)

        self.state = 'idle'
        self.state_detail = None

    def _capture_unguided(self, num_caps, exposure, cooldown_s, name, seq_attr, pattern, target_dir):
        last_capture = self.last_capture
        for n in xrange(num_caps):
            try:
                seqno = getattr(self, seq_attr)
                logger.info("Starting %s %d", name, seqno)
                self.state = 'capturing'
                self.state_detail = '%s %d' % (name, seqno)
                self.ccd.expose(exposure)
                time.sleep(exposure)
                if self.save_on_client:
                    blob = self.ccd.pullBLOB(self.ccd_name)
                    path = os.path.join(self.base_dir, target_dir, pattern % seqno)
                    with open(path, 'wb') as f:
                        f.write(blob)
                logger.info("Finished %s %d", name, self.flat_seq)

                setattr(self, seq_attr, seqno + 1)

                if self._stop:
                    break

                self.state = 'cooldown'
                time.sleep(cooldown_s)

                if not self.save_on_client:
                    last_capture = self.wait_capture_ready(last_capture, min(cooldown_s, 1))
            except Exception:
                self.state = 'cooldown after error'
                self.state_detail = None
                logger.exception("Error capturing %s", name)
                time.sleep(cooldown_s)

        self.state = 'idle'
        self.state_detail = None

    def capture_flats(self, num_caps, exposure):
        self.ccd.setFlat()
        self.ccd.setUploadSettings(
            upload_dir=os.path.join(self.base_dir, self.flat_target_dir),
            image_type='flat')
        self._capture_unguided(
            num_caps, exposure, self.flat_cooldown_s,
            'flat', 'flat_seq', self.flat_pattern, self.flat_target_dir)

    def capture_darks(self, num_caps, exposure):
        self.ccd.setDark()
        self.ccd.setUploadSettings(
            upload_dir=os.path.join(self.base_dir, self.dark_target_dir),
            image_type='dark')
        self._capture_unguided(
            num_caps, exposure, self.cooldown_s,
            'dark', 'dark_seq', self.dark_pattern, self.dark_target_dir)

    def capture_dark_flats(self, num_caps, exposure):
        self.ccd.setDark()
        self.ccd.setUploadSettings(
            upload_dir=os.path.join(self.base_dir, self.dark_flat_target_dir),
            image_type='flat_dark')
        self._capture_unguided(
            num_caps, exposure, self.flat_cooldown_s,
            'dark_flat', 'flat_dark_seq', self.dark_flat_pattern, self.dark_flat_target_dir)

    def stop(self):
        self._stop = True

    def restart(self):
        self._stop = False


class InteractiveGuider(object):

    def __init__(self, guider_process, guider_controller, ccd_name='CCD1', capture_seq=None):
        self.guider = guider_process
        self.controller = guider_controller
        self.ccd_name = ccd_name
        self.capture_seq = capture_seq
        self.capture_thread = None
        self.stop = False

    def get_helpstring(self):
        helpstring = []
        for name in sorted(dir(self)):
            if not name.startswith('cmd_'):
                continue
            cmdhelp = filter(None, getattr(self, name).__doc__.splitlines())
            indent = min(len(re.match(r"(^ *).*$", l).group(1)) for l in cmdhelp)
            cmdhelp = [l[indent:].rstrip() for l in cmdhelp]
            if not cmdhelp[-1]:
                del cmdhelp[-1:]
            helpstring.extend(cmdhelp)
        return '\n'.join(helpstring)

    def parse_coord(self, coord):
        from cvastrophoto.util import coords

        return coords.parse_coord_string(coord)

    def run(self):
        self.cmd_help()

        while not self.stop:
            cmd = raw_input("\ncvguide> ")
            if not cmd:
                continue

            cmd = cmd.split()
            cmd, args = cmd[0], cmd[1:]

            cmdmethod = getattr(self, 'cmd_' + cmd, None)
            if cmdmethod is None:
                logger.error("Unrecognized command %r", cmd)
                self.cmd_help()
            else:
                try:
                    cmdmethod(*args)
                except Exception:
                    logger.exception("Error executing %s", cmd)

    def cmd_help(self):
        """help: print this help string"""
        helpstring = self.get_helpstring()
        print("""
Commands:

%(helpstring)s

Coordinates are given as RA,DEC,EPOCH with RA given as HH:MM:SS.sssh
orr DEG:MM:SS.sssd, and DEC given as DEG:MM:SS.sssd, no spaces.
Eg: 09:12:17.55h,38:48:06.4d

In those representations, d stands for degrees, h for hours. It's
possible to give explicit per-component units, as:
09h12m17s,38d48m06s or simply use fractional numbers, as
9.327483h,38.78837d.

""" % dict(helpstring=helpstring))

    def cmd_start(self, wait=False):
        """start: start guiding, calibrate if necessary"""
        logger.info("Start guiding")
        self.guider.start_guiding(wait=wait)

    def cmd_stop(self, wait=False):
        """stop: stop guiding"""
        logger.info("Stop guiding")
        self.guider.stop_guiding(wait=wait)

    def cmd_capture(self, exposure, dither_interval=None, dither_px=None):
        """
        capture N [D P]: start capturing N-second subs,
            dither P pixels every D subs
        """
        if self.capture_thread is not None:
            logger.info("Already capturing")

        if dither_interval is not None:
            self.capture_seq.dither_interval = int(dither_interval)
        if dither_px is not None:
            self.capture_seq.dither_px = float(dither_px)

        logger.info("Starting capture")

        self.capture_seq.restart()

        self.capture_thread = threading.Thread(
            target=self.capture_seq.capture,
            args=(float(exposure),))
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def cmd_stop_capture(self, wait=True):
        """stop_capture: stop capturing"""
        logger.info("Stopping capture")
        self.capture_seq.stop()
        if wait:
            self.capture_thread.join()
            logger.info("Stopped capture")
        self.capture_thread = None

    def cmd_capture_flats(self, exposure, num_frames):
        """
        capture_flats T N: start capturing N flats of T-second
        """
        self._capture_unguided(exposure, num_frames, 'flats', 'capture_flats')

    def cmd_capture_darks(self, exposure, num_frames):
        """
        capture_darks T N: start capturing N darks of T-second
        """
        self._capture_unguided(exposure, num_frames, 'darks', 'capture_darks')

    def cmd_capture_dark_flats(self, exposure, num_frames):
        """
        capture_dark_flats T N: start capturing N dark flats of T-second
        """
        self._capture_unguided(exposure, num_frames, 'dark_flats', 'capture_dark_flats')

    def _capture_unguided(self, exposure, num_frames, kind, method):
        if self.capture_thread is not None:
            logger.info("Already capturing")

        logger.info("Starting capture: %s", kind)

        self.capture_seq.restart()

        self.capture_thread = threading.Thread(
            target=getattr(self.capture_seq, method),
            args=(int(num_frames), float(exposure),))
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def cmd_halt(self):
        """halt: stop guiding (and all movement)"""
        logger.info("Halt guiding")
        self.guider.controller.paused = True
        self.guider.stop_guiding(wait=False)

    def cmd_pause_drift(self):
        """pause_drift: stop drift guiding (keep issuing explicit guide pulses)"""
        logger.info("Pause drift guiding")
        self.guider.controller.paused_drift = True

    def cmd_resume_drift(self):
        """resume_drift: resume drift guiding"""
        logger.info("Resumed drift guiding")
        self.guider.controller.paused_drift = False

    def cmd_update_calibration(self):
        """update_calibration: given an initial calibration has been done, update it"""
        logger.info("Initiating calibration update")
        self.guider.update_calibration(wait=False)

    def cmd_calibrate(self):
        """calibrate: reset calibration data and recalibrate from scratch"""
        logger.info("Initiating recalibration")
        self.guider.calibrate(wait=False)

    def cmd_dark(self, n):
        """dark N: take N darks and calibrate"""
        from cvastrophoto.image.base import ImageAccumulator

        n = int(n)
        dark = ImageAccumulator(dtype=numpy.float32)
        self.guider.ccd.setDark()
        for i in xrange(n):
            logger.info("Taking dark %d/%d", i+1, n)
            self.guider.ccd.expose(self.guider.calibration.guide_exposure)
            dimg = self.guider.ccd.pullImage(self.ccd_name).rimg.raw_image
            dark += dimg
        logger.info("Setting master dark")
        self.guider.master_dark = self.guider.calibration.master_dark = dark.average
        del dark, dimg
        self.guider.ccd.setLight()
        logger.info("Done taking master dark")

    def cmd_goto(self, to_, from_=None, speed=None, wait=False, use_guider=False):
        """
        goto to [from speed]: Move to "to" coordinates, assuming the scope is currently
            pointed at "from", and that it moves at "speed" times sideral. If a goto
            mount is connected, a slew command will be given and only "to" is necessary.
            Otherwise, guiding commands will be issued and from/speed are mandatory.
        """
        if self.guider.telescope is not None and not use_guider:
            to_gc = self.parse_coord(to_)

            if self.guider.state.startswith('guiding'):
                self.guider.stop_guiding(wait=True)

            logger.info("Slew to %s", to_gc)
            self.guider.telescope.trackTo(to_gc.ra.hour, to_gc.dec.degree)
            if wait:
                time.sleep(0.5)
                self.guider.telescope.waitSlew()
                logger.info("Slew finished")
        elif from_ and speed:
            to_gc = self.parse_coord(to_)
            from_gc = self.parse_coord(from_)

            logger.info("Shifting from %s to %s", from_gc, to_gc)

            from cvastrophoto.util import coords
            from_gc, to_gc = coords.equalize_frames(from_gc, from_gc, to_gc)

            ra_off, dec_off = from_gc.spherical_offsets_to(to_gc)

            logger.info("Shifting will take %s RA %s DEC", ra_off.hms, dec_off.dms)
            self.guider.shift(dec_off.arcsec, -ra_off.hour * 3600, speed)
        else:
            logger.error("Without a mount connected, from and speed are mandatory")

    def cmd_goto_solve(self, ccd_name, to_, speed, tolerance=60, from_=None, max_steps=10, exposure=8):
        """
        goto_solve ccd to speed [tolerance [from]]: Like goto, but more precise since it will use
            the configured solver to plate-solve and accurately center the given coordinates
            in the selected ccd.

            If from isn't given and the mount isn't a goto mount, and initial plate solve
            will be used to find the current coordinates.

            Valid cameras: guide main
            Tolerance: requested precision in arc-seconds
        """
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        if self.guider.state.startswith('guiding'):
            self.guider.stop_guiding(wait=True)
        if self.capture_thread is not None:
            self.cmd_stop_capture()

        to_gc = self.parse_coord(to_)
        from_gc = self.parse_coord(from_) if from_ is not None else None
        prev_gc = None
        use_guider = self.guider.telescope is None

        # In any case, when we solve, we assume we're close to the target
        # - If using guider, we must be close (guider is slow)
        # - If using goto, the first goto command will supposedly leave us nearby
        hint = (0, 0, to_gc.ra.degree, to_gc.dec.degree)

        if use_guider and speed and from_gc is None:
            # Do an initial plate solving to find our current location
            success, solver, path, coords, kw = self.cmd_solve(ccd_name, exposure, hint=hint, allsky=True)
            if not success:
                return

            fx, fy, fra, fdec = coords

            from_gc = SkyCoord(ra=fra, dec=fdec, unit=u.degree)

        if max_steps > 0:
            self.cmd_goto(to_gc, from_gc, speed, wait=True)

        for i in range(max_steps):
            time.sleep(5)

            success, solver, path, coords, kw = self.cmd_solve(ccd_name, exposure, hint=hint)
            if not success:
                break

            x, y, ra, dec = coords
            sgc = SkyCoord(ra=ra, dec=dec, unit=u.degree)

            d = sgc.separation(to_gc)
            if d.arcsec < tolerance:
                logger.info("Reached target (d=%s)", d.dms)
                break

            if not use_guider and prev_gc is not None and prev_gc.separation(sgc).arcsec < tolerance:
                # Switch to guider steps
                use_guider = True
                logger.info("Goto too imprecise, switching to guider steps")

                if not self.guider.controller.paused_drift:
                    # If drift is paused, calibration won't be precise,
                    # so we manage with existing calibration instead
                    if self.guider.calibration.is_ready:
                        logger.info("Updating calibration")
                        self.guider.update_calibration(wait=True)
                    else:
                        logger.info("Calibrating")
                        self.guider.calibrate(wait=True)

            logger.info("Centering target %s(d=%s)",
                "using guider steps " if use_guider else "",
                d.dms)

            prev_gc = from_gc = sgc

            if self.guider.telescope is not None:
                self.guider.telescope.syncTo(ra, dec)
            self.cmd_goto(to_gc, from_gc, speed, wait=True, use_guider=use_guider)

    def _parse_ccdsel(self, ccd_name):
        ccd_name = ccd_name.lower()
        if ccd_name == 'guide':
            return self.guider.ccd
        elif ccd_name == 'main':
            return self.capture_seq.ccd

    def cmd_move(self, we, ns, speed):
        """
        move RA DEC speed: Move the specified amount of RA seconds W/E and DEC arc-seconds
            N/S (needs calibration) assuming the mount moves at the specified speed.
            Don't use this while guiding, use shift instead.
        """
        self.guider.move(float(ns), float(we), float(speed))

    def cmd_shift(self, we, ns, speed):
        """
        shift RA DEC speed: Shift the specified amount of RA seconds W/E and DEC arc-seconds
            N/S (needs calibration) assuming the mount moves at the specified speed.
            Stops guiding and then re-starts it after the shift has been executed.
        """
        self.guider.shift(float(ns), float(we), float(speed))

    def cmd_dither(self, px):
        """
        dither amount: Shift up to the specified amount of pixels randomly, in a random
            direction. Stops guiding and then re-starts it after the shift has been executed.
        """
        self.guider.dither(float(px))

    def cmd_dither_stop(self):
        """
        dither_stop: Stop dithering. If dithering doesn't stabilize, this reeturns the
            guider to normal mode.
        """
        self.guider.stop_dither()

    def cmd_exit(self):
        """exit: exit the program"""
        self.stop = True

    def show_device_properties(self, device):
        logger.info("Properties for %s:", device.name)
        for propname, val in sorted(device.properties.items()):
            prop = device.getAnyProperty(propname)
            if prop is None:
                valstr = repr(val)
            else:
                valstr = []
                for i, v in enumerate(val):
                    p = prop[i] if i < len(prop) else None
                    if p is None:
                        plabel = 'unk'
                    else:
                        plabel = p.label
                    valstr.append('%s=%r' % (plabel, v))
                valstr = ",".join(valstr)
            logger.info("    %s: [%s]", propname, valstr)

    def cmd_show_cam(self):
        """show_cam: Show camera properties"""
        self.show_device_properties(self.guider.ccd)

    def cmd_show_icam(self):
        """show_icam: Show imaging camera properties"""
        self.show_device_properties(self.capture_seq.ccd)

    def cmd_show_mount(self):
        """show_mount: Show mount properties"""
        if self.guider.telescope is None:
            logger.info("No mount connected")
        else:
            self.show_device_properties(self.guider.telescope)

    def cmd_show_image_header(self):
        """show_image_header: Show guide cam image properties"""
        img_header = self.guider.img_header or self.guider.calibration.img_header

        if img_header is None:
            self.guider.request_snap(wait=True)
            img_header = self.guider.img_header or self.guider.calibration.img_header

        if img_header is None:
            logger.warning("Can't get snapshot, start guider")
            return

        logger.info("Image properties for %s:", self.guider.ccd.name)
        for propname, val in img_header.items():
            logger.info("    %s: %r", propname, val)

    def cmd_show_controller(self):
        """show_controller: Show controller state"""
        logger.info("Controller drift %.4f%% N/S %.4f%% W/E pulse period %.4fs",
            self.guider.controller.ns_drift,
            self.guider.controller.we_drift,
            self.guider.controller.pulse_period)

    def cmd_snap(self):
        """
        snap: Take a snapshot with the guidecam.
            Snapshots are saved to guide_snap.jpg
        """
        self.guider.request_snap(wait=False)

    def cmd_snap_gamma(self, gamma):
        """snap_gamma: Set snapshot gamma, higher values brighten the image."""
        gamma = float(gamma)
        self.guider.snap_gamma = self.guider.calibration.snap_gamma = gamma

    def cmd_snap_bright(self, bright):
        """snap_bright: Set snapshot brightness, higher values brighten the image."""
        bright = float(bright)
        self.guider.snap_bright = self.guider.calibration.snap_bright = bright

    def cmd_start_trace(self):
        """
        start_trace: Start taking traces.
            Traces are cumulative snapshots, of each guide exposure all added
            together. Traces are saved to guide_trace.jpg.
        """
        self.guider.start_trace()

    def cmd_stop_trace(self):
        """stop_trace: Stop taking traces."""
        self.guider.stop_trace()

    def cmd_gain(self, gain):
        """gain N: Set guide camera gain to N."""
        gain = float(gain)
        if 'CCD_GAIN' in self.guider.ccd.properties:
            # Some drivers have a CCD_GAIN property
            self.guider.ccd.setNumber('CCD_GAIN', gain)
        elif 'CCD_CONTROLS' in self.guider.ccd.properties:
            # Some other drivers have a CCD_CONTROLS with multiple settings
            self.guider.ccd.setNumber('CCD_CONTROLS', {'Gain': gain})
        else:
            # If none is present, it may not have arrived yet. Use the stadard-ish CCD_GAIN.
            self.guider.ccd.setNumber('CCD_GAIN', gain)

    def cmd_offset(self, offset):
        """offset N: Set guide camera offset to N."""
        offset = float(offset)
        if 'CCD_CONTROLS' in self.guider.ccd.properties:
            # Some other drivers have a CCD_CONTROLS with multiple settings
            self.guider.ccd.setNumber('CCD_CONTROLS', {'Offset': offset})

    def cmd_exposure(self, exposure):
        """exposure N: Set guide camera exposure to N seconds."""
        self.guider.calibration.guide_exposure = float(exposure)

    def cmd_gui(self):
        """gui: Start the graphical user interface"""
        import cvastrophoto.gui.app
        self.gui = cvastrophoto.gui.app.launch_app(self)

    def cmd_aggression(self, aggression):
        """aggression A: Change aggression to A"""
        self.guider.aggressiveness = float(aggression)

    def cmd_drift_aggression(self, aggression):
        """drift_aggression A: Change drift aggression to A"""
        self.guider.drift_aggressiveness = float(aggression)

    def cmd_solve(self, ccd_name='guide', exposure=8, hint=None, allsky=False, path=None):
        """solve [camera [exposure]]: Plate-solve and find image coordinates"""
        from cvastrophoto.platesolve import astap
        from cvastrophoto.util import imgscale

        telescope = self.guider.telescope
        st4 = self.guider.controller.st4
        info_source = telescope or st4
        ccd = self._parse_ccdsel(ccd_name)
        if ccd is None:
            logger.error("Invalid CCD selected")
            return

        solver = astap.ASTAPSolver()
        solver.tolerance = 'high'

        if allsky:
            solver.search_radius = 90

        if ccd is self.guider.ccd:
            # Request a snapshot and process it
            if path is None:
                self.guider.request_snap()
                path = 'guide_snap.fit'
            fl = self.guider.calibration.eff_guider_fl
        else:
            if path is None:
                # Backup properties
                orig_upload_mode = ccd.properties.get("UPLOAD_MODE")
                orig_transfer_fmt = ccd.properties.get("CCD_TRANSFER_FORMAT")

                try:
                    # Configure for FITS-to-Client transfer
                    ccd.setNarySwitch("UPLOAD_MODE", "Client")
                    if "CCD_TRANSFER_FORMAT" in ccd.properties:
                        ccd.setNarySwitch("CCD_TRANSFER_FORMAT", 0)

                    # Capture a frame and use it
                    ccd.expose(int(exposure))
                    blob = ccd.pullBLOB(self.guider.ccd_name)
                    path = 'solve_snap.fit'
                    with open(path, 'wb') as f:
                        f.write(blob.getblobdata())

                finally:
                    # Restore upload mode
                    ccd.setSwitch("UPLOAD_MODE", orig_upload_mode)
                    if "CCD_TRANSFER_FORMAT" in ccd.properties:
                        ccd.setSwitch("CCD_TRANSFER_FORMAT", orig_transfer_fmt)

            fl = self.guider.calibration.eff_imaging_fl

        # Compute hint
        l, t, r, b = ccd.properties['CCD_FRAME'][:4]
        w = r - l
        h = b - t
        rx = w / 2.0
        ry = h / 2.0

        if hint is None and info_source is not None:
            coords = info_source.properties.get('EQUATORIAL_EOD_COORD')
            if coords:
                ra, dec = coords
                ra = solver.ra_h_to_deg(ra)
                hint = (rx, ry, ra, dec)
        elif hint is not None:
            rx, ry, ra, dec = hint
        else:
            rx = ry = ra = dec = None

        image_scale = fov = None
        pixsz = self.guider.calibration.eff_guider_pixel_size
        if pixsz and fl:
            image_scale = imgscale.compute_image_scale(fl, pixsz)
        if image_scale and h:
            fov = h * image_scale / 3600.0

        kw = dict(hint=hint, fov=fov, image_scale=image_scale)
        success = solver.solve(path, **kw)

        if success:
            sx, sy, sra, sdec = coords = solver.get_coords(path)
            sra = solver.ra_deg_to_h(sra)
            logger.info("Successfully platesolved at coordinates: %r RA %r DEC", sra, sdec)
            if hint is not None:
                ra = solver.ra_deg_to_h(ra)
                logger.info("Original hint coordinates: %r RA %r DEC", ra, dec)
                logger.info("Effective shift: %r RA %r DEC", sra - ra, sdec - dec)
        else:
            logger.info("Plate solving failed")
            coords = None

        return success, solver, path, coords, kw

    def cmd_annotate(self, ccd_name='guide', exposure=8, **kw):
        """annotate [camera [exposure]]: Take a snapshot, and annotate it"""
        success, solver, path, coords, kw = self.cmd_solve(ccd_name, exposure, **kw)

        if success:
            self.last_annotate = annotated = solver.annotate(path, **kw)
            annotated.close()
            logger.info("Annotated image at %s", annotated.name)

            return annotated

    def add_snap_listener(self, listener):
        self.guider.add_snap_listener(listener)

    @property
    def last_capture(self):
        if self.capture_seq is None:
            # Use guide cam settings
            return CaptureSequence(self, self.guider.ccd).last_capture
        else:
            return self.capture_seq.last_capture
