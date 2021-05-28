# -*- coding: utf-8 -*-
from __future__ import print_function, division

from past.builtins import xrange, basestring, raw_input
import threading
import functools
import itertools
import datetime
import time
import logging
import os.path
import numpy
import re
import bisect
import subprocess
import tempfile

import PIL.Image


logger = logging.getLogger(__name__)


def add_opts(subp):
    ap = subp.add_parser('guide', help="Start an interactive guider process")

    ap.add_argument('--darklib', help='Location of the main dark library', default=None)
    ap.add_argument('--biaslib', help='Location of the bias library', default=None)

    ap.add_argument('--phdlog', '-L', help='Write a PHD2-style log file in PATH', metavar='PATH')
    ap.add_argument('--pulselog', help='Write a detailed pulse log file in PATH', metavar='PATH')
    ap.add_argument('--exposure', '-x', help='Guiding exposure length', default=4.0, type=float)
    ap.add_argument('--gain', '-G', help='Guiding CCD gain', type=float)
    ap.add_argument('--offset', '-O', help='Guiding CCD offset', type=float)
    ap.add_argument('--imaging-gain', '-Gi', help='Imaging CCD gain', type=float)
    ap.add_argument('--imaging-offset', '-Oi', help='Imaging CCD offset', type=float)
    ap.add_argument('--autostart', '-A', default=False, action='store_true',
        help='Start guiding immediately upon startup')
    ap.add_argument('--pepa-sim', default=False, action='store_true',
        help='Simulate PE/PA')
    ap.add_argument('--debug-tracks', default=False, action='store_true',
        help='Save debug tracking images to ./Tracks/guide_*.jpg')
    ap.add_argument('--client-timeout', type=float, help='Default timeout waiting for server')

    ap.add_argument('--ra-aggression', '-ara', type=float,
        help='Defines how strongly it will apply immediate corrections')
    ap.add_argument('--dec-aggression', '-adec', type=float,
        help='Defines how strongly it will apply immediate corrections')
    ap.add_argument('--ra-drift-aggression', '-adra', type=float,
        help='Defines the learn rate of the drift model')
    ap.add_argument('--dec-drift-aggression', '-addec', type=float,
        help='Defines the learn rate of the drift model')
    ap.add_argument('--history-length', '-H', type=int,
        help='Defines how long a memory should be used for the drift model, in steps')

    ap.add_argument('--min-pulse', '-Pm', type=float,
        help='Defines the minimum pulse that ought to be sent to the mount, in seconds.')
    ap.add_argument('--min-pulse-ra', '-Pmr', type=float,
        help='Defines the minimum RA pulse that ought to be sent to the mount, in seconds.')
    ap.add_argument('--min-pulse-dec', '-Pmd', type=float,
        help='Defines the minimum DEC pulse that ought to be sent to the mount, in seconds.')
    ap.add_argument('--max-pulse', '-PM', type=float,
        help='Defines the maximum pulse that ought to be sent to the mount, in seconds.')
    ap.add_argument('--target-pulse', '-Pt', type=float,
        help='Defines the optimal pulse that should be sent to the mount, in seconds.')

    ap.add_argument('--track-distance', '-d', type=int, default=192,
        help=(
            'The maximum search distance. The default should be fine. '
            'Lower values consume less resources'
        ))
    ap.add_argument('--backlash-track-distance', '-db', type=int, default=384,
        help=(
            'The maximum search distance during backlash calibration. The default should be fine. '
            'Lower values consume less resources, larger values allow accurate measurement '
            'of larger backlash.'
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
    ap.add_argument('--cfw', help='The name of the color filter wheel', metavar='CFW')
    ap.add_argument('--cfw-max-pos', type=int, metavar='CFWMXP',
        help="The number of positions in the CFW if the driver doesn't report it")
    ap.add_argument('--focus', help='The name of the imaging focuser', metavar='FOCUS')
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
    from cvastrophoto.guiding import controller, guider, calibration, phdlogging, pulselogging
    import cvastrophoto.guiding.simulators.mount
    from cvastrophoto.rops.tracking import correlation, extraction
    from cvastrophoto.image import rgb
    from cvastrophoto.library.darks import DarkLibrary
    from cvastrophoto.library.bias import BiasLibrary

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

    if opts.pulselog:
        pulselogger = pulselogging.PulseLogger(os.path.join(
            opts.pulselog,
            'cvastrophoto_pulselog_%s.txt' % datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')))
        pulselogger.start()
    else:
        pulselogger = None

    bias_library = BiasLibrary(opts.biaslib)
    dark_library = DarkLibrary(opts.darklib)

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
    cfw = indi_client.waitCFW(opts.cfw) if opts.cfw else None
    focuser = indi_client.waitFocuser(opts.focus) if opts.focus else None

    if opts.cfw_max_pos:
        cfw.set_maxpos(opts.cfw_max_pos)

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

    if cfw:
        logger.info("Connecting CFW")
        cfw.connect()

    if focuser:
        logger.info("Connecting Focuser")
        focuser.connect()

    st4.waitConnect(False)
    ccd.waitConnect(False)
    ccd.subscribeBLOB(ccd_name)

    # CFW takes longer to connect usually, so we'll wait after initializing everything else

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

    # We'll need the guider CCD's blobs
    ccd.waitPropertiesReady()
    ccd.setNarySwitch("TELESCOPE_TYPE", "Guide", quick=True, optional=True)

    logger.info("Detecting CCD info")
    ccd.detectCCDInfo(ccd_name)
    if imaging_ccd is not None:
        imaging_ccd.detectCCDInfo(iccd_name)
    logger.info("Detected CCD info")

    ccd.setUploadClient()

    if cfw:
        cfw.waitConnect(False)

    if focuser:
        focuser.waitConnect(False)

    tracker_class = functools.partial(correlation.CorrelationTrackingRop,
        track_distance=opts.track_distance,
        resolution=opts.track_resolution,
        luma_preprocessing_rop=extraction.ExtractStarsRop(
            rgb.Templates.LUMINANCE, copy=False, quick=True))

    backlash_tracker_class = functools.partial(correlation.CorrelationTrackingRop,
        track_distance=opts.backlash_track_distance,
        resolution=opts.track_resolution,
        luma_preprocessing_rop=extraction.ExtractStarsRop(
            rgb.Templates.LUMINANCE, copy=False, quick=True))

    if opts.pepa_sim:
        controller_class = cvastrophoto.guiding.simulators.mount.PEPASimGuiderController
        controller_class.we_speed = opts.pepa_ra_speed
        controller_class.ns_speed = opts.pepa_dec_speed
    else:
        controller_class = controller.GuiderController
    guider_controller = controller_class(telescope, st4, pulselogger=pulselogger)

    if opts.min_pulse:
        guider_controller.min_pulse_ra = guider_controller.min_pulse_dec = opts.min_pulse
    if opts.min_pulse_ra:
        guider_controller.min_pulse_ra = opts.min_pulse_ra
    if opts.min_pulse_dec:
        guider_controller.min_pulse_dec = opts.min_pulse_dec
    if opts.max_pulse:
        guider_controller.max_pulse = opts.max_pulse
    if opts.target_pulse:
        guider_controller.target_pulse = opts.target_pulse

    calibration_seq = calibration.CalibrationSequence(
        telescope, guider_controller, ccd, ccd_name, tracker_class,
        phdlogger=phdlogger, backlash_tracker_class=backlash_tracker_class)
    calibration_seq.guide_exposure = opts.exposure
    if opts.guide_fl:
        calibration_seq.guider_fl = opts.guide_fl
    if opts.imaging_fl:
        calibration_seq.imaging_fl = opts.imaging_fl
    guider_process = guider.GuiderProcess(
        telescope, calibration_seq, guider_controller, ccd, ccd_name, tracker_class,
        phdlogger=phdlogger, dark_library=dark_library, bias_library=bias_library)
    guider_process.save_tracks = opts.debug_tracks
    if opts.ra_aggression:
        guider_process.ra_aggressivenes = opts.ra_aggression
    if opts.dec_aggression:
        guider_process.dec_aggressivenes = opts.dec_aggression
    if opts.ra_drift_aggression:
        guider_process.ra_drift_aggressiveness = opts.ra_drift_aggression
    if opts.dec_drift_aggression:
        guider_process.dec_drift_aggressiveness = opts.dec_drift_aggression
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
        capture_seq = CaptureSequence(
            guider_process, imaging_ccd, iccd_name,
            phdlogger=phdlogger, cfw=cfw, focuser=focuser,
            dark_library = dark_library, bias_library=bias_library)
        capture_seq.save_on_client = False
        capture_seq.save_native = opts.save_native

        imaging_ccd.waitPropertiesReady()

        imaging_ccd.setUploadLocal()
        if imaging_ccd is not ccd:
            imaging_ccd.setNarySwitch("TELESCOPE_TYPE", "Primary", quick=True, optional=True)
        if opts.save_dir or opts.save_prefix:
            if opts.save_dir:
                capture_seq.base_dir = opts.save_dir
            imaging_ccd.setUploadSettings(
                upload_dir=os.path.join(capture_seq.base_dir, capture_seq.target_dir),
                image_type='light')
        if imaging_ccd.transfer_format is not None:
            if opts.save_native:
                imaging_ccd.setTransferFormatNative()
            else:
                imaging_ccd.setTransferFormatFits()
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
    if opts.imaging_gain:
        iguider.cmd_igain(opts.imaging_gain)
    if opts.imaging_offset:
        iguider.cmd_ioffset(opts.imaging_offset)

    iguider.run()

    logger.info("Shutting down")
    guider_process.stop()
    guider_controller.stop()
    indi_client.stopWatchdog()
    iguider.destroy()

    logger.info("Exit")


class AbortError(Exception):
    pass


class CaptureSequence(object):

    dither_interval = 5
    dither_px = 20
    stabilization_s_min = 5
    stabilization_s = 10
    stabilization_s_max = 30
    stabilization_px = 4
    cooldown_s = 10
    flat_cooldown_s = 1
    filter_change_timeout = 10

    save_on_client = False
    save_native = True
    master_dark = None

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

    sound_play_command = "aplay"
    finish_sound = os.path.join(os.path.dirname(__file__), "..", "..", "..", "resources", "sounds", "ding.wav")

    def __init__(self, guider_process, ccd, ccd_name='CCD1', phdlogger=None, cfw=None, focuser=None,
            dark_library=None, bias_library=None):
        self.guider = guider_process
        self.ccd = ccd
        self.ccd_name = ccd_name
        self.cfw = cfw
        self.focuser = focuser
        self.state = 'idle'
        self.state_detail = None
        self.phdlogger = phdlogger
        self.new_capture = False
        self.dark_library = dark_library
        self.bias_library = bias_library
        self.sleep = time.sleep
        self._stop = False

    def _play_sound(self, which):
        return subprocess.Popen([self.sound_play_command, which]).wait()

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
        except OSError:
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
            self.sleep(sleep_time)
            cur_last_capture = self.last_capture
        return cur_last_capture

    def _parse_filter_sequence(self, filter_sequence):
        if isinstance(filter_sequence, list):
            if isinstance(filter_sequence[0], basestring):
                filter_sequence = map(self.cfw.filter_names.index, filter_sequence)
        elif isinstance(filter_sequence, basestring):
            filter_sequence = list(map(int, filter_sequence))
        return filter_sequence

    def _change_filter(self, filter_sequence, force=False, next_filter=None, image_type='light', target_dir=None):
        if self.cfw is not None and filter_sequence is not None:
            if next_filter is None:
                next_filter = next(filter_sequence)
            if next_filter != self.cfw.curpos:
                logger.info("Changing filter to pos %d", next_filter)
                self.cfw.set_curpos(next_filter, wait=True, timeout=self.filter_change_timeout)
                if self.cfw.curpos != next_filter:
                    logger.warning("Filter change unsuccessful, continuing with errors")
                force = True
        if force:
            if target_dir is None:
                target_dir = self.target_dir
            self.ccd.setUploadSettings(
                upload_dir=os.path.join(self.base_dir, target_dir),
                image_type=image_type,
                image_suffix=self.cfw.curfilter if self.cfw is not None else None)
        return force

    def _init_filter_sequence(self, filter_sequence, set_type, image_type, target_dir=None):
        if filter_sequence is not None:
            filter_sequence = self._parse_filter_sequence(filter_sequence)
            raw_filter_sequence = filter_sequence
            filter_set = set(filter_sequence)
            filter_sequence = iter(itertools.cycle(filter_sequence))
        else:
            filter_set = None
            raw_filter_sequence = filter_sequence

        def change_filter(force=False, next_filter=None):
            if self._change_filter(filter_sequence, force, next_filter, image_type, target_dir=target_dir):
                set_type()

        # Cycle through the filter sequence in fast succession
        # The point of this is to make sure the CFW ends up in a consistent, reproducible
        # position at the start of the sequence. This will make flats more successful.
        if filter_sequence is not None:
            for next_filter in raw_filter_sequence:
                change_filter(False, next_filter)

        change_filter(True)

        return change_filter, filter_set, raw_filter_sequence

    def init_capture(self):
        if self.save_on_client:
            self.ccd.setUploadClient()
        else:
            self.ccd.setUploadLocal()
        if self.ccd.transfer_format is not None:
            if self.save_native:
                self.ccd.setTransferFormatNative()
            else:
                self.ccd.setTransferFormatFits()

    def capture(self, exposure, number=None, filter_sequence=None, filter_exposures=None):
        next_dither = self.dither_interval
        last_capture = self.last_capture

        change_filter, filter_set, filter_sequence = self._init_filter_sequence(
            filter_sequence, self.ccd.setLight, 'light')

        logger.info("Filter exposures: %r", filter_exposures)

        self.init_capture()

        while not self._stop and (number is None or number > 0):
            try:
                sub_exposure = exposure
                if self.cfw is not None:
                    cur_filter_name = self.cfw.curfilter
                    if filter_exposures is not None:
                        sub_exposure = filter_exposures.get(
                            cur_filter_name, filter_exposures.get(self.cfw.curpos, exposure))
                else:
                    cur_filter_name = ''

                logger.info("Starting sub exposure %d %s exposure %s", self.start_seq, cur_filter_name, sub_exposure)
                self.state = 'capturing'
                self.state_detail = 'sub %d %s' % (self.start_seq, cur_filter_name)

                if self.phdlogger is not None:
                    try:
                        self.phdlogger.info("Sub %d start", self.start_seq)
                    except Exception:
                        logger.exception("Error writing to PHD log")

                self.ccd.expose(sub_exposure)
                self.sleep(sub_exposure)

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

                if number is not None:
                    number -= 1

                if self._stop or (number is not None and not number):
                    self.new_capture = True
                    break

                self.state = 'cooldown'
                if next_dither > 0:
                    self.sleep(self.cooldown_s)
                else:
                    # Shorter sleep in case the main cam is still exposing
                    self.sleep(min(self.cooldown_s, 1))

                if not self.save_on_client:
                    last_capture = self.wait_capture_ready(last_capture, min(self.cooldown_s, 1))
                self.new_capture = True

                # Use stabilization time to change filter
                # Don't change filter before capture is ready or it could ruin an ongoing capture
                # if it somehow took longer than it was supposed to
                change_filter()

                # Even if we don't stabilize in s_max time, it's worth waiting
                # half the exposure length. If stabiliztion delays a bit and we
                # start shooting, we'll waste "exposure" sub time, so we might
                # as well spend it waiting.
                # If we're forced to wait for longer, however, it's better to be
                # exposing, in case things do stabilize and the sub turns out
                # usable anyway.
                stabilization_s_max = max(self.stabilization_s_max, sub_exposure / 2)

                if next_dither <= 0:
                    self.state = 'dither'
                    self.state_detail = 'start'
                    logger.info("Starting dither")
                    self.guider.dither(self.dither_px)

                    self.state_detail = 'wait stable'
                    self.sleep(self.stabilization_s)
                    self.guider.wait_stable(self.stabilization_px, self.stabilization_s, stabilization_s_max)
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
                self.sleep(self.cooldown_s)

        self.state = 'idle'
        self.state_detail = None

    def _capture_unguided(self, num_caps, exposure, cooldown_s, name, seq_attr, pattern, target_dir):
        if not self.save_on_client:
            last_capture = self.last_capture

        self.init_capture()

        for n in xrange(num_caps):
            try:
                seqno = getattr(self, seq_attr)
                logger.info("Starting %s %d", name, seqno)
                self.state = 'capturing'
                self.state_detail = '%s %d' % (name, seqno)
                self.ccd.expose(exposure)
                self.sleep(exposure)
                if self.save_on_client:
                    blob = self.ccd.pullBLOB(self.ccd_name)
                    path = os.path.join(self.base_dir, target_dir, pattern % seqno)
                    with open(path, 'wb') as f:
                        f.write(blob)
                logger.info("Finished %s %d", name, seqno)

                setattr(self, seq_attr, seqno + 1)

                if self._stop:
                    self.new_capture = True
                    break

                self.state = 'cooldown'
                self.sleep(cooldown_s)

                if not self.save_on_client:
                    last_capture = self.wait_capture_ready(last_capture, min(cooldown_s, 1))
                self.new_capture = True
            except Exception:
                self.state = 'cooldown after error'
                self.state_detail = None
                logger.exception("Error capturing %s", name)
                self.sleep(cooldown_s)

        self.state = 'idle'
        self.state_detail = None

    def find_exposure(self, target_adu, exposures, cooldown_s):
        lo = 0
        hi = len(exposures) - 1

        orig_transfer_format = self.ccd.transfer_format
        self.ccd.setUploadClient()
        self.ccd.setTransferFormatFits(quick=True, optional=orig_transfer_format is None)

        try:
            med = lo
            loadu = None
            while lo < hi:
                if self._stop:
                    return None

                exposure = exposures[med]
                logger.info("Testing exposure %g", exposure)
                self.ccd.expose(exposure)
                img = self.ccd.pullImage(self.ccd_name)
                img.name = 'test_exp_%s' % (exposure,)
                imgpp = img.postprocessed
                if len(imgpp.shape) < 3:
                    imgpp = imgpp.reshape(imgpp.shape + (1,) * (3 - len(img.shape)))

                adu = 0
                for c in xrange(imgpp.shape[2]):
                    adu = max(adu, numpy.median(imgpp[:,:,c]))

                logger.info("Got %d ADU at exposure %g", adu, exposure)

                if adu < target_adu:
                    lo = med
                    loadu = adu
                elif adu == target_adu:
                    lo = med
                    loadu = adu
                    break
                else:
                    hi = med - 1

                if loadu is not None:
                    loexp = exposures[lo]
                    hiexp = exposures[hi]
                    medexp = loexp * target_adu / loadu
                    med = min(hi, bisect.bisect_right(exposures, medexp, lo=lo, hi=hi))
                else:
                    lo = med
                    break
        finally:
            if orig_transfer_format is not None:
                self.ccd.setTransferFormat(orig_transfer_format)

        exposure = exposures[lo]

        return exposure

    def capture_flats(self, num_caps, exposure, target_adu=None, set_upload_settings=True, notify_finish=True):
        import cvastrophoto.constants.exposure

        self.ccd.setFlat()

        if exposure is None and target_adu is not None:
            self.state = 'Calibrate exposure'
            exposure = self.find_exposure(
                target_adu,
                cvastrophoto.constants.exposure.FLAT_EXPOSURE_VALUES,
                self.flat_cooldown_s)
            if exposure is None:
                self.state = 'idle'
                self.state_detail = None
                logger.info("Aborting flat exposures")
                return
            logger.info("Optimum flat exposure set to %g", exposure)

        if set_upload_settings:
            self.ccd.setUploadSettings(
                upload_dir=os.path.join(self.base_dir, self.flat_target_dir),
                image_type='flat')
        self._capture_unguided(
            num_caps, exposure, self.flat_cooldown_s,
            'flat', 'flat_seq', self.flat_pattern, self.flat_target_dir)

        if notify_finish:
            self._play_sound(self.finish_sound)

        self.state = 'idle'
        self.state_detail = None

        return exposure

    def auto_flats(self, num_caps, target_adu, filter_sequence=None):
        change_filter, filter_set, filter_sequence = self._init_filter_sequence(
            filter_sequence, self.ccd.setFlat, 'flat', target_dir=self.flat_target_dir)
        done_set = set()
        exposures = set()

        if filter_sequence is None:
            exposure = self.capture_flats(num_caps, None, target_adu, notify_finish=False)
            exposures.add(exposure)
        else:
            for next_filter in filter_sequence:
                if next_filter not in done_set:
                    change_filter(False, next_filter)
                    exposure = self.capture_flats(num_caps, None, target_adu, False, notify_finish=False)
                    exposures.add(exposure)
                    done_set.add(next_filter)
                if done_set >= filter_set:
                    break

        logger.info("Exposures used: %r", exposures)
        logger.info("Check if you need new darks")
        self._play_sound(self.finish_sound)

        return exposures

    def _measure_focus(self, exposure, state, snap_callback=None):
        import cvastrophoto.image
        from cvastrophoto.image.rgb import Templates
        from cvastrophoto.rops.measures import fwhm, focus

        self.ccd.expose(exposure)
        img = self.ccd.pullImage(self.ccd_name)
        img.name = 'test_focus'

        pos = self.focuser.absolute_position

        if snap_callback is not None:
            try:
                snap_callback(img)
            except Exception:
                logger.exception("Error in focus snap callback")

        if img.postprocessing_params is not None:
            img.postprocessing_params.half_size = True
        imgpp = img.postprocessed

        if len(imgpp.shape) > 2:
            imgpp = numpy.average(imgpp, axis=2)
            img.close()
            img = Templates.LUMINANCE

        fwhm_rop = state.get('fwhm_rop', None)
        if fwhm_rop is None:
            state['fwhm_rop'] = fwhm_rop = fwhm.FWHMMeasureRop(img, quick=True)
        fwhm_value = fwhm_rop.measure_scalar(imgpp)

        focus_rop = state.get('focus_rop', None)
        if focus_rop is None:
            state['focus_rop'] = focus_rop = focus.FocusMeasureRop(img, quick=True)
        focus_value = focus_rop.measure_scalar(imgpp)

        logger.info("Measured focus: pos=%s fwhm=%g contrast=%g", pos, fwhm_value, focus_value)
        return pos, fwhm_value, focus_value

    def _probe_focus(
            self,
            direction, initial_step, min_step, max_step, max_steps, exposure, initial_sample, current_sample, state,
            accel=1.25):
        ipos, ifwhm, ifocus = initial_sample
        pos, fwhm, focus = current_sample

        max_fwhm = min(fwhm + ifwhm, fwhm * 3)

        logger.info(
            "Focus probe: direction=%s ipos=%s istep=%s max_step=%s accel=%g max_fwhm=%g",
            direction, pos, initial_step, max_step, accel, max_fwhm,
        )

        # Move forward until we're consistently above max_fwhm
        step = initial_step
        count = 0
        confirm_count = 3
        nstep = 0
        samples = state.setdefault('samples', [])

        def apply_focus_step(img):
            logger.info("Moving focus position by %s", direction * step)
            self.focuser.moveRelative(direction * step)

        # Initial move
        apply_focus_step(None)
        self.focuser.waitMoveDone(30)

        while nstep < max_steps and (fwhm < max_fwhm or count < confirm_count):
            if self._stop:
                raise AbortError("Aborted by user")

            pos, fwhm, focus = current_sample = self._measure_focus(exposure, state, snap_callback=apply_focus_step)
            samples.append(current_sample)
            nstep += 1

            if fwhm >= max_fwhm:
                count += 1
                step = max(min_step, int(step / accel))
            else:
                step = min(max_step, int(step * accel))

            self.focuser.waitMoveDone(30)

    def __find_best_focus(self, samples):
        import sklearn.linear_model
        import sklearn.preprocessing
        import sklearn.pipeline

        best_sample_focus = max(samples, key=lambda sample:sample[2])

        best_focus = best_sample_focus[2]
        best_focus_fwhm = best_sample_focus[1]

        model = sklearn.pipeline.Pipeline([
            ('poly', sklearn.preprocessing.PolynomialFeatures(degree=2)),
            ('linear', sklearn.linear_model.RidgeCV(alphas=numpy.logspace(0, 1, 12), cv=2))
        ])
        X = numpy.array([[sample[0]] for sample in samples])
        Y = numpy.array([[sample[2]] for sample in samples])
        model.fit(X, Y ** 0.25)
        Xfull = numpy.arange(int(X.min()), int(X.max()))
        Xfull = Xfull.reshape((Xfull.size, 1))
        Yfull = model.predict(Xfull) ** 4
        best_focus_ix = Yfull[:,0].argmax()
        best_focus_pos = int(Xfull[best_focus_ix,0])

        # Check FWHM, only samples above median focus ranking (other samples tend to be inaccurate)
        median_focus = numpy.median([s[2] for s in samples])
        best_samples = [s for s in samples if s[2] >= median_focus]

        best_sample_fwhm = min(best_samples, key=lambda sample:sample[1])

        # TODO: Compute best focus by interpolating Y
        # (the model accurately predicts the vertex location but not necessarily the score)
        best_sample_focus = best = (best_focus_pos, best_focus_fwhm, best_focus)

        return best_sample_fwhm, best_sample_focus, best, model

    def _find_best_focus(self, state):
        samples = state['samples']
        best_fwhm, best_focus, best, focus_model = self.__find_best_focus(samples)

        state.update(dict(
            best_focus=best_focus,
            best_fwhm=best_fwhm,
            best=best,
            focus_model=focus_model,
            min_pos=min(samples)[0],
            max_pos=max(samples)[0],
        ))
        return best

    def _show_focus_curve(self, state):
        X = numpy.linspace(state['min_pos'], state['max_pos'], 200)
        X = X.reshape((X.size, 1))
        Y = state['focus_model'].predict(X) ** 4

        # Temporary hack until we can have a proper UI for this
        import matplotlib.pyplot as plt
        samples = numpy.array(sorted(state['samples'])).T
        best_focus = numpy.array([state['best_focus']]).T
        best_fwhm = numpy.array([state['best_fwhm']]).T
        best = numpy.array([state['best']]).T
        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].scatter(samples[0], samples[1], c="#008000", marker="+")
        ax[1].scatter(samples[0], samples[2], c="#000080", marker=".")
        ax[0].scatter(best_fwhm[0], best_fwhm[1], c="#00F000", marker="*")
        ax[1].scatter(best_focus[0], best_focus[2], c="#0000F0", marker="o")
        ax[0].scatter(best[0], best[1], c="#00F000", marker="1")
        ax[1].scatter(best[0], best[2], c="#0000F0", marker="1")
        ax[1].plot(X[:,0], Y[:,0], c="#000040", linestyle="dashed")
        ax[1].set_xlabel('position')
        ax[1].set_ylabel('contrast')
        ax[0].set_ylabel('FWHM')
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            fig.savefig(f)
            f.seek(0)
            PIL.Image.open(f).show()

    def auto_focus(self, initial_step, min_step, max_step, max_steps, exposure, notify_finish=True, show_curve=True):
        self.ccd.setLight()

        orig_transfer_format = self.ccd.transfer_format
        self.ccd.setUploadClient()
        self.ccd.setTransferFormatFits(quick=True, optional=orig_transfer_format is None)

        try:
            initial_pos = self.focuser.absolute_position
            samples = []
            state = {'samples': samples}
            logger.info("Initiating autofocus with exposure %g, starting position %s", exposure, initial_pos)

            initial_pos, fwhm, focus = initial_sample = self._measure_focus(exposure, state)
            samples.append(initial_sample)
            logger.info("Initial focus values: pos=%s fwhm=%g contrast=%g", initial_pos, fwhm, focus)

            self._probe_focus(
                1,
                initial_step, min_step, max_step, max_steps,
                exposure, initial_sample, initial_sample, state)

            self.focuser.setAbsolutePosition(initial_pos)
            self.focuser.waitMoveDone(60)

            self._probe_focus(
                -1, initial_step, min_step, max_step, max_steps,
                exposure, initial_sample, initial_sample, state)

            best_pos, best_fwhm, best_focus = best_sample = self._find_best_focus(state)
            logger.info("Best focus at pos=%s fwhm=%g contrast=%g", best_pos, best_fwhm, best_focus)
            self.focuser.setAbsolutePosition(best_pos)
            self.focuser.waitMoveDone(60)

            best_pos, best_fwhm, best_focus = best_sample = self._measure_focus(exposure, state)
            logger.info("Measured best focus at pos=%s fwhm=%g contrast=%g", best_pos, best_fwhm, best_focus)
            logger.info("Autofocusing finished")
        except AbortError:
            logger.info("Focus routine aborted")
            state.clear()

        finally:
            if orig_transfer_format is not None:
                self.ccd.setTransferFormat(orig_transfer_format)

            if notify_finish:
                self._play_sound(self.finish_sound)

            self.state = 'idle'
            self.state_detail = None

        if state:
            self.auto_focus_state = state

            if show_curve:
                try:
                    self._show_focus_curve(state)
                except Exception:
                    logger.exception("Error showing focus curve")

        return exposure

    def capture_darks(self, num_caps, exposure):
        self.ccd.setDark()
        self.ccd.setUploadSettings(
            upload_dir=os.path.join(self.base_dir, self.dark_target_dir),
            image_type='dark')
        self._capture_unguided(
            num_caps, exposure, self.cooldown_s,
            'dark', 'dark_seq', self.dark_pattern, self.dark_target_dir)
        self._play_sound(self.finish_sound)

    def capture_dark_flats(self, num_caps, exposure):
        self.ccd.setDark()
        self.ccd.setUploadSettings(
            upload_dir=os.path.join(self.base_dir, self.dark_flat_target_dir),
            image_type='flat_dark')
        self._capture_unguided(
            num_caps, exposure, self.flat_cooldown_s,
            'dark_flat', 'flat_dark_seq', self.dark_flat_pattern, self.dark_flat_target_dir)
        self._play_sound(self.finish_sound)

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
        self.goto_state = None
        self.goto_state_detail = None
        self.gui = None

    def get_helpstring(self):
        helpstring = []
        for name in sorted(dir(self)):
            if not name.startswith('cmd_'):
                continue
            cmdhelp = list(filter(None, getattr(self, name).__doc__.splitlines()))
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

    def destroy(self):
        if self.gui is not None:
            self.gui.shutdown()

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

    def cmd_capture(self, exposure, dither_interval=None, dither_px=None, number=None,
            filter_sequence=None, filter_exposures=None):
        """
        capture N [D P [L]]: start capturing N-second subs,
            dither P pixels every D subs. Capture up to L subs.
        """
        if self.capture_thread is not None:
            logger.info("Already capturing")

        if dither_interval is not None:
            self.capture_seq.dither_interval = int(dither_interval)
        if dither_px is not None:
            self.capture_seq.dither_px = float(dither_px)
        if number is not None:
            number = int(number)

        logger.info("Starting capture")

        self.capture_seq.restart()

        self.capture_thread = threading.Thread(
            target=self.capture_seq.capture,
            args=(float(exposure), number, filter_sequence or None, filter_exposures or None))
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

    def cmd_auto_flats(self, num_frames, target_adu, filter_sequence=None):
        """
        auto_flats N ADU [SEQ]: start capturing N flats targeting ADU median ADU
            optionally following the filter sequence SEQ
        """
        self._auto_flats(num_frames, target_adu, filter_sequence)

    def cmd_auto_focus(self, initial_step, min_step, max_step, max_steps, exposure):
        """
        auto_focus STEP_INI STEP_MIN STEP_MAX MAX_N_STEPS EXPOSURE:
            Automatically find the best focus point (requires an autofocuser)
        """
        self._auto_focus(initial_step, min_step, max_step, max_steps, exposure)

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

    def _auto_flats(self, num_frames, target_adu, filter_sequence):
        if self.capture_thread is not None:
            logger.info("Already capturing")

        logger.info("Starting capture: auto flats")

        self.capture_seq.restart()

        self.capture_thread = threading.Thread(
            target=self.capture_seq.auto_flats,
            args=(int(num_frames), float(target_adu), filter_sequence))
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def _auto_focus(self, initial_step, min_step, max_step, max_steps, exposure):
        if self.capture_thread is not None:
            logger.info("Already capturing")

        logger.info("Starting capture: auto focus")

        self.capture_seq.restart()

        self.capture_thread = threading.Thread(
            target=self.capture_seq.auto_focus,
            args=(int(initial_step), int(min_step), int(max_step), int(max_steps), float(exposure)))
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

    def cmd_goto(self, to_, from_=None, speed=None, wait=False, use_guider=False, set_state=True):
        """
        goto to [from speed]: Move to "to" coordinates, assuming the scope is currently
            pointed at "from", and that it moves at "speed" times sideral. If a goto
            mount is connected, a slew command will be given and only "to" is necessary.
            Otherwise, guiding commands will be issued and from/speed are mandatory.
        """
        if set_state:
            self.goto_state = 'Slew'
        try:
            if self.guider.telescope is not None and not use_guider:
                to_gc = self.parse_coord(to_)

                if self.guider.state.startswith('guiding'):
                    self.guider.stop_guiding(wait=True)

                logger.info("Slew to %s", to_gc)
                if set_state:
                    self.goto_state_detail = 'Go-to'
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

                dec_dir = -self.guider.calibration.dec_handedness

                logger.info("Shifting will take %s RA %s DEC", ra_off.hms, dec_off.dms)
                if set_state:
                    self.goto_state_detail = 'Shift'
                self.guider.shift(dec_dir * dec_off.arcsec, -ra_off.hour * 3600, speed)
            else:
                logger.error("Without a mount connected, from and speed are mandatory")
        finally:
            if set_state:
                self.goto_state = self.goto_state_detail = None

    def cmd_goto_solve(self, ccd_name, to_, speed,
            tolerance=60, from_=None,
            max_steps=10, exposure=8, recalibrate=True):
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

        self.goto_state = 'Slew and solve'

        try:
            if use_guider and speed and from_gc is None:
                # Do an initial plate solving to find our current location
                self.goto_state_detail = 'Blind platesolve'
                success, solver, path, coords, hdu, kw = self.cmd_solve(ccd_name, exposure, hint=hint, allsky=True)
                if not success:
                    return

                fx, fy, fra, fdec = coords

                from_gc = SkyCoord(ra=fra, dec=fdec, unit=u.degree)

            if max_steps > 0:
                self.goto_state_detail = 'Approach %d/%d' % (1, max_steps)
                self.cmd_goto(to_gc, from_gc, speed, wait=True, set_state=False)

            for i in range(max_steps):
                time.sleep(5)

                self.goto_state_detail = 'Solve %d/%d' % (i + 1, max_steps)
                success, solver, path, coords, hdu, kw = self.cmd_solve(ccd_name, exposure, hint=hint)
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
                        self.goto_state_detail = 'Calibration'
                        if self.guider.calibration.is_ready:
                            if recalibrate:
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
                self.goto_state_detail = 'Approach %d/%d' % (i + 1, max_steps)
                self.cmd_goto(to_gc, from_gc, speed, wait=True, use_guider=use_guider, set_state=False)
        finally:
            self.goto_state = self.goto_state_detail = None

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
        shift RA DEC [speed]: Shift the specified amount of RA seconds W/E and DEC arc-seconds
            N/S (needs calibration) assuming the mount moves at the specified speed.
            Stops guiding and then re-starts it after the shift has been executed.
        """
        calibration = self.guider.calibration
        if not speed and calibration.is_ready:
            speed = calibration.guide_speed
            logger.info("Computed guiding speed %.3f", speed)
        self.guider.shift(float(ns), float(we), float(speed))

    def cmd_shift_pixels(self, x, y, speed):
        """
        shift_pixels RA DEC [speed]: Shift the specified amount of guide pixels
            using calibration data to figure out how to translate to RA/DEC shift.
            Stops guiding and then re-starts it after the shift has been executed.
        """
        calibration = self.guider.calibration
        if not calibration.is_ready or not calibration.image_scale:
            logger.error("Calibration not ready for pixel shift")
            return

        if not speed:
            speed = calibration.guide_speed
            logger.info("Computed guiding speed %.3f", speed)
        we, ns = calibration.project_ec((float(y), float(x)))

        self.guider.shift(ns, we, speed, seconds=True)

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

    def cmd_set_reference(self, x, y):
        """
        set_reference x y: Set tracking reference point to given x-y coordinates.
        """
        self.guider.set_reference((float(y), float(x)))

    def cmd_set_backlash(self, ra_backlash, dec_backlash):
        """
        set_backlash RA DEC [RATE]: Set backlash compensation max gear state for
            RA and DEC.
        """
        self.guider.calibration.set_backlash(float(dec_backlash), float(ra_backlash))

    def cmd_flip_pier_side(self):
        """
        flip_pier_side: Flip calibration data after a manual side of pier change
        """
        self.guider.calibration.flip_pier_side()
        self.guider.controller.flip_pier_side()
        logger.info("Flipped calibration data")

    def _get_component(self, component):
        if component == 'guider':
            return self.guider
        elif component == 'controller':
            return self.guider.controller
        elif component == 'calibration':
            return self.guider.calibration
        elif component == 'backlash':
            from cvastrophoto.guiding import backlash
            return backlash.BacklashCompensation
        else:
            logger.warn("Unrecognized component %s", component)

    def cmd_set_param(self, component, name, value):
        """
        set_param component name value: Set a parameter to a new value
        """
        component_obj = self._get_component(component)
        if component_obj is None:
            return

        NONE = object()
        curval = getattr(component_obj, name, NONE)
        if curval is NONE:
            logger.warn("Unrecognized parameter %s of %s", name, component)
            return

        value = type(curval)(value)
        setattr(component_obj, name, value)
        logger.info("Set %s.%s = %r", component, name, value)
        if self.guider.phdlogger is not None:
            self.guider.phdlogger.info("Set %s.%s = %r", component, name, value)

    def cmd_show_param(self, component, name):
        """
        show_param component name: Show a parameter's value
        """
        component_obj = self._get_component(component)
        if component_obj is None:
            return

        NONE = object()
        curval = getattr(component_obj, name, NONE)
        if curval is NONE:
            logger.warn("Unrecognized parameter %s of %s", name, component)
            return
        logger.info("%s.%s = %r", component, name, curval)

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

    def cmd_gain(self, gain, ccd=None):
        """gain N: Set guide camera gain to N."""
        gain = float(gain)
        if ccd is None:
            ccd = self.guider.ccd
        ccd.set_gain(gain)

    def cmd_offset(self, offset, ccd=None):
        """offset N: Set guide camera offset to N."""
        offset = float(offset)
        if ccd is None:
            ccd = self.guider.ccd
        ccd.set_offset(offset)

    def cmd_igain(self, gain, ccd=None):
        """igain N: Set imaging camera gain to N."""
        self.cmd_gain(gain, ccd=self.capture_seq.ccd)

    def cmd_ioffset(self, offset, ccd=None):
        """offset N: Set imaging camera offset to N."""
        self.cmd_offset(offset, ccd=self.capture_seq.ccd)

    def cmd_exposure(self, exposure):
        """exposure N: Set guide camera exposure to N seconds."""
        self.guider.calibration.guide_exposure = float(exposure)

    def cmd_gui(self):
        """gui: Start the graphical user interface"""
        import cvastrophoto.gui.app
        self.gui = cvastrophoto.gui.app.launch_app(self)

    def cmd_aggression(self, ra_aggression, dec_aggression):
        """aggression A_RA A_DEC: Change aggression to A"""
        self.guider.ra_aggressiveness = float(ra_aggression)
        self.guider.dec_aggressiveness = float(dec_aggression)

    def cmd_drift_aggression(self, ra_aggression, dec_aggression):
        """drift_aggression A_RA A_DEC: Change drift aggression to A (ra/dec)"""
        self.guider.ra_drift_aggressiveness = float(ra_aggression)
        self.guider.dec_drift_aggressiveness = float(dec_aggression)

    def cmd_log(self, *msg):
        """log MESSAGE: Write a custom message to the guide log"""
        if self.guider.phdlogger is not None:
            self.guider.phdlogger.info('%s', ' '.join(msg))

    def cmd_solve(self, ccd_name='guide', exposure=8, hint=None, allsky=False, path=None):
        """solve [camera [exposure]]: Plate-solve and find image coordinates"""
        from cvastrophoto.platesolve import astap
        from cvastrophoto.util import imgscale
        from cvastrophoto.image import Image

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

        dark_library = None
        bias_library = None

        if ccd is self.guider.ccd:
            # Request a snapshot and process it
            # NOTE: Guider snaps are already denoised
            if path is None:
                self.guider.request_snap()
                path = 'guide_snap.fit'
            fl = self.guider.calibration.eff_guider_fl
        else:
            if path is None:
                # Backup properties
                orig_upload_mode = ccd.upload_mode
                orig_transfer_fmt = ccd.transfer_format

                try:
                    # Configure for FITS-to-Client transfer
                    ccd.setUploadClient()
                    if orig_transfer_fmt is not None:
                        ccd.setTransferFormatFits()

                    # Capture a frame and use it
                    ccd.expose(int(exposure))
                    blob = ccd.pullBLOB(self.guider.ccd_name)
                    path = 'solve_snap.fit'
                    with open(path, 'wb') as f:
                        f.write(blob.getblobdata())

                    # Use capture sequence's dark libraries
                    if self.capture_seq is not None:
                        dark_library = self.capture_seq.dark_library
                        bias_library = self.capture_seq.bias_library

                finally:
                    # Restore upload mode
                    ccd.setUploadMode(orig_upload_mode, optional=orig_upload_mode is None)
                    if orig_transfer_fmt is not None:
                        ccd.setTransferFormat(orig_transfer_fmt)

            fl = self.guider.calibration.eff_imaging_fl

        # Compute hint
        l, t, r, b = ccd.properties['CCD_FRAME'][:4]
        w = r - l
        h = b - t
        rx = w / 2.0
        ry = h / 2.0

        if not hint:
            hint = None
        if isinstance(hint, basestring):
            hint_gc = self.parse_coord(hint)
            rx, ry, ra, dec = hint = (rx, ry, hint_gc.ra.degree, hint_gc.dec.degree)
            logger.info("Using manual hint: %r", hint)
        elif hint is None and info_source is not None:
            coords = info_source.properties.get('EQUATORIAL_EOD_COORD')
            if coords:
                ra, dec = coords
                ra = solver.ra_h_to_deg(ra)
                hint = (rx, ry, ra, dec)
                logger.info("Using hint from %s: %r", getattr(info_source, 'name', 'UNK'), hint)
        elif hint is not None:
            rx, ry, ra, dec = hint
            logger.info("Using explicit hint %r", hint)
        else:
            rx = ry = ra = dec = None

        if not hint:
            logger.warning("Solving hint unavailable, plate solving unlikely to succeed, and likely to be slow")

        image_scale = fov = None
        if ccd_name == 'guide':
            pixsz = self.guider.calibration.eff_guider_pixel_size
        else:
            pixsz = ccd.properties.get('CCD_INFO', [None]*5)[2]

        if pixsz and fl:
            image_scale = imgscale.compute_image_scale(fl, pixsz)
        if image_scale and h:
            fov = h * image_scale / 3600.0

        if dark_library is not None or bias_library is not None:
            master_dark = None
            img = Image.open(path, mode='update')
            if dark_library is not None:
                master_dark = dark_library.get_master(dark_library.classify_frame(path), raw=img)
            if master_dark is None and bias_library is not None:
                master_dark = bias_library.get_master(bias_library.classify_frame(path), raw=img)
            if master_dark is not None:
                # Denoise file in-place
                img.denoise([master_dark], entropy_weighted=False)
            img.close()
            del img

        kw = dict(hint=hint, fov=fov, image_scale=image_scale)
        success = solver.solve(path, **kw)

        if success:
            hdu = solver.get_solve_data(path)
            sx, sy, sra, sdec = coords = solver.get_coords(path, hdu=hdu)
            sra = solver.ra_deg_to_h(sra)
            logger.info("Successfully platesolved at coordinates: %r RA %r DEC", sra, sdec)
            if hint is not None:
                ra = solver.ra_deg_to_h(ra)
                logger.info("Original hint coordinates: %r RA %r DEC", ra, dec)
                logger.info("Effective shift: %r RA %r DEC", sra - ra, sdec - dec)
        else:
            logger.info("Plate solving failed")
            coords = hdu = None

        return success, solver, path, coords, hdu, kw

    def cmd_annotate(self, ccd_name='guide', exposure=8, **kw):
        """annotate [camera [exposure]]: Take a snapshot, and annotate it"""
        solve_callback = kw.pop('solve_callback', None)
        success, solver, path, coords, hdu, kw = self.cmd_solve(ccd_name, exposure, **kw)

        if solve_callback is not None:
            solve_callback(success, solver, path, coords, hdu, **kw)

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
