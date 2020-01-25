# -*- coding: utf-8 -*-
from __future__ import print_function

import functools
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

    ap.add_argument('--exposure', '-x', help='Guiding exposure length', default=4.0, type=float)
    ap.add_argument('--autostart', '-A', default=False, action='store_true',
        help='Start guiding immediately upon startup')
    ap.add_argument('--pepa-sim', default=False, action='store_true',
        help='Simulate PE/PA')
    ap.add_argument('--debug-tracks', default=False, action='store_true',
        help='Save debug tracking images to ./Tracks/guide_*.jpg')

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

    ap.add_argument('indi_addr', metavar='HOSTNAME:PORT', help='Indi server address',
        default='localhost:7624', nargs='?')

def main(opts, pool):
    import cvastrophoto.devices.indi
    from cvastrophoto.devices.indi import client
    from cvastrophoto.guiding import controller, guider, calibration
    import cvastrophoto.guiding.simulators.mount
    from cvastrophoto.rops.tracking.correlation import CorrelationTrackingRop

    if opts.guide_on_ccd:
        guide_st4 = opts.guide_ccd
    elif opts.guide_on_mount:
        guide_st4 = opts.mount
    elif opts.guide_st4:
        guide_st4 = opts.guide_st4
    else:
        logger.error("Either --guide-on-ccd, --guide-on-mount or --guide-st4 must be specified")
        return 1

    indi_host, indi_port = opts.indi_addr.split(':')
    indi_port = int(indi_port)

    indi_client = client.IndiClient()
    indi_client.setServer(indi_host, indi_port)
    indi_client.connectServer()

    telescope = indi_client.waitTelescope(opts.mount) if opts.mount else None
    st4 = indi_client.waitST4(guide_st4)
    ccd = indi_client.waitCCD(opts.guide_ccd)
    ccd_name = 'CCD1'

    if telescope is not None:
        logging.info("Connecting telescope")
        telescope.waitConnect()

    logging.info("Connecting ST4")
    st4.waitConnect()

    logging.info("Connecting CCD")
    ccd.waitConnect()
    ccd.subscribeBLOB('CCD1')

    if telescope is not None and opts.mount == 'Telescope Simulator':
        ra = float(os.getenv('RA', repr((279.23473479 * 24.0)/360.0) ))
        dec = float(os.getenv('DEC', repr(+38.78368896) ))

        logging.info("Slew to target")
        telescope.trackTo(ra, dec)
        time.sleep(10)
        logging.info("Slewd to target")

    logging.info("Detecting CCD info")
    ccd.detectCCDInfo(ccd_name)
    logging.info("Detected CCD info")

    tracker_class = functools.partial(CorrelationTrackingRop,
        track_distance=opts.track_distance,
        resolution=opts.track_resolution)

    if opts.pepa_sim:
        controller_class = cvastrophoto.guiding.simulators.mount.PEPASimGuiderController
    else:
        controller_class = controller.GuiderController
    guider_controller = controller_class(telescope, st4)

    if opts.min_pulse:
        guider_controller.min_pulse = opts.min_pulse
    if opts.max_pulse:
        guider_controller.max_pulse = opts.max_pulse
    if opts.target_pulse:
        guider_controller.target_pulse = opts.target_pulse

    calibration_seq = calibration.CalibrationSequence(telescope, guider_controller, ccd, ccd_name, tracker_class)
    calibration_seq.guide_exposure = opts.exposure
    guider_process = guider.GuiderProcess(telescope, calibration_seq, guider_controller, ccd, ccd_name, tracker_class)
    guider_process.save_tracks = opts.debug_tracks
    if opts.aggression:
        guider_process.aggressivenes = opts.aggression
    if opts.drift_aggression:
        guider_process.drift_aggressiveness = opts.drift_aggression
    if opts.history_length:
        guider_process.history_length = opts.history_length

    guider_controller.start()
    guider_process.start()

    if opts.autostart:
        guider.start_guiding(wait=False)

    iguider = InteractiveGuider(guider_process, guider_controller, ccd_name)
    iguider.run()

    logging.info("Shutting down")
    guider_process.stop()
    guider_controller.stop()

    logging.info("Exit")


class InteractiveGuider(object):

    def __init__(self, guider_process, guider_controller, ccd_name='CCD1'):
        self.guider = guider_process
        self.controller = guider_controller
        self.ccd_name = ccd_name
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

    def run(self):
        helpstring = self.get_helpstring()

        while not self.stop:
            cmd = raw_input("""
Commands:

%(helpstring)s

> """ % dict(helpstring=helpstring))
            if not cmd:
                continue

            cmd = cmd.split()
            cmd, args = cmd[0], cmd[1:]

            cmdmethod = getattr(self, 'cmd_' + cmd, None)
            if cmdmethod is None:
                logger.error("Unrecognized command %r", cmd)
            else:
                try:
                    cmdmethod(*args)
                except Exception:
                    logger.exception("Error executing %s", cmd)

    def cmd_start(self):
        """start: start guiding, calibrate if necessary"""
        logger.info("Start guiding")
        self.guider.start_guiding(wait=False)

    def cmd_stop(self):
        """stop: stop guiding"""
        logger.info("Stop guiding")
        self.guider.stop_guiding(wait=False)

    def cmd_halt(self):
        """halt: stop guiding (and all movement)"""
        logger.info("Halt guiding")
        self.guider.controller.paused = True
        self.guider.stop_guiding(wait=False)

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

    def cmd_move(self, we, ns, speed):
        """
        move RA DEC speed: Move the specified amount of RA seconds W/E and DEC arc-seconds
            N/S (needs calibration) assuming the mount moves at the specified speed.
            Don't use this while guiding, use shift instead.
        """
        self.guider.move(float(ns), float(we), float(speed))

    def cmd_exit(self):
        """exit: exit the program"""
        self.stop = True

    def show_device_properties(self, device):
        logger.info("Properties for %s:", device.name)
        for propname, val in device.properties.items():
            logger.info("    %s: %r", propname, val)

    def cmd_show_cam(self):
        """show_cam: Show camera properties"""
        self.show_device_properties(self.guider.ccd)

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
        logger.info("Controller drift %.4f%% N/S %.4f%% W/E",
            self.guider.controller.ns_drift,
            self.guider.controller.we_drift)

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