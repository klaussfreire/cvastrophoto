# -*- coding: utf-8 -*-
from __future__ import print_function

import functools
import time
import logging
import os.path


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
    ap.add_argument('--guide-on-ccd', '-G', action='store_true', help='A shorthand to set ST4=GCCD')
    ap.add_argument('--mount', '-gmount', help='The name of the mount interface')

    ap.add_argument('indi_addr', metavar='HOSTNAME:PORT', help='Indi server address', default='localhost:7624')

def main(opts, pool):
    import cvastrophoto.devices.indi
    from cvastrophoto.devices.indi import client
    from cvastrophoto.guiding import controller, guider, calibration
    import cvastrophoto.guiding.simulators.mount
    from cvastrophoto.rops.tracking.correlation import CorrelationTrackingRop

    guide_ccd = opts.guide_ccd if opts.guide_on_ccd else opts.guide_st4
    if not guide_ccd:
        logger.error("Either --guide-on-ccd or --guide-st4 must be specified")
        return 1

    indi_host, indi_port = opts.indi_addr.split(':')
    indi_port = int(indi_port)

    indi_client = client.IndiClient()
    indi_client.setServer(indi_host, indi_port)
    indi_client.connectServer()

    telescope = indi_client.waitTelescope(opts.mount) if opts.mount else None
    st4 = indi_client.waitST4(guide_ccd)
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
    icontroller = controller_class(telescope, st4)
    icalibration = calibration.CalibrationSequence(telescope, icontroller, ccd, ccd_name, tracker_class)
    iguider = guider.GuiderProcess(telescope, icalibration, icontroller, ccd, ccd_name, tracker_class)
    iguider.save_tracks = opts.save_tracks
    if opts.aggression:
        iguider.aggressivenes = opts.aggression
    if opts.drift_aggression:
        iguider.drift_aggressiveness = opts.drift_aggression
    if opts.history_length:
        iguider.history_length = opts.history_length

    icontroller.start()
    iguider.start()

    if opts.autostart:
        iguider.start_guiding(wait=False)

    while True:
        cmd = raw_input("start, stop, exit> ")
        if cmd == "start":
            iguider.start_guiding(wait=False)
        elif cmd == "stop":
            iguider.stop_guiding(wait=False)
        elif cmd == "update-calibration":
            iguider.update_calibration(wait=False)
        elif cmd == "calibrate":
            iguider.calibrate(wait=False)
        elif cmd == "exit":
            break

    logging.info("Shutting down")
    iguider.stop()
    icontroller.stop()

    logging.info("Exit")
