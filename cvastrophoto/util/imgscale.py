# -*- coding: utf-8 -*-
import math

tan_arcsec = math.tan(1.0 / 3600.0 * math.pi / 180.0)

def compute_image_scale(scope_fl_mm, pixel_um):
    return (pixel_um / 1000.0) / (scope_fl_mm * tan_arcsec)
