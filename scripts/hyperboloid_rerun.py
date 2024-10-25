#!/usr/bin/env python
"""
visualize a hyperboloid in rerun
"""
import argparse
import colorsys

import numpy as np
import rerun as rr


def xyz_to_poincare(x, y, z):
    # make each point a vector with origin (0, 0, -1), then scale
    # so they are on the z=0 plane
    sc = 1.0 / (z + 1)
    px = x * sc
    py = y / (z + 1.0)
    pz = 0.0
    return px, py, pz


def main():
    parser = argparse.ArgumentParser(description="Simple example of a ROS node that republishes to Rerun.")
    rr.script_add_args(parser)
    args, unknown_args = parser.parse_known_args()

    rr.script_setup(args, "hyperboloid")  # , recording_id=recording_id)

    extent = 10.0
    strips = []
    poincare_strips = []
    colors = []
    for ind, x in enumerate(np.arange(-extent, extent, extent / 20.0)):
        rgb = colorsys.hsv_to_rgb((ind * 0.01) % 1.0, 0.5, 0.5)
        color = [int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)]
        strip = []
        poincare_strip = []
        for y in np.arange(-extent, extent, 0.25):
            z = np.sqrt(1.0 + x * x + y * y)
            strip.append([x, y, z])
            poincare_strip.append(xyz_to_poincare(x, y, z))

        strips.append(strip)
        poincare_strips.append(poincare_strip)
        colors.append(color)

    for ind, y in enumerate(np.arange(-extent, extent, extent / 20.0)):
        rgb = colorsys.hsv_to_rgb((ind * 0.01) % 1.0, 0.5, 0.5)
        color = [int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)]
        strip = []
        poincare_strip = []
        for x in np.arange(-extent, extent, 0.25):
            z = np.sqrt(1.0 + x * x + y * y)
            strip.append([x, y, z])
            poincare_strip.append(xyz_to_poincare(x, y, z))

        strips.append(strip)
        poincare_strips.append(poincare_strip)
        colors.append(color)

    rr_hyperboloid_strips = rr.LineStrips3D(strips, colors=colors)
    rr.log("hyperboloid", rr_hyperboloid_strips)
    rr_poincare_strips = rr.LineStrips3D(poincare_strips, colors=colors)
    rr.log("poincare", rr_poincare_strips)


if __name__ == "__main__":
    main()
