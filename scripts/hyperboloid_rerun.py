#!/usr/bin/env python
"""
visualize a hyperboloid in rerun
"""
import argparse
import colorsys

import numpy as np
import rerun as rr


def main():
    parser = argparse.ArgumentParser(description="Simple example of a ROS node that republishes to Rerun.")
    rr.script_add_args(parser)
    args, unknown_args = parser.parse_known_args()

    rr.script_setup(args, "hyperboloid")  # , recording_id=recording_id)

    strips = []
    colors = []
    for ind, x in enumerate(np.arange(-4.0, 4.0, 0.25)):
        rgb = colorsys.hsv_to_rgb((ind * 0.01) % 1.0, 0.5, 0.5)
        color = [int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)]
        strip = []
        for y in np.arange(-4.0, 4.0, 0.25):
            z = np.sqrt(1.0 + x * x + y * y)
            strip.append([x, y, z])
        strips.append(strip)
        colors.append(color)

    for ind, y in enumerate(np.arange(-4.0, 4.0, 0.25)):
        rgb = colorsys.hsv_to_rgb((ind * 0.01) % 1.0, 0.5, 0.5)
        color = [int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)]
        strip = []
        for x in np.arange(-4.0, 4.0, 0.25):
            z = np.sqrt(1.0 + x * x + y * y)
            strip.append([x, y, z])
        strips.append(strip)
        colors.append(color)

    rr_marker = rr.LineStrips3D(strips, colors=colors)
    rr.log("hb_a", rr_marker)


if __name__ == "__main__":
    main()
