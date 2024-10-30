#!/usr/bin/env python
"""
visualize a hyperboloid in rerun
"""
import argparse
import colorsys

import numpy as np
import rerun as rr

from hyperbolic.poincare import Point, Transform
from poincare_plane import make_node_tree, Node


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

    all_nodes = []

    root = Node(name="root", p_scale=3.0 / 5.0)
    all_nodes = make_node_tree(root, levels=4)

    num = 50
    for i in range(1, num):
        fr = i / num
        x = -0.75 + fr * 0.75 * 2.0
        y = 0.0
        root_rot = Transform.rotation(deg=90)
        root_offset = Transform.translation(Point(x, y))
        root_transform = Transform.merge(root_offset, root_rot)

        root.apply_transform(offset_transform=root_transform)

        strips = []
        poincare_strips = []
        colors = []
        h_colors = []

        for node in all_nodes:
            rgb = colorsys.hsv_to_rgb((node.depth * 0.17) % 1.0, 0.35, 0.35)
            color = [int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)]
            h_colors.append(color)

            hyperboloid_strip = []
            for x, y, z in zip(node.xh, node.yh, node.zh):
                hyperboloid_strip.append([x, y, z])
            strips.append(hyperboloid_strip)

            rgb = colorsys.hsv_to_rgb((node.depth * 0.17) % 1.0, 0.5, 0.5)
            color = [int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)]
            colors.append(color)

            poincare_strip = []
            for x, y in zip(node.xs, node.ys):
                z = 0.0
                poincare_strip.append(xyz_to_poincare(x, y, z))
            poincare_strips.append(poincare_strip)

        rr_hyperboloid_strips = rr.LineStrips3D(strips, colors=h_colors)
        rr.log("hyperboloid", rr_hyperboloid_strips)
        rr_poincare_strips = rr.LineStrips3D(poincare_strips, colors=colors)
        rr.log("poincare", rr_poincare_strips)

    if False:
        extent = 10.0
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


if __name__ == "__main__":
    main()
