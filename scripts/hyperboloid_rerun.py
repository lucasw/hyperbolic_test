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


def make_translated(root, all_nodes, x, y):
    root_rot = Transform.rotation(deg=0)
    root_offset = Transform.translation(Point(x, y))
    root_transform = Transform.merge(root_offset, root_rot)

    rr.log("transform", rr.TextDocument(f"{x:0.2f} {y:0.2f} -> {root_transform}"))

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


"""
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
"""


def main():
    parser = argparse.ArgumentParser(description="Simple example of a ROS node that republishes to Rerun.")
    rr.script_add_args(parser)
    args, unknown_args = parser.parse_known_args()

    rr.script_setup(args, "hyperboloid")  # , recording_id=recording_id)

    all_nodes = []

    p = 5
    q = 4
    # set p_scale to 4 or so to get a circle
    root = Node(name="root", p=p, q=q, p_scale=1)  # 3.0 / 5.0)
    all_nodes = make_node_tree(root, levels=4)

    # the range of translation is 0 to the closest part of the edge of a p-polygon
    # which varies with the angle (given the orientation of the polygon)
    # theta = pi / p
    x_min = -root.r
    x_max = root.r * np.cos(root.half_angle)
    print(f"translation range: {x_min} - {x_max}")
    r_num = 20
    angle_num = 6
    for j in range(0, angle_num + 1):
        angle = root.half_angle * j / angle_num
        for i in range(0, r_num + 1):
            # TODO(lucasw) this isn't right- check that it even works without the hyperbolic transform
            # dist_to_edge = root.r / (np.cos(angle) + np.tan(angle) * np.sin(angle))
            dist_to_edge = root.r
            fr = dist_to_edge * i / r_num
            if j % 2 == 1:
                fr = dist_to_edge - fr
            x = - fr * np.cos(angle)
            y = - fr * np.sin(angle)
            make_translated(root, all_nodes, x, y)
            rr.log("angle", rr.TextDocument(f"angle: {angle:0.2f}, dist: {fr:0.2f}, dist to edge: {dist_to_edge:0.2f}"))


if __name__ == "__main__":
    main()
