"""
Lucas Walter
Octoboer 2024

Adapted from hyperbolic example
https://github.com/cduck/hyperbolic/blob/master/examples/poincare.ipynb


"""

import copy
import math

import numpy as np
# from hyperbolic import euclid, util
from hyperbolic.euclid import Arc, Line
from hyperbolic.poincare import Point, Polygon, Transform


# TODO(lucasw) I think this is the radius of the tesselating p polygon with q
# polygons at each corner intersection on a poincare circle?
def hyp_poly_edge_construct(p, q):
    pi, pi2 = math.pi, math.pi * 2
    th = pi2 / q
    phi = pi2 / p
    ang1 = pi - phi / 2 - th / 2 - pi / 2
    ang2 = th / 2 + pi / 2
    a = math.sin(ang2) / math.sin(ang1)
    b = math.sin(phi / 2) / math.sin(ang1)
    r_p = math.sqrt(1 / (a**2 - b**2))
    r_c = a * r_p
    r_from_c = b * r_p
    # return r_c, r_from_c
    t1 = pi - math.asin(r_c / (r_from_c / math.sin(phi / 2)))
    t2 = pi - t1 - phi / 2
    r = math.sin(t2) * (r_from_c / math.sin(phi / 2))
    return r


# skip >= 1
# increasing p_scale approximates a circle
def construct_poly_vertices(p, q, deg_offset=0, skip=1, p_scale=1):
    r = hyp_poly_edge_construct(p, q)
    p2 = p * p_scale
    return [
        Point.from_polar_euclid(r, deg=-skip * i * 360 / p2 + deg_offset)
        for i in range(int(p2))
    ]


def construct_poly_from_points(pt_list):
    p = len(pt_list)
    e_list = [Line.from_points(*pt_list[i], *pt_list[(i + 1) % p]) for i in range(p)]
    return Polygon(e_list, join=True)


def get_poly_xy(poly, num=5):
    xs = []
    ys = []
    for edge in poly.edges:
        arc = edge.proj_shape
        if isinstance(arc, Arc):
            for i in range(num):  # for deg in np.arange(arc.start_deg, arc.end_deg, 1.0):
                # Arc ought to have a convenience function for this
                # clockwise
                if arc.cw:
                    d0 = arc.start_deg
                    d1 = arc.end_deg
                    if d1 < d0:
                        d1 += 360
                else:
                    d0 = arc.start_deg
                    d1 = arc.end_deg
                    if d1 > d0:
                        d1 -= 360

                deg = d0 + i * (d1 - d0) / num
                x = arc.cx + arc.r * math.cos(math.radians(deg))
                y = arc.cy + arc.r * math.sin(math.radians(deg))
                xs.append(x)
                ys.append(y)
        elif isinstance(arc, Line):
            xs.append(arc.x1)
            ys.append(arc.y1)
        else:
            print(type(arc))
        # pt0 = edge.start_point()
        # xs.append(pt0.x)
        # ys.append(pt0.y)
        # pt0 = edge.midpoint_euclid()
        # xs.append(pt0.x)
        # ys.append(pt0.y)

    xs.append(xs[0])
    ys.append(ys[0])

    return xs, ys


class Node:
    """construct a tree of polygon nodes
    avoid creating the same polygon in the same position when adding children

    parent and parent_side must be provided together
    """
    def __init__(self, name, depth=0, parent=None, parent_side=None, p=5, q=4, p_scale=1):
        self.name = name
        self.depth = depth
        self.p = p
        self.p_scale = p_scale
        self.r = hyp_poly_edge_construct(p, q)
        self.half_angle = np.pi / p

        self.neighbors = {}
        for i in range(p):
            self.neighbors[i] = None

        self.parent_side = parent_side
        self.parent = parent
        self.neighbors[0] = parent

        # TODO(lucasw) this is the same for every polygon
        vertices = construct_poly_vertices(p, q)
        self.vertices = vertices

        if p_scale == 1:
            self.vertices_to_render = None
        else:
            # look at making other shapes but still with the 5,4 tiling
            # increasing p makes
            vertices_to_render = construct_poly_vertices(p, q, p_scale=self.p_scale)
            self.vertices_to_render = vertices_to_render

        self.children = {}

    def apply_transform(self, offset_transform=None):
        transform = Transform.identity()

        if self.parent is not None:
            # use the first edge as the shift origin
            t0 = self.vertices[0]
            t1 = self.vertices[1]
            transform = Transform.shift_origin(t0, t1)

            t0 = self.parent.polygon.vertices[(self.parent_side + 1) % self.p]
            t1 = self.parent.polygon.vertices[self.parent_side]
            trans_to_side = Transform.translation(t0, t1)
            transform = Transform.merge(transform, trans_to_side)
        if offset_transform is not None:
            transform = Transform.merge(transform, offset_transform)

        transformed_vertices = transform.apply_to_list(self.vertices)
        self.polygon = Polygon.from_vertices(transformed_vertices)

        if self.vertices_to_render is not None:
            vertices_to_render = self.vertices_to_render
        else:
            vertices_to_render = copy.deepcopy(self.vertices)

        # visualize some extra points
        vertices_to_render.append(Point(0, 0))
        if False:
            # make an edge mid-point
            # TODO(lucasw) this isn't right
            # x = self.r * np.cos(self.half_angle)
            # y = x * np.sin(self.half_angle)
            p0 = vertices_to_render[0]
            p1 = vertices_to_render[-2]
            x = (p0.x + p1.x) / 2.0
            y = (p0.y + p1.y) / 2.0
            vertices_to_render.append(Point(x, y))

        transformed_vertices = transform.apply_to_list(vertices_to_render)
        self.polygon_to_render = Polygon.from_vertices(transformed_vertices)

        self.xs, self.ys = get_poly_xy(self.polygon_to_render)
        minx = np.min(self.xs)
        miny = np.min(self.ys)
        maxx = np.max(self.xs)
        maxy = np.max(self.ys)
        self.cx = minx + 0.5 * (maxx - minx)
        self.cy = miny + 0.5 * (maxy - miny)

        # hyperboloid coordinates
        if True:
            xp = self.xs
            yp = self.ys

            # instead of the unit circle poincare disk, convert back into
            # hyperboloid coordinates
            # zh^2 - xh^2 - yh^2 = 1.0
            # zh^2 = 1.0 + xh^2 + yh^2
            # zh = sqrt(1.0 + xh^2 + yh^2)

            # convert to hyperboloid coords, solve for scalar a which projects
            # the poincare point back onto the hyperboloid
            # (xh, yh, zh) = (0, 0, -1) + a * (xp, yp, 1.0)
            # xh = a * xp
            # yh = a * yp
            # zh = a - 1.0
            # (a - 1.0)^2 = 1.0 + (a * xp)^2 + (a * yp)^2
            # a = 2.0 / (1 - xp^2 - yp^2)
            xp2 = np.multiply(xp, xp)
            yp2 = np.multiply(yp, yp)
            a = np.divide(2.0, 1.0 - xp2 - yp2)
            self.zh = a - 1.0
            self.xh = np.multiply(a, xp)
            self.yh = np.multiply(a, yp)

        for child in self.children.values():
            child.apply_transform()

    def neighbor_names(self):
        text = "["
        for ind, neighbor in self.neighbors.items():
            text += f"{ind}"
            if neighbor is None:
                text += " None, "
            else:
                text += f" '{neighbor.name}', "  # {id(neighbor):0x},"
        text += "]"
        return text

    def get_ind(self, src):
        """return index of node in neighbors, if it is a neighbor"""
        for ind, node in self.neighbors.items():
            # print(f"{id(node)} {id(src)} {id(node) == id(src)}")
            if node == src:
                return ind
        print(f"{src.name} {id(src):0x} not a neighbor of {self.name} {self.neighbor_names()}")
        return None

    def get_left(self, src):
        """get the neighbor node to the left of the src node
        """
        src_ind = self.get_ind(src)
        if src_ind is None:
            return None, None

        # TODO(lucasw) which is actually left and which is actually right?
        # maybe if consistent it won't matter
        left_ind = (src_ind - 1) % self.p
        left = self.neighbors[left_ind]
        if left is None:
            return None, None
        ind_from_left = left.get_ind(self)
        return left, ind_from_left

    def get_right(self, src):
        """get the neighbor node to the right of the src node
        """
        src_ind = self.get_ind(src)
        if src_ind is None:
            return None, None

        right_ind = (src_ind + 1) % self.p
        right = self.neighbors[right_ind]
        if right is None:
            return None, None
        ind_from_right = right.get_ind(self)
        return right, ind_from_right

    def get_left_left(self, src):
        """get the left neighbor of the left neighbor from the entrypoint of src"""
        left, _ = self.get_left(src)
        if left is None:
            return None, None
        left_left, ind_from_left_left = left.get_left(self)
        return left_left, ind_from_left_left

    def get_right_right(self, src):
        """get the right neighbor of the right neighbor from the entrypoint of src"""
        right, _ = self.get_right(src)
        if right is None:
            return None, None
        right_right, ind_from_right_right = right.get_right(self)
        return right_right, ind_from_right_right

    def add_children(self, prefix="c"):
        # print(f"{self.name} add children")
        for i in range(self.p):
            if self.neighbors[i] is not None:
                # print(f"{i} existing neighbor {self.neighbors[i].name}")
                continue
            left_i = (i - 1) % self.p
            left = self.neighbors[left_i]
            if left is not None:
                # print(f"{self.name} {i} {left_i} {left.name} get left left")
                left3, ind_from_left3 = left.get_left_left(self)
                if left3 is not None:
                    self.neighbors[i] = left3
                    left3.neighbors[(ind_from_left3 - 1) % self.p] = self
                    continue

            right_i = (i + 1) % self.p
            right = self.neighbors[right_i]
            if right is not None:
                # print(f"{self.name} {i} get right right")
                right3, ind_from_right3 = right.get_right_right(self)
                if right3 is not None:
                    self.neighbors[i] = right3
                    right3.neighbors[(ind_from_right3 + 1) % self.p] = self
                    continue

            node = Node(name=f"{prefix}{i}", depth=self.depth + 1, parent=self, parent_side=i, p_scale=self.p_scale)
            self.children[i] = node
            self.neighbors[i] = node
        return self.children.values()

    def plot_recursive(self, ax):
        text = f"{self.name}, neighbors: "
        text += self.neighbor_names()
        if self.parent is not None:
            text += f" parent: {self.parent.name}"
        print(text)
        ax.plot(self.xs, self.ys)
        ax.text(self.cx, self.cy, self.name, fontsize=6)
        for child in self.children.values():
            child.plot_recursive(ax)


def make_node_tree(root: Node, levels=2) -> list[Node]:
    parents = [root]
    all_nodes = [root]
    for level in range(levels):
        children = []
        for i, parent in enumerate(parents):
            children.extend(parent.add_children(f"{level}_{i}_"))
            all_nodes.extend(children)
        print(f"cur depth: {children[0].depth}")
        print(f"level: {level}, all: {len(all_nodes)}, children: {len(children)}")
        parents = children
    return all_nodes


def example():
    root_rot = Transform.rotation(deg=90)
    root_offset = Transform.translation(Point(0.0, 0.0))
    root_transform = Transform.merge(root_offset, root_rot)
    root = Node(name="root", offset_transform=root_transform)
    children = []
    children.extend(root.add_children("a"))
    print(len(children))

    grand_children = []
    for i, child in enumerate(children):
        prefix = f"b{i}"
        grand_children.extend(child.add_children(prefix))
    print(len(grand_children))

    if False:
        great_grand_children = []
        for i, child in enumerate(grand_children):
            prefix = f"c{i}"
            great_grand_children.extend(child.add_children(prefix))
        print(len(great_grand_children))
