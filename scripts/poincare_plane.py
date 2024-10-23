"""
Lucas Walter
Octoboer 2024

Adapted from hyperbolic example
https://github.com/cduck/hyperbolic/blob/master/examples/poincare.ipynb


"""

import math

import numpy as np
# from hyperbolic import euclid, util
from hyperbolic.euclid import Arc, Line
from hyperbolic.poincare import Point, Polygon, Transform


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


def construct_poly_vertices(p, q, deg_offset=0, skip=1):
    r = hyp_poly_edge_construct(p, q)
    return [
        Point.from_polar_euclid(r, deg=-skip * i * 360 / p + deg_offset)
        for i in range(p)
    ]


def construct_poly_from_points(pt_list):
    p = len(pt_list)
    e_list = [Line.from_points(*pt_list[i], *pt_list[(i + 1) % p]) for i in range(p)]
    return Polygon(e_list, join=True)


def get_poly_xy(poly):
    xs = []
    ys = []
    for edge in poly.edges:
        arc = edge.proj_shape
        if isinstance(arc, Arc):
            num = 8
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
    """
    def __init__(self, name, depth=0, offset_transform=None, parent=None, parent_side=None, p=5, q=4):
        self.name = name
        self.depth = depth
        self.p = p

        self.neighbors = {}
        for i in range(p):
            self.neighbors[i] = None

        self.parent = parent
        self.neighbors[0] = parent

        # TODO(lucasw) this is the same for every polygon
        vertices = construct_poly_vertices(p, q)

        transform = Transform.identity()
        if self.parent is not None:

            # use the first edge as the shift origin
            t0 = vertices[0]
            t1 = vertices[1]
            transform = Transform.shift_origin(t0, t1)

            t0 = parent.polygon.vertices[(parent_side + 1) % p]
            t1 = parent.polygon.vertices[parent_side]
            trans_to_side = Transform.translation(t0, t1)
            transform = Transform.merge(transform, trans_to_side)
        if offset_transform is not None:
            transform = Transform.merge(transform, offset_transform)

        transformed_vertices = transform.apply_to_list(vertices)
        self.polygon = Polygon.from_vertices(transformed_vertices)

        self.xs, self.ys = get_poly_xy(self.polygon)
        minx = np.min(self.xs)
        miny = np.min(self.ys)
        maxx = np.max(self.xs)
        maxy = np.max(self.ys)
        self.cx = minx + 0.5 * (maxx - minx)
        self.cy = miny + 0.5 * (maxy - miny)

        self.children = {}

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

            node = Node(name=f"{prefix}{i}", depth=self.depth + 1, parent=self, parent_side=i)
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
