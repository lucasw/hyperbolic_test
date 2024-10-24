#!/usr/bin/env python
"""
pip install ursina

Adapted from ursina sample fps.py
"""
from hyperbolic.poincare import Point, Transform
import numpy as np
# from ursina import *
from ursina import (
    BoxCollider,
    DirectionalLight,
    EditorCamera,
    Entity,
    Mesh,
    Sky,
    # Text,
    Ursina,
    Vec3,
)
from ursina import (
    application,
    camera,
    color,
    destroy,
    distance_xz,
    held_keys,
    hsv,
    invoke,
    mouse,
    random,
    raycast,
    time,
)
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.shaders import lit_with_shadows_shader
from poincare_plane import get_poly_xy, Node


def make_verts(p0, p1, flip_normal=True):
    x0, y0, z0 = p0[0], p0[1], p0[2]
    x1, y1, z1 = p1[0], p1[1], p1[2]
    # make two triangles worth
    if flip_normal:
        vts = ((x1, y1, z1), (x1, y0, z1), (x0, y1, z0),
               (x1, y0, z1), (x0, y0, z0), (x0, y1, z0),
               )
        uvs = ((1, 1), (0, 0), (0, 1),
               (1, 0), (0, 0), (0, 1),
               )
    else:
        vts = ((x1, y1, z1), (x0, y1, z0), (x1, y0, z1),
               (x1, y0, z1), (x0, y1, z0), (x0, y0, z0),
               )
        uvs = ((1, 1), (0, 1), (0, 0),
               (1, 0), (0, 1), (0, 0),
               )

    pt1 = np.array([x0, y1, z0])
    pt2 = np.array([x1, y0, z1])
    v0 = pt1 - p0
    v0 = v0 / np.sqrt(np.sum(v0**2))
    v1 = pt2 - p0
    v1 = v1 / np.sqrt(np.sum(v1**2))
    normal = np.cross(v0, v1)
    if flip_normal:
        normal *= -1.0
    # to tuple
    normal = (normal[0], normal[1], normal[2])
    normals = (normal, normal, normal,
               normal, normal, normal)

    return vts, normals, uvs


def make_mesh(pos, p0, p1, hue):
    """
    x01, y01, z01 are relative to xyz
    x is left right
    y is up down
    z is forward back
    """
    position = (pos[0], pos[1], pos[2])

    vertices, normals, uvs = make_verts(p0, p1)
    mesh = Entity(position=position,
                  model=Mesh(
                      vertices=vertices,
                      uvs=uvs,
                      normals=normals,
                      colors=[hsv(hue, 0.5, 0.5), hsv(hue, 0.5, 0.5), hsv(hue * 0.8, 0.5, 0.5),
                              hsv(hue * 0.8, 0.5, 0.5), hsv(hue, 0.5, 0.5), hsv(hue * 0.8, 0.5, 0.5),
                              ],
                      mode='triangle'),
                  # TODO(lucasw) need to shift each wall a little so there isn't a z-buffer
                  # fight with the wall for the adjacent node
                  # double_sided=True
                  )
    return mesh


class Game(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pause_handler = Entity(ignore_paused=True, input=self.pause_input)

        gun = Entity(model='cube', parent=camera, position=(.5, -.25, .25), scale=(.3, .2, 1),
                     origin_z=-.5, color=color.red, on_cooldown=False)
        gun.muzzle_flash = Entity(parent=gun, z=1, world_scale=.5, model='quad',
                                  color=color.yellow, enabled=False)
        self.gun = gun

        self.editor_camera = EditorCamera(enabled=False, ignore_paused=True)

        self.player = FirstPersonController(model='cube', z=-10, color=color.orange,
                                            origin_y=-.5, speed=8, collider='box')
        self.player.collider = BoxCollider(self.player, Vec3(0, 1, 0), Vec3(0, 2, 0))

        self.count = 0
        # Text(parent=self.surface, text='quad_with_usv_and_normals_and_vertex_colors',
        #      y=1, scale=10, origin=(0,-.5))

        root_rot = Transform.rotation(deg=90)
        root_offset = Transform.translation(Point(0.0, 0.0))
        root_transform = Transform.merge(root_offset, root_rot)
        self.root_node = Node(name="root", offset_transform=root_transform)

        children = []
        children.extend(self.root_node.add_children("a"))

        grand_children = []
        for i, child in enumerate(children):
            prefix = f"b{i}"
            grand_children.extend(child.add_children(prefix))

        all_nodes = []
        all_nodes.append(self.root_node)
        all_nodes.extend(children)
        all_nodes.extend(grand_children)

        self.meshes = []

        sc = 20.0
        for ind, node in enumerate(all_nodes):
            # ground meshes, make a triangle fan around the center
            xs, zs = get_poly_xy(node.polygon, num=1)
            cx = node.cx * sc
            cz = node.cy * sc
            y0 = 0.05
            vts = (
                (xs[0] * sc, y0, zs[0] * sc), (cx, y0, cz), (xs[1] * sc, y0, zs[1] * sc),
                (xs[1] * sc, y0, zs[1] * sc), (cx, y0, cz), (xs[2] * sc, y0, zs[2] * sc),
                (xs[2] * sc, y0, zs[2] * sc), (cx, y0, cz), (xs[3] * sc, y0, zs[3] * sc),
                (xs[3] * sc, y0, zs[3] * sc), (cx, y0, cz), (xs[4] * sc, y0, zs[4] * sc),
                (xs[4] * sc, y0, zs[4] * sc), (cx, y0, cz), (xs[0] * sc, y0, zs[0] * sc),
            )
            uvs = (
                (1, 0), (0, 0), (0, 1),
                (0, 1), (0, 0), (1, 0),
                (1, 0), (0, 0), (0, 1),
                (0, 1), (0, 0), (1, 0),
                (1, 0), (0, 0), (0, 1),
            )

            normal = (0, 1, 0)
            normals = (
                normal, normal, normal,
                normal, normal, normal,
                normal, normal, normal,
                normal, normal, normal,
                normal, normal, normal,
            )

            hue = 97
            sat = 0.1
            var = 0.1
            mesh = Entity(
                position=(0, 0, 0),
                model=Mesh(
                    vertices=vts,
                    uvs=uvs,
                    normals=normals,
                    colors=[
                        hsv(hue, sat, var), hsv(hue, 0.01, 0.01), hsv(hue, sat, var),
                        hsv(hue, sat, var), hsv(hue, 0.01, 0.01), hsv(hue, sat, var),
                        hsv(hue, sat, var), hsv(hue, 0.01, 0.01), hsv(hue, sat, var),
                        hsv(hue, sat, var), hsv(hue, 0.01, 0.01), hsv(hue, sat, var),
                        hsv(hue, sat, var), hsv(hue, 0.01, 0.01), hsv(hue, sat, var),
                    ],
                    mode='triangle'),
                # TODO(lucasw) need to shift each wall a little so there isn't a z-buffer
                # fight with the wall for the adjacent node
                # double_sided=True,
            )
            self.meshes.append(mesh)
            if ind % 2 == 0:
                continue
            xs, zs = get_poly_xy(node.polygon, num=3)
            num = len(xs)
            for i0 in range(num):
                i1 = (i0 + 1) % num
                x0 = xs[i0] * sc
                z0 = zs[i0] * sc
                x1 = xs[i1] * sc
                z1 = zs[i1] * sc
                y0 = -0.1
                y1 = 2.0

                p0a = np.array([x0, y0, z0])
                p1a = np.array([x1, y1, z1])

                p0b = np.array([0, 0, 0])
                p1b = p1a - p0a
                mesh = make_mesh(p0a, p0b, p1b, hue=(node.depth * 20) % 360)
                self.meshes.append(mesh)

    def update(self):
        # TODO(lucasw) can't modify vertices live, just replace the old mesh
        # self.surface.model.vertices = make_verts(sc)
        # print("destroy mesh")
        # if self.mesh is not None:
        #     destroy(self.mesh)
        #     self.mesh = None

        # print(type(self.surface.model.vertices))
        if held_keys['left mouse']:
            self.shoot()

        self.count += 1

    def shoot(self):
        if not self.gun.on_cooldown:
            # print('shoot')
            self.gun.on_cooldown = True
            self.gun.muzzle_flash.enabled = True
            from ursina.prefabs.ursfx import ursfx
            ursfx([(0.0, 0.0), (0.1, 0.9), (0.15, 0.75), (0.3, 0.14), (0.6, 0.0)],
                  volume=0.5, wave='noise', pitch=random.uniform(-13, -12), pitch_change=-12, speed=3.0)
            invoke(self.gun.muzzle_flash.disable, delay=.05)
            invoke(setattr, self.gun, 'on_cooldown', False, delay=.15)
            if mouse.hovered_entity and hasattr(mouse.hovered_entity, 'hp'):
                mouse.hovered_entity.hp -= 10
                mouse.hovered_entity.blink(color.red)

    def pause_input(self, key):
        if key == 'tab':    # press tab to toggle edit/play mode
            editor_camera = self.editor_camera
            editor_camera.enabled = not editor_camera.enabled

            self.player.visible_self = editor_camera.enabled
            self.player.cursor.enabled = not editor_camera.enabled
            self.gun.enabled = not editor_camera.enabled
            mouse.locked = not editor_camera.enabled
            editor_camera.position = self.player.position

            application.paused = editor_camera.enabled


# from ursina.prefabs.health_bar import HealthBar
class Enemy(Entity):
    def __init__(self, shootables_parent, player, **kwargs):
        super().__init__(parent=shootables_parent, model='cube', scale_y=2,
                         origin_y=-.5, color=color.light_gray, collider='box', **kwargs)
        self.health_bar = Entity(parent=self, y=1.2, model='cube', color=color.red, world_scale=(1.5, .1, .1))
        self.max_hp = 100
        self.hp = self.max_hp
        self.player = player

    def update(self):
        dist = distance_xz(self.player.position, self.position)
        if dist > 40:
            return

        self.health_bar.alpha = max(0, self.health_bar.alpha - time.dt)

        self.look_at_2d(self.player.position, 'y')
        hit_info = raycast(self.world_position + Vec3(0, 1, 0), self.forward, 30, ignore=(self,))
        # print(hit_info.entity)
        if hit_info.entity == self.player:
            if dist > 2:
                self.position += self.forward * time.dt * 5

    @property
    def hp(self):
        return self._hp

    @hp.setter
    def hp(self, value):
        self._hp = value
        if value <= 0:
            destroy(self)
            return

        self.health_bar.world_scale_x = self.hp / self.max_hp * 1.5
        self.health_bar.alpha = 1


def main():
    app = Ursina()

    random.seed(0)
    Entity.default_shader = lit_with_shadows_shader

    ground = Entity(model='plane', collider='box', scale=64, texture='grass', texture_scale=(4, 4))
    print(ground)

    game = Game()

    shootables_parent = Entity()
    mouse.traverse_target = shootables_parent

    if False:  # for i in range(16):
        Entity(model='quad',  # 'cube',
               origin_y=-.5,
               scale=2,
               texture='brick',
               texture_scale=(1, 2),
               x=random.uniform(-8, 8),
               z=random.uniform(-8, 8) + 8,
               collider='box',
               scale_y=random.uniform(2, 3),
               color=color.hsv(0, 0, random.uniform(.9, 1))
               )

    # Enemy()
    num_enemies = 0
    enemies = [Enemy(shootables_parent, game.player, x=x * 4) for x in range(num_enemies)]
    print(len(enemies))

    sun = DirectionalLight()
    sun.look_at(Vec3(1, -1, -1))
    Sky()

    app.run()


if __name__ == "__main__":
    main()
