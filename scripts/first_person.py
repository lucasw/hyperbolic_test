#!/usr/bin/env python
"""
pip install ursina

Adapted from ursina sample fps.py
"""
import math

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
    invoke,
    mouse,
    random,
    raycast,
    time,
)
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.shaders import lit_with_shadows_shader


def make_verts(xsc=1.0):
    vts = ((0.5 * xsc, 0.5, 0.0),
           (-0.5 * xsc, 0.5, 0.0),
           (-0.5 * xsc, -0.5, 0.0),
           (0.5 * xsc, -0.5, 0.0),
           (0.5 * xsc, 0.5, 0.0),
           (-0.5 * xsc, -0.5, 0.0),
           )
    return vts


def make_mesh(x, y, z, xsc=1.0):
    """
    x is left right
    y is up down
    z is forward back
    """
    mesh = Entity(position=(x, y, z),
                  model=Mesh(
                      vertices=make_verts(xsc),
                      uvs=((1, 1), (0, 1),
                           (0, 0), (1, 0),
                           (1, 1), (0, 0),
                           ),
                      normals=[(-0.0, 0.0, -1.0),
                               (-0.0, 0.0, -1.0),
                               (-0.0, 0.0, -1.0),
                               (-0.0, 0.0, -1.0),
                               (-0.0, 0.0, -1.0),
                               (-0.0, 0.0, -1.0),
                               ],
                      colors=[color.red, color.yellow,
                              color.green, color.cyan,
                              color.blue, color.magenta,
                              ],
                      mode='triangle'),
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
        self.player.collider = BoxCollider(self.player, Vec3(0, 1, 0), Vec3(1, 2, 1))

        self.count = 0
        self.mesh = make_mesh(0.0, 1.0, 0.0, 1.0)
        # Text(parent=self.surface, text='quad_with_usv_and_normals_and_vertex_colors',
        #      y=1, scale=10, origin=(0,-.5))

    def update(self):
        sc = 1.0 + abs(math.sin(self.count / 10.0))
        # TODO(lucasw) can't modify vertices live, just replace the old mesh
        # self.surface.model.vertices = make_verts(sc)
        # print("destroy mesh")
        destroy(self.mesh)
        self.mesh = None
        self.mesh = make_mesh(0, 1, 0, sc)

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

    for i in range(16):
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