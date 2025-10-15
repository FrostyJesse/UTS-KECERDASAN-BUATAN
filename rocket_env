%%writefile rocket_env.py
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import math

class SimpleRocketEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode='human'):
        super().__init__()
        self.render_mode = render_mode

        self.dt, self.m, self.g = 0.025, 3.0, 9.81
        self.F_main, self.F_side = 400.0, 200.0
        self.screen_w, self.screen_h = 960, 480
        self.floor_y = 10.0
        self.w, self.h = 30.0, 60.0
        self.I = (1/12) * self.m * (self.w**2 + self.h**2)
        self.b_linear, self.b_angular = 0.1, 0.05

        tx_init = int(random.random() * 500)
        self.target_pos = np.array([300.0 + tx_init, 40.0], np.float32)
        self.target_w, self.target_h = 240.0, 80.0
        self.target_min_x, self.target_max_x = 300.0, 800.0
        self.target_vx = 50.0

        self.launch_pad_pos = np.array([100.0, 50.0], dtype=np.float32)
        self.pad_w, self.pad_h = 40.0, 40.0

        high = np.array([self.screen_w, self.screen_h,
                         np.finfo(np.float32).max, np.finfo(np.float32).max,
                         1., 1., np.finfo(np.float32).max,
                         self.screen_w, self.screen_h], np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("Simple Rocket Env")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16)

            base = os.path.dirname(__file__)
            load = lambda fn: pygame.image.load(os.path.join(base, fn)).convert_alpha()
            self.rocket_img = pygame.transform.scale(load("rocket.png"), (int(self.w), int(self.h)))
            self.target_img = pygame.transform.scale(load("target.png"), (int(self.target_w), int(self.target_h)))

        self.state = None
        self.last_action = 0
        self.max_steps = 800
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([100., 50., 0., 60., 0., 1., 0., 0., 0.], np.float32) / self._normalizer()
        self.target_vx *= random.choice([1, -1])
        self.last_action = 0
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        self.step_count += 1
        x, y, vx, vy, sinθ, cosθ, ω, dx, dy = self.state * self._normalizer()
        Fx = Fy = torque = 0.0
        θ = np.arctan2(sinθ, cosθ)

        if action == 1:
            Fx = self.F_main * np.sin(θ)
            Fy = self.F_main * np.cos(θ)
        elif action in (2, 3):
            s = 1 if action == 2 else -1
            torque = s * self.F_side * (self.h/2)
            Fx += -s * self.F_side * np.cos(θ)
            Fy += -s * self.F_side * np.sin(θ)

        Fx += -self.b_linear * vx
        Fy += -self.b_linear * vy
        torque += -self.b_angular * ω

        vy += ((Fy/self.m) - self.g) * self.dt
        vx += (Fx/self.m) * self.dt
        y += vy * self.dt
        x += vx * self.dt

        ω += (torque / self.I) * self.dt
        θ += ω * self.dt
        sinθ, cosθ = np.sin(θ), np.cos(θ)

        θ = (θ + np.pi) % (2*np.pi) - np.pi
        vx, vy = np.clip([vx, vy], -120, 120)
        ω = np.clip(ω, -20, 20)

        tx, ty = self.target_pos
        tvx = self.target_vx * self.dt
        tx += tvx
        if tx < self.target_min_x or tx > self.target_max_x:
            self.target_vx *= -1
        self.target_pos[0] = tx
        dx, dy = x - self.target_pos[0], y - self.target_pos[1]

        landed = False
        if y <= self.floor_y:
            y, vy = self.floor_y, 0.0
            landed = True

        tx, ty = self.target_pos
        half_w, half_h = self.target_w/2, self.target_h/2
        target_collide = tx-half_w <= x <= tx+half_w and ty-half_h <= y <= ty+half_h

        D_max = math.hypot(self.screen_w, self.screen_h)
        V_max = 120.0
        distance = math.sqrt(dx**2 + dy**2)
        angle = abs(θ)
        speed = math.sqrt(vx**2 + vy**2)

        r = 0.01
        r -= 2.0 * (distance / D_max)
        r -= 1.5 * (angle / math.pi)
        r -= 0.5 * (speed / V_max)

        terminated, truncated = False, False
        if target_collide and angle < 0.2 and speed < 0.1:
            r += 100.0
            terminated = True
        elif landed or x < 0 or x > self.screen_w or y > self.screen_h:
            r -= 50.0
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        self.state = np.array([x, y, vx, vy, sinθ, cosθ, ω, dx, dy], np.float32) / self._normalizer()
        self.last_action = action
        return self.state, r, terminated, truncated, {}

    def _normalizer(self):
        return np.array([self.screen_w, self.screen_h, 120., 120., 1., 1., 20., self.screen_w, self.screen_h])
