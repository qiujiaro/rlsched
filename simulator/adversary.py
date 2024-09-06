"""Robust optimisation hooks (paper Eq.5-6).

AdversarialDisturbance approximates the inner  min_{xi in Omega}  of Eq.(6) greedily per step:
K candidate disturbances are drawn from Omega(eps); each is evaluated on a cloned simulator with the
agent's action, and the one yielding the lowest reward is applied to the real transition.
"""
import gymnasium as gym
import numpy as np

from .env import sample_disturbance


class AdversarialDisturbance(gym.Wrapper):
    def __init__(self, env, eps=0.3, k=4, p_active=1.0):
        super().__init__(env)
        self.eps, self.k, self.p_active = eps, k, p_active

    def step(self, action):
        base = self.env.unwrapped
        rng = base.np_random
        if self.eps <= 0 or rng.random() > self.p_active:
            base.set_disturbance(None)
            return self.env.step(action)
        worst, worst_r = None, np.inf
        for _ in range(self.k):
            xi = sample_disturbance(rng, self.eps, base.M)
            sim = base.clone()
            sim.set_disturbance(xi)
            _, r, *_ = sim.step(action)
            if r < worst_r:
                worst, worst_r = xi, r
        base.set_disturbance(worst)
        obs, r, term, trunc, info = self.env.step(action)
        info["xi_surge"] = worst["surge"]
        return obs, r, term, trunc, info
