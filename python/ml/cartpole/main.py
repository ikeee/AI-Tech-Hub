"""
CartPole 强化学习（Python 最简实现）。

对应浏览器端 policy gradient 的参考实现：手写倒立摆环境 + 线性策略 + 交叉熵
方法（CEM）训练，不依赖 gym，几秒钟即可学会平衡。

依赖安装：
    pip install numpy

用法：
    python main.py [generations]
"""

import sys

import numpy as np

GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
POLE_LENGTH = 0.5
FORCE_MAG = 10.0
TAU = 0.02


def step(state, action):
    x, v, theta, omega = state
    force = FORCE_MAG if action == 1 else -FORCE_MAG
    costheta, sintheta = np.cos(theta), np.sin(theta)
    temp = (force + MASS_POLE * POLE_LENGTH * omega * omega * sintheta) / TOTAL_MASS
    theta_acc = (GRAVITY * sintheta - costheta * temp) / (
        POLE_LENGTH * (4 / 3 - MASS_POLE * costheta * costheta / TOTAL_MASS)
    )
    x += TAU * v
    v += TAU * temp
    theta += TAU * omega
    omega += TAU * theta_acc
    done = x < -2.4 or x > 2.4 or theta < -0.209 or theta > 0.209
    return np.array([x, v, theta, omega]), done


def run_episode(w, max_steps=500):
    state = np.array([0.0, 0.0, 0.05, 0.0])
    for i in range(max_steps):
        action = 1 if np.dot(w, state) > 0 else 0
        state, done = step(state, action)
        if done:
            return i + 1
    return max_steps


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    generations = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rng = np.random.default_rng(42)
    mean = np.zeros(4)
    std = np.ones(4)
    best = 0
    for g in range(generations):
        candidates = mean + rng.normal(0, 1, (16, 4)) * std
        scores = np.array([run_episode(w) for w in candidates])
        elite = candidates[np.argsort(scores)[-4:]]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-4
        best = max(best, int(scores.max()))
        print(f"第 {g + 1} 代: 最高 {int(scores.max())} / 平均 {scores.mean():.1f}", flush=True)
        if best >= 490:
            print(f"已学会平衡！共 {g + 1} 代", flush=True)
            break


if __name__ == "__main__":
    main()
