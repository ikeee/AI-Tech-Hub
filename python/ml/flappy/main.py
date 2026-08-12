"""
Flappy Bird 神经进化（Python 最简实现）。

对应浏览器端遗传算法 + 神经网络训练小鸟的参考实现：每只鸟是一个
小型前馈网络（权重为基因），通过适应度（存活步数）选择精英并变异进化。

依赖安装：
    pip install numpy

用法：
    python main.py [generations]
"""

import sys

import numpy as np

W, H = 480, 360
BIRD_X, GRAVITY, JUMP_V = 100, 0.45, -8.5
PIPE_W, PIPE_GAP, PIPE_SPEED = 60, 150, 2.5
INPUT = 3
HIDDEN = 6
POP = 20


def make_net(rng):
    return [rng.normal(0, 1, (HIDDEN, INPUT)) * 0.4,
            rng.normal(0, 1, (HIDDEN,)) * 0.4,
            rng.normal(0, 1, (1, HIDDEN)) * 0.4,
            rng.normal(0, 1, (1,)) * 0.4]


def forward(net, x):
    h = np.tanh(net[0] @ x + net[1])
    return float(1 / (1 + np.exp(-(net[2] @ h + net[3])[0])))


def mutate(net, rng, rate=0.25, amount=0.5):
    return [w + rng.choice([0, 1], size=w.shape, p=[1 - rate, rate]) * rng.normal(0, amount, w.shape)
            for w in net]


def run_bird(net, rng):
    y, vy = H / 2, 0.0
    pipe_x, pipe_gap = W + 100, H / 2
    score = 0
    steps = 0
    while steps < 20000:
        steps += 1
        pipe_x -= PIPE_SPEED
        if pipe_x < -PIPE_W:
            pipe_x = W + 100
            pipe_gap = rng.uniform(90, H - 90)
            score += 1
        x_in = (pipe_x - BIRD_X) / W
        gap_in = (pipe_gap - y) / H
        if forward(net, np.array([y / H, x_in, gap_in])) > 0.5:
            vy = JUMP_V
        vy += GRAVITY
        y += vy
        if y < 8 or y > H - 8:
            break
        if BIRD_X + 10 > pipe_x and BIRD_X - 10 < pipe_x + PIPE_W:
            if y - 8 < pipe_gap - PIPE_GAP / 2 or y + 8 > pipe_gap + PIPE_GAP / 2:
                break
    return score, steps


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    generations = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rng = np.random.default_rng(42)
    pop = [make_net(rng) for _ in range(POP)]
    best = 0
    for g in range(generations):
        scores = [run_bird(net, rng)[0] for net in pop]
        best = max(best, max(scores))
        order = np.argsort(scores)[::-1]
        elites = [pop[i] for i in order[:4]]
        # 保留前 2 名精英原样，其余变异
        pop = [elites[i] if i < 2 else mutate(elites[i % 4], rng) for i in range(POP)]
        print(f"第 {g + 1} 代: 最高 {max(scores)} / 平均 {np.mean(scores):.2f}", flush=True)
        if best >= 30:
            print(f"已学会飞行！共 {g + 1} 代", flush=True)
            break


if __name__ == "__main__":
    main()
