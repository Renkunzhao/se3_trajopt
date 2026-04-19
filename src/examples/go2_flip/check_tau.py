#!/usr/bin/env python3
"""Analyze tau usage against Go2 effort limits from an exported CSV."""

import sys
import numpy as np

# Go2 per-joint effort limits (N·m), URDF order hip/thigh/calf per leg
GO2_EFFORT_LIMITS = np.array([23.7, 23.7, 45.43] * 4)
DT = 0.02


def analyze(csv_path: str, dt: float = DT) -> None:
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    tau_cols = [i for i, h in enumerate(header) if h.startswith("tau_")]
    if not tau_cols:
        raise ValueError(f"No tau_* columns in {csv_path}")
    tau_names = [header[i].replace("tau_", "") for i in tau_cols]
    tau = data[:, tau_cols]
    limits = GO2_EFFORT_LIMITS

    abs_tau = np.abs(tau)
    peaks = abs_tau.max(axis=0)
    peak_idx = abs_tau.argmax(axis=0)
    usage = 100 * peaks / limits

    print(f"file: {csv_path}")
    print(f"frames: {len(data)}, dt={dt}s → T={len(data)*dt:.2f}s\n")
    print(f'{"joint":<20}{"|τ|_max":>10}{"limit":>10}{"usage%":>10}{"peak_t":>8}')
    print("-" * 58)
    order = np.argsort(-usage)
    for i in order:
        mark = ""
        if usage[i] > 99:
            mark = " ←SAT"
        elif usage[i] > 95:
            mark = " !"
        elif usage[i] > 85:
            mark = "  ~"
        print(
            f"{tau_names[i]:<20}{peaks[i]:>10.2f}{limits[i]:>10.2f}"
            f"{usage[i]:>9.1f}%{peak_idx[i]*dt:>7.2f}s{mark}"
        )

    sat95 = (abs_tau >= 0.95 * limits).sum(axis=0)
    sat99 = (abs_tau >= 0.99 * limits).sum(axis=0)
    any95 = [(n, s95, s99) for n, s95, s99 in zip(tau_names, sat95, sat99) if s95 > 0]
    if any95:
        print("\n frames >=95% / >=99% limit:")
        for n, s95, s99 in any95:
            print(f"  {n}: {s95} / {s99}")

    top = order[0]
    print(f"\npeak: {tau_names[top]} @ {usage[top]:.1f}% (t={peak_idx[top]*dt:.2f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <csv_path> [csv_path ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        analyze(path)
        if len(sys.argv) > 2:
            print()
