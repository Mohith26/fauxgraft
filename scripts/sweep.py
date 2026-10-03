"""Run the full label-budget sweep with N parallel CPU workers (resumable).

    python3 -u scripts/sweep.py --workers 4 --threads 4

Seed 0 for every (k, arm) runs first so a partial sweep is already readable.
"""

import argparse
import itertools
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KS = [1, 2, 4, 8, 16]
ARMS = ["real", "naive", "fauxgraft"]
SEEDS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--runs", default=os.environ.get("FAUXGRAFT_RUNS", os.path.join(ROOT, "runs")))
    a = ap.parse_args()

    jobs = [(k, arm, s, a.iters) for s, k, arm in itertools.product(SEEDS, KS, ARMS)]
    jobs += [(0, "real", s, 6000) for s in (0, 1)]  # all-labels reference

    def run(job):
        k, arm, seed, iters = job
        cmd = [sys.executable, "-m", "fauxgraft.train", "--k", str(k), "--arm", arm,
               "--seed", str(seed), "--iters", str(iters), "--threads", str(a.threads),
               "--runs", a.runs]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        last = [l for l in p.stdout.splitlines() if l.strip()][-1:] or [p.stderr[-300:]]
        print(f"[{arm} k={k} s={seed}] {last[0]}", flush=True)

    with ThreadPool(a.workers) as pool:
        pool.map(run, jobs, chunksize=1)


if __name__ == "__main__":
    main()
