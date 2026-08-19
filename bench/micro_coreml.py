#!/usr/bin/env python3
"""Microbenchmark one exported CoreML graph across compute units.

    env/venv/bin/python bench/micro_coreml.py [--graph memattn-b1] [--iters 30]

Times `predict()` on pre-made numpy inputs, so the number is CoreML's own
compute + its input copy, with no torch conversion in the loop.
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import coremltools as ct

UNITS = {"ALL": ct.ComputeUnit.ALL, "GPU": ct.ComputeUnit.CPU_AND_GPU,
         "NE": ct.ComputeUnit.CPU_AND_NE, "CPU": ct.ComputeUnit.CPU_ONLY}


def inputs_for(man, graph):
    B = int(graph.split("-b")[-1]) if "-b" in graph else 1
    T, M, H, D = man["tokens"], man["memlen"], man["hidden"], man["mem_dim"]
    S = man["image_size"]
    r = np.random.RandomState(0)
    if graph.startswith("memattn"):
        return {"curr": r.randn(T, B, H).astype(np.float32),
                "memory": r.randn(M, B, D).astype(np.float32),
                "curr_pos": r.randn(T, B, H).astype(np.float32),
                "memory_pos": r.randn(M, B, D).astype(np.float32)}
    if graph.startswith("memenc"):
        return {"pix_feat": r.randn(B, H, S // 16, S // 16).astype(np.float32),
                "mask": r.rand(B, 1, S, S).astype(np.float32),
                "is_obj": np.ones((B, 1), np.float32)}
    return {"image": r.randn(1, 3, S, S).astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "env", "coreml"))
    ap.add_argument("--graph", default="memattn-b1")
    ap.add_argument("--units", default="ALL,GPU,NE")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.dir, "manifest.json")))
    path = os.path.join(a.dir, man["graphs"][a.graph]["path"]) \
        if a.graph in man.get("graphs", {}) else os.path.join(a.dir, a.graph + ".mlpackage")
    x = inputs_for(man, a.graph)
    print("graph=%s inputs=%s" % (a.graph, {k: v.shape for k, v in x.items()}))
    for u in a.units.split(","):
        t = time.perf_counter()
        try:
            m = ct.models.MLModel(path, compute_units=UNITS[u])
        except Exception as e:
            print("  %-4s load failed: %s" % (u, e))
            continue
        load = time.perf_counter() - t
        for _ in range(3):
            m.predict(x)
        t = time.perf_counter()
        for _ in range(a.iters):
            m.predict(x)
        ms = 1000 * (time.perf_counter() - t) / a.iters
        print("  %-4s  %7.2f ms/call   (load %.1fs)" % (u, ms, load))
        del m


if __name__ == "__main__":
    main()
