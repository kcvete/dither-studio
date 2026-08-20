#!/usr/bin/env python3
"""What the jobs/ janitor does, checked against fabricated job directories.

    env/venv/bin/python server/jobsgc_check.py

Nothing here touches the real jobs/ -- every case builds its own tree in a
temp directory with the mtimes it needs, runs the same GC class server.py
runs, and asserts on what is left on disk. Fast enough to run on every change;
verify.mjs run `gc` then checks the same policy against the live server.
"""
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobsgc                                                # noqa: E402

DAY = 86400.0
FAILED = []


def check(name, ok, detail=""):
    print("%s  %s%s" % ("ok  " if ok else "FAIL", name,
                        "" if ok else "  -- " + str(detail)))
    if not ok:
        FAILED.append(name)


def mkjob(root, jid, *, age_days, filename=None, mb=1, frames=3,
          source="source.webm", masks=True, renders=True):
    """One job directory, `mb` megabytes of it, last used `age_days` ago."""
    d = os.path.join(root, jid)
    os.makedirs(os.path.join(d, "frames"), exist_ok=True)
    blob = b"\0" * (mb * (1 << 20) // max(1, frames))
    for i in range(frames):
        with open(os.path.join(d, "frames", "%04d.jpg" % i), "wb") as f:
            f.write(blob)
    if masks:
        os.makedirs(os.path.join(d, "masks", "1"), exist_ok=True)
        with open(os.path.join(d, "masks", "1", "0000.png"), "wb") as f:
            f.write(b"\0" * 1024)
    if source:
        with open(os.path.join(d, source), "wb") as f:
            f.write(b"\0" * (1 << 20))
    if renders:
        with open(os.path.join(d, "out.mp4"), "wb") as f:
            f.write(b"\0" * (1 << 20))
    meta = {"job": jid, "n_frames": frames, "w": 16, "h": 16, "fps": 30,
            "created": time.time() - age_days * DAY}
    if source:
        meta["source"] = source
    if filename:
        meta["filename"] = filename
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    stamp = time.time() - age_days * DAY
    for sub in ("frames", os.path.join("masks", "1"), "masks", ""):
        p = os.path.join(d, sub)
        if os.path.isdir(p):
            os.utime(p, (stamp, stamp))
    return d


def gc_for(root, **kw):
    kw.setdefault("budget_mb", 1_000_000)     # effectively no budget
    kw.setdefault("max_age_days", 14)
    kw.setdefault("keep_hours", 48)
    kw.setdefault("enabled", True)
    return jobsgc.GC(root, log=lambda s: None, **kw)


def case_age():
    """Old normal job goes; recent one stays; a camera job keeps its original."""
    root = tempfile.mkdtemp(prefix="gc-age-")
    try:
        mkjob(root, "oldnormal", age_days=30, filename="parkour.mp4")
        mkjob(root, "recent", age_days=0.5, filename="parkour.mp4")
        mkjob(root, "old-but-inside-14d", age_days=9, filename="parkour.mp4")
        mkjob(root, "camjob", age_days=30, filename="camera-101500.webm")
        mkjob(root, "photojob", age_days=30, filename="photo-101500.png",
              frames=1, source=None, masks=True)

        rep = gc_for(root).run("test")

        check("age · a normal job past 14 days is deleted",
              not os.path.exists(os.path.join(root, "oldnormal")),
              rep)
        check("age · a job used in the last 48 h is untouched",
              os.path.isdir(os.path.join(root, "recent", "frames")), rep)
        check("age · a 9-day-old job is untouched",
              os.path.isdir(os.path.join(root, "old-but-inside-14d", "frames")),
              rep)

        cam = os.path.join(root, "camjob")
        check("camera · the directory survives", os.path.isdir(cam), rep)
        check("camera · source.webm survives",
              os.path.exists(os.path.join(cam, "source.webm")), rep)
        check("camera · meta.json survives",
              os.path.exists(os.path.join(cam, "meta.json")), rep)
        check("camera · frames/ is gone",
              not os.path.exists(os.path.join(cam, "frames")), rep)
        check("camera · masks/ is gone",
              not os.path.exists(os.path.join(cam, "masks")), rep)
        check("camera · the render is gone",
              not os.path.exists(os.path.join(cam, "out.mp4")), rep)
        check("camera · it is reported as trimmed, not deleted",
              rep["trimmed"] == ["camjob"] or "camjob" in rep["trimmed"], rep)

        # a photo has no source file: the picture IS frames/0000.jpg
        pho = os.path.join(root, "photojob")
        check("photo · frames/0000.jpg survives -- it is the only copy",
              os.path.exists(os.path.join(pho, "frames", "0000.jpg")), rep)
        check("photo · masks/ is gone",
              not os.path.exists(os.path.join(pho, "masks")), rep)

        check("age · bytes freed is more than zero",
              rep["freed_bytes"] > 1 << 20, rep["freed_bytes"])

        # second sweep: an already-trimmed camera job is not re-reported
        rep2 = gc_for(root).run("test")
        check("camera · a trimmed job is not trimmed twice",
              rep2["trimmed"] == [] and rep2["freed_bytes"] == 0, rep2)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def case_budget():
    """Over budget, the oldest goes first and only until it fits."""
    root = tempfile.mkdtemp(prefix="gc-budget-")
    try:
        # 4 jobs x ~4 MB, all older than the keep window, none past the age
        # limit. Budget 9 MB, so the two oldest have to go and no more.
        for i, age in enumerate((10, 8, 6, 4)):
            mkjob(root, "j%d-age%d" % (i, age), age_days=age, mb=2,
                  filename="clip.mp4")
        before = gc_for(root).usage()["bytes"]
        rep = gc_for(root, budget_mb=9, max_age_days=365).run("test")
        left = sorted(os.listdir(root))

        check("budget · something was evicted", len(rep["deleted"]) > 0, rep)
        check("budget · oldest first",
              rep["deleted"] == sorted(rep["deleted"],
                                       key=lambda x: -int(x.split("age")[1])),
              rep["deleted"])
        check("budget · the newest job survives", "j3-age4" in left, left)
        check("budget · it stopped at the budget",
              rep["usage_bytes"] <= 9 * (1 << 20), rep)
        check("budget · it did not delete everything", len(left) >= 2, left)
        check("budget · freed bytes add up",
              abs((before - rep["freed_bytes"]) - rep["usage_bytes"]) < 4096,
              (before, rep["freed_bytes"], rep["usage_bytes"]))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def case_protected():
    """The 48 h window beats the budget: a job in use is never evicted."""
    root = tempfile.mkdtemp(prefix="gc-prot-")
    try:
        mkjob(root, "inuse", age_days=0, mb=4, filename="clip.mp4")
        mkjob(root, "ancient", age_days=40, mb=4, filename="clip.mp4")
        rep = gc_for(root, budget_mb=1, max_age_days=365).run("test")
        check("protected · a job used minutes ago survives a budget sweep",
              os.path.isdir(os.path.join(root, "inuse", "frames")), rep)
        check("protected · the ancient one goes instead",
              not os.path.exists(os.path.join(root, "ancient")), rep)
        check("protected · it stays over budget rather than eat live work",
              rep["usage_bytes"] > 1 << 20 and rep["protected"] == 1, rep)

        # ... and the stamp alone is enough, with an old directory mtime
        d = mkjob(root, "stamped", age_days=40, mb=4, filename="clip.mp4")
        g = gc_for(root, budget_mb=1, max_age_days=1)
        g.touch("stamped")
        old = time.time() - 40 * DAY
        os.utime(d, (old, old))          # dir mtime lies; the stamp does not
        rep = g.run("test")
        check("protected · the .access stamp alone keeps a job alive",
              os.path.isdir(os.path.join(root, "stamped", "frames")), rep)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def case_hardlink():
    """/reextract links the source instead of copying it. Freed bytes must not
    count a file that another job still holds a link to."""
    root = tempfile.mkdtemp(prefix="gc-link-")
    try:
        a = mkjob(root, "aaa", age_days=30, mb=1, filename="clip.mp4")
        b = mkjob(root, "bbb", age_days=0, mb=1, filename="clip.mp4")
        os.remove(os.path.join(b, "source.webm"))
        os.link(os.path.join(a, "source.webm"), os.path.join(b, "source.webm"))
        g = gc_for(root)
        usage = g.usage()["bytes"]
        naive = sum(os.path.getsize(os.path.join(dp, f))
                    for j in (a, b) for dp, _d, fs in os.walk(j) for f in fs)
        rep = g.run("test")
        check("hardlink · the shared source is counted once in usage",
              abs(naive - usage - (1 << 20)) < 4096, (naive, usage))
        check("hardlink · deleting one link frees nothing for that file",
              rep["freed_bytes"] < 3 * (1 << 20), rep["freed_bytes"])
        check("hardlink · the surviving job still has its source",
              os.path.exists(os.path.join(b, "source.webm")), rep)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def case_seq():
    """seq-* rasterise directories are ordinary throwaways."""
    root = tempfile.mkdtemp(prefix="gc-seq-")
    try:
        d = os.path.join(root, "seq-abc123")
        os.makedirs(d)
        for fn in ("in.dots.gz", "out.mp4"):
            with open(os.path.join(d, fn), "wb") as f:
                f.write(b"\0" * (1 << 20))
        old = time.time() - 30 * DAY
        os.utime(d, (old, old))
        rep = gc_for(root).run("test")
        check("seq · an old seq-* directory is deleted whole",
              not os.path.exists(d) and rep["deleted"] == ["seq-abc123"], rep)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def case_disabled():
    root = tempfile.mkdtemp(prefix="gc-off-")
    try:
        mkjob(root, "old", age_days=99, filename="clip.mp4")
        g = gc_for(root, enabled=False)
        g.start()
        time.sleep(0.2)
        check("off · DV_JOBS_GC=0 never starts the sweep",
              os.path.isdir(os.path.join(root, "old")), os.listdir(root))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    for fn in (case_age, case_budget, case_protected, case_hardlink,
               case_seq, case_disabled):
        print("--- " + fn.__name__)
        fn()
    print()
    if FAILED:
        print("%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
        sys.exit(1)
    print("all green")
