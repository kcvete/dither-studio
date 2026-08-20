#!/usr/bin/env python3
"""jobs/ garbage collection -- the janitor for the scratch directory.

Every upload, every re-extract, every suite run leaves a directory behind under
jobs/, and until this module nothing ever removed one: two days of use grew it
to 5.4 GB across 323 directories, most of them throwaways from verify.mjs and
duplicates of the same clip. This sweeps it.

The policy, both knobs env-tunable:

    DV_JOBS_BUDGET_MB      2048   how much disk jobs/ may hold
    DV_JOBS_MAX_AGE_DAYS     14   how long an untouched job lives
    DV_JOBS_KEEP_HOURS       48   a job used this recently is never touched
    DV_JOBS_GC_EVERY_H        6   how often the background sweep runs
    DV_JOBS_GC                1   0 turns the whole thing off

A sweep runs on server startup and every DV_JOBS_GC_EVERY_H hours after that:

  1. anything past the age limit goes,
  2. then, while jobs/ is still over budget, the oldest goes, one at a time.

Two things keep it from eating work in progress or work that cannot be redone:

  * A job used inside DV_JOBS_KEEP_HOURS is off limits in both passes. "Used"
    is the newest of the directory's own mtime and the `.access` stamp, which
    server.py writes on any /api/jobs/<id>/* call -- reading frames does not
    otherwise touch anything on disk, so a job someone is looking at right now
    would look ancient without it. 48 h is far longer than any flow, which is
    why a GC firing in the middle of a track -> render cannot hurt it.

  * A job whose meta `filename` starts with `camera-` or `photo-` was recorded
    in the browser and exists nowhere else. Those are TRIMMED, not deleted:
    frames/, masks/, polish/, preview/ and every render output go, while
    meta.json and the original -- source.webm for a recording, frames/0000.jpg
    for a photo, which is the only copy of the picture there is -- stay. The
    trimmed job is still re-extractable through /api/jobs/<id>/reextract.

Everything else, seq-* rasterise dirs included, is deleted whole.

Byte accounting is hard-link aware: /reextract links the source into the new
job instead of copying it, so a file with more than one link still on disk
frees nothing when one copy is removed, and this counts it that way.
"""
import json
import os
import shutil
import threading
import time

STAMP = ".access"                 # last-access marker, written by touch()
KEEP_NAMES = ("camera-", "photo-")
TRIM_DIRS = ("frames", "masks", "polish", "preview")
TOUCH_EVERY_S = 60                # don't rewrite the stamp on every frame GET
MB = 1 << 20


def _env(name, default, cast=float):
    try:
        return cast(os.environ.get(name, "").strip() or default)
    except ValueError:
        return cast(default)


# ------------------------------------------------------------------ sizes
def _walk_bytes(root, seen=None):
    """Bytes under `root`. A hard-linked file counts once per `seen` set."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            try:
                st = os.lstat(os.path.join(dirpath, fn))
            except OSError:
                continue
            if st.st_nlink > 1 and seen is not None:
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
            total += st.st_size
    return total


def _freeable(path):
    """Bytes the volume actually gets back if `path` goes away -- which is
    nothing at all for a file another job still links to."""
    total = 0
    for dirpath, _d, filenames in os.walk(path):
        for fn in filenames:
            try:
                st = os.lstat(os.path.join(dirpath, fn))
            except OSError:
                continue
            if st.st_nlink <= 1:
                total += st.st_size
    return total


def _rm(path):
    """Remove a file or a tree. Returns the bytes freed."""
    try:
        st = os.lstat(path)
    except OSError:
        return 0
    if os.path.isdir(path) and not os.path.islink(path):
        freed = _freeable(path)
        shutil.rmtree(path, ignore_errors=True)
        return 0 if os.path.exists(path) else freed
    freed = st.st_size if st.st_nlink <= 1 else 0
    try:
        os.remove(path)
    except OSError:
        return 0
    return freed


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f TB" % n


# --------------------------------------------------------------- the sweep
class GC:
    """One janitor over one jobs/ directory. server.py makes exactly one."""

    def __init__(self, jobs_dir, budget_mb=None, max_age_days=None,
                 keep_hours=None, every_h=None, enabled=None, log=None):
        self.root = jobs_dir
        self.budget_mb = float(budget_mb if budget_mb is not None
                               else _env("DV_JOBS_BUDGET_MB", 2048))
        self.max_age_days = float(max_age_days if max_age_days is not None
                                  else _env("DV_JOBS_MAX_AGE_DAYS", 14))
        self.keep_hours = float(keep_hours if keep_hours is not None
                                else _env("DV_JOBS_KEEP_HOURS", 48))
        self.every_h = float(every_h if every_h is not None
                             else _env("DV_JOBS_GC_EVERY_H", 6))
        self.enabled = bool(int(_env("DV_JOBS_GC", 1, int))
                            if enabled is None else enabled)
        self._log = log or (lambda s: print(s, flush=True))
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()   # one sweep at a time
        self._touched = {}          # jid -> when the stamp was last written
        self.last = None            # the last run's report
        self.next_at = 0.0
        self._thread = None

    # ------------------------------------------------------------- access
    def touch(self, jid):
        """Mark a job as in use. Cheap enough to call on every API hit: the
        stamp is only rewritten once a minute per job."""
        if not jid or os.path.sep in jid or jid.startswith("."):
            return
        now = time.time()
        with self._lock:
            if now - self._touched.get(jid, 0.0) < TOUCH_EVERY_S:
                return
            self._touched[jid] = now
        d = os.path.join(self.root, jid)
        if not os.path.isdir(d):
            return
        try:
            with open(os.path.join(d, STAMP), "w") as f:
                f.write("%d\n" % now)
        except OSError:
            pass

    # -------------------------------------------------------------- scan
    def scan(self):
        """Every job directory: bytes, when it was last used, what it is.

        `usage` is the du-honest total (a hard link counted once); the per-job
        `bytes` are counted per job, so summing them can exceed it when two
        jobs share a source. The budget loop works off the total and subtracts
        what each eviction really freed, so the double count never matters.
        """
        jobs, seen = [], set()
        try:
            names = sorted(os.listdir(self.root))
        except OSError:
            return [], 0
        usage = 0
        for name in names:
            d = os.path.join(self.root, name)
            if name.startswith(".") or not os.path.isdir(d):
                continue
            usage += _walk_bytes(d, seen)
            used = 0.0
            try:
                used = os.stat(d).st_mtime
            except OSError:
                pass
            try:
                used = max(used, os.stat(os.path.join(d, STAMP)).st_mtime)
            except OSError:
                pass
            meta = {}
            try:
                with open(os.path.join(d, "meta.json")) as f:
                    meta = json.load(f)
            except Exception:                                # noqa: BLE001
                meta = {}
            fn = str(meta.get("filename") or "").strip().lower()
            jobs.append({
                "id": name, "dir": d,
                "bytes": _walk_bytes(d, set()),
                "used": used,
                "keep": fn.startswith(KEEP_NAMES),
                "trimmed": bool(meta.get("gc_trimmed")),
                "source": meta.get("source") or "",
            })
        with self._lock:
            for j in jobs:
                t = self._touched.get(j["id"])
                if t:
                    j["used"] = max(j["used"], t)
        return jobs, usage

    def usage(self):
        jobs, total = self.scan()
        return {"bytes": total, "jobs": len(jobs)}

    # ---------------------------------------------------------- eviction
    def _trim(self, job):
        """A camera recording or a photo past its date: throw away everything
        that can be rebuilt, keep the thing that cannot."""
        d = job["dir"]
        # A still has no source file -- upload_image writes the picture straight
        # into frames/0000.jpg and that IS the original. Keep it.
        has_source = bool(job["source"]) and os.path.exists(
            os.path.join(d, job["source"]))
        freed = 0
        for sub in TRIM_DIRS:
            if sub == "frames" and not has_source:
                continue
            freed += _rm(os.path.join(d, sub))
        try:
            for fn in sorted(os.listdir(d)):
                if fn.startswith("out.") or fn.startswith("in."):
                    freed += _rm(os.path.join(d, fn))
        except OSError:
            pass
        # Record it, and put the mtime back: a trim is the janitor's doing, not
        # the user's, and must not make the job look freshly used.
        try:
            p = os.path.join(d, "meta.json")
            with open(p) as f:
                meta = json.load(f)
            meta["gc_trimmed"] = int(time.time())
            with open(p, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:                                    # noqa: BLE001
            pass
        try:
            os.utime(d, (job["used"], job["used"]))
        except OSError:
            pass
        return freed

    def _evict(self, job):
        """Delete or trim one job. Returns (what, bytes freed)."""
        if job["keep"]:
            if job["trimmed"]:
                return None, 0
            return "trimmed", self._trim(job)
        freed = _freeable(job["dir"])
        shutil.rmtree(job["dir"], ignore_errors=True)
        if os.path.exists(job["dir"]):
            return None, 0
        return "deleted", freed

    # --------------------------------------------------------------- run
    def run(self, reason="periodic"):
        # Two sweeps at once would race each other's rmtree and double-count
        # what they freed. The timer and the button share this lock.
        with self._run_lock:
            return self._run(reason)

    def _run(self, reason):
        t0 = time.perf_counter()
        now = time.time()
        jobs, usage = self.scan()
        before = usage
        budget = self.budget_mb * MB
        keep_cut = now - self.keep_hours * 3600.0
        age_cut = now - self.max_age_days * 86400.0
        deleted, trimmed, freed = [], [], 0

        def evict(j):
            nonlocal freed
            what, n = self._evict(j)
            if what is None:
                return False
            freed += n
            (deleted if what == "deleted" else trimmed).append(j["id"])
            return True

        # oldest first, and never anything used inside the keep window
        live = sorted((j for j in jobs if j["used"] < keep_cut),
                      key=lambda j: j["used"])
        done = set()

        # 1. past the age limit
        for j in live:
            if j["used"] < age_cut and evict(j):
                done.add(j["id"])
        usage = before - freed

        # 2. still over budget: oldest first until it fits
        for j in live:
            if usage <= budget:
                break
            if j["id"] in done:
                continue
            was = freed
            if evict(j):
                done.add(j["id"])
                usage -= freed - was

        el = time.perf_counter() - t0
        rep = {
            "at": now, "reason": reason,
            "deleted": deleted, "trimmed": trimmed,
            "freed_bytes": freed,
            "usage_bytes": max(0, usage), "usage_before_bytes": before,
            "jobs": len(jobs) - len(deleted),
            "protected": sum(1 for j in jobs if j["used"] >= keep_cut),
            "elapsed_s": round(el, 3),
        }
        with self._lock:
            self.last = rep
        self._log(
            "[gc] %s: %d deleted, %d trimmed, %s freed -- %s / %s in %d jobs "
            "(%d protected, <%gh) in %.2fs"
            % (reason, len(deleted), len(trimmed), human(freed),
               human(rep["usage_bytes"]), human(budget), rep["jobs"],
               rep["protected"], self.keep_hours, el))
        return rep

    def status(self):
        with self._lock:
            last = dict(self.last) if self.last else None
        u = self.usage()
        return {
            "enabled": self.enabled,
            "budget_mb": self.budget_mb,
            "max_age_days": self.max_age_days,
            "keep_hours": self.keep_hours,
            "every_h": self.every_h,
            "usage_bytes": u["bytes"],
            "usage_mb": round(u["bytes"] / MB, 1),
            "jobs": u["jobs"],
            "over_budget": u["bytes"] > self.budget_mb * MB,
            "last_run": last,
            "next_run_in_s": (max(0, round(self.next_at - time.time()))
                              if self.next_at else None),
        }

    # ------------------------------------------------------------ thread
    def start(self):
        """Sweep now, then every `every_h` hours, on a daemon thread."""
        if not self.enabled or self._thread:
            if not self.enabled:
                self._log("[gc] disabled (DV_JOBS_GC=0)")
            return
        every = max(60.0, self.every_h * 3600.0)

        def loop():
            while True:
                try:
                    self.run("startup" if self.last is None else "periodic")
                except Exception as e:                       # noqa: BLE001
                    self._log("[gc] sweep failed: %s" % e)
                self.next_at = time.time() + every
                time.sleep(every)

        self._thread = threading.Thread(target=loop, name="jobs-gc", daemon=True)
        self._thread.start()
