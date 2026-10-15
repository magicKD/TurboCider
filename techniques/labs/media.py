"""Actual bounded producer/consumer and trace-accounting learning experiments."""
import hashlib
from queue import Queue
from threading import Lock, Thread
from fractions import Fraction

try:
    from .common import report
except ImportError:
    from common import report


def bounded_frames(seed=17):
    frames, width, height, capacity = 25, 16, 12, 2
    def frame(index):
        return bytes((index * 17 + pixel + seed) % 256 for pixel in range(width * height * 4))
    reference = hashlib.sha256(b"".join(frame(index) for index in range(frames))).hexdigest()
    queue, lock = Queue(maxsize=capacity), Lock()
    live, peak, seen, failures = 0, 0, [], []
    def producer():
        nonlocal live, peak
        try:
            for index in range(frames):
                pixels = frame(index)
                with lock:
                    live += 1
                    peak = max(live, peak)
                queue.put((index, pixels))
                del pixels
        except BaseException as error:
            failures.append(error)
        finally:
            queue.put(None)
    thread = Thread(target=producer)
    thread.start()
    digest = hashlib.sha256()
    try:
        while True:
            item = queue.get()
            if item is None:
                break
            index, pixels = item
            digest.update(pixels)
            seen.append(index)
            del item, pixels
            with lock:
                live -= 1
    finally:
        thread.join()
    if failures:
        raise failures[0]
    assert digest.hexdigest() == reference and seen == list(range(frames)) and live == 0 and peak <= capacity + 2
    fps = Fraction(24000, 1001)
    return report("media-backpressure", {"frames": frames, "RGBA_shape": [height, width, 4],
                  "frame_sha_equal": True, "observed_live_frame_buffers": peak,
                  "maximum_by_contract": capacity + 2, "buffered_payload_bytes": frames * width * height * 4,
                  "bounded_payload_upper_bytes": (capacity + 2) * width * height * 4,
                  "timebase": {"fps": str(fps), "last_frame_pts_seconds": str(Fraction(frames - 1, 1) / fps)},
                  "limits": "Actual Python queue/hash exercise, not codec memory, VAE inference, MP4 export or long-model-video qualification."},
                  scope="actual CPU bounded queue with synthetic RGBA frames")


def union_seconds(intervals):
    ordered = sorted(intervals)
    total, start, end = 0., None, None
    for left, right in ordered:
        if left < 0 or right < left:
            raise ValueError("invalid interval")
        if start is None:
            start, end = left, right
        elif left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return total + (end - start if start is not None else 0)


def profiling_scopes(seed=17):
    trace = [{"name": "request", "start": 0., "end": 15., "scope": "host"},
             {"name": "gpu", "start": 2., "end": 10., "scope": "device_assumption"},
             {"name": "ane", "start": 3., "end": 9., "scope": "device_assumption"},
             {"name": "finish_wait", "start": 8., "end": 10., "scope": "host"},
             {"name": "decode_export", "start": 10., "end": 15., "scope": "host"}]
    naive = sum(row["end"] - row["start"] for row in trace)
    union = union_seconds([(row["start"], row["end"]) for row in trace])
    assert naive > union == 15.
    return report("profiling-scopes", {"synthetic_timeline": trace, "wrong_sum_nested_spans": naive,
                  "timeline_union": union, "finish_wait_is_not_ane_alone": True,
                  "power_example": {"baseline_watts": 30, "baseline_seconds": 10, "baseline_joules": 300,
                                    "candidate_watts": 25, "candidate_seconds": 14, "candidate_joules": 350},
                  "note": "Intervals/power are invented to teach accounting. Feed genuine timestamps with aligned clocks for real analysis."},
                  scope="synthetic timing/energy accounting witness")
