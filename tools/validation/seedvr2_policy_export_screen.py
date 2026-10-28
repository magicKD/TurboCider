#!/usr/bin/env python3
"""Verify every decoded frame of the t25/24-FPS native policy ABBA exports."""
import argparse
import json
from pathlib import Path

from seedvr2_app_media_screen import PixelStats, inspect_video, read_frame, video_frames
from seedvr2_framework_screen import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("fresh export screen output required")
    cohort = json.loads(args.cohort.read_text())
    if (cohort.get("schema") != "seedvr2-native-revision-request-abba-v1" or cohort.get("status") != "complete" or
        cohort.get("order") != ["before","after","after","before"] or len(cohort.get("cohorts",[])) != 4):
        parser.error("complete four-process ABBA cohort required")
    workload = cohort["workload"]
    if workload["kind"] != "video" or workload["frames"] != 25 or workload["fps"] != 24:
        parser.error("this screen qualifies only complete t25/24-FPS video cohorts")
    size = workload["size"]
    result = dict(schema="seedvr2-policy-export-screen-v1",status="running",cohort_sha256=sha256(args.cohort),pairs=[],
                  scope="all cold/warm MP4 decoded frames; container hashes are separate; not original-FP32 tensor qualification")
    try:
        for first,second in ((0,1),(3,2)):
            observations = [json.loads(Path(cohort["cohorts"][index]["report"]).read_text()) for index in (first,second)]
            if any(item["status"] != "complete" for item in observations) or len(observations[0]["runs"]) != len(observations[1]["runs"]):
                raise ValueError("incomplete/mismatched policy observers")
            for index,(left,right) in enumerate(zip(observations[0]["runs"],observations[1]["runs"])):
                if left["metrics"]["seed"] != right["metrics"]["seed"]:
                    raise ValueError("policy outputs do not share the same sampling seed")
                # The observer's row.output is ffprobe metadata, not a path.
                paths = [Path(item["metrics"]["output"]) for item in (left,right)]
                hashes = [sha256(path) for path in paths]
                if hashes != [item["output_sha256"] for item in (left,right)]:
                    raise ValueError("policy output changed since observation")
                for path in paths: inspect_video(path,size)
                stats = PixelStats()
                with video_frames(paths[0],size) as a, video_frames(paths[1],size) as b:
                    while True:
                        x,y = read_frame(a,size*size*3),read_frame(b,size*size*3)
                        if x is None or y is None:
                            if x is not None or y is not None: raise ValueError("policy frame counts differ")
                            break
                        stats.add(x,y)
                values = stats.result()
                result["pairs"].append(dict(cohorts=[first,second],request=index,paths=list(map(str,paths)),
                                            artifact_sha256=hashes,**values))
                if stats.frames != 25 or not stats.exact:
                    raise ValueError("policy decoded frames are not completely bitwise equal")
        result["status"] = "complete"
        print(json.dumps(dict(status="complete",pairs=len(result["pairs"]),all_frames_bitwise_equal=True)),flush=True)
    except BaseException as error:
        result.update(status="failed",failure=str(error))
        raise
    finally:
        write_json(args.output,result)


if __name__ == "__main__": main()
