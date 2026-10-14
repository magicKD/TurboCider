#!/usr/bin/env python3
"""Summarize completed original-boundary VAE diagnostics, not speed qualification.

Synchronization alters cadence. Inclusive mid blocks overlap their child rows.
History buffers may also own live operator inputs/outputs; these metrics are not
disjoint physical-memory categories and must never be summed into a RAM budget.
"""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
from seedvr2_framework_screen import sha256,write_json


def expected_counts(frames):
    if type(frames) is not int or frames<1 or (frames-1)%4:
        raise ValueError("complete padded decoder frame count required")
    latent=(frames-1)//4+1;slices=1+max(0,latent-2)
    counts={"decoder.mid_block.inclusive":slices,
        "decoder.up_blocks.0.projection_shuffle_conv":slices,
        "decoder.up_blocks.1.upsamplers.0.projection_shuffle_conv":(frames+1)//2,
        "decoder.up_blocks.2.projection_shuffle_conv":frames,
        "decoder.conv_norm_out_conv_out":frames}
    for prefix,number,calls in (("decoder.mid_block",2,slices),
        *(("decoder.up_blocks."+str(block),3,slices if block<2 else frames) for block in range(4))):
        for resnet in range(number):
            for suffix in ("norm1_conv1","norm2_conv2_residual"):
                counts[f"{prefix}.resnets.{resnet}.{suffix}"]=calls
    return counts


def validate_trace(rows,observation):
    if (observation.get("quality_passed") is not True or observation.get("bitwise_equal") is not True or
        observation.get("route")!="baseline" or observation.get("diagnostic_phase_count")!=len(rows)):
        raise ValueError("completed byte-exact full-route phase observation required")
    counts=expected_counts(observation["frames"])
    if Counter(row.get("key") for row in rows)!=Counter(counts):
        raise ValueError("phase trace does not cover every stage/frame/slice")
    previous_peak=0
    for index,row in enumerate(rows):
        if (row.get("schema")!="tc_seedvr2_vae_phase_profile_v1" or row.get("index")!=index or
            type(row.get("next_output_frame")) is not int or not 0<=row["next_output_frame"]<observation["frames"] or
            row.get("inclusive")!=(row["key"]=="decoder.mid_block.inclusive") or
            not math.isfinite(row.get("seconds",float("nan"))) or row["seconds"]<0):
            raise ValueError("malformed or incomplete phase observation")
        for name in ("input_shape","output_shape"):
            shape=row.get(name)
            if not isinstance(shape,list) or len(shape)!=5 or any(type(x) is not int or x<1 for x in shape):
                raise ValueError("phase tensor geometry differs")
        for name in ("active_before","active_after","peak_before","peak_after","cache_after",
            "history_unique_before","history_unique_after","history_logical_after","history_unavailable_after"):
            if type(row.get(name)) is not int or row[name]<0:raise ValueError("invalid phase allocation accounting")
        if (row["peak_before"]<row["active_before"] or row["peak_after"]<max(row["peak_before"],row["active_after"],previous_peak) or
            row["history_unique_before"]>row["active_before"] or row["history_unique_after"]>row["active_after"] or
            row["history_unavailable_after"]):
            raise ValueError("phase memory high-water/history accounting differs")
        previous_peak=row["peak_after"]
    if previous_peak!=observation.get("mlx_active_peak_bytes"):
        raise ValueError("phase trace does not account for the decoder maximum")
    for key in ("decoder.up_blocks.2.projection_shuffle_conv","decoder.conv_norm_out_conv_out"):
        if [row["next_output_frame"] for row in rows if row["key"]==key]!=list(range(observation["frames"])):
            raise ValueError("output frame phase order differs")


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        if not row["inclusive"]:groups[row["key"]].append(row)
    timings=[dict(key=key,calls=len(values),seconds=sum(row["seconds"] for row in values),
        initial_seconds=sum(row["seconds"] for row in values if row["initial"])) for key,values in groups.items()]
    timings.sort(key=lambda row:row["seconds"],reverse=True)
    highwaters=[row for row in rows if not row["inclusive"] and row["peak_after"]>row["peak_before"]]
    return dict(noninclusive_phase_seconds=sum(row["seconds"] for row in timings),timings=timings,
        maximum_history_unique_bytes=max(row["history_unique_after"] for row in rows),
        highest_peak_region=max(highwaters,key=lambda row:row["peak_after"]),new_high_water_regions=highwaters,
        memory_scope="MLX active/cumulative high water and deduplicated allocator buffer extents; history/input/output may alias; not disjoint or physical RAM budgets",
        timing_scope="original evaluation boundaries plus diagnostic synchronization, norm/conv and upsample/shuffle/conv bundles; inclusive mid-block children excluded from sums; not paired warm speed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh phase summary required")
    screen=json.loads(args.screen.read_text())
    if screen.get("status")!="complete" or len(screen.get("routes",[]))!=1:parser.error("one completed full-route screen required")
    receipt=screen["routes"][0];observation=receipt["observation"]
    if receipt["exit_code"]:parser.error("failed decoder is not phase evidence")
    trace=Path(receipt["command"][3])/"phases.jsonl"
    rows=[json.loads(line) for line in trace.read_text().splitlines()];validate_trace(rows,observation)
    report=dict(schema="seedvr2-vae-phase-summary-v1",status="complete",qualified=False,
        screen_sha256=sha256(args.screen),trace_sha256=sha256(trace),export_sha256=receipt["export_sha256"],
        probe_sha256=screen["probe_sha256"],checkpoint_sha256=screen["checkpoint_sha256"],
        input_sha256=screen["prefix_fixture_sha256"],frames=observation["frames"],**summarize(rows))
    write_json(args.output,report)
    print(json.dumps(dict(status="complete",frames=report["frames"],history=report["maximum_history_unique_bytes"],
        highest_peak_region=report["highest_peak_region"],slowest=report["timings"][:5])),flush=True)


if __name__=="__main__":main()
