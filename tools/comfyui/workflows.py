#!/usr/bin/env python3
"""Create ready-to-import ComfyUI workflows from existing TurboCider registrations."""
import argparse
import json
import os
from pathlib import Path
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bindings/python"))
from turbocider_http import HTTPClient


def workflow(installation, url, lora=None):
    label = lambda i: f'{i["id"]} | {i["modelID"]} | {i["name"]}'
    model = installation["modelID"]
    steps = 6 if lora and "viggle" in lora["name"].lower() else (20 if model == "qwen-image-2.1" else 8 if model.startswith("z-image") else 4)
    advanced = '{"allow_approximation":true}' if lora else '{}'
    def node(key, kind, x, y, size, widgets, inputs, outputs):
        return {"id": key, "type": kind, "pos": [x, y], "size": size, "flags": {}, "order": key - 1, "mode": 0,
                "inputs": inputs, "outputs": outputs, "properties": {"Node name for S&R": kind}, "widgets_values": widgets}
    nodes = [node(1, "TurboCiderModel", 40, 80, [470, 180], [label(installation), url, "TURBOCIDER_API_TOKEN", ""], [],
                  [{"name": "backend", "type": "TURBOCIDER_MODEL", "links": [1]}, {"name": "capabilities", "type": "STRING", "links": None}])]
    links = []
    if lora:
        nodes.append(node(2, "TurboCiderLoRA", 40, 330, [470, 190], [label(lora), 1.0, "auto", ""],
                          [{"name": "backend", "type": "TURBOCIDER_MODEL", "link": 1}], [{"name": "TURBOCIDER_MODEL", "type": "TURBOCIDER_MODEL", "links": [2]}]))
        links += [[1, 1, 0, 2, 0, "TURBOCIDER_MODEL"], [2, 2, 0, 3, 0, "TURBOCIDER_MODEL"]]
    else:
        links.append([1, 1, 0, 3, 0, "TURBOCIDER_MODEL"])
    nodes.append(node(3, "TurboCiderGenerate", 570, 80, [430, 470],
                      ["A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.", 512, 512, steps, 42, "fixed", 1, advanced, 1800],
                      [{"name": "backend", "type": "TURBOCIDER_MODEL", "link": 2 if lora else 1}],
                      [{"name": "images", "type": "IMAGE", "links": [3]}, {"name": "alpha_mask", "type": "MASK", "links": None}, {"name": "jobs", "type": "STRING", "links": None}]))
    nodes.append(node(4, "SaveImage", 1060, 80, [420, 470], ["TurboCider/" + model],
                      [{"name": "images", "type": "IMAGE", "link": 3}], []))
    links.append([3, 3, 0, 4, 0, "IMAGE"])
    return {"id": str(uuid.uuid4()), "revision": 0, "last_node_id": 4, "last_link_id": 3, "nodes": nodes, "links": links,
            "groups": [], "config": {}, "extra": {"ds": {"scale": 0.8, "offset": [30, 30]}}, "version": 0.4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.environ.get("TURBOCIDER_API_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    catalog = HTTPClient(args.api_url, os.environ.get("TURBOCIDER_API_TOKEN", "")).catalog()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = {m["id"]: m for m in catalog["models"]}
    for installation in catalog["installations"]:
        if models[installation["modelID"]]["output"] != "image":
            continue
        name = installation["modelID"] + "-" + installation["id"][:8]
        target = args.output_dir / (name + ".json")
        target.write_text(json.dumps(workflow(installation, args.api_url), indent=2, ensure_ascii=False))
        print(target)
        for lora in catalog["loras"]:
            if lora["modelID"] == installation["modelID"] and ("viggle" in lora["name"].lower() or "distill_patch" in lora["name"]):
                target = args.output_dir / (name + "-lora-" + lora["id"][:8] + ".json")
                target.write_text(json.dumps(workflow(installation, args.api_url, lora), indent=2, ensure_ascii=False))
                print(target)


if __name__ == "__main__":
    main()
