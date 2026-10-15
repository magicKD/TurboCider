#!/usr/bin/env python3
"""Embed the sole image-workflow JSON catalog; no model or device access."""
import argparse
import json
from pathlib import Path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate catalog key: " + key)
        result[key] = value
    return result


def render(source):
    value = json.loads(source.read_text(), object_pairs_hook=unique_object)
    expected = {"playground." + name for name in ("outfit", "identity", "face", "outpaint", "transparent")}
    workflows = value["workflows"]
    if value["schema_version"] != 1 or value["model"] != "qwen-image-2.1" or len(workflows) != 5 or {w["id"] for w in workflows} != expected:
        raise ValueError("invalid image-workflow catalog identity")
    for workflow in workflows:
        for key in ("title", "detail", "default_instruction"):
            if not isinstance(workflow[key], str) or not workflow[key]:
                raise ValueError("invalid workflow " + key)
        roles = workflow["roles"]
        names = {r["id"] for r in roles}
        if len(names) != len(roles):
            raise ValueError("duplicate workflow role")
        for role in roles:
            if type(role["required"]) is not bool or any(not isinstance(role[k], str) or not role[k] for k in ("id", "title", "detail")):
                raise ValueError("invalid workflow role metadata")
        if not workflow["modes"]:
            raise ValueError("workflow requires a mode")
        for mode in workflow["modes"]:
            if mode["operation"] not in ("image.generate", "image.edit") or not mode["prompt_prefix"]:
                raise ValueError("invalid workflow mode")
            if not set(mode["required_roles"] + mode["absent_roles"]).issubset(names):
                raise ValueError("mode refers to an unknown role")
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if "\0" in body or ')TC_WORKFLOW"' in body:
        raise ValueError("invalid catalog string delimiter")
    return '#pragma once\n// Generated from native/workflows/image_workflows.json; do not edit.\nnamespace tc {\ninline constexpr char image_workflow_catalog_json[] = R"TC_WORKFLOW(' + body + ')TC_WORKFLOW";\n}\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = render(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != content:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
