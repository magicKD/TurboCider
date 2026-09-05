"""Execution-plan selection with fail-closed heterogeneous routing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from turbocider.errors import PlanUnavailableError
from turbocider.device import DeviceProfileCatalog
from turbocider.models import (
    ApproximationMode,
    ExecutionMode,
    ExecutionPlan,
    GenerationRequest,
    ModelDescriptor,
)


_QUALITY_RANK = {
    ApproximationMode.EXACT: 0,
    ApproximationMode.VALIDATED: 1,
    ApproximationMode.EXPERIMENTAL: 2,
}


def _requirement_failures(
    model: ModelDescriptor,
    plan: ExecutionPlan,
    request: GenerationRequest,
    *,
    enforce_device_profile: bool = True,
) -> List[str]:
    requirements = plan.requirements
    failures: List[str] = []

    for key in requirements.get("config_paths", []):
        value = model.config.get(str(key))
        if not value or not Path(str(value)).exists():
            failures.append("missing config path %s=%s" % (key, value or "<unset>"))

    for key in requirements.get("config_path_lists", []):
        value = model.config.get(str(key), [])
        if isinstance(value, str):
            value = [item for item in value.split(os.pathsep) if item]
        if not value or not all(Path(str(item)).exists() for item in value):
            failures.append("missing config path list %s" % key)

    for name in requirements.get("environment", []):
        if not os.environ.get(str(name)):
            failures.append("missing environment variable %s" % name)

    if requirements.get("persistent") and not request.policy.persistent:
        failures.append("plan requires persistent=true")

    modes = requirements.get("modes", [])
    if modes and request.resolved_mode not in modes:
        failures.append("mode %s is unsupported" % request.resolved_mode)

    excluded_modes = requirements.get("excluded_modes", [])
    if request.resolved_mode in excluded_modes:
        failures.append("mode %s is unsupported" % request.resolved_mode)

    shapes = requirements.get("shapes", [])
    if shapes:
        current = {
            "width": request.output.width,
            "height": request.output.height,
            "frames": request.output.frames,
            "fps": request.output.fps,
        }
        matched = any(
            all(current.get(key) == int(value) for key, value in shape.items())
            for shape in shapes
        )
        if not matched:
            failures.append(
                "unsupported shape %dx%dx%d"
                % (request.output.width, request.output.height, request.output.frames)
            )

    option_requirements = requirements.get("engine_options", {})
    for namespace, keys in option_requirements.items():
        options = request.engine_options.get(namespace, {})
        for key in keys:
            if not isinstance(options, dict) or not options.get(key):
                failures.append("missing engine_options.%s.%s" % (namespace, key))

    device_profiles = requirements.get("device_profiles", [])
    if enforce_device_profile and device_profiles:
        report = DeviceProfileCatalog().match(device_profiles)
        if not report["matched"]:
            detail = "; ".join(
                "%s: %s" % (item["id"], ", ".join(item["failures"]))
                for item in report["profiles"]
            )
            failures.append("ANE device profile mismatch (%s)" % detail)

    return failures


def available_plans(
    model: ModelDescriptor,
    request: GenerationRequest,
) -> List[Tuple[ExecutionPlan, List[str]]]:
    return [
        (
            plan,
            _requirement_failures(
                model,
                plan,
                request,
                enforce_device_profile=request.policy.execution is ExecutionMode.AUTO,
            ),
        )
        for plan in model.plans
    ]


def choose_plan(model: ModelDescriptor, request: GenerationRequest) -> ExecutionPlan:
    requested_quality = _QUALITY_RANK[request.policy.approximation]
    candidates: List[ExecutionPlan] = []
    rejected = []

    for plan in model.plans:
        failures = _requirement_failures(
            model,
            plan,
            request,
            enforce_device_profile=request.policy.execution is ExecutionMode.AUTO,
        )
        if request.policy.execution is not ExecutionMode.AUTO:
            if plan.execution is not request.policy.execution:
                continue
        elif not plan.production:
            continue

        if _QUALITY_RANK[plan.quality] > requested_quality:
            failures.append(
                "quality %s exceeds requested %s"
                % (plan.quality.value, request.policy.approximation.value)
            )

        if plan.profile not in ("*", request.policy.profile.value):
            failures.append(
                "profile %s does not match %s"
                % (plan.profile, request.policy.profile.value)
            )

        if failures:
            rejected.append("%s: %s" % (plan.id, "; ".join(failures)))
        else:
            candidates.append(plan)

    if not candidates and request.policy.execution is ExecutionMode.AUTO:
        # A model may only declare a quality plan while callers ask for a
        # preview profile. A production wildcard/exact plan remains preferable
        # to failing, but no approximation level may be silently widened.
        for plan in model.plans:
            failures = _requirement_failures(model, plan, request)
            if not plan.production or _QUALITY_RANK[plan.quality] > requested_quality:
                continue
            if not failures:
                candidates.append(plan)

    if not candidates:
        detail = " | ".join(rejected) if rejected else "no matching plan declared"
        raise PlanUnavailableError(
            "no execution plan for %s (%s): %s"
            % (model.id, request.policy.execution.value, detail)
        )

    # Higher priority wins. For equal priority, prefer the least approximate
    # plan and then a hybrid plan only when the plugin explicitly ranked it.
    candidates.sort(
        key=lambda item: (
            item.priority,
            -_QUALITY_RANK[item.quality],
            item.execution is ExecutionMode.GPU_ANE,
        ),
        reverse=True,
    )
    return candidates[0]


def device_compatibility(plan: ExecutionPlan) -> dict:
    profile_ids = plan.requirements.get("device_profiles", [])
    if not profile_ids:
        return {"matched": True, "matched_profile": None, "profiles": []}
    return DeviceProfileCatalog().match(profile_ids)
