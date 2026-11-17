#!/usr/bin/env python3
"""Validate all .github YAML files: parse + structural sanity for workflows."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path("/home/z/my-project/VoiceCall")
WF = ROOT / ".github" / "workflows"

problems: list[str] = []


def norm_on(data: dict) -> dict:
    """PyYAML (YAML 1.1) parses the bare key `on:` as boolean True."""
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


workflows = sorted(WF.glob("*.yml"))
other_yamls = sorted((ROOT / ".github").glob("*.yml")) + [ROOT / "mkdocs.yml"]

for f in workflows:
    try:
        data = norm_on(yaml.safe_load(f.read_text(encoding="utf-8")))
    except yaml.YAMLError as exc:
        problems.append(f"{f.name}: YAML parse error: {exc}")
        continue
    if "jobs" not in data:
        problems.append(f"{f.name}: missing 'jobs'")
        continue
    if "on" not in data:
        problems.append(f"{f.name}: missing 'on' trigger block")
        continue
    for jname, job in data["jobs"].items():
        if not isinstance(job, dict):
            problems.append(f"{f.name}: job '{jname}' is not a mapping")
            continue
        has_steps = "steps" in job
        uses_call = "uses" in job
        if not (has_steps or uses_call):
            problems.append(f"{f.name}: job '{jname}' has neither steps nor uses")
        if has_steps:
            for i, step in enumerate(job["steps"]):
                if not isinstance(step, dict) or not ("uses" in step or "run" in step):
                    problems.append(
                        f"{f.name}: job '{jname}' step #{i} has no uses/run")
        if "runs-on" not in job and not uses_call:
            problems.append(f"{f.name}: job '{jname}' missing runs-on")
    names = set(data["jobs"])
    print(f"OK: {f.name} — {len(names)} job(s): {', '.join(sorted(names))[:90]}")

for f in other_yamls:
    try:
        yaml.safe_load(f.read_text(encoding="utf-8"))
        print(f"OK: {f.relative_to(ROOT)} — parses")
    except yaml.YAMLError as exc:
        problems.append(f"{f}: YAML parse error: {exc}")

for t in sorted((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml")):
    data = yaml.safe_load(t.read_text(encoding="utf-8"))
    if t.name == "config.yml":
        continue
    for key in ("name", "body"):
        if key not in data:
            problems.append(f"{t.name}: issue form missing '{key}'")
    labels = data.get("labels") or []
    print(f"OK: issue form {t.name} — {len(data.get('body', []))} field(s), "
          f"labels={labels}")

print(f"\nworkflow files: {len(workflows)} | other YAML: {len(other_yamls)}")
if problems:
    print("PROBLEMS:")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("ALL YAML FILES VALID")
