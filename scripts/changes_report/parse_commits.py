#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract the complete commit history of the current branch into commits.json.

One entry per commit with: full/short hash, author and committer identities,
date, subject, resolved tags, and the COMPLETE per-file numstat
(additions / deletions / binary) for every file touched by that commit.
The JSON is the single source of truth for every other stage of the pipeline
(charts, HTML map, CHANGES.md, wiki pages, release notes).
"""
import argparse
import json
import re
import subprocess
from collections import OrderedDict
from pathlib import Path

from common import classify, commit_type

LOG_FORMAT = ("%H|%h|%an|%ae|%cn|%ce|%ad|%s")


def git(repo, *args):
    res = subprocess.run(["git", "-C", str(repo), *args], check=True,
                         capture_output=True, text=True)
    return res.stdout


def collect_tags(repo):
    """Return {commit_full_sha: [ {name, date}, ... ]} for all refs/tags."""
    out = git(repo, "for-each-ref", "refs/tags",
              "--format=%(refname:short)|%(objectname)|%(*objectname)|%(creatordate:iso8601-strict)")
    tags = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        name, obj, star, cdate = parts[0], parts[1], parts[2], parts[3]
        target = star or obj          # annotated tags: *objectname is the commit
        tags.setdefault(target, []).append({"name": name, "date": cdate[:10]})
    for sha in tags:
        tags[sha].sort(key=lambda t: t["date"])
    return tags


def parse(repo):
    raw = git(repo, "log", "HEAD", "--date=format:%Y-%m-%d %H:%M",
              f"--pretty=format:COMMIT|{LOG_FORMAT}", "--numstat")
    tags = collect_tags(repo)
    commits, cur = [], None
    numstat_re = re.compile(r"^(\d+|\-)\t(\d+|\-)\t(.+)$")
    for line in raw.splitlines():
        if not line.strip():
            continue
        if line.startswith("COMMIT|"):
            fields = line.split("|", 8)
            full, short, an, ae, cn, ce, ad, subject = fields[1:9]
            cur = OrderedDict()
            cur["hash"] = full
            cur["short"] = short
            cur["author_name"] = an
            cur["author_email"] = ae
            cur["committer_name"] = cn
            cur["committer_email"] = ce
            cur["date"] = ad                       # "YYYY-MM-DD HH:MM"
            cur["subject"] = subject
            cur["tags"] = [t["name"] for t in tags.get(full, [])]
            cur["tag_dates"] = [t["date"] for t in tags.get(full, [])]
            cur["files"] = []
            commits.append(cur)
            continue
        m = numstat_re.match(line)
        if m and cur is not None:
            ins, dele, path = m.group(1), m.group(2), m.group(3)
            binary = ins == "-" or dele == "-"
            cur["files"].append({
                "path": path,
                "ins": 0 if binary else int(ins),
                "del": 0 if binary else int(dele),
                "binary": binary,
                "cat": classify(path),
            })
    # chronological order: oldest first
    commits.reverse()
    for i, c in enumerate(commits, 1):
        c["seq"] = i
        c["total_ins"] = sum(f["ins"] for f in c["files"])
        c["total_del"] = sum(f["del"] for f in c["files"])
        c["n_files"] = len(c["files"])
        cats = {}
        for f in c["files"]:
            cats[f["cat"]] = cats.get(f["cat"], 0) + 1
        c["cats"] = cats
        label, tcls, _bg, _brd = commit_type(c["subject"])
        c["type_label"] = label
        c["type_class"] = tcls
    return commits, tags


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", help="path to the git repository")
    ap.add_argument("--out", default="dist/changes-report/commits.json",
                    help="output JSON path")
    args = ap.parse_args()

    commits, tags = parse(args.repo)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(commits, f, ensure_ascii=False, indent=1)

    uniq = set()
    for c in commits:
        for fl in c["files"]:
            uniq.add(fl["path"])
    print(f"commits: {len(commits)}")
    print(f"file-change events: {sum(c['n_files'] for c in commits)}")
    print(f"unique files: {len(uniq)}")
    print(f"insertions: +{sum(c['total_ins'] for c in commits):,}")
    print(f"deletions: -{sum(c['total_del'] for c in commits):,}")
    print(f"tags: {sum(len(v) for v in tags.values())}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
