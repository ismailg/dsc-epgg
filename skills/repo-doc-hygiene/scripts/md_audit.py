#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "iwr-results",
    "hetzner-results",
    "skills",
}

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


@dataclass
class DocInfo:
    path: Path
    roles: list[str]
    strong_roles: list[str]
    claims: list[str]
    broken_links: list[str]
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit repository Markdown files for broken links, role conflicts, and ownership drift."
    )
    parser.add_argument("root", nargs="?", default=".", help="Repository root to scan.")
    parser.add_argument(
        "--include-outputs",
        action="store_true",
        help="Include outputs/ directories in the scan.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of plain text.",
    )
    return parser.parse_args()


def iter_markdown_files(root: Path, include_outputs: bool) -> Iterable[Path]:
    for path in root.rglob("*.md"):
        parts = set(path.parts)
        if parts & DEFAULT_EXCLUDE_DIRS:
            continue
        if not include_outputs and "outputs" in parts:
            continue
        yield path


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def score_roles(path: Path, text: str) -> dict[str, int]:
    name = path.name.lower()
    upper_name = path.name.upper()
    lower = text.lower()
    scores: dict[str, int] = defaultdict(int)

    if name == "readme.md":
        scores["entry_point"] += 3
    if "read first" in lower or "start with" in lower:
        scores["entry_point"] += 1

    if "data_map" in name or "registry" in name:
        scores["artifact_registry"] += 3
    if "canonical local roots" in lower or "canonical remote roots" in lower:
        scores["artifact_registry"] += 2
    if "fetch convention" in lower or "artifact registry" in lower:
        scores["artifact_registry"] += 1

    if "next_steps" in name or "status" in name or "handoff" in name:
        scores["handoff_status"] += 3
    if "current gating run" in lower or "short handoff" in lower:
        scores["handoff_status"] += 2
    if "current status" in lower or "live status" in lower:
        scores["handoff_status"] += 1

    if "todo" in name or "checklist" in name:
        scores["active_checklist"] += 3
    if "- [ ]" in text:
        scores["active_checklist"] += 2

    if upper_name == "AGENTS.MD" or "must not" in lower or "parity contract" in lower:
        scores["rules_contract"] += 3

    if "overview" in name or "didactic" in name or "summary" in name:
        scores["interpretation"] += 2
    if "what this covers" in lower or "interpretation" in lower:
        scores["interpretation"] += 1

    if (
        "historical artifact" in lower
        or "historical implementation-plan" in lower
        or "do not use this file for current status" in lower
        or "this is a historical" in lower
    ):
        scores["historical"] += 3

    if "tracker" in name:
        scores["tracker"] += 3

    return scores


def detect_claims(text: str) -> list[str]:
    lower = text.lower()
    claims = []
    regex_patterns = {
        "source_of_truth": [
            r"this file is the canonical artifact map",
            r"this file is the source of truth",
        ],
        "active_checklist": [
            r"this checklist is the current paper-facing todo",
            r"this file is the active checklist for manuscript-facing work",
        ],
        "current_status": [
            r"this file is the short handoff note",
            r"^## current gating run\b",
            r"this file is the live status ledger",
        ],
        "read_first": [
            r"^## read first\b",
            r"\bstart with:\b",
        ],
    }
    for claim, patterns in regex_patterns.items():
        if any(re.search(pattern, lower, flags=re.MULTILINE) for pattern in patterns):
            claims.append(claim)
    return claims


def resolve_link(path: Path, target: str) -> bool:
    target = target.strip()
    if not target or target.startswith("#"):
        return True
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", target):
        return True
    if target.startswith("mailto:"):
        return True
    clean = target.split("#", 1)[0]
    if not clean:
        return True
    candidate = Path(clean)
    if candidate.is_absolute():
        return candidate.exists()
    return (path.parent / candidate).exists()


def find_broken_links(path: Path, text: str) -> list[str]:
    broken = []
    for match in LINK_RE.finditer(text):
        target = match.group(1)
        if not resolve_link(path, target):
            broken.append(target)
    return broken


def classify_doc(path: Path, text: str) -> DocInfo:
    scores = score_roles(path, text)
    roles = sorted([role for role, score in scores.items() if score > 0])
    strong_roles = sorted([role for role, score in scores.items() if score >= 3])
    claims = detect_claims(text)
    broken_links = find_broken_links(path, text)
    warnings: list[str] = []

    singleton_roles = {"entry_point", "artifact_registry", "handoff_status", "active_checklist"}
    if len([r for r in strong_roles if r in singleton_roles]) > 1:
        warnings.append("multiple_singleton_roles")
    if "tracker" in strong_roles and "historical" not in strong_roles:
        pending_count = text.lower().count("status: pending")
        if pending_count >= 3:
            warnings.append("tracker_may_be_stale")
    if broken_links:
        warnings.append("broken_links")

    return DocInfo(
        path=path,
        roles=roles,
        strong_roles=strong_roles,
        claims=claims,
        broken_links=broken_links,
        warnings=warnings,
    )


def build_report(root: Path, include_outputs: bool) -> dict:
    docs = [classify_doc(path, read_text(path)) for path in sorted(iter_markdown_files(root, include_outputs))]
    singleton_roles = {"entry_point", "artifact_registry", "handoff_status", "active_checklist"}
    role_owners: dict[str, list[str]] = defaultdict(list)
    claim_owners: dict[str, list[str]] = defaultdict(list)

    for doc in docs:
        for role in doc.strong_roles:
            if role in singleton_roles and "historical" not in doc.strong_roles:
                role_owners[role].append(str(doc.path))
        for claim in doc.claims:
            if "historical" not in doc.strong_roles:
                claim_owners[claim].append(str(doc.path))

    duplicate_roles = {role: owners for role, owners in role_owners.items() if len(owners) > 1}
    duplicate_claims = {claim: owners for claim, owners in claim_owners.items() if len(owners) > 1}

    warning_counts = Counter()
    for doc in docs:
        warning_counts.update(doc.warnings)

    return {
        "root": str(root.resolve()),
        "doc_count": len(docs),
        "warnings": dict(warning_counts),
        "duplicate_roles": duplicate_roles,
        "duplicate_claims": duplicate_claims,
        "docs": [
            {
                "path": str(doc.path),
                "roles": doc.roles,
                "strong_roles": doc.strong_roles,
                "claims": doc.claims,
                "warnings": doc.warnings,
                "broken_links": doc.broken_links,
            }
            for doc in docs
        ],
    }


def print_text_report(report: dict) -> None:
    print(f"root: {report['root']}")
    print(f"markdown_files: {report['doc_count']}")

    if report["warnings"]:
        print("\nwarning_counts:")
        for key, value in sorted(report["warnings"].items()):
            print(f"- {key}: {value}")

    if report["duplicate_roles"]:
        print("\nduplicate_singleton_roles:")
        for role, owners in sorted(report["duplicate_roles"].items()):
            print(f"- {role}:")
            for owner in owners:
                print(f"  - {owner}")

    if report["duplicate_claims"]:
        print("\nduplicate_claims:")
        for claim, owners in sorted(report["duplicate_claims"].items()):
            print(f"- {claim}:")
            for owner in owners:
                print(f"  - {owner}")

    print("\ndocs:")
    for doc in report["docs"]:
        roles = ",".join(doc["strong_roles"] or doc["roles"] or ["unclassified"])
        warnings = ",".join(doc["warnings"]) if doc["warnings"] else "-"
        print(f"- {doc['path']} :: roles={roles} :: warnings={warnings}")
        if doc["broken_links"]:
            for link in doc["broken_links"]:
                print(f"    broken_link: {link}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    report = build_report(root, include_outputs=args.include_outputs)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)


if __name__ == "__main__":
    main()
