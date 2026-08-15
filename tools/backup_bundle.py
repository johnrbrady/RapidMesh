#!/usr/bin/env python
"""Bundle the whole repository to a second physical disk, and verify the result.

WP-0.2. A `git bundle --all` is a single file holding every ref and every
object needed to reconstruct them — `git clone <bundle>` gives back a working
repository. That makes it the cheapest honest backup for a repo whose value is
entirely in its history.

Run it after each authorised commit:

    python tools/backup_bundle.py D:\\Backups\\RapidMesh

There is no scheduling here on purpose. Nothing this script writes is ever
committed; only the script itself lives in the repository.

**What a bundle does not contain.** `--all` bundles refs. Uncommitted working
tree changes, staged-but-uncommitted work, stashes and untracked files are not
in it. The script says so loudly when the tree is dirty rather than letting a
green "verified" line imply cover it does not give.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Only files this script itself produces are ever eligible for deletion. The
# pattern is anchored and the date/sha groups are fixed-width, so a hand-placed
# file that merely starts with "rapidmesh-" is not matched and not removed.
BUNDLE_RE = re.compile(r"^rapidmesh-\d{8}-[0-9a-f]{7,40}\.bundle$")

REPO = Path(__file__).resolve().parent.parent


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command inside the repository and capture its output."""
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bundle this repository to a destination disk and verify the bundle.",
    )
    parser.add_argument("dest", type=Path, help="Destination directory, on a second physical disk.")
    parser.add_argument(
        "--keep",
        type=int,
        default=10,
        metavar="N",
        help="How many bundles to retain, newest first (default: 10).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Say what would happen; write and delete nothing.",
    )
    parser.add_argument(
        "--same-drive-ok",
        action="store_true",
        help="Permit a destination on the same drive as the repository. Defeats the "
        "point of an off-disk backup; require an explicit reason before using it.",
    )
    args = parser.parse_args()

    if args.keep < 1:
        die("--keep must be at least 1.")

    dest: Path = args.dest.expanduser().resolve()

    # A bundle inside the repository is lost with the repository, and would also
    # end up as untracked clutter in every `git status`.
    if dest == REPO or REPO in dest.parents:
        die(f"destination is inside the repository ({dest}). Back up to a separate disk.")

    if not args.same_drive_ok and dest.drive and dest.drive.upper() == REPO.drive.upper():
        die(
            f"destination drive {dest.drive} is the repository's own drive. The whole point "
            f"of WP-0.2 is a second physical disk. Pass --same-drive-ok to override."
        )

    if not dest.is_dir():
        die(f"destination does not exist or is not a directory: {dest}")

    # Identify what is about to be bundled.
    head = git("rev-parse", "--short", "HEAD").stdout.strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    dirty = git("status", "--porcelain").stdout.strip()

    stamp = datetime.now(UTC).strftime("%Y%m%d")
    out = dest / f"rapidmesh-{stamp}-{head}.bundle"

    print(f"repository   {REPO}")
    print(f"HEAD         {head} on {branch}")
    print(f"destination  {out}")

    if dirty:
        changed = len(dirty.splitlines())
        print(
            f"WARNING      working tree is NOT clean ({changed} entry(s)). A bundle holds "
            f"committed refs only,\n             so those changes will NOT be in this backup."
        )

    if args.dry_run:
        print("dry-run      no bundle written, nothing deleted")
        return 0

    # `git bundle create` refuses to overwrite nothing — same date and same HEAD
    # means identical content, so rewriting is harmless and keeps the run idempotent.
    created = git("bundle", "create", str(out), "--all", check=False)
    if created.returncode != 0:
        die(f"git bundle create failed:\n{created.stderr.strip()}")
    size_mb = out.stat().st_size / 1e6
    print(f"created      {out.name}  ({size_mb:,.1f} MB)")

    # Verify against the real repository — this is the step that makes the backup
    # a claim rather than a hope.
    verified = git("bundle", "verify", str(out), check=False)
    verify_text = (verified.stdout + verified.stderr).strip()
    print("verify       " + "\n             ".join(verify_text.splitlines()))
    if verified.returncode != 0:
        die(f"git bundle verify FAILED for {out}. The backup is not trustworthy.")

    # Retention. Only this script's own output is considered.
    bundles = sorted(
        (p for p in dest.iterdir() if p.is_file() and BUNDLE_RE.match(p.name)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    keep, drop = bundles[: args.keep], bundles[args.keep :]
    print(f"retained     {len(keep)} bundle(s), newest first (--keep {args.keep})")
    for p in drop:
        p.unlink()
        print(f"deleted      {p.name}")
    if not drop:
        print("deleted      nothing")

    print("OK           bundle written and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
