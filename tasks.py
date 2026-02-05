"""
Pyinvoke tasks.py file for automating releases.

Usage:
    inv release --dry-run    # Preview release steps
    inv release              # Full release
    inv publish              # Just publish to PyPI
    inv set-ver              # Just update version
    inv update-changelog     # Just update changelog

Requirements:
    pip install invoke twine build
    gh auth login  # For GitHub releases
"""

import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

from invoke import task
from invoke.exceptions import Exit

# Configuration
REPO_OWNER = "TRI-AMDD"
REPO_NAME = "beep"
MAIN_BRANCH = "master"
STABLE_BRANCH = "stable"
VERSION_FILE = Path("beep/__init__.py")
PYPROJECT_FILE = Path("pyproject.toml")
CHANGELOG_FILE = Path("CHANGES.md")

# CalVer format: YYYY.M.D.H (no zero-padding)
NEW_VER = datetime.datetime.now().strftime("%Y.%-m.%-d.%-H")


def get_current_version():
    """Read current version from __init__.py."""
    content = VERSION_FILE.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    raise Exit("Could not find version in beep/__init__.py")


def run_command(ctx, cmd, dry_run=False, warn=False, hide=False):
    """Run a command, optionally in dry-run mode."""
    if dry_run:
        print(f"  [DRY-RUN] Would run: {cmd}")
        return None
    return ctx.run(cmd, warn=warn, hide=hide)


def check_prerequisites(ctx):
    """Check that all required tools are available."""
    checks = [
        ("git --version", "git"),
        ("gh --version", "GitHub CLI (gh)"),
        ("python -m build --version", "build (pip install build)"),
        ("twine --version", "twine (pip install twine)"),
    ]

    missing = []
    for cmd, name in checks:
        result = ctx.run(cmd, warn=True, hide=True)
        if result.exited != 0:
            missing.append(name)

    if missing:
        print("Missing required tools:")
        for tool in missing:
            print(f"  - {tool}")
        raise Exit("Please install missing tools before releasing.")


def check_git_state(ctx, dry_run=False):
    """Verify git is in a clean state on the correct branch."""
    # Check we're on the main branch
    result = ctx.run("git branch --show-current", hide=True)
    current_branch = result.stdout.strip()
    if current_branch != MAIN_BRANCH:
        raise Exit(f"Must be on '{MAIN_BRANCH}' branch, currently on '{current_branch}'")

    # Check for uncommitted changes
    result = ctx.run("git status --porcelain", hide=True)
    if result.stdout.strip():
        print("Warning: Uncommitted changes detected:")
        print(result.stdout)
        if not dry_run:
            response = input("Continue anyway? [y/N] ")
            if response.lower() != 'y':
                raise Exit("Aborted due to uncommitted changes")

    # Check we're up to date with remote
    ctx.run("git fetch origin", hide=True)
    result = ctx.run(f"git rev-list HEAD...origin/{MAIN_BRANCH} --count", hide=True)
    if int(result.stdout.strip()) > 0:
        print(f"Warning: Local branch differs from origin/{MAIN_BRANCH}")
        if not dry_run:
            response = input("Continue anyway? [y/N] ")
            if response.lower() != 'y':
                raise Exit("Aborted - please sync with remote first")


@task
def lint(ctx, fix=False):
    """
    Run ruff linting.

    Args:
        fix: Auto-fix issues if possible
    """
    fix_flag = "--fix" if fix else ""
    ctx.run(f"ruff check beep {fix_flag}")
    print("Linting passed!")


@task
def test(ctx, quick=False):
    """
    Run pytest.

    Args:
        quick: Skip slow tests
    """
    marks = "-m 'not slow'" if quick else ""
    ctx.run(f"pytest beep {marks} --tb=short")
    print("Tests passed!")


@task
def set_ver(ctx, version=None, dry_run=False):
    """
    Update version in all relevant files.

    Args:
        version: Version string (default: CalVer based on current time)
        dry_run: Preview changes without writing
    """
    version = version or NEW_VER
    print(f"Setting version to: {version}")

    # Update beep/__init__.py
    content = VERSION_FILE.read_text()
    new_content = re.sub(
        r'__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{version}"',
        content
    )
    if dry_run:
        print(f"  [DRY-RUN] Would update {VERSION_FILE}")
    else:
        VERSION_FILE.write_text(new_content)
        print(f"  Updated {VERSION_FILE}")

    # Update pyproject.toml
    if PYPROJECT_FILE.exists():
        content = PYPROJECT_FILE.read_text()
        new_content = re.sub(
            r'version\s*=\s*"[^"]+"',
            f'version = "{version}"',
            content
        )
        if dry_run:
            print(f"  [DRY-RUN] Would update {PYPROJECT_FILE}")
        else:
            PYPROJECT_FILE.write_text(new_content)
            print(f"  Updated {PYPROJECT_FILE}")


@task
def update_changelog(ctx, version=None, dry_run=False):
    """
    Create preliminary changelog from git commits.

    Args:
        version: Version string for the release
        dry_run: Preview changes without writing
    """
    version = version or NEW_VER
    current_ver = get_current_version()

    print(f"Generating changelog for {current_ver} -> {version}")

    # Get commits since last version
    try:
        output = subprocess.check_output(
            ["git", "log", "--pretty=format:%s", f"v{current_ver}..HEAD"],
            stderr=subprocess.DEVNULL
        )
        commits = output.decode("utf-8").strip().split("\n")
    except subprocess.CalledProcessError:
        # No previous tag, get recent commits
        output = subprocess.check_output(
            ["git", "log", "--pretty=format:%s", "-20"]
        )
        commits = output.decode("utf-8").strip().split("\n")

    # Filter out merge commits and format
    lines = [f"* {c}" for c in commits if c and not c.startswith("Merge")]

    if not lines:
        print("  No commits found for changelog")
        return

    # Create new changelog entry
    header = f"\nv{version}\n" + "-" * (len(version) + 1) + "\n"
    new_entry = header + "\n".join(lines[:20]) + "\n"  # Limit to 20 entries

    print("  New changelog entry:")
    for line in new_entry.split("\n")[:10]:
        print(f"    {line}")
    if len(lines) > 10:
        print(f"    ... and {len(lines) - 10} more")

    if dry_run:
        print(f"  [DRY-RUN] Would prepend to {CHANGELOG_FILE}")
        return

    # Prepend to changelog
    if CHANGELOG_FILE.exists():
        existing = CHANGELOG_FILE.read_text()
        CHANGELOG_FILE.write_text(new_entry + "\n" + existing)
    else:
        CHANGELOG_FILE.write_text(new_entry)

    print(f"  Updated {CHANGELOG_FILE}")

    # Open for editing (cross-platform)
    if sys.platform == "darwin":
        ctx.run(f"open {CHANGELOG_FILE}", warn=True)
    elif sys.platform == "linux":
        ctx.run(f"xdg-open {CHANGELOG_FILE}", warn=True)
    else:
        print(f"  Please review {CHANGELOG_FILE}")


@task
def build(ctx, dry_run=False):
    """
    Build distribution packages using modern PEP 517 build.

    Args:
        dry_run: Preview without building
    """
    print("Building distribution packages...")

    # Clean old builds
    run_command(ctx, "rm -rf dist/ build/ *.egg-info", dry_run=dry_run, warn=True)

    # Build using PEP 517
    run_command(ctx, "python -m build", dry_run=dry_run)

    if not dry_run:
        print("  Built packages:")
        for f in Path("dist").glob("*"):
            print(f"    {f.name}")


@task
def publish(ctx, dry_run=False, test_pypi=False):
    """
    Upload release to PyPI using twine.

    Args:
        dry_run: Preview without uploading
        test_pypi: Upload to TestPyPI instead
    """
    repo_flag = "--repository testpypi" if test_pypi else ""
    target = "TestPyPI" if test_pypi else "PyPI"

    print(f"Publishing to {target}...")

    if not Path("dist").exists() or not list(Path("dist").glob("*")):
        print("  No dist/ directory found, building first...")
        build(ctx, dry_run=dry_run)

    cmd = f"twine upload {repo_flag} dist/*"
    run_command(ctx, cmd, dry_run=dry_run)

    if not dry_run:
        print(f"  Successfully uploaded to {target}")


@task
def tag_release(ctx, version=None, dry_run=False):
    """
    Create git tag and push.

    Args:
        version: Version to tag
        dry_run: Preview without tagging
    """
    version = version or NEW_VER
    tag = f"v{version}"

    print(f"Creating git tag: {tag}")

    # Commit version changes
    run_command(
        ctx,
        f'git commit -a -m "Release {tag}"',
        dry_run=dry_run,
        warn=True
    )

    # Create annotated tag
    run_command(
        ctx,
        f'git tag -a {tag} -m "Release {tag}"',
        dry_run=dry_run
    )

    # Push tag
    run_command(ctx, "git push origin --tags", dry_run=dry_run)
    run_command(ctx, "git push origin", dry_run=dry_run)


@task
def merge_stable(ctx, dry_run=False):
    """
    Merge main branch into stable branch.

    Args:
        dry_run: Preview without merging
    """
    print(f"Merging {MAIN_BRANCH} into {STABLE_BRANCH}...")

    run_command(ctx, f"git checkout {STABLE_BRANCH}", dry_run=dry_run)
    run_command(ctx, "git pull origin", dry_run=dry_run, warn=True)
    run_command(ctx, f"git merge {MAIN_BRANCH}", dry_run=dry_run)
    run_command(ctx, "git push origin", dry_run=dry_run)
    run_command(ctx, f"git checkout {MAIN_BRANCH}", dry_run=dry_run)


@task
def release_github(ctx, version=None, dry_run=False):
    """
    Create GitHub release using gh CLI.

    Args:
        version: Version for the release
        dry_run: Preview without creating release
    """
    version = version or NEW_VER
    tag = f"v{version}"

    print(f"Creating GitHub release: {tag}")

    # Get release notes from changelog
    notes = ""
    if CHANGELOG_FILE.exists():
        content = CHANGELOG_FILE.read_text()
        # Extract notes for this version
        pattern = rf"v{re.escape(version)}\n-+\n(.*?)(?=\nv\d|$)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            notes = match.group(1).strip()

    if not notes:
        notes = f"Release {tag}"

    # Create release using gh CLI
    notes_escaped = notes.replace('"', '\\"')
    cmd = f'gh release create {tag} --repo {REPO_OWNER}/{REPO_NAME} --title "{tag}" --notes "{notes_escaped}"'

    if dry_run:
        print(f"  [DRY-RUN] Would run: gh release create {tag}")
        print(f"  Release notes preview:\n{notes[:500]}")
    else:
        result = ctx.run(cmd, warn=True)
        if result.exited == 0:
            print(f"  Created release: https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/tag/{tag}")
        else:
            print("  Warning: GitHub release creation failed")


@task
def release(ctx, dry_run=False, no_test=False, no_lint=False, version=None):
    """
    Run full release sequence.

    Args:
        dry_run: Preview all steps without executing
        no_test: Skip running tests
        no_lint: Skip linting
        version: Specific version (default: CalVer)
    """
    version = version or NEW_VER

    print("=" * 60)
    print(f"BEEP Release Process - v{version}")
    print("=" * 60)

    if dry_run:
        print("[DRY-RUN MODE - No changes will be made]\n")

    # Pre-flight checks
    print("\n[1/9] Checking prerequisites...")
    if not dry_run:
        check_prerequisites(ctx)
    print("  OK")

    print("\n[2/9] Checking git state...")
    check_git_state(ctx, dry_run=dry_run)
    print("  OK")

    # Quality checks
    if not no_lint:
        print("\n[3/9] Running linter...")
        if not dry_run:
            lint(ctx)
        else:
            print("  [DRY-RUN] Would run: ruff check beep")
    else:
        print("\n[3/9] Skipping linter (--no-lint)")

    if not no_test:
        print("\n[4/9] Running tests...")
        if not dry_run:
            test(ctx)
        else:
            print("  [DRY-RUN] Would run: pytest beep")
    else:
        print("\n[4/9] Skipping tests (--no-test)")

    # Version and changelog
    print("\n[5/9] Updating version...")
    set_ver(ctx, version=version, dry_run=dry_run)

    print("\n[6/9] Updating changelog...")
    update_changelog(ctx, version=version, dry_run=dry_run)

    # Build and publish
    print("\n[7/9] Building packages...")
    build(ctx, dry_run=dry_run)

    print("\n[8/9] Publishing to PyPI...")
    publish(ctx, dry_run=dry_run)

    # Git operations
    print("\n[9/9] Git operations...")
    tag_release(ctx, version=version, dry_run=dry_run)
    merge_stable(ctx, dry_run=dry_run)
    release_github(ctx, version=version, dry_run=dry_run)

    print("\n" + "=" * 60)
    if dry_run:
        print("DRY-RUN COMPLETE - No changes were made")
        print("Run without --dry-run to execute the release")
    else:
        print(f"RELEASE COMPLETE - v{version}")
        print(f"  PyPI: https://pypi.org/project/beep/{version}/")
        print(f"  GitHub: https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/tag/v{version}")
    print("=" * 60)
