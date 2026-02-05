# BEEP Administration Guide

This guide is for BEEP maintainers and covers the release process.

## Prerequisites

### Required Tools

```bash
# Python 3.10+ environment
conda create -n beep python=3.11
conda activate beep

# Install BEEP in development mode
git clone https://github.com/TRI-AMDD/beep
cd beep
pip install -e ".[dev]"

# GitHub CLI for releases
# macOS: brew install gh
# Ubuntu: sudo apt install gh
gh auth login
```

### PyPI Authentication

BEEP uses PyPI trusted publishing (recommended) or API tokens:

**Option 1: Trusted Publishing (Recommended)**
- Configure on PyPI: https://pypi.org/manage/project/beep/settings/publishing/
- No local credentials needed - GitHub Actions handles authentication

**Option 2: API Token**
```bash
# Create token at: https://pypi.org/manage/account/token/
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-xxxxx  # Your API token
```

## Release Process

### Quick Release (Automated)

The easiest way to release is using the invoke tasks with dry-run first:

```bash
# Preview what will happen (no changes made)
inv release --dry-run

# Full release (runs tests, linting, publishes to PyPI, creates GitHub release)
inv release

# Skip tests for faster release (use with caution)
inv release --no-test
```

### Manual Release Steps

If you prefer manual control:

```bash
# 1. Update version
inv set-ver

# 2. Update and review changelog
inv update-changelog
# Edit CHANGES.md as needed

# 3. Run quality checks
inv lint
inv test

# 4. Build and publish
inv build
inv publish

# 5. Create git tag and GitHub release
inv tag-release
inv release-github
```

### Available Invoke Tasks

```bash
inv --list                    # Show all available tasks

# Individual tasks
inv lint                      # Run ruff linting
inv lint --fix                # Auto-fix lint issues
inv test                      # Run pytest
inv test --quick              # Skip slow tests
inv build                     # Build packages (PEP 517)
inv publish                   # Upload to PyPI
inv publish --test-pypi       # Upload to TestPyPI first
inv set-ver                   # Update version (CalVer)
inv set-ver --version 1.0.0   # Set specific version
inv update-changelog          # Generate changelog from commits
inv tag-release               # Create and push git tag
inv merge-stable              # Merge master into stable branch
inv release-github            # Create GitHub release
```

### CI/CD Automated Releases

Releases can also be triggered automatically via GitHub Actions:

1. Update version and changelog locally
2. Commit and push changes
3. Create and push a version tag:

```bash
git tag -a v2025.2.5 -m "Release v2025.2.5"
git push origin --tags
```

The GitHub Actions workflow will automatically:
- Build the package
- Run tests on Python 3.10, 3.11, 3.12
- Publish to PyPI
- Create a GitHub release

## Version Scheme

BEEP uses Calendar Versioning (CalVer): `YYYY.M.D.H`

- `YYYY` - Full year
- `M` - Month (no zero padding)
- `D` - Day (no zero padding)
- `H` - Hour (no zero padding)

Example: `2025.2.5.14` = February 5, 2025 at 2 PM

## Post-Release Checklist

After releasing:

1. Verify PyPI: https://pypi.org/project/beep/
2. Verify GitHub release: https://github.com/TRI-AMDD/beep/releases
3. Test installation: `pip install beep==VERSION`
4. Update documentation if needed: `mkdocs build` and push to `docs/`

## Troubleshooting

### Release Failed Mid-way

If a release fails partway through:

```bash
# Check current state
git status
git log --oneline -5

# If tag was created but not pushed:
git tag -d vX.Y.Z.H  # Delete local tag
# Then retry release

# If published to PyPI but GitHub release failed:
inv release-github --version X.Y.Z.H
```

### Reverting a Release

PyPI releases cannot be deleted, only yanked:

```bash
# Yank a release (marks as not recommended)
pip install twine
twine yank beep==X.Y.Z.H

# Delete GitHub release
gh release delete vX.Y.Z.H --yes
git push --delete origin vX.Y.Z.H
```
