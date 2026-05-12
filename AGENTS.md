# AGENTS.md

Conventions for AI agents and contributors working on `egt`. Supplements `README.md` (user-facing) and `CHANGELOG.md` (release history).

## Publishing a release

End-to-end procedure. All steps run on `main`; no release branch or release-prep PR.

1. **Make sure `[Unreleased]` has the changes you're about to ship.** Every substantive PR is expected to append its own bullets to the `[Unreleased]` section of `CHANGELOG.md` under the appropriate Keep-a-Changelog heading (`### Added`, `### Changed`, etc.) as part of the same PR. Do NOT open a follow-up PR for CHANGELOG bookkeeping.
2. **Confirm `main` is green** for the commit you're about to bump from:
   ```bash
   gh run list --branch main --limit 1
   ```
   Expect `completed success`. Wait for any in-progress run.
3. **Run the bump on `main`.** Choose patch / minor / major per the [Semver](#semantic-versioning) section below:
   ```bash
   git checkout main && git pull --ff-only
   bump-my-version bump <patch|minor|major>
   ```
   This atomically:
   - bumps `version` in `pyproject.toml`,
   - bumps `__version__` in `src/egt/__init__.py`,
   - rotates `CHANGELOG.md` — the `## [Unreleased]` heading stays in place (now empty) and a new `## [X.Y.Z] - YYYY-MM-DD` section is inserted below it, carrying all the bullets that were under Unreleased,
   - commits with message `Bump version: <old> → <new>`,
   - creates tag `vX.Y.Z` against that commit.
4. **Push the bump commit + tag:**
   ```bash
   git push --follow-tags
   ```
5. **Wait for the push-to-main CI run** on the new commit to go `completed success`. This is the last safety net before publishing.
6. **Cut the GitHub Release:**
   ```bash
   gh release create vX.Y.Z --title "vX.Y.Z" --notes-file <body.md>
   ```
   `body.md` is the `[X.Y.Z]` section content from `CHANGELOG.md`, optionally tightened with a `## Highlights` TL;DR up top (3–5 bullets).
7. **`publish.yml` fires** on `release: published`, runs a sanity test matrix, builds the wheel + sdist, and uploads to PyPI via OIDC trusted publishing.

Three local commands; one CLI release call; the rest is automated.

## CHANGELOG conventions

Format: [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

Per-release sections (use only the ones that have entries):
- `### Added` — new features, subcommands, flags, modules.
- `### Changed` — behavior change to existing features. **Breaking changes go here in bold** with explicit migration instructions (e.g. "Update Snakefiles invoking the old subcommand name.").
- `### Deprecated` — features still working but slated for removal.
- `### Removed` — features deleted in this release.
- `### Fixed` — bug fixes.
- `### Security` — security-relevant fixes.
- `### Internal` — refactors, CI / tooling changes, no user impact.

Prose style:
- Plain prose bullets. No PR-number references (cluttered; the GitHub Release links back to the merge commits anyway).
- Lead with the user-visible identifier (`egt foo --bar`, file name, module name) so a reader skimming the CHANGELOG can find their concern instantly.
- For each substantive change in a PR, append the bullet(s) to `[Unreleased]` as part of that PR.

## Semantic versioning

This project follows [SemVer 2.0.0](https://semver.org/spec/v2.0.0.html).

- **Patch (`0.2.X`):** bug fixes, doc-only, internal refactors, dependency bumps that don't change behavior.
- **Minor (`0.X.0`):** new subcommands, new CLI flags, new optional kwargs. Additive only.
- **Major (`X.0.0`):** breaking changes — CLI rename, removed flag, schema change, removed subcommand. Anything that requires a caller-side edit.

Pre-1.0 caveat (current state): a breaking change may ride a minor bump rather than forcing a 1.0. Always flag it in **bold** under `### Changed` in the CHANGELOG with migration guidance regardless of the bump category.

## Version-bump infrastructure

The `[tool.bumpversion]` block in `pyproject.toml` is the source of truth. It keeps these in lockstep:
- `pyproject.toml`'s `version = "..."`
- `src/egt/__init__.py`'s `__version__ = "..."`
- `CHANGELOG.md`'s `## [Unreleased]` → versioned section rotation

**Never edit version strings or rotate the CHANGELOG by hand.** Always run `bump-my-version bump`. Manual edits desync the files and the next bump refuses to run until the drift is reconciled.

## publish.yml architecture

Lives at `.github/workflows/publish.yml`. Triggers on `release: published`.

Two jobs:

1. **`test`** — 4-cell matrix (Ubuntu 3.10/3.11/3.12 + macOS 3.12) as a sanity re-check on the tagged commit.
2. **`publish`** — depends on `test`:
   - `python -m build` produces wheel + sdist.
   - `twine check dist/*` validates metadata.
   - `pypa/gh-action-pypi-publish@release/v1` uploads via OIDC trusted publishing.

The `publish` job runs in the **`pypi` GitHub environment**, which is restricted to tag refs matching `v*`. The environment is the choke point preventing non-release pushes (or arbitrary workflows) from reaching the PyPI write surface. The publish step also declares `permissions: id-token: write` so the action can exchange a short-lived OIDC token for PyPI auth — no long-lived API token.

The 4-cell publish-time matrix is intentionally smaller than the 7-cell PR / push-to-main CI matrix. Main CI is the primary correctness gate; publish-time tests are a final sanity re-check, not the gate.

## PyPI trusted publishing — one-time setup

If trusted publishing is ever lost (rebuilt repo, renamed project, etc.):

### PyPI side

https://pypi.org/manage/project/egt/settings/publishing/ → "Add a new trusted publisher" → GitHub:
- **Owner:** `conchoecia`
- **Repository name:** `egt`
- **Workflow filename:** `publish.yml`
- **Environment name:** `pypi`

### GitHub side

https://github.com/conchoecia/egt/settings/environments → **New environment** → name it `pypi`.

In the environment configuration:
- "Deployment branches and tags" → switch from "All branches" to "Selected branches and tags".
- **Add a rule of type `Tag` with pattern `v*`.** Branch rules alone do NOT cover tag refs — `release: published` triggers run on tags, so the env must explicitly allow tag refs.
- Optionally: add yourself (or another maintainer) as a required reviewer for extra paranoia. Every publish then requires a manual approval click before the PyPI upload happens.

### Verification

After both halves are configured, the next release should publish without manual intervention. If the publish job fails with `Tag "vX.Y.Z" is not allowed to deploy to pypi due to environment protection rules`, the GitHub env's tag rule is missing or its pattern doesn't match the tag.

## Common failure modes

- **`Tag "vX.Y.Z" is not allowed to deploy to pypi due to environment protection rules.`**
  GitHub env's Deployment branches and tags rule is missing, has the wrong type (Branch instead of Tag), or its pattern doesn't include the tag (e.g. `v0.*` won't match `v1.0.0`). Fix in https://github.com/conchoecia/egt/settings/environments/pypi.

- **`Trusted publishing exchange failure: OpenID Connect token retrieval failed ... missing or insufficient OIDC token permissions, the ACTIONS_ID_TOKEN_REQUEST_TOKEN environment variable was unset.`**
  The `publish` job is missing `permissions: id-token: write`. Should already be present in `publish.yml`; if regressed, restore it at the job level.

- **`bump-my-version` refuses to run** with `Git working directory is not clean` or `Specified version (X.Y.Z) does not match last tagged version`.
  Clean the working tree first (commit/stash/discard). The "last tagged" warning is informational and benign as long as `current_version` in `pyproject.toml` matches the version actually in the file content.

- **publish step shows red, but `pip install egt==X.Y.Z` works.**
  The wheel uploaded fine; a post-upload step (sigstore attestation, OIDC verification of attestations) failed. Check PyPI for the version before re-running. Often the only missing artifact is the attestation, not the wheel itself.

## Anti-patterns

- **Don't open a separate PR for CHANGELOG rotation or version bump.** CHANGELOG entries belong in the substantive PR that introduces the change. The bump runs on main as part of the release flow, not on a branch.
- **Don't `twine upload` manually.** Always cut a GitHub Release; `publish.yml` handles the upload. Manual uploads bypass the trusted-publishing audit trail.
- **Don't tag before the PR has merged.** Tag a release only from a commit that's on `main` and has green CI.
- **Don't edit `pyproject.toml`'s `version` or `src/egt/__init__.py`'s `__version__` by hand.** Use `bump-my-version`. Manual edits create the drift that breaks subsequent bumps.
- **Don't commit straight to `main`** for code or test changes. Code changes always go through a branch + PR + green CI before merge. Direct-to-`main` commits are reserved for trivial docs (this file, typo fixes), `bump-my-version`'s output, and the coverage-badge auto-commit bot.
