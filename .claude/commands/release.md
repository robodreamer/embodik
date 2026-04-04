# EmbodiK Release Workflow

Execute the full release workflow for embodik: test, build, version bump, changelog, commit, push, and publish to PyPI.

## Steps

1. **Validate**: Run `pixi run test` (full test suite). If changes are tightly scoped, a focused test command is acceptable.

2. **Build**: Run `pixi run build` as a sanity check.

3. **Version bump**: Determine bump type from the changes:
   - **patch** (`0.x.Y`): bug fixes, docs, refactors with no new API surface
   - **minor** (`0.X.0`): new public API, new features, new examples
   - **major** (`X.0.0`): breaking changes to existing API
   
   Run `pixi run version --bump <patch|minor|major>` (updates `pyproject.toml`).
   Sync `pixi.toml` version to match `pyproject.toml` if not auto-synced.

4. **Update CHANGELOG.md**: Add a new section at the top:
   ```
   ## [<version>] - <YYYY-MM-DD>
   ### Added
   - ...
   ### Changed
   - ...
   ### Fixed
   - ...
   ```

5. **Commit**: Stage changed files and commit with message: `release: <version> <brief summary>`

6. **Push**: `git push`

7. **Publish to PyPI** (unless user says "push only" or "skip publish"):
   - `pixi run build-dist`
   - `pixi run upload-pypi`
   - Verify: `curl -sSf https://pypi.org/pypi/embodik/json | python -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"` should match.

## Guardrails

- If `pixi run version --bump patch` only updates `pyproject.toml`, manually sync `pixi.toml`.
- Include PyPI publish for any version-bumped release unless user explicitly opts out.
- Follow repository git safety (no force push, no destructive commands).

## Arguments

$ARGUMENTS - Optional: bump type (patch/minor/major) and/or "push only" to skip publish
