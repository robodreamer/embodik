# Project context for Claude

## Environment rule (always apply)

- Always run project scripts and tooling through the Pixi environment.
- Use `pixi run <task>` for commands from `pixi.toml`.
- For ad-hoc executables, use `pixi run <command>` so execution still happens in
  the Pixi-managed environment.

## Push workflow (any code change)

- Validate changes before commit:
  - `pixi run test`
  - or a focused test command when scoped changes are made.
- Build/install sanity check:
  - `pixi run build`
- Stage and commit:
  - `git add <files>`
  - `git commit -m "<message>"`
- Push branch:
  - `git push`

## Patch release workflow (bug fix release)

1. Bump version with pixi task:
   - `pixi run version --bump patch`
2. Update `CHANGELOG.md` with release date + fixed items.
3. Keep metadata in sync:
   - update `pixi.toml` project version to match `pyproject.toml`.
4. Run verification:
   - `pixi run test` (or targeted tests for the fix)
   - `pixi run build-dist`
5. Publish:
   - `pixi run upload-pypi`
6. Commit release files:
   - `CHANGELOG.md`
   - `pyproject.toml`
   - `pixi.toml`
   - code/tests touched by the fix
7. Push:
   - `git push`

## Notes

- Prefer small, focused commits.
- Do not publish (`upload-pypi`) for non-release commits.
- If `pixi run version --bump patch` only updates `pyproject.toml`, manually sync
  `pixi.toml`.
