# Contributing

We use short-lived feature branches, rebase to stay current with `main`, and
squash-merge PRs to keep history clean.

## Basics
- No direct pushes to `main`. Open a PR and use "Squash and merge".
- One task/feature per branch. Keep branches short-lived.
- Keep your branch up to date with `main` via rebase (not merge).

## Setup (once)
- Ensure your local `main` tracks `origin/main`.
- Recommended config:
  ```bash
  git config pull.rebase true
  git config rebase.autoStash true
  git config --global rerere.enabled true
  ```
  - pull.rebase=true: `git pull` uses rebase by default.
  - rebase.autoStash=true: stashes unstaged changes during rebase/pull.
  - rerere: remembers conflict resolutions.

## Start work
```bash
git switch main
git pull --rebase
git switch -c feature/short-description
```

## Keep your branch fresh while working
```bash
git fetch origin
git rebase origin/main
# If conflicts:
#   fix files, then:
#   git add <resolved-files>
#   git rebase --continue
git push --force-with-lease
```

## Before opening or merging a PR
```bash
git fetch origin
git rebase origin/main
git push --force-with-lease
# Open PR → choose "Squash and merge" when approved
```

## Dependency changes (uv + pyproject.toml)
- Add runtime dep: `uv add <pkg>`
- Add dev dep: `uv add --dev <pkg>`
- Update lock: `uv lock`
- Sync env: `uv sync`
- If multiple PRs touch deps, merge one first; the other rebases on `main` and runs `uv lock` again.

## Handling conflicts
- During rebase, fix conflicts, `git add`, then `git rebase --continue`.
- To abort a rebase: `git rebase --abort`.
- If two PRs touch the same files, coordinate order in chat; second PR rebases after the first merges.

## Do/Don’t
- Do open small PRs early (drafts welcome).
- Do use `git push --force-with-lease` on your own feature branches after rebases.
- Don’t force-push `main` or other shared branches.
- Don’t hand-edit lockfiles—let tools regenerate them.

## Optional niceties
- Interactive cleanup before PR: `git rebase -i origin/main`
- GitHub settings (recommended):
  - Require PRs for `main`
  - Require branches to be up to date before merging
  - Enable auto-merge after checks pass
