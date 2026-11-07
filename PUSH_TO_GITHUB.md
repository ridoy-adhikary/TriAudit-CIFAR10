# Push TriAudit to GitHub (from your laptop)

Remote repo: https://github.com/ridoy-adhikary/TriAudit-CIFAR10

This package already contains **one commit on `main`**. On your laptop you mainly need to **authenticate and push**.

## 1) Unzip

```bash
unzip TriAudit-CIFAR10-for-laptop.zip
cd TriAudit
```

Confirm the local commit exists:

```bash
git log -1 --oneline
git status
git remote -v
```

You should see something like:

```text
2768970 Initial TriAudit release for GitHub.
origin  https://github.com/ridoy-adhikary/TriAudit-CIFAR10.git
```

If the remote is missing or wrong:

```bash
git remote remove origin 2>/dev/null
git remote add origin https://github.com/ridoy-adhikary/TriAudit-CIFAR10.git
```

## 2) Push (first time)

Create a GitHub **Personal Access Token** (classic) with `repo` scope:
GitHub → Settings → Developer settings → Personal access tokens

Then:

```bash
git push -u origin main
```

When prompted:

- **Username:** `ridoy-adhikary`
- **Password:** paste the **token** (not your GitHub account password)

After a successful push, open:
https://github.com/ridoy-adhikary/TriAudit-CIFAR10

## 3) If GitHub repo already has a README / commit

Only if push is rejected as non-fast-forward:

```bash
git pull origin main --rebase
git push -u origin main
```

Avoid `git push --force` unless you intentionally want to overwrite the GitHub history.

## 4) Later commits (after you edit files)

```bash
git status
git add -A
git commit -m "Describe why you changed things."
git push
```

Keep ignoring large local artifacts (already in `.gitignore`):

- `dataset/raw/` (CIFAR download)
- `*.pt` checkpoints
- `.cursor/`, `__pycache__/`, orchestrator logs

## 5) Optional: SSH instead of HTTPS

```bash
git remote set-url origin git@github.com:ridoy-adhikary/TriAudit-CIFAR10.git
git push -u origin main
```

Requires your laptop SSH key added to GitHub.

## What’s in this zip

Included:

- source code, configs, tests, scripts
- reports/figures + research report
- eval JSON summaries / metrics (not checkpoints)
- split indices
- `.git` history (so you can push immediately)

Excluded (too large / local-only):

- CIFAR-10 raw download under `dataset/raw/`
- model `*.pt` checkpoints
- Cursor/IDE files, Python caches, orchestrator logs
