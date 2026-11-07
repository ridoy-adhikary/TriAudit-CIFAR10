# TriAudit site (GitHub Pages)

Static report site for the project.

## Local preview

Open `docs/index.html` in a browser, or from the repo root:

```bash
python -m http.server 8000 --directory docs
```

Then visit `http://localhost:8000`.

## Enable on GitHub

1. Push the `docs/` folder to `main`.
2. Repo → **Settings** → **Pages**.
3. Source: **Deploy from a branch**.
4. Branch: `main` / folder: `/docs`.
5. Save. Site URL will look like:

`https://ridoy-adhikary.github.io/TriAudit-CIFAR10/`
