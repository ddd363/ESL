# ESL Website (Jekyll + Decap CMS)

This repo now includes a full static-site stack:

- **Jekyll** for templating and data-driven pages
- **Decap CMS** for browser-based GUI editing (`/admin`)
- **GitHub Actions + Pages** for automatic deployment

## Where to edit

### GUI editor (WordPress-like)

- Local-first workflow (no OAuth required):
	1. Run the local web server and local Decap backend.
	2. Open `http://127.0.0.1:8000/admin/`.
	3. Edit **Site Content → Home Page**.
	4. Publish in CMS, then commit/push with Git.

This updates `_data/home.yml` and, after `git push`, GitHub Pages redeploys automatically.

### Code editor

- Page template: `index.html`
- Content data: `_data/home.yml`
- CMS config: `admin/config.yml`
- Jekyll config: `_config.yml`

## Local preview

### Option 1: quick static preview (no Jekyll rendering)

```bash
cd "/Users/spc/PYTHON NOTEBOOKS/ESL"
python3 -m http.server 8000
```

Open `http://localhost:8000`

### Option 2: full Jekyll preview (recommended)

Requires Ruby and Bundler.

```bash
cd "/Users/spc/PYTHON NOTEBOOKS/ESL"
bundle install
bundle exec jekyll serve
```

Open `http://127.0.0.1:4000/ESL/`

### Local CMS mode (no OAuth needed)

With `local_backend: true`, you can run Decap locally and edit content without hosted OAuth setup.

```bash
cd "/Users/spc/PYTHON NOTEBOOKS/ESL"
python3 -m http.server 8000
```

```bash
cd "/Users/spc/PYTHON NOTEBOOKS/ESL"
npx decap-server
```

Then open `http://127.0.0.1:8000/admin/`.

After editing, publish your changes and push to GitHub:

```bash
cd "/Users/spc/PYTHON NOTEBOOKS/ESL"
git add -A
git commit -m "CMS: update website content"
git push origin main
```

## Notes

- `baseurl` is set to `/ESL` for GitHub project pages.
- Local bead tracking files are ignored via `.gitignore` (`.beads/`).
- Hosted `/admin` login via OAuth proxy is optional and only needed for remote browser-only editing.
