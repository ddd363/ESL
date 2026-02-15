# ESL Website (Jekyll + Decap CMS)

This repo now includes a full static-site stack:

- **Jekyll** for templating and data-driven pages
- **Decap CMS** for browser-based GUI editing (`/admin`)
- **GitHub Actions + Pages** for automatic deployment

## Where to edit

### GUI editor (WordPress-like)

- Open: `https://ddd363.github.io/ESL/admin/`
- Sign in with GitHub
- Edit **Site Content → Home Page**
- Save and publish

This updates `_data/home.yml` and triggers a new Pages deployment.

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

## Notes

- `baseurl` is set to `/ESL` for GitHub project pages.
- Local bead tracking files are ignored via `.gitignore` (`.beads/`).
