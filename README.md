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

Production note: Decap's GitHub login needs an OAuth provider endpoint. This repo is configured for the GitHub backend, but you still need to wire an auth provider (for example Netlify OAuth flow or a self-hosted Decap auth endpoint) for hosted `/admin` login to complete.

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
npx decap-server
```

Then open `http://127.0.0.1:4000/ESL/admin/` while Jekyll is running.

## Notes

- `baseurl` is set to `/ESL` for GitHub project pages.
- Local bead tracking files are ignored via `.gitignore` (`.beads/`).
