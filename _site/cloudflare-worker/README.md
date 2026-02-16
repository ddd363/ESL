Cloudflare Worker OAuth proxy for Decap CMS

This worker implements two endpoints used by Decap CMS's GitHub backend:

- `/auth` - redirects the user into GitHub's OAuth authorize flow
- `/callback` - exchanges the code for an access token and posts it back to the opener window

Quick deploy steps (summary):

1. Create a GitHub OAuth App (https://github.com/settings/developers -> OAuth Apps)
   - Homepage URL: your worker URL (e.g. https://my-auth.example.workers.dev)
   - Authorization callback URL: https://<your-worker-domain>/callback
   - Copy `CLIENT_ID` and `CLIENT_SECRET`.

2. Deploy the worker with secrets (using Wrangler):

```bash
npm install -g wrangler
npx wrangler login
# edit wrangler.toml to set name and route, then publish
wrangler secret put CLIENT_ID
wrangler secret put CLIENT_SECRET
wrangler publish
```

3. Update `admin/config.yml` in this repo:

- set `backend.base_url: https://<your-worker-domain>`
- ensure `backend.auth_endpoint: auth`
- set `backend.site_domain` to your worker hostname if required by config

4. Open `/admin` on your Pages site, click Login with GitHub and complete OAuth.

Notes:
- Keep `CLIENT_SECRET` private (use Wrangler secrets or Cloudflare dashboard secrets).
- This template is intentionally minimal; consider adding CSRF `state` verification and more robust error handling in production.