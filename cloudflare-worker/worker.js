addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

const CLIENT_ID = ''; // set via Cloudflare secret or replace at deploy time
const CLIENT_SECRET = '';
const REPO = 'ddd363/ESL';

async function handleRequest(request) {
  const url = new URL(request.url)
  const path = url.pathname.replace(/\/+$/, '')

  if (path === '/auth') {
    // Redirect user to GitHub authorize page
    const state = Math.random().toString(36).slice(2)
    const params = new URLSearchParams({
      client_id: CLIENT_ID,
      redirect_uri: `${url.origin}/callback`,
      scope: 'repo',
      state,
    })
    return Response.redirect(`https://github.com/login/oauth/authorize?${params.toString()}`)
  }

  if (path === '/callback') {
    const qs = url.searchParams
    const code = qs.get('code')
    if (!code) return new Response('Missing code', { status: 400 })

    // Exchange code for access token (server-side, keep secrets here)
    const tokenResp = await fetch('https://github.com/login/oauth/access_token', {
      method: 'POST',
      headers: { 'Accept': 'application/json', 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: CLIENT_ID, client_secret: CLIENT_SECRET, code }),
    })
    const tokenJson = await tokenResp.json()
    const accessToken = tokenJson.access_token
    if (!accessToken) return new Response('Auth failed', { status: 400 })

    // Send token back to opener window via postMessage
    const html = `<!doctype html><html><body>
<script>
  (function(){
    window.opener.postMessage({type:'decap-auth', token: '${accessToken}'}, '*');
    window.close();
  })();
</script>
</body></html>`

    return new Response(html, { headers: { 'Content-Type': 'text/html' } })
  }

  return new Response('Not found', { status: 404 })
}
