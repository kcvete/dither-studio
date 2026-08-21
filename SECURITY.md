# Security

## Threat model, honestly

**The browser engine sends nothing anywhere.** Decode, tracking, dithering and
encoding all happen in the tab — no account, no upload, no telemetry, and nothing
fetched from a CDN at run time. Your frames never leave the machine.

**The optional local server is a local tool.** `server/server.py` binds
`127.0.0.1` and, with no `DV_API_KEY` set, is deliberately wide open on that
interface: `/api/*` needs no credential, CORS defaults to `*`, and uploaded
sources, extracted frames and rendered output live under `jobs/<id>/` until the
janitor sweeps them (2 GB / 14 days by default). Treat anything that can reach
the port as able to read and write `jobs/`.

**Exposing it publicly is your responsibility.** Set `DV_API_KEY` — every
`/api/*` request then needs `Authorization: Bearer <key>` or gets a 401 — narrow
`DV_CORS_ORIGINS` from its `*` default, and put TLS, rate limiting and a much
shorter `DV_JOBS_MAX_AGE_DAYS` in front of it. There is no billing, no accounts,
no per-tenant isolation and no rate limiting in this repository, on purpose.

**Model weights are not in git.** `setup.sh` clones EdgeTAM at a pinned commit,
downloads its checkpoint and fetches onnxruntime-web from npm — upstream supply
chains, to audit as you would any dependency.

## Reporting a vulnerability

Open a private security advisory:
https://github.com/kcvete/dither-studio/security/advisories/new

Please do not open a public issue for something exploitable. Expect a reply
within a couple of weeks; this is a small project. No PGP key, no bug bounty.
