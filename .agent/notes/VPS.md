# VPS — Carte système (forge.tail0b8aa8.ts.net)

> Dernière révision : 2026-07-06 (mission sysadmin, validée Bob).
> Debian 13 (trixie) · disque `/dev/sda1` 99G · `/` et `/home` sont le **même** filesystem.
> `bat` = uid 1000, `sudo NOPASSWD:ALL`, membre du groupe `docker`.

## 1. PROD — NE JAMAIS INTERROMPRE

Sert **Agora** publiquement via le funnel Tailscale (`:443`).

| Élément | Valeur |
|---------|--------|
| Checkout | `/home/bat/projects/Analyse-des-consultations-citoyennes` (branche `main`, clone dédié deploy) |
| Backend | systemd **user** `agora-backend.service` → `uv run … uvicorn backend.server:app --host 0.0.0.0 --port 8010` |
| Frontend | systemd **user** `agora-frontend.service` → `npm run preview -- --host 0.0.0.0 --port 5180 --strictPort` (vite preview) |
| EnvironmentFile | `…/var/deploy.env` (secrets, `600`) |
| Interpréteur | venv **éphémère** sous `~/.cache/uv/builds-v0/.tmp*/` — ⚠️ ne pas purger `.cache/uv` à l'aveugle |
| CI/CD | `actions.runner.1Batmain-Analyse-des-consultations-citoyennes.agora-vps.service` (system) — checkout in-place |
| Exposition | funnel Tailscale `:443` → **seul point public** |
| Healthcheck | `curl -sS https://forge.tail0b8aa8.ts.net/` doit renvoyer **200** après TOUT changement |

Raccourci nav : `/home/bat/prod` → symlink vers le checkout (les services utilisent le chemin absolu, le symlink n'affecte rien).

## 2. DEV & LAB

| Chemin | Rôle | Git |
|--------|------|-----|
| `/home/bat/agora-dev` | Hub dev Agora (branche `dev`) — **héberge les worktrees** dans `.git/worktrees/` | même remote que prod : `github.com:1Batmain/Analyse-des-consultations-citoyennes` |
| `/home/bat/agora-syntheses` | worktree d'agora-dev (`feat/insights-cache-hero`) — dev backend `:8011` (localhost) + vite `:5181`, up ~2j | worktree |
| `/home/bat/agora-stance-ab` | worktree d'agora-dev (`feat/stance-cible-ab`) | worktree |
| `/home/bat/agora-worktrees/{claims-prod,front-redesign,live-front,mappolish,p4-llm,titles}` | worktrees d'agora-dev par feature | worktrees |
| `/home/bat/forge` | **Lab R&D** (8.5G). Contient `dummy` + worktrees `dummy-*` | repo `dummy` |
| `/home/bat/forge/dummy` | **App dummy** (PROTÉGÉE) — MCP server `python -m mcp_server.server` sur `:8765`, up ~20j, venv **vivant** `.venv` (5.1G) | repo principal (`main`) |
| `/home/bat/forge/dummy-{3d,3d-asset,frontend,labels,embed2,capture,researcher}` | worktrees de `dummy` par branche | worktrees |
| `/home/bat/agora-backups` | backups (PROTÉGÉ) |
| `/home/bat/actions-runner` | runner GitHub (PROTÉGÉ) |

Raccourcis nav : `/home/bat/dev/agora` → `agora-dev`, `/home/bat/dev/dummy` → `forge/dummy`, `/home/bat/dev/lab` → `forge`.

⚠️ **Les worktrees ne se déplacent PAS** : leur `.git` pointe en chemin absolu vers `agora-dev/.git/worktrees/…` (ou `forge/dummy/.git`). Pour retirer un worktree : `git -C <hub> worktree remove <path>`, jamais `rm`/`mv`.

## 3. Ports (écoute)

| Port | Bind | Process | Rôle |
|------|------|---------|------|
| 443 | tailscale0 | tailscaled | **funnel public** → PROD |
| 8010 | 0.0.0.0 | uvicorn | PROD backend |
| 5180 | 0.0.0.0 | vite preview | PROD frontend |
| 8011 | 127.0.0.1 | uvicorn | dev backend (agora-syntheses) |
| 5181 | 0.0.0.0 | vite | dev frontend (agora-syntheses) |
| 8765 | 0.0.0.0 | python | dummy MCP server (recommandé : rebind `127.0.0.1`) |
| 11434 | 127.0.0.1 | ollama | LLM local |
| 22 | 0.0.0.0 | sshd | SSH — **bloqué au public par ufw** (tailnet-only) |
| 53 | localhost | systemd-resolved | DNS |

## 4. Sécurité (audit 2026-07-06)

**Posture globale : saine.** Seule surface publique = funnel Tailscale `:443`.

- **ufw** actif : `default deny incoming`, autorise **uniquement** l'interface `tailscale0`. Tous les bind `0.0.0.0` (SSH inclus) sont donc injoignables depuis l'Internet public.
- **SSH** : `PermitRootLogin no`, `PasswordAuthentication no`, pubkey-only, `PermitEmptyPasswords no`. Clés présentes (`bat`, `root`). ✅
- **fail2ban** actif (jail `sshd`). **unattended-upgrades** actif.
- **Secrets** tous en `600 bat:bat` : `projects/.../var/deploy.env`, `agora-dev/var/{deploy.env,mistral.key}`. ✅
- Aucun fichier world-writable sensible ; aucun conteneur Docker exposé (seule image : `hello-world`).

**Risques résiduels (informés, acceptés / recommandations) :**
- `bat` = `sudo NOPASSWD:ALL` + groupe `docker` (≈ root). Design voulu ; ne pas exposer davantage.
- MCP dummy `:8765` bind `0.0.0.0` inutilement → recommandé rebind `127.0.0.1` (ufw couvre déjà).
- 8 paquets tiers upgradables (docker*, gh, tailscale). ⚠️ **Ne pas upgrader tailscale à chaud** (blip funnel = coupure PROD) — faire en fenêtre de maintenance.

## 5. Procédures

- **Healthcheck PROD** : `curl -sS -o /dev/null -w '%{http_code}\n' https://forge.tail0b8aa8.ts.net/` → 200.
- **Restart PROD** (si nécessaire) : `systemctl --user restart agora-backend agora-frontend` puis healthcheck.
- **Logs PROD** : `journalctl --user -u agora-backend -f`.
- **Nettoyage sûr récurrent** : `journalctl --vacuum-size=200M` ; `sudo apt-get clean`. Ne PAS purger `.cache/{uv,huggingface}` (interpréteur/embeddings PROD vivants) ni `forge/dummy/.venv` (MCP vivant).
- **Corbeille de tri** : `/home/bat/_corbeille_YYYYMMDD/` + `MANIFEST.md` ; rien n'est `rm` sans validation Bob.
