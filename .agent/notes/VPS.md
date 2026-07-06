# Infra VPS — documentée HORS du repo

La carte du VPS (services, ports, sécurité, procédures, journal d'interventions) vit dans
**`/home/bat/infra/VPS.md`**, tenue par l'agent **VPS-admin** (fenêtre tmux `vps-admin`).

Principe de frontière (décision Bob, 2026-07-06) : tout ce qui est relatif à la MACHINE
(environnement, services, sécurité, disque — multi-projets) appartient aux spécifications
du VPS ; ce repo ne documente que ce qui est PROPRE au projet Agora (voir `DEV_PROD.md` :
flux dev→prod, promotion des caches — des concepts projet, pas des specs machine).
