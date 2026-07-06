# `.agent/` — guide pour tout agent (et humain) qui travaille sur Agora

Tu lances un agent (Claude Code, Cursor, autre) sur ce dépôt ? **Commence ici.** Ce
dossier centralise tout le contexte non-évident : architecture, façon de travailler,
notes de R&D, et le ledger des tâches. Le code, lui, est documenté à côté du code.

---

## 1. Ce qu'est Agora
Un outil d'analyse de **consultations citoyennes** : il ingère des milliers de réponses
en texte libre et en fait **émerger les grands thèmes** — de façon automatisée,
transparente, souveraine (modèles ouverts), et bon marché.

Pipeline : `claims (extraction LLM verbatim) → embeddings (nomic-v2) → graphe k-NN →
Leiden → hiérarchie variance-adaptative → nommage/enrichissement/insights (Mistral, caché)
→ opinion (cible de clivage + stance)`.

## 2. Carte du dépôt
| Dossier | Rôle |
|---|---|
| `backend/` | API FastAPI (:8010) **SERVE-only** + les builds (`build_analysis`, `build_opinion`, `build_cache`) |
| `pipeline/` | le cœur algo : `ingest/`, `claims/`, `embed/`, `cluster/` |
| `frontend/` | app Vite/React (:5180), contrat de types dans `src/redesign/contract.ts` |
| `.github/` | CI (tests sur PR) + Deploy (auto-déploiement sur push `main`) |
| `.agent/notes/` | **notes de R&D / décisions** (algorithmes, modèles, arbitrages) — à lire avant de toucher au domaine concerné |
| `research/` | harnais d'expériences (verdicts one-off) |
| `data/`, `var/` | données brutes (gitignoré) · secrets (gitignoré) |

## 3. Comment on travaille (à respecter)
- **Dev / prod séparés** — voir `.agent/notes/DEV_PROD.md`. On CODE dans un clone dev
  (a la clé Mistral) ; la **prod** sert le cache **sans clé** (aucun appel LLM au runtime).
- **Flux** : branche → **PR vers `main`** → la CI (pytest + build) doit passer → merge →
  la prod **se met à jour toute seule**. Ne pousse jamais d'état local non commité (le
  déploiement fait `reset --hard origin/main`).
- **R&D pilotée par verdict** : toute idée « on pourrait… » se **mesure** avant d'être
  adoptée, et le verdict (OUI/NON + pourquoi) est écrit. Les `.agent/notes/` et la mémoire
  du projet sont pleins de « NON, testé, voici pourquoi » — lis-les pour ne pas refaire.
- **Généricité** : zéro nom de corpus en dur. Tout est dérivé des données (Agora doit
  marcher sur des centaines de consultations inconnues).
- **Verbatim & traçabilité** : les claims sont des extraits VERBATIM ancrés sur
  `text_clean` (PII masquée) ; on peut toujours remonter du thème à la phrase du citoyen.
- **Sécurité** : secrets dans `var/` (jamais commités) ; en public, backend `AGORA_PUBLIC=1`
  fail-closed (lectures ouvertes, mutations/compute 403).

## 4. Démarrer en local
```bash
./scripts/setup.sh      # installe tout (deps back+front, caches, .env) — voir CONTRIBUTING.md
make dev                # lance backend :8010 + front :5180
```

## 5. Où trouver quoi
- **Décisions/algorithmes** → `.agent/notes/` (ADAPTIVE, HDBSCAN, MISTRAL, NAMING_SWITCH,
  MULTIDATASET, DEVELOP, REALDATA) + `DEV_PROD.md`.
- **Contrat d'API front↔back** → `frontend/src/redesign/contract.ts`.
- **Défi hackathon** → `hackathon-an-2026/DEFI.md`.
