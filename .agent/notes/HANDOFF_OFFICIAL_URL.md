# Handoff → architecte : lien « consultation officielle » manquant sur la page de synthèse

**Statut :** à traiter hors lane FRONT-COSMETIC (touche le backend/cache — hors périmètre cosmétique).
**Signalé par :** lane FRONT-COSMETIC (session 2026-07-08), sur remontée de Bob.

## Symptôme
Sur la page de synthèse (`ConsultationOverview`), le lien **« Voir la consultation
officielle ↗ »** ne s'affiche pour aucune consultation.

## Diagnostic (le front n'est PAS en cause)
Le lien est piloté par la donnée, pas codé en dur (règle de généricité). La chaîne :

1. `frontend/src/redesign/ConsultationOverview.tsx:141-147` — affiche le lien **si**
   `dataset.official_url` est présent. Code **correct et intact**.
2. `backend/recluster.py:234-235` — propage `official_url` vers `/datasets`
   **seulement s'il figure dans le `meta.json`** du descripteur.
3. `backend/cache/*/meta.json` — **aucun** ne contient `official_url`
   (vérifié : `tiktok`, `granddebat`, … n'ont que
   `id, label, n_nodes, languages, source, question, context, n_responses`).

→ Le champ n'est jamais servi faute de donnée source : le lien reste donc masqué
(comportement voulu du front : pas de donnée = pas de lien).

## Fix proposé (lane backend/data)
Ajouter `"official_url": "https://…"` (l'URL officielle réelle de chaque
consultation) dans les `backend/cache/<consult>/meta.json` concernés. Le schéma le
supporte déjà : `backend/consultation_schema.py:47` (`official_url: NotRequired[str]`).
Aucune modif front nécessaire — le lien apparaîtra automatiquement.

## Pourquoi pas fait ici
- `backend/cache/*/meta.json` = **backend + cache**, explicitement hors du périmètre
  FRONT-COSMETIC.
- Les URLs officielles réelles ne sont pas connues de cette lane — à ne pas inventer.
