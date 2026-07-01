# Lane stream — backend FastAPI · WS · replay simulator

Owns: `backend/`. Port `:8010`. Implémente le protocole WS du contrat.
**Phase 2** (live) — démarre après le pipeline batch + viz batch (décision : batch
d'abord). En Phase 1, le front consomme un `GraphPayload` statique sans ce backend.

## T-S1 · App FastAPI + santé
- Goal : squelette FastAPI sur `:8010`, endpoints santé + snapshot REST.
- Accept : `GET /health` ok ; ne touche aucun port interdit.
- Deps : contrat figé.

## T-S2 · Protocole WS + snapshot
- Goal : émettre idea_added / edges_added / cluster_updated / merged / split ;
  `snapshot` pour late-joiners.
- Accept : un client WS reçoit une séquence valide ; reconnect → snapshot.
- Deps : T-S1, nlp T-N4.

## T-S3 · Replay simulator
- Goal : rejouer le JSONL canonique (TikTok 33k) comme un flux temps réel
  (cadence réglable) → alimente le fast/slow path. (Pivot si fork #1 = vrai online.)
- Accept : `replay --rate N` produit une montée en charge animable.
- Deps : T-S2, data T-D4.

## T-S4 · Projection 2D pour l'animation
- Goal : coords (x,y) des nœuds. Défaut : UMAP batch périodique + placement
  provisoire des nouveaux points près de leurs k-NN, transitions interpolées côté front.
- Accept : pas de "saut" visuel brutal ; refresh périodique cohérent.
- Deps : T-S2, nlp T-N2.
