# Veille embedders — contenders permissifs vs nomic-v2 (le servi)

**Date** : 2026-07-07 · **Branche** : `research/bench-jina` · **Suite de** [`bench_jina.md`](bench_jina.md)
**Résultat : 3 rivaux Apache DÉPASSENT ou ÉGALENT nomic** sur le gold — à évaluer pour adoption
(gain modéré mais réel), contrairement à JINA (barré licence). qwen3 & e5-large : **NON mesuré**.

Protocole IDENTIQUE au témoin (gold x-stance n=2214 de/fr/it équilibré, rang-kNN k=15 → Leiden
res=1.0, seed=42, CPU, bootstrap 4×). **nomic-v2 re-validé au chiffre près** (NMI langue 0.008 /
thème 0.407). Scorecard brute complète : [`quality_report_veille.md`](quality_report_veille.md).

## 0. Périmètre (licences vérifiées API HF, commits épinglés)

| Modèle | Licence | Chargé ? | Note |
|---|:--:|:--:|---|
| granite-embedding-97m-multilingual-r2 | **Apache-2.0** | ✅ | ModernBERT natif |
| granite-embedding-311m-multilingual-r2 | **Apache-2.0** | ✅ | ModernBERT natif |
| snowflake-arctic-embed-l-v2.0 | **Apache-2.0** | ✅ | XLM-R natif |
| Qwen3-Embedding-0.6B | **Apache-2.0** | ✅ | LLM, last-token |
| multilingual-e5-large-instruct | **MIT** | ✅ | XLM-R |
| **gte-multilingual-base** | Apache-2.0 | ❌ | `IndexError` dans le code `trust_remote_code` vs transformers moderne — **même fragilité que JINA**. Écarté. |

## 1. Résultats chiffrés (gold x-stance)

Trié par **NMI(thème)** ↑ (récupération du thème = le cœur). Latence = smoke mono-modèle
(propre) ; le banc complet (7 modèles en série) a mesuré des latences ~2× plus hautes sous
contention mémoire — l'**ordre** tient, les valeurs absolues sont indicatives.

| Modèle | Licence | dim | **NMI thème ↑** | NMI langue ↓ | Cohér. ↑ | Pureté thème ↑ | ~ms/txt CPU |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **arctic-l** | Apache | 1024 | **0.455** | 0.004 | -0.118 | 0.682 | ~120 |
| **granite-311m-r2** | Apache | 768 | **0.437** | 0.004 | -0.103 | 0.703 | ~180 |
| qwen3-0.6b | Apache | 1024 | 0.408 | 0.005 | -0.093 | 0.673 | ~370 🔴 |
| **nomic-v2** *(témoin)* | Apache | 768 | 0.407 | 0.008 | -0.106 | 0.649 | ~45 |
| **granite-97m-r2** | Apache | 384 | 0.406 | 0.004 | -0.111 | 0.684 | ~57 |
| e5-large-instruct | MIT | 1024 | 0.271 | **0.504** | -0.077 | 0.482 | ~330 |
| e5-small *(piège)* | MIT | 384 | 0.048 | **0.812** | -0.159 | 0.215 | ~9 |

Composite pondéré (mixité 30/cohér. 25/thème 20/sil. 10/stab. 10/mod. 5) : **granite-311m-r2
0.834** > qwen3 0.804 > arctic-l 0.802 > granite-97m 0.738 > **nomic 0.722** > e5-large 0.474 > e5-small 0.193.

## 2. Lecture — 3 pistes d'adoption réelles, 2 fausses pistes

**Les gagnants (Apache, adoptables) :**
- **arctic-l** — meilleure récupération de thème (**0.455**, +0.048 vs nomic ≈ +12 %), mixité
  langue parfaite (0.004). Coût : **dim 1024** (+33 % de caches) et **~2,5× plus lent** à embed.
- **granite-311m-r2** — **gagnant composite**. Bat nomic sur le thème (0.437, +0.03), cohérence,
  pureté, modularité, silhouette. **Même dim 768** que nomic → **caches drop-in** (pas de +volume).
  ModernBERT **natif** (Apache propre, zéro code custom rotté). Coût : **~4× plus lent** à embed.
- **granite-97m-r2** — **parité qualité** avec nomic (thème 0.406 ≈ 0.407) à **DEMI-dimension**
  (384 → caches 2× plus petits, moins de RAM au service), vitesse ~comparable. Le pari « même
  qualité, empreinte réduite ».

**Les fausses pistes (mesurées, écartées) :**
- **qwen3-0.6b** — malgré 600M params, **AUCUN gain de thème** (0.408 ≈ nomic) et **~8-18× plus
  lent** sur CPU. Le « plafond qualité » n'en est pas un pour ce clustering. Rejeté par la mesure
  (bon exemple d'anti-hype : le plus gros ≠ le meilleur ici).
- **e5-large-instruct** — la famille e5 **ségrège encore par langue** même en large-instruct
  (NMI langue **0.504**, thème 0.271). Le « piège langue » d'e5 est un **trait de famille**, pas
  un artefact de petite taille. Confirme la leçon e5-small.

## 3. Verdict & recommandation

**La veille est productive** : 3 embedders **Apache** battent (arctic-l, granite-311m) ou égalent
à moindre empreinte (granite-97m) le témoin nomic — tous **libres et déployables** (aucune ficelle
type CC-BY-NC de JINA). **Mais le gain thème reste modéré (+0.03 à +0.05 NMI)** et l'adoption
impose toujours : **re-embedder TOUS les caches servis** (au débit CPU 2-4× de nomic → plusieurs
heures/dataset) **+ re-valider le témoin** end-to-end.

Ce n'est donc pas un « adopter maintenant », mais **une piste qui vaut une décision dédiée** :

| Si l'objectif est… | Candidat | Prix |
|---|---|---|
| Meilleure qualité brute | **arctic-l** (thème 0.455) | dim 1024, embed ~2,5× |
| Gain qualité, **caches inchangés** (dim 768), Apache propre | **granite-311m-r2** (0.437) | embed ~4× |
| **Empreinte réduite** (dim 384) à qualité nomic | **granite-97m-r2** | embed ~comparable |

**Reco** : prochaine étape = benché ces 3 sur un **dataset FR servi réel** (ex. granddebat/tiktok,
métriques internes + inspection qualitative des clusters — le gold x-stance est suisse/court), et
chiffrer le **coût de re-embed** vs le gain, avant tout GO. Ne rien changer en prod d'ici là.

## 4. Notes techniques
- gte-multilingual-base : `trust_remote_code` casse (IndexError) → **préférer les archis natives**
  (ModernBERT/XLM-R) pour la robustesse long-terme, leçon confirmée par JINA.
- Latences du banc gonflées par la contention (7 modèles en série) ; smoke mono-modèle plus fiable.
- Registre : 6 contenders ajoutés à `pipeline/embed/registry.py` (épinglés). Défaut **inchangé** (nomic-v2).
- Aucun cache servi ré-embeddé. Rien sur `main`/prod.
