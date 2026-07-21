"""Test DÉCISIF — ré-embedder la SYNTHÈSE d'un cluster rapproche-t-il les redondances ?

Hypothèse (Bob) : la redondance entre thèmes frères est SÉMANTIQUE, pas géométrique — les
5 clusters d'« addiction » sont éloignés dans l'espace des claims (0.38-0.53) parce que
l'embedding encode la FORME (dopamine / désinstaller / drogue). Une SYNTHÈSE LLM normalise la
forme vers le sens ; ré-embedder les synthèses devrait donc RAPPROCHER les 5 addictions
(surface→sens), tout en gardant SÉPARÉS les 3 clusters « filles » (contrôle parental ≠
prédateurs ≠ image du corps — même public, sujets différents).

Verdict attendu si l'approche récursive tient :
  intra-addiction (synthèses) ≫ intra-addiction (claims 0.38-0.53)  ET  addiction↔filles reste bas.

    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra embed-contender --extra faiss \
        python research/synthesis_embed_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend import analysis as A
from backend.build_analysis import load_dataset
from pipeline.cluster.layers import centre
from pipeline.cluster.mistral_client import chat
from pipeline.embed.embedder import embed

SYNTH_MODEL = "mistral-small-latest"


def _topic(label: str) -> str | None:
    t = set(label.lower().split())
    if t & {"application", "appli", "scroller", "dopamine", "dépendance", "addiction", "addictif"}:
        return "ADDICTION"
    if t & {"fille", "enfants", "fils", "parental", "corps"}:
        return "FILLES"
    return None


def _synthesise(claims: list[str]) -> str:
    extraits = "\n".join(f"- {c[:200]}" for c in claims[:15])
    msg = [
        {"role": "system", "content":
         "Tu résumes un groupe de témoignages citoyens. Donne en UNE phrase le SUJET COMMUN "
         "dont ils parlent, en langage ABSTRAIT et général — le fond, pas la forme ni les "
         "détails. Pas de préambule, juste la phrase."},
        {"role": "user", "content": f"Témoignages :\n{extraits}\n\nSujet commun (une phrase) :"},
    ]
    return chat(msg, model=SYNTH_MODEL, temperature=0.1, max_tokens=80).strip()


def main() -> None:
    model = json.loads(Path("backend/cache/tiktok/claims.json").read_text())["model"]
    tree = A.build_theme_tree(load_dataset("tiktok"), model=model, seed=42)
    leaves = [n for n in tree.nodes.values() if not n.children]

    targets = []
    for n in sorted(leaves, key=lambda x: -x.n_avis):
        topic = _topic(n.label)
        if topic:
            claims = n.representative_claims or [tree.prepared.claim_texts[i] for i in n.members[:15]]
            targets.append((topic, " ".join(n.label.split()[:3]), claims))

    print(f"{len(targets)} clusters cibles (addiction + filles). Génération des synthèses…\n")
    synths = []
    for topic, name, claims in targets:
        s = _synthesise(claims)
        synths.append(s)
        print(f"  [{topic:<9}] {name:<26} → {s}")

    # Embedding LOCAL des synthèses (même modèle que les claims), recentré entre elles.
    V = embed(synths, model_id="nomic-v2").astype(np.float64)
    Vc = centre(V)
    S = Vc @ Vc.T
    topics = [t for t, _, _ in targets]

    def _avg(a, b, same):
        vals = [S[i, j] for i in range(len(topics)) for j in range(i + 1, len(topics))
                if (topics[i] == a and topics[j] == b) or (topics[i] == b and topics[j] == a)
                if (i != j)]
        return float(np.mean(vals)) if vals else float("nan")

    intra_add = np.mean([S[i, j] for i in range(len(topics)) for j in range(i + 1, len(topics))
                         if topics[i] == topics[j] == "ADDICTION"])
    intra_fil = np.mean([S[i, j] for i in range(len(topics)) for j in range(i + 1, len(topics))
                         if topics[i] == topics[j] == "FILLES"])
    cross = np.mean([S[i, j] for i in range(len(topics)) for j in range(i + 1, len(topics))
                     if topics[i] != topics[j]])

    print("\n=== Cosinus des SYNTHÈSES (espace recentré) ===")
    print(f"  intra-ADDICTION : {intra_add:.3f}   (claims : 0.38-0.53)")
    print(f"  intra-FILLES    : {intra_fil:.3f}")
    print(f"  ADDICTION↔FILLES: {cross:.3f}")
    print(f"\n  → surface→sens CONFIRMÉ si intra-ADDICTION ≫ 0.5 ET nettement > ADDICTION↔FILLES")
    print(f"  → séparation préservée si intra-FILLES n'écrase pas la distinction (filles = sujets distincts)")

    Path("research/synthesis_embed_results.json").write_text(json.dumps(
        {"targets": [(t, n) for t, n, _ in targets], "synths": synths,
         "intra_addiction": round(intra_add, 4), "intra_filles": round(intra_fil, 4),
         "cross": round(cross, 4)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
