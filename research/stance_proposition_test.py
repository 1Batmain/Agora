"""Test ciblé : le prompt de stance corrigé résout-il la confusion sujet↔proposition ?

Cas : tiktok n15 « TikTok et ses effets contrastés », objet « réguler l'utilisation de TikTok ».
Actuel : 97% défavorable (= critique de TikTok). Attendu après fix : majorité FAVORABLE
(décrire les méfaits = soutenir la régulation). On re-classe les MÊMES claims avec l'ancien
et le nouveau prompt, on compare.
"""
import os
from collections import Counter

from pipeline.cluster import mistral_client
from backend.build_opinion import STANCE_SYSTEM as OLD_SYS, _leaf_claims, MODEL
from backend.analysis import build_theme_tree
from backend.recluster import dataset_descriptor, load_cache
import json

# --- Prompt CORRIGÉ : juge le SOUTIEN À L'ACTION, pas le sentiment envers le sujet. ---
NEW_SYS = (
    "Tu es analyste de consultations citoyennes. On te donne UNE CIBLE — une PROPOSITION "
    "D'ACTION débattable (p. ex. « réguler l'utilisation de TikTok ») — et des CONTRIBUTIONS "
    "citoyennes verbatim. Pour chaque contribution, classe si son auteur SOUTIENT ou S'OPPOSE "
    "À CETTE ACTION (et NON son sentiment envers le sujet) :\n"
    "  - \"favorable\"   : la contribution VA DANS LE SENS de l'action — elle la réclame, OU "
    "elle décrit un PROBLÈME/méfait que cette action viserait à corriger (décrire les dangers "
    "de X = soutenir une action pour réguler/limiter X) ;\n"
    "  - \"defavorable\" : la contribution S'OPPOSE à l'action — elle défend le sujet tel quel, "
    "juge l'action inutile/excessive/nuisible, ou refuse toute intervention ;\n"
    "  - \"nuance\"      : position ambivalente/conditionnelle, ou aucune position claire sur "
    "l'ACTION elle-même.\n"
    "ATTENTION : ne confonds JAMAIS un sentiment négatif ENVERS LE SUJET avec une opposition à "
    "l'action. Quelqu'un qui critique/subit TikTok est FAVORABLE à « réguler TikTok ».\n"
    "Indique aussi ta CONFIANCE : \"high|medium|low\". Réponds en JSON strict : "
    "{\"results\":[{\"i\":<int>,\"stance\":\"favorable|defavorable|nuance\",\"confidence\":"
    "\"high|medium|low\",\"justif\":\"<≤14 mots>\"}]}. Une entrée par contribution, dans l'ordre."
)


def classify(cible, items, system):
    lines = [f"[{i}] {t}" for i, t in items]
    user = f"CIBLE : {cible}\n\nCONTRIBUTIONS (réponds pour chaque [indice]) :\n" + "\n".join(lines)
    raw = mistral_client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=MODEL, temperature=0.0, max_tokens=1500, json_mode=True)
    out = {}
    for rec in json.loads(raw).get("results", []):
        out[int(rec["i"])] = (rec.get("stance"), rec.get("justif", ""))
    return out


ds_id = "tiktok"
ideas, vecs, weights = load_cache(ds_id)
ds = type("D", (), {"id": ds_id, "ideas": ideas, "vecs": vecs, "weights": weights,
                    "descriptor": dataset_descriptor(ds_id, ideas)})()
tree = build_theme_tree(ds)
node = tree.nodes["n15"]
claims = _leaf_claims(node, tree.prepared)  # (idx, avis_id, text)
items = [(j, c[2]) for j, c in enumerate(claims)]
cible = "réguler l'utilisation de TikTok"
print(f"n15 : {len(items)} claims · cible : « {cible} »\n")

for label, sysp in [("ANCIEN", OLD_SYS), ("NOUVEAU", NEW_SYS)]:
    res = classify(cible, items, sysp)
    dist = Counter(s for s, _ in res.values())
    print(f"=== {label} ===  {dict(dist)}")
    for j in list(res)[:5]:
        s, jf = res[j]
        print(f"   [{s}] {jf[:80]}  ⟵ « {items[j][1][:60]} »")
    print()
