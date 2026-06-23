# Témoin mistral-large — découpe des claims (C1)

> Worker **TEMOIN** · branche `work/temoin` · 2026-06-23
> But : produire la **meilleure découpe contiguë** (PAS de multi-spans ici) du dataset
> **tiktok**, ré-extraite par **mistral-large**, pour jugement visuel de Bob sur `:5180`.

## Ce qui a changé

`pipeline/claims/extract.py` → `CLAIM_SYS` réécrit autour de **4 exigences** + few-shot
tirés des cas réels de Bob (cf. `queue/iteration-feedback-distill.md`, C1/C2) :

1. **SÉLECTIVITÉ** — n'extraire que la SUBSTANCE (grief / opinion / proposition).
   Laisser le cadrage, le narratif, les annonces (« pour illustrer… », « mes doléances
   sont triples : », politesses, anecdote de contexte).
2. **REGROUPEMENT** — ne pas fragmenter une idée : contraste (« X et non Y »),
   justification (« … parce que … »), condition (« si…, alors… »), énumération qui
   détaille UNE idée → **un seul claim**.
3. **SUJET + POSITION** — chaque claim porte, à lui seul, SA thématique ET la POSITION
   du citoyen (prérequis de la future stance ; un fragment qui ampute l'un est inutile).
4. **VERBATIM strict** — sous-chaîne exacte, rien d'ajouté/corrigé (fautes comprises).

Few-shot (principes, pas thèmes — généricité) : « élus qui représentent l'intérêt des
citoyens **et non** … » = 1 claim ; « Plus de respect, d'honnêteté … » = 1 ; énumération
« que ce soit X comme Y sur Z » = 1.

Modèle : `mistral-large-latest` (API Mistral, EU). Verbatim **100 %** sur l'échantillon
(toutes les portions s'ancrent comme sous-chaîne exacte → zéro hallucination, garanti
par `align_spans`).

## Avant / après (même modèle mistral-large, même avis)

### tiktok:29 — fragmentation & cadrage
*Avis : « Tiktok est une plateforme qui montre principalement du contenue qui pousse à la
culture du vide, même si des comptes peuvent être intéressant…, la plupart du contenu peut
même parfois être très choquant, les propos sont violents… : la plateforme crée une boucle
d'addiction et après c'est dur de s'en sortir, à part supprimer il n'y a pas grand chose. »*

- **AVANT — 6 claims** (sur-découpe) :
  1. « Tiktok … culture du vide »
  2. « même si des comptes peuvent être intéressant… » ← **concession/cadrage, pas un grief**
  3. « la plupart du contenu peut même parfois être très choquant »
  4. « les propos sont violents et haineux ou enferme… » ← coupé du 3 (même idée)
  5. « la plateforme crée une boucle d'addiction et après c'est dur de s'en sortir »
  6. « à part supprimer il n' y a pas grand chose à faire » ← coupé du 5
- **APRÈS — 3 claims** (regroupé, concession écartée) :
  1. « Tiktok … culture du vide »
  2. « la plupart du contenu peut même parfois être très choquant, les propos sont violents
     et haineux ou enferme l'utilisateur dans un mal être particulier »
  3. « la plateforme crée une boucle d'addiction et après c'est dur de s'en sortir, à part
     supprimer il n' y a pas grand chose à faire »

### tiktok:20 — narratif anecdotique aspiré
- **AVANT** : « Cela peut entrainer une dépendance, **ce qui fut mon cas lorsque j'étais au
  lycée** » (anecdote collée) ; le 1ᵉ claim démarre sur le cadrage « Sentiment de mal-être
  expliqué par le fait que… ».
- **APRÈS** : « Cela peut entrainer une dépendance » (substance seule) ; 1ᵉ claim recadré
  sur « regarder des vidéos courtes pendant 1h fait perdre notre productivité… ».

### tiktok:256 — cadrage en tête de claim
- **AVANT** : « **Malgré toute notre vigilance**, nos enfants sont confrontés… » ; phrase de
  récit « Ils nous posent des questions pour évacuer la gêne… » gardée entière.
- **APRÈS** : « nos enfants sont confrontés à des contenus choquants et sans aucun intérêt » ;
  « certains sujets les bouleversent et les questionne sur notre société à la dérive ».

### tiktok:236 — déjà bon, stable
Avis long & argumenté → 5 claims dans les deux cas (idées réellement distinctes :
constat → jugement → demande → nuance → proposition). Le regroupement ne sur-fusionne pas.

## Bilan

- **Moins de fragments**, **moins de méta/narratif**, idées complètes (contraste +
  justification gardés ensemble), **verbatim 100 %**.
- La découpe reste **contiguë** (mono-span) — le multi-spans viendra en C2/C3.
- Repli inchangé : un avis dont aucune portion ne s'ancre → 1 claim = avis entier (jamais
  perdu) ; un avis pur narratif → liste vide → repli avis entier.

## Reproduire

```
export MISTRAL_API_KEY=$(cat var/mistral.key)
uv run python -m backend.build_analysis --dataset tiktok --reextract --model mistral-large-latest
```
puis explorer sur `:8010` (API) / `:5180` (front) — surlignages des portions par macro.
