"""Cache d'embeddings ADRESSÉ PAR CONTENU — un vecteur, une clé, indépendant des autres.

## Le problème résolu

Le cache d'embeddings de claims était empreinté par un sha256 de la **concaténation de tous
les textes** : une seule entrée nouvelle invalidait **tout le corpus**. Acceptable en batch
(on rebâtit de toute façon), rédhibitoire en LIVE — chaque tour de parole aurait re-embeddé
les 36 000 claims du corpus.

Ici, la clé d'un vecteur est `sha256(embedder ‖ texte)`. Trois propriétés en découlent, et
chacune corrige un défaut réel de l'empreinte globale :

1. **Incrémental** — un texte nouveau coûte UN embedding, pas N.
2. **Insensible à l'ORDRE** — l'empreinte globale changeait si les claims étaient simplement
   permutés, alors que les vecteurs, eux, étaient identiques. C'est exactement l'avalanche de
   régénération déjà rencontrée sur les clés de titres (corrigée là-bas en triant la clé).
3. **Dédup gratuite** — deux textes identiques partagent une clé, donc un seul calcul.

L'embedder fait partie de la clé : changer d'embedder DOIT manquer le cache (des vecteurs de
modèles différents vivent dans des espaces incomparables — cf. le garde-dimension de
`backend/submissions.py`, posé après une régression réelle).

## Format sur disque

Un `.npz` unique : `keys` (U64, une clé par ligne) et `vecs` (float32, alignés). Un fichier
par magasin, pas un fichier par vecteur — à 36 000 claims, l'un est instantané et l'autre
inutilisable.

Le magasin est **append-only** : il conserve les vecteurs de textes disparus. C'est voulu
(un rebuild qui revient en arrière les retrouve), et borné par la taille du corpus. Utiliser
`compact()` pour reprendre la place quand un cache a beaucoup dérivé.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

# Version du FORMAT (pas du contenu) : permet de reconnaître un magasin d'une génération
# antérieure et de le migrer au lieu de tout re-calculer.
FORMAT_VERSION = "v1"

KEY_DTYPE = "U64"


def key_for(text: str, embedder: str) -> str:
    """Clé d'un vecteur = hash de (embedder, texte). Stable, indépendante du contexte."""
    raw = f"{embedder}\x00{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def keys_for(texts: Sequence[str], embedder: str) -> list[str]:
    return [key_for(t, embedder) for t in texts]


class VectorStore:
    """Magasin clé→vecteur en mémoire, persistable en `.npz`."""

    def __init__(self, dim: int | None = None) -> None:
        self._index: dict[str, int] = {}
        self._rows: list[np.ndarray] = []
        self.dim = dim

    # -- lecture ------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def get(self, key: str) -> np.ndarray | None:
        i = self._index.get(key)
        return None if i is None else self._rows[i]

    # -- écriture ------------------------------------------------------------------ #

    def put(self, key: str, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float32).ravel()
        if self.dim is None:
            self.dim = int(vec.shape[0])
        elif vec.shape[0] != self.dim:
            # Un magasin ne mélange JAMAIS deux espaces : un cosinus inter-espaces n'a aucun
            # sens et planterait plus loin, à un endroit où la cause serait illisible.
            raise ValueError(
                f"dimension {vec.shape[0]} incompatible avec le magasin (dim={self.dim}) — "
                "un changement d'embedder impose un magasin distinct."
            )
        i = self._index.get(key)
        if i is None:
            self._index[key] = len(self._rows)
            self._rows.append(vec)
        else:
            self._rows[i] = vec

    def put_many(self, keys: Sequence[str], vecs: np.ndarray) -> None:
        if len(keys) != len(vecs):
            raise ValueError(f"{len(keys)} clés pour {len(vecs)} vecteurs")
        for k, v in zip(keys, vecs):
            self.put(k, v)

    # -- persistance --------------------------------------------------------------- #

    @classmethod
    def load(cls, path: Path | str) -> "VectorStore":
        """Charge un magasin. Fichier absent/illisible/d'un autre format → magasin VIDE.

        Jamais d'exception sur un cache abîmé : un cache est une optimisation, sa perte doit
        coûter du temps de calcul, pas une panne.
        """
        store = cls()
        p = Path(path)
        if not p.exists():
            return store
        try:
            data = np.load(p, allow_pickle=False)
            keys = [str(k) for k in data["keys"]]
            vecs = data["vecs"].astype(np.float32)
        except (OSError, KeyError, ValueError):
            return store
        if len(keys) != len(vecs):
            return store
        store.dim = int(vecs.shape[1]) if vecs.size else None
        store._index = {k: i for i, k in enumerate(keys)}
        store._rows = [vecs[i] for i in range(len(vecs))]
        return store

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._index.keys()), dtype=KEY_DTYPE)
        if self._rows:
            order = list(self._index.values())
            vecs = np.stack([self._rows[i] for i in order]).astype(np.float32)
        else:
            vecs = np.zeros((0, self.dim or 1), dtype=np.float32)
        # Écriture ATOMIQUE : un `save` interrompu ne doit pas laisser un magasin tronqué
        # que le prochain `load` accepterait silencieusement.
        # ⚠️ On passe un DESCRIPTEUR à `np.savez`, pas un chemin : avec un chemin, numpy
        # ajoute `.npz` quand l'extension manque (« s.npz.tmp » → « s.npz.tmp.npz »), et le
        # `replace` échouait sur un fichier inexistant.
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, keys=keys, vecs=vecs, version=np.str_(FORMAT_VERSION))
        tmp.replace(p)

    def compact(self, keep: Iterable[str]) -> "VectorStore":
        """Nouveau magasin restreint aux clés `keep` (celles encore utilisées)."""
        out = VectorStore(dim=self.dim)
        for k in keep:
            v = self.get(k)
            if v is not None:
                out.put(k, v)
        return out


def embed_cached(
    texts: Sequence[str],
    *,
    embedder: str,
    path: Path | str,
    embed_fn: Callable[[list[str]], np.ndarray],
    save: bool = True,
) -> tuple[np.ndarray, int]:
    """Embeddings de `texts`, ALIGNÉS sur l'ordre d'entrée, en ne calculant que les manquants.

    Renvoie `(matrice, n_calculés)`. `embed_fn` reçoit la liste des textes RÉELLEMENT
    manquants (dédupliqués) et doit renvoyer leurs vecteurs dans le même ordre.

    C'est le point d'entrée unique : l'appelant n'a pas à connaître le format du magasin.
    """
    store = VectorStore.load(path)
    keys = keys_for(texts, embedder)

    # Textes à calculer : manquants du magasin, DÉDUPLIQUÉS (deux claims identiques dans le
    # même lot ne doivent pas être embeddés deux fois).
    missing: dict[str, str] = {}
    for k, t in zip(keys, texts):
        if k not in store and k not in missing:
            missing[k] = t

    if missing:
        new_keys = list(missing.keys())
        new_vecs = np.asarray(embed_fn([missing[k] for k in new_keys]), dtype=np.float32)
        if len(new_vecs) != len(new_keys):
            raise ValueError(
                f"embed_fn a renvoyé {len(new_vecs)} vecteurs pour {len(new_keys)} textes"
            )
        store.put_many(new_keys, new_vecs)
        if save:
            store.save(path)

    if not texts:
        return np.zeros((0, store.dim or 1), dtype=np.float32), 0

    out = np.stack([store.get(k) for k in keys]).astype(np.float32)
    return out, len(missing)


def migrate_from_ordered(
    path: Path | str,
    texts: Sequence[str],
    vecs: np.ndarray,
    embedder: str,
) -> int:
    """Adopte un cache ORDONNÉ de l'ancien format (vecteurs alignés sur `texts`).

    Évite de re-payer l'embedding d'un corpus déjà calculé au moment de la bascule : à
    36 275 claims sur granddebat, la migration coûte une écriture de fichier au lieu d'une
    passe GPU complète. Renvoie le nombre de vecteurs adoptés.
    """
    vecs = np.asarray(vecs, dtype=np.float32)
    if len(texts) != len(vecs):
        return 0
    store = VectorStore.load(path)
    store.put_many(keys_for(texts, embedder), vecs)
    store.save(path)
    return len(texts)


__all__ = [
    "VectorStore", "embed_cached", "key_for", "keys_for",
    "migrate_from_ordered", "FORMAT_VERSION",
]
