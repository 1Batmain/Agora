"""Cache d'embeddings adressé par contenu — le verrou de l'incrémental.

L'ancien cache empreintait la CONCATÉNATION de tous les textes : un claim nouveau
invalidait les 36 275 vecteurs de granddebat. Inacceptable pour le pipeline live, où un
tour de parole arrive toutes les ~50 secondes.

Ces tests verrouillent les quatre propriétés qui font la différence (incrémentalité,
insensibilité à l'ordre, dédup, isolation par embedder) plus la migration sans re-calcul.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.embed import vector_store as VS


class Compteur:
    """Faux embedder : renvoie un vecteur déterministe par texte et COMPTE les appels.

    Déterministe par texte (pas par ordre d'appel) — sinon on ne saurait pas distinguer
    « valeur reprise du cache » de « valeur recalculée à l'identique ».
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.batches: list[list[str]] = []

    @property
    def n_embedded(self) -> int:
        return sum(len(b) for b in self.batches)

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.batches.append(list(texts))
        return np.stack([self._vec(t) for t in texts])

    def _vec(self, text: str) -> np.ndarray:
        seed = abs(hash(text)) % (2**31)
        return np.random.RandomState(seed).rand(self.dim).astype(np.float32)


# --------------------------------------------------------------------------------- #
# 1. Incrémentalité — la raison d'être du module
# --------------------------------------------------------------------------------- #

def test_un_texte_nouveau_ne_coute_quun_embedding(tmp_path):
    p = tmp_path / "s.npz"
    emb = Compteur()
    VS.embed_cached(["a", "b", "c"], embedder="e", path=p, embed_fn=emb)
    assert emb.n_embedded == 3

    VS.embed_cached(["a", "b", "c", "d"], embedder="e", path=p, embed_fn=emb)
    assert emb.batches[-1] == ["d"], "seul le texte NOUVEAU doit être recalculé"
    assert emb.n_embedded == 4


def test_rien_de_nouveau_ne_coute_rien(tmp_path):
    p = tmp_path / "s.npz"
    emb = Compteur()
    VS.embed_cached(["a", "b"], embedder="e", path=p, embed_fn=emb)
    _, n = VS.embed_cached(["a", "b"], embedder="e", path=p, embed_fn=emb)
    assert n == 0 and emb.n_embedded == 2


def test_les_vecteurs_repris_sont_identiques(tmp_path):
    p = tmp_path / "s.npz"
    emb = Compteur()
    v1, _ = VS.embed_cached(["a", "b", "c"], embedder="e", path=p, embed_fn=emb)
    v2, _ = VS.embed_cached(["a", "b", "c", "d"], embedder="e", path=p, embed_fn=emb)
    assert np.allclose(v1, v2[:3])


# --------------------------------------------------------------------------------- #
# 2. Ordre — l'ancien cache tombait sur une simple permutation
# --------------------------------------------------------------------------------- #

def test_une_permutation_ne_coute_rien(tmp_path):
    """Même défaut que l'avalanche de re-titrage : une clé sensible à l'ordre régénère tout
    alors que RIEN n'a changé. Ici, l'ordre est structurellement hors de la clé."""
    p = tmp_path / "s.npz"
    emb = Compteur()
    VS.embed_cached(["a", "b", "c"], embedder="e", path=p, embed_fn=emb)
    _, n = VS.embed_cached(["c", "a", "b"], embedder="e", path=p, embed_fn=emb)
    assert n == 0


def test_la_sortie_suit_lordre_dentree(tmp_path):
    p = tmp_path / "s.npz"
    emb = Compteur()
    droit, _ = VS.embed_cached(["a", "b", "c"], embedder="e", path=p, embed_fn=emb)
    permute, _ = VS.embed_cached(["c", "a", "b"], embedder="e", path=p, embed_fn=emb)
    assert np.allclose(permute[0], droit[2])
    assert np.allclose(permute[1], droit[0])
    assert np.allclose(permute[2], droit[1])


# --------------------------------------------------------------------------------- #
# 3. Dédup et embedder
# --------------------------------------------------------------------------------- #

def test_textes_identiques_embeddes_une_seule_fois(tmp_path):
    p = tmp_path / "s.npz"
    emb = Compteur()
    vecs, n = VS.embed_cached(["même", "même", "même"], embedder="e", path=p, embed_fn=emb)
    assert n == 1 and emb.n_embedded == 1
    assert vecs.shape[0] == 3 and np.allclose(vecs[0], vecs[2])


def test_changer_dembedder_manque_le_cache(tmp_path):
    """Deux embedders vivent dans des espaces INCOMPARABLES : réutiliser serait un bug
    silencieux (cf. le garde-dimension de backend/submissions.py, posé après régression)."""
    p_a, p_b = tmp_path / "a.npz", tmp_path / "b.npz"
    emb = Compteur()
    VS.embed_cached(["a"], embedder="arctic-l", path=p_a, embed_fn=emb)
    _, n = VS.embed_cached(["a"], embedder="nomic-v2", path=p_b, embed_fn=emb)
    assert n == 1
    assert VS.key_for("a", "arctic-l") != VS.key_for("a", "nomic-v2")


def test_dimension_incompatible_leve(tmp_path):
    store = VS.VectorStore()
    store.put("k1", np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        store.put("k2", np.zeros(16, dtype=np.float32))


# --------------------------------------------------------------------------------- #
# 4. Persistance et robustesse
# --------------------------------------------------------------------------------- #

def test_le_magasin_survit_au_disque(tmp_path):
    p = tmp_path / "s.npz"
    emb = Compteur()
    v1, _ = VS.embed_cached(["a", "b"], embedder="e", path=p, embed_fn=emb)
    rechargé = VS.VectorStore.load(p)
    assert len(rechargé) == 2
    assert np.allclose(rechargé.get(VS.key_for("a", "e")), v1[0])


def test_un_cache_abime_ne_fait_pas_planter(tmp_path):
    """Un cache est une OPTIMISATION : le perdre doit coûter du calcul, jamais une panne."""
    p = tmp_path / "s.npz"
    p.write_bytes(b"ceci n'est pas un npz")
    store = VS.VectorStore.load(p)
    assert len(store) == 0

    emb = Compteur()
    _, n = VS.embed_cached(["a"], embedder="e", path=p, embed_fn=emb)
    assert n == 1


def test_fichier_absent_donne_un_magasin_vide(tmp_path):
    assert len(VS.VectorStore.load(tmp_path / "jamais-ecrit.npz")) == 0


def test_liste_vide(tmp_path):
    vecs, n = VS.embed_cached([], embedder="e", path=tmp_path / "s.npz",
                              embed_fn=Compteur())
    assert n == 0 and vecs.shape[0] == 0


def test_embed_fn_incoherent_leve(tmp_path):
    """Un embedder qui renvoie le mauvais nombre de vecteurs doit LEVER : sans ce garde,
    l'alignement claim↔vecteur casserait en silence, et tout le clustering avec."""
    with pytest.raises(ValueError, match="vecteurs"):
        VS.embed_cached(["a", "b"], embedder="e", path=tmp_path / "s.npz",
                        embed_fn=lambda ts: np.zeros((1, 8), dtype=np.float32))


def test_compact_ne_garde_que_lutile(tmp_path):
    p = tmp_path / "s.npz"
    emb = Compteur()
    VS.embed_cached(["a", "b", "c"], embedder="e", path=p, embed_fn=emb)
    store = VS.VectorStore.load(p)
    réduit = store.compact([VS.key_for("a", "e")])
    assert len(réduit) == 1 and VS.key_for("b", "e") not in réduit


# --------------------------------------------------------------------------------- #
# 5. Migration depuis l'ancien format — sans re-calculer
# --------------------------------------------------------------------------------- #

def test_migration_adopte_un_cache_ordonne_sans_recalcul(tmp_path):
    """Sans migration, la bascule imposerait de ré-embedder tous les corpus déjà bâtis."""
    p = tmp_path / "s.npz"
    textes = ["a", "b", "c"]
    anciens = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)

    assert VS.migrate_from_ordered(p, textes, anciens, "e") == 3

    emb = Compteur(dim=4)
    vecs, n = VS.embed_cached(textes, embedder="e", path=p, embed_fn=emb)
    assert n == 0, "après migration, plus rien à embedder"
    assert np.allclose(vecs, anciens), "les vecteurs d'origine sont conservés à l'identique"


def test_migration_refuse_un_cache_desaligne(tmp_path):
    """Longueurs incohérentes ⇒ on n'adopte RIEN plutôt que d'aligner de travers."""
    assert VS.migrate_from_ordered(tmp_path / "s.npz", ["a", "b"],
                                   np.zeros((3, 4), dtype=np.float32), "e") == 0


def test_migration_du_format_herite_par_lendpoint(tmp_path):
    """Bout en bout : un `claims_emb.npz` de l'ANCIEN format est adopté, pas recalculé."""
    from backend.claims_endpoint import _emb_fingerprint, _migrate_legacy_emb_cache

    p = tmp_path / "claims_emb.npz"
    textes = ["premier claim", "second claim"]
    vecs = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    with open(p, "wb") as fh:                      # format hérité : vecs + empreinte globale
        np.savez(fh, vecs=vecs,
                 fingerprint=np.str_(_emb_fingerprint("arctic-l", textes)))

    assert _migrate_legacy_emb_cache(p, textes, "arctic-l") == 2

    emb = Compteur(dim=4)
    out, n = VS.embed_cached(textes, embedder="arctic-l", path=p, embed_fn=emb)
    assert n == 0 and np.allclose(out, vecs)


def test_un_cache_herite_perime_nest_pas_adopte(tmp_path):
    """Empreinte qui ne correspond plus ⇒ les vecteurs ne sont PAS ceux de ces textes."""
    from backend.claims_endpoint import _migrate_legacy_emb_cache

    p = tmp_path / "claims_emb.npz"
    with open(p, "wb") as fh:
        np.savez(fh, vecs=np.zeros((2, 4), dtype=np.float32),
                 fingerprint=np.str_("une-empreinte-qui-ne-correspond-pas"))
    assert _migrate_legacy_emb_cache(p, ["autre", "textes"], "arctic-l") == 0
