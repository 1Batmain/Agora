"""Moteur de REJEU — déroule une séance à sa cadence réelle, ou accélérée.

Le corpus porte un timecode par prise de parole (`stime`, en secondes depuis l'ouverture,
90,4 % de couverture) : on peut donc rejouer un débat **à sa cadence exacte**, sans simuler
d'horloge. Mesuré : ~330 tours sur ~5 h, soit un tour toutes les ~50 s, contre quelques
secondes de traitement — environ 10× de marge.

Le rejeu écrit un INSTANTANÉ après chaque tour. C'est le contrat avec l'affichage : le
serveur ne fait que lire ce fichier, il ne calcule rien. Écriture atomique, donc un lecteur
ne voit jamais un demi-état.

Usage :
    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra contender --extra embed-contender \\
      --extra faiss python -m live.replay data/an/*.xml --speed 60 --bootstrap 40
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from live.pipeline import LiveConfig, bootstrap, process_turn
from live.state import LiveState
from live.transcript import Turn, parse_session, read_session_meta


@dataclass
class ReplayConfig:
    """`speed` = facteur d'accélération ; 0 (ou None) → aussi vite que possible.

    `max_sleep` borne l'attente entre deux tours : une suspension de séance d'une heure ne
    doit pas figer la démo. C'est un paramètre de CONFORT du rejeu, pas une propriété des
    données — d'où la trace explicite quand il mord.
    """

    speed: float = 60.0
    max_sleep: float = 5.0
    snapshot_path: Path = Path("var/live-cache/snapshot.json")
    transcript_path: Path = Path("var/live-cache/transcript.jsonl")


def _wait(prev: Turn | None, cur: Turn, cfg: ReplayConfig, log) -> None:
    if not cfg.speed or prev is None:
        return
    if prev.stime is None or cur.stime is None:
        return
    delta = (cur.stime - prev.stime) / cfg.speed
    if delta <= 0:
        return
    if delta > cfg.max_sleep:
        log(f"[replay] attente {delta:.0f}s bornée à {cfg.max_sleep}s "
            f"(suspension de séance ?)")
        delta = cfg.max_sleep
    time.sleep(delta)


def append_transcript(turn: Turn, path: Path) -> None:
    """Journal du flux ENTRANT, distinct de la sortie du pipeline.

    L'affichage montre les deux côte à côte : ce qui a été dit, et ce que le pipeline en a
    fait. Les séparer permet de voir le décalage — c'est précisément ce qu'on veut observer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "id": turn.id, "seq": turn.seq, "stime": turn.stime,
            "speaker": turn.speaker, "speaker_id": turn.speaker_id,
            "text": turn.text, "n_reactions": len(turn.reactions),
        }, ensure_ascii=False) + "\n")


def replay(turns: list[Turn], live_cfg: LiveConfig, replay_cfg: ReplayConfig, *,
           n_bootstrap: int, question: str | None = None, session: str = "",
           topic: str = "", log=print) -> LiveState:
    """Amorce sur les `n_bootstrap` premiers tours, puis déroule le reste en direct."""
    if len(turns) <= n_bootstrap:
        raise ValueError(
            f"{len(turns)} tours pour un amorçage de {n_bootstrap} — il n'en resterait "
            "aucun à rejouer."
        )
    head, tail = turns[:n_bootstrap], turns[n_bootstrap:]

    replay_cfg.transcript_path.parent.mkdir(parents=True, exist_ok=True)
    replay_cfg.transcript_path.write_text("", encoding="utf-8")   # repart d'un flux vierge
    for t in head:
        append_transcript(t, replay_cfg.transcript_path)

    state = bootstrap(head, live_cfg, question=question, session=session, topic=topic, log=log)
    state.write_snapshot(replay_cfg.snapshot_path)
    log(f"[replay] amorçage terminé · {len(state.themes)} thèmes · "
        f"{len(tail)} tours à rejouer (×{replay_cfg.speed or '∞'})")

    prev: Turn | None = head[-1] if head else None
    t0 = time.time()
    for i, turn in enumerate(tail, 1):
        _wait(prev, turn, replay_cfg, log)
        append_transcript(turn, replay_cfg.transcript_path)
        try:
            process_turn(state, turn, live_cfg, question=question, log=log)
        except Exception as exc:                      # un tour raté ne casse pas la séance
            log(f"[replay] ⚠️ tour {turn.seq} échoué ({type(exc).__name__}: {exc}) — on continue")
        state.write_snapshot(replay_cfg.snapshot_path)
        if i % 10 == 0:
            d = state.drift()
            log(f"[replay] {i}/{len(tail)} tours · {state.claims_seen} claims · "
                f"dérive récente {d['recent']:.0%} · {time.time()-t0:.0f}s écoulées")
        prev = turn

    log(f"[replay] terminé · {state.turns_seen} tours · {state.claims_seen} claims · "
        f"{time.time()-t0:.0f}s")
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Rejoue une séance dans le pipeline live.")
    ap.add_argument("xml", nargs="+", help="compte(s) rendu(s) XML de séance")
    ap.add_argument("--speed", type=float, default=60.0,
                    help="facteur d'accélération (0 = aussi vite que possible)")
    ap.add_argument("--bootstrap", type=int, default=40, help="tours d'amorçage")
    ap.add_argument("--limit", type=int, default=0, help="borne le nombre de tours rejoués")
    ap.add_argument("--profile", default="live", help="profil de pipeline")
    ap.add_argument("--cache", default="var/live-cache", help="dossier de travail live")
    ap.add_argument("--no-stance", action="store_true",
                    help="assignation seule (aucun appel de stance) — utile pour un essai à blanc")
    args = ap.parse_args()

    turns: list[Turn] = []
    topics: list[str] = []
    for src in args.xml:
        turns.extend(parse_session(src))
        meta = read_session_meta(src)
        topics.extend(meta.topics[:1])
    turns.sort(key=lambda t: (t.session, t.seq))
    if args.limit:
        turns = turns[:args.limit]

    cache = Path(args.cache)
    live_cfg = LiveConfig(profile_name=args.profile, cache_dir=cache,
                          stance_enabled=not args.no_stance)
    replay_cfg = ReplayConfig(speed=args.speed,
                              snapshot_path=cache / "snapshot.json",
                              transcript_path=cache / "transcript.jsonl")

    topic = topics[0] if topics else ""
    question = f"Faut-il adopter : {topic} ?" if topic else None
    print(f"[replay] {len(turns)} tours · sujet « {topic } » · profil {args.profile}"
          f"{' · SANS stance' if args.no_stance else ''}", flush=True)

    replay(turns, live_cfg, replay_cfg, n_bootstrap=args.bootstrap, question=question,
           session=turns[0].session if turns else "", topic=topic)


if __name__ == "__main__":
    main()
