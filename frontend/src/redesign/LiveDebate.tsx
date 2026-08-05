import { useEffect, useRef, useState } from 'react';
import { Header } from './Header';
import {
  fetchLiveSnapshot,
  fetchLiveTranscript,
  type LiveSnapshot,
  type LiveTurn,
  type SpeakerRow,
} from './liveApi';

/** Cadence de rafraîchissement — le rejeu écrit un instantané par tour de parole. */
const POLL_MS = 2000;

/** `stime` est une DURÉE depuis l'ouverture de séance, pas une heure : on la formate en h:mm:ss. */
function elapsed(stime: number | null): string {
  if (stime == null) return '—';
  const s = Math.max(0, Math.round(stime));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}:${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

const pct = (x: number | null | undefined) =>
  x == null ? '—' : `${Math.round(x * 100)} %`;

/** Barre pour / contre / nuance. Réutilise les couleurs de `OpinionBar` (mêmes conventions). */
function StanceBar({ fav, def, nu }: { fav: number; def: number; nu: number }) {
  const total = fav + def + nu;
  if (!total) return <span className="live__muted">—</span>;
  const w = (x: number) => `${((100 * x) / total).toFixed(1)}%`;
  return (
    <span
      className="live__bar"
      title={`${fav} pour · ${def} contre · ${nu} nuance`}
      aria-label={`${fav} pour, ${def} contre, ${nu} nuance`}
    >
      <i className="live__seg live__seg--fav" style={{ width: w(fav) }} />
      <i className="live__seg live__seg--def" style={{ width: w(def) }} />
      <i className="live__seg live__seg--nu" style={{ width: w(nu) }} />
    </span>
  );
}

function positionClass(p: SpeakerRow['position']): string {
  if (p === 'favorable') return 'live__pill--fav';
  if (p === 'défavorable') return 'live__pill--def';
  return 'live__pill--nu';
}

/**
 * Vue LIVE — un débat rejoué, et la synthèse qui se construit dessus.
 *
 * À gauche le flux ENTRANT (ce qui est dit), à droite la sortie du pipeline (thèmes,
 * positions). Les séparer est le but de la page : on veut VOIR le décalage entre la parole
 * et ce que la machine en fait.
 *
 * ⚠️ Vue de RECHERCHE, pas un produit. Trois réserves affichées dans la page elle-même,
 * parce qu'elles changent la lecture des chiffres :
 *   1. une position ne vient JAMAIS d'une proximité de texte, toujours d'un jugement modèle ;
 *   2. le discours rapporté (citer l'adversaire pour le réfuter) peut inverser une position,
 *      et le garde-fou n'est pas encore validé par un banc ;
 *   3. une part des thèmes est procédurale (joute parlementaire) : mesuré, 47 % des claims
 *      classés y tombent, et les positions qu'on y lit n'ont pas de sens.
 */
export function LiveDebate({ onHome, onAbout }: { onHome?: () => void; onAbout?: () => void }) {
  const [snap, setSnap] = useState<LiveSnapshot | null>(null);
  const [turns, setTurns] = useState<LiveTurn[]>([]);
  const [online, setOnline] = useState<boolean | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      const [s, t] = await Promise.all([fetchLiveSnapshot(), fetchLiveTranscript(25)]);
      if (cancelled) return;
      setOnline(s != null);
      if (s) setSnap(s);
      if (t) setTurns(t.turns);
    };
    tick();
    timer.current = window.setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      if (timer.current) window.clearInterval(timer.current);
    };
  }, []);

  const themeLabel = new Map((snap?.themes ?? []).map((t) => [t.id, t.label]));
  const activeThemes = (snap?.theme_opinion ?? []).filter((t) => t.n_claims > 0);
  const drift = snap?.drift;

  return (
    <div className="agora live">
      <Header onHome={onHome} onAbout={onAbout} />

      <div className="live__body">
        <div className="live__topbar">
          <div>
            <h1 className="live__title">Synthèse live d'un débat</h1>
            <p className="live__topic">
              {snap?.topic || 'Aucun rejeu en cours'}
              <span className="live__exp">expérimental</span>
            </p>
          </div>
          <div className="live__stats">
            <span>
              tours <strong>{snap?.turns_seen ?? 0}</strong>
            </span>
            <span>
              claims <strong>{snap?.claims_seen ?? 0}</strong>
            </span>
            <span>
              thèmes <strong>{snap?.themes.length ?? 0}</strong>
            </span>
            <span>
              hors thèmes <strong>{pct(drift?.recent)}</strong>
            </span>
            <span className={online ? 'live__dot live__dot--on' : 'live__dot'}>
              {online === null ? 'connexion…' : online ? 'en direct' : 'hors ligne'}
            </span>
          </div>
        </div>

        {online === false && (
          <div className="live__offline">
            <strong>Aucun instantané disponible.</strong> Lance un rejeu&nbsp;:
            <code>python -m live.replay data/raw/an/&lt;séance&gt;.xml --speed 60 --bootstrap 30</code>
            puis le serveur live <code>uvicorn live.server:app --port 8020</code>.
          </div>
        )}

        <div className="live__grid">
          <section className="live__panel">
            <h2 className="live__h2">Ce qui est dit — flux entrant</h2>
            <div className="live__scroll">
              {turns.length === 0 ? (
                <p className="live__muted">En attente du rejeu…</p>
              ) : (
                [...turns].reverse().map((t) => (
                  <article key={t.id} className="live__turn">
                    <div className="live__turn-head">
                      <span className="live__who">{t.speaker}</span>
                      <span className="live__when">{elapsed(t.stime)}</span>
                    </div>
                    <p className="live__said">
                      {t.text.slice(0, 280)}
                      {t.text.length > 280 ? '…' : ''}
                    </p>
                  </article>
                ))
              )}
            </div>
          </section>

          <div className="live__col">
            <section className="live__panel">
              <h2 className="live__h2">Positions par thème</h2>
              {activeThemes.length === 0 ? (
                <p className="live__muted">Aucun claim rattaché pour l'instant.</p>
              ) : (
                <div className="live__tablewrap">
                  <table className="live__table">
                    <thead>
                      <tr>
                        <th>Thème et objet de clivage</th>
                        <th className="live__num">claims</th>
                        <th>pour / contre</th>
                      </tr>
                    </thead>
                    <tbody>
                      {activeThemes.map((t) => (
                        <tr key={t.theme_id}>
                          <td>
                            <strong>{t.label}</strong>
                            <span className="live__cleavage">{t.cleavage}</span>
                          </td>
                          <td className="live__num">{t.n_claims}</td>
                          <td>
                            <StanceBar fav={t.favorable} def={t.defavorable} nu={t.nuance} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {drift && (
                <p className="live__note">
                  {drift.excess > 0.25 ? (
                    <>
                      <strong className="live__warn">
                        Dérive thématique : {pct(drift.excess)} au-delà du bruit de fond.
                      </strong>{' '}
                      Des sujets échappent aux thèmes gelés — une restructuration se justifierait.
                    </>
                  ) : (
                    <>
                      Hors thèmes : {pct(drift.recent)} des claims récents, pour un bruit de fond
                      attendu de {pct(drift.baseline)}. <strong>Un taux brut ne s'interprète
                      pas</strong> : dans un débat parlementaire, l'essentiel des claims hors
                      thème n'est pas un sujet nouveau mais de la joute. Seul l'excès signale.
                    </>
                  )}
                </p>
              )}
            </section>

            <section className="live__panel">
              <h2 className="live__h2">Positions par orateur</h2>
              {(snap?.speakers ?? []).length === 0 ? (
                <p className="live__muted">Pas encore assez de prises de parole jugées.</p>
              ) : (
                <div className="live__tablewrap">
                  <table className="live__table">
                    <thead>
                      <tr>
                        <th>Orateur</th>
                        <th>Thème</th>
                        <th>Position</th>
                        <th className="live__num">claims</th>
                        <th>pour / contre</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(snap?.speakers ?? []).map((s) => (
                        <tr key={`${s.speaker_id}-${s.theme_id}`}>
                          <td>{s.speaker}</td>
                          <td className="live__muted">
                            {themeLabel.get(s.theme_id) ?? s.theme_id}
                          </td>
                          <td>
                            <span className={`live__pill ${positionClass(s.position)}`}>
                              {s.position}
                            </span>
                          </td>
                          <td className="live__num">{s.n_claims}</td>
                          <td>
                            <StanceBar fav={s.favorable} def={s.defavorable} nu={s.nuance} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <p className="live__note">
                Une position n'est <strong>jamais</strong> déduite d'une proximité de texte :
                elle vient d'un jugement modèle sur une proposition débattable. Mesuré, le
                clustering ne recouvre le clivage qu'à NMI&nbsp;≈&nbsp;0,05 — l'embedding capte
                le sujet, pas la position.
              </p>
              <p className="live__note live__note--warn">
                <strong>Réserves.</strong> Le discours rapporté (citer l'adversaire pour le
                réfuter) peut inverser une position, et le garde-fou n'est pas encore validé par
                un banc. Par ailleurs, une partie des thèmes relève de la joute parlementaire
                plutôt que du fond&nbsp;: mesuré, 47&nbsp;% des claims classés y tombent, et les
                positions qu'on y lit n'ont pas de sens.
              </p>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
