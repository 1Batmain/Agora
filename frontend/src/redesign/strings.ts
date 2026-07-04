/**
 * SINGLE SOURCE of FR copy for the dataset INDICES dashboard (labels + one-line
 * explanations), the noise-cluster label, and the app locale.
 *
 * The backend ships PURE DATA for indices (`{key, value, detail}`) and a MACHINE
 * key for the noise cluster (`__noise__`). All UI strings — and the number
 * formatting locale — live here so the copy is declared in ONE place (no `DICT`
 * duplicated inside `IndicesDashboard`).
 */

/** App-wide locale for `toLocaleString` (number/percent formatting). */
export const LOCALE = 'fr-FR';

/** Stable MACHINE key the backend emits for the HDBSCAN noise cluster's label. */
export const NOISE_KEY = '__noise__';
/** Localised label shown for the noise cluster (cf. `NOISE_KEY`). */
export const NOISE_LABEL = 'Non classé';

/**
 * key → human label (FR). Covers the derived indices ({key,value,detail} from
 * the backend) plus the loose record / mock keys, so every shape renders.
 */
export const INDEX_LABELS: Record<string, string> = {
  // Derived indices served by /analysis (data-only: {key, value, detail}).
  effusion: 'Effusion (variété des avis)',
  concentration: 'Concentration',
  consensus: 'Cohésion sémantique',
  structuration: 'Structuration',
  // Métriques de CONFIANCE servies au SERVE-TIME par /analysis (couverture + fidélité).
  couverture: 'Couverture',
  fidelite_verbatim: 'Fidélité verbatim',
  // Loose record / mock keys (permissive {key: number} shape).
  participants: 'Participants',
  n_participants: 'Participants',
  variete: 'Effusion (variété)',
  convergence: 'Convergence',
  convergence_cumulee: 'Convergence cumulée',
  'convergence_cumulée': 'Convergence cumulée',
  diversity: 'Diversité des opinions',
  diversite: 'Diversité des opinions',
  consensus_global: 'Consensus global',
  polarization: 'Polarisation',
  polarisation: 'Polarisation',
  coverage: 'Couverture',
  n_avis: 'Avis analysés',
  n_themes: 'Thèmes émergents',
  n_claims: 'Arguments extraits',
  n_clusters: 'Clusters',
};

/** Numbers an index `detail` may carry (used to rebuild the rich phrase). */
export type IndexDetail = Record<string, number> | null | undefined;

/**
 * Short static fallback hints — used for the loose record shape, which carries
 * NO `detail` (mock + permissive `{key: number}`). Reproduces the former DICT.
 */
const HINTS: Record<string, string> = {
  participants: 'Nombre de personnes ayant contribué à la consultation.',
  n_participants: 'Nombre de personnes ayant contribué à la consultation.',
  effusion: 'Variété des opinions exprimées (0 = unanime, 1 = très foisonnant).',
  variete: 'Variété des opinions exprimées (0 = unanime, 1 = très foisonnant).',
  convergence: 'Degré de convergence des idées exprimées.',
  convergence_cumulee:
    'Accord agrégé sur l’ensemble des contributions (1 = forte convergence).',
  'convergence_cumulée':
    'Accord agrégé sur l’ensemble des contributions (1 = forte convergence).',
  diversity: 'Variété des thèmes exprimés (0 = unanime, 1 = très éclaté).',
  diversite: 'Variété des thèmes exprimés (0 = unanime, 1 = très éclaté).',
  consensus:
    'Proximité des contributions d’un thème dans l’espace sémantique — PAS un accord d’opinion.',
  consensus_global:
    'Proximité des contributions d’un thème dans l’espace sémantique — PAS un accord d’opinion.',
  concentration: 'Part des avis captée par les plus gros thèmes (1 = très concentré).',
  polarization: 'Opposition entre pôles d’opinion.',
  polarisation: 'Opposition entre pôles d’opinion.',
  coverage: 'Part des avis rattachés à un thème.',
  couverture: 'Part des contributions rattachées à un thème (le reste = non classé).',
  fidelite_verbatim:
    'Part des arguments extraits qui sont des citations exactes (zéro trahison).',
  n_avis: 'Nombre total de contributions citoyennes.',
  n_themes: 'Nombre de thèmes de premier niveau.',
  n_claims: 'Nombre de prises de position verbatim.',
  n_clusters: 'Nombre de regroupements détectés.',
};

/**
 * One-line explanation for an index. When the backend `detail` carries the
 * numbers, rebuilds the RICH phrase (interpolated — IDENTICAL to the copy the
 * backend used to ship). Otherwise falls back to a short static hint.
 */
export function indexExplanation(key: string, detail?: IndexDetail): string | undefined {
  const d = detail || {};
  switch (key) {
    case 'effusion':
      if (typeof d.effective_themes === 'number' && typeof d.n_themes === 'number') {
        return (
          `Les avis se répartissent sur ~${d.effective_themes.toFixed(1)} sujets ` +
          `effectifs (sur ${d.n_themes} thèmes). Proche de 1 = parole foisonnante, ` +
          `voix équilibrées entre sujets ; proche de 0 = un sujet domine tout.`
        );
      }
      break;
    case 'concentration':
      if (typeof d.top_share === 'number') {
        return (
          `Le thème dominant capte ${Math.round(d.top_share * 100)} % des voix. ` +
          `Proche de 1 = débat accaparé par un sujet ; proche de 0 = dispersé.`
        );
      }
      break;
    case 'consensus':
      return (
        `À quel point les contributions d’un thème sont PROCHES dans l’espace ` +
        `sémantique — PAS un accord d’opinion. Pondérée par la population (les petits ` +
        `thèmes pèsent moins). Proche de 1 = thèmes très cohésifs ; proche de 0 = épars.`
      );
    case 'couverture':
      if (typeof d.classes === 'number' && typeof d.total === 'number') {
        return (
          `${Math.round((d.classes / Math.max(d.total, 1)) * 100)} % des ` +
          `${d.total.toLocaleString(LOCALE)} contributions sont rattachées à un thème ; ` +
          `${(d.noise ?? d.total - d.classes).toLocaleString(LOCALE)} restent non classées. ` +
          `Proche de 1 = la carte capte presque toute la parole.`
        );
      }
      break;
    case 'fidelite_verbatim':
      if (typeof d.n_verbatim === 'number' && typeof d.n_claims === 'number') {
        return (
          `${d.n_verbatim.toLocaleString(LOCALE)} arguments sur ` +
          `${d.n_claims.toLocaleString(LOCALE)} sont des citations EXACTES (tous les ` +
          `extraits + la cible ancrés au texte source). Proche de 1 = zéro trahison du verbatim.`
        );
      }
      break;
    case 'structuration':
      if (typeof d.share === 'number') {
        return (
          `${Math.round(d.share * 100)} % des voix relèvent de thèmes à facettes ` +
          `(subdivisés en sous-thèmes). Proche de 1 = sujets riches/complexes ; ` +
          `0 = débat plat.`
        );
      }
      break;
  }
  return HINTS[key];
}

/** Nettoie l'emphase markdown littérale (**gras**, *italique*, y compris non
 *  fermée en fin de titre) des chaînes affichées en texte brut — les artefacts
 *  bakés avant le strip backend en portent encore. */
export const stripMd = (s: string): string =>
  s.replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1').replace(/\*{1,3}/g, '').trim();
