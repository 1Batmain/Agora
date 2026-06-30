/**
 * Dérive un NUAGE de points 2D depuis la grille de densité PRÉCALCULÉE (`/density`,
 * KDE sur la projection UMAP cachée `umap2d.npy`). C'est la même source de cache que
 * le « Paysage 3D » — donc ZÉRO calcul à la requête et AUCUN appel `/recluster` : le
 * front se contente de relire le cache de densité et de l'échantillonner en points.
 *
 * Chaque cellule de la grille reçoit un nombre de points ∝ sa densité (budget total
 * `TARGET`), jittérés dans la cellule (jitter déterministe → rendu stable). La couleur
 * suit la rampe Bleu France par hauteur (cohérente avec Density3D : vallées claires →
 * pics foncés). Le résultat alimente `Scatter2D` (vue de dessus de la distribution).
 */
import type { DensityPayload } from './densityApi';
import type { ScatterPoint } from './Scatter2D';

// Rampe Bleu France : vallées #ececfe → pics #000091 (identique à Density3D).
const LOW = [0xec, 0xec, 0xfe];
const HIGH = [0x00, 0x00, 0x91];
// Budget de points du nuage : assez dense pour lire les amas, assez léger pour le canvas.
const TARGET = 2800;

/** Couleur de la rampe pour une hauteur normalisée `h` ∈ [0, 1]. */
function ramp(h: number): string {
  const t = Math.max(0, Math.min(1, h));
  const r = Math.round(LOW[0] + (HIGH[0] - LOW[0]) * t);
  const g = Math.round(LOW[1] + (HIGH[1] - LOW[1]) * t);
  const b = Math.round(LOW[2] + (HIGH[2] - LOW[2]) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

/** Pseudo-aléa déterministe ∈ [0, 1) à partir d'un entier (jitter reproductible). */
function rnd(seed: number): number {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

/**
 * Échantillonne la grille de densité en un nuage de points (vue UMAP 2D de dessus).
 * Renvoie `[]` si le payload est absent/vide → l'appelant grise proprement l'option.
 */
export function densityScatterPoints(payload: DensityPayload | null): ScatterPoint[] {
  if (!payload) return [];
  const { nx, nz, x_range, z_range, heights, zmax } = payload;
  const norm = zmax > 0 ? zmax : 1;

  let total = 0;
  for (let iz = 0; iz < nz; iz++) {
    for (let ix = 0; ix < nx; ix++) total += Math.max(0, heights[iz]?.[ix] ?? 0);
  }
  if (total <= 0) return [];

  const dx = (x_range[1] - x_range[0]) / Math.max(1, nx - 1);
  const dz = (z_range[1] - z_range[0]) / Math.max(1, nz - 1);
  const pts: ScatterPoint[] = [];
  let seed = 1;
  for (let iz = 0; iz < nz; iz++) {
    for (let ix = 0; ix < nx; ix++) {
      const raw = Math.max(0, heights[iz]?.[ix] ?? 0);
      if (raw <= 0) continue;
      const count = Math.round((TARGET * raw) / total);
      if (count <= 0) continue;
      const color = ramp(raw / norm);
      const cx = x_range[0] + ix * dx;
      const cz = z_range[0] + iz * dz;
      for (let k = 0; k < count; k++) {
        pts.push({
          x: cx + (rnd(seed++) - 0.5) * dx,
          z: cz + (rnd(seed++) - 0.5) * dz,
          cluster_id: null,
          color,
        });
      }
    }
  }
  return pts;
}
