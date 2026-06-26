import type { SpatialTheme } from './contract';
import { NOISE_KEY, NOISE_LABEL } from './strings';

/**
 * Canonical human caption for a theme — the SINGLE source of truth for what label
 * the UI shows (bubbles, tooltip, breadcrumb, insights/citations headers).
 *
 * Point 4 of the UX brief: the bubble/label is the LLM-generated `title`. We fall
 * back to the keyword stub `label` only while the backend hasn't emitted a title
 * yet, so the UI never shows an empty caption (graceful repli).
 *
 * The backend tags the noise cluster with a MACHINE key (`__noise__`); we localise
 * it here so the copy stays in `strings.ts` and the bubble reads naturally.
 */
export function themeCaption(t: Pick<SpatialTheme, 'title' | 'label'>): string {
  const caption = t.title?.trim() || t.label;
  return caption === NOISE_KEY ? NOISE_LABEL : caption;
}
