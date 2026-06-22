import type { SpatialTheme } from './contract';

/**
 * Canonical human caption for a theme — the SINGLE source of truth for what label
 * the UI shows (bubbles, tooltip, breadcrumb, insights/citations headers).
 *
 * Point 4 of the UX brief: the bubble/label is the LLM-generated `title`. We fall
 * back to the keyword stub `label` only while the backend hasn't emitted a title
 * yet, so the UI never shows an empty caption (graceful repli).
 */
export function themeCaption(t: Pick<SpatialTheme, 'title' | 'label'>): string {
  return t.title?.trim() || t.label;
}
