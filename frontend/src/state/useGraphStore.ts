import { create } from 'zustand';

/**
 * Minimal UI state for the batch swarm view. Kept deliberately generic (no
 * business-domain coupling) so the Phase 2 live wiring can extend it freely.
 *
 * The drill-down is a macro→sub tree:
 * - `expandedMacroId` is the open macro-theme in the panel; it also drives the
 *   swarm emphasis (highlight every node whose `macro_id` matches) until a
 *   sub-theme is picked.
 * - `selectedClusterId` is the open LEAF (sub-theme): its member ideas are
 *   listed and the swarm narrows the emphasis to that single sub-community.
 * - `hoveredNodeId` is the node under the cursor (tooltip / soft highlight).
 */
interface GraphUIState {
  expandedMacroId: number | null;
  selectedClusterId: number | null;
  hoveredNodeId: string | null;
  toggleMacro: (macroId: number) => void;
  selectCluster: (clusterId: number | null) => void;
  /** Click a node → open its macro then its sub-theme. */
  focusNode: (macroId: number | null, clusterId: number) => void;
  hover: (nodeId: string | null) => void;
}

export const useGraphStore = create<GraphUIState>((set) => ({
  expandedMacroId: null,
  selectedClusterId: null,
  hoveredNodeId: null,
  toggleMacro: (macroId) =>
    set((s) => {
      // Toggle the macro; collapsing it also clears any selected sub-theme.
      const open = s.expandedMacroId === macroId;
      return {
        expandedMacroId: open ? null : macroId,
        selectedClusterId: open ? null : s.selectedClusterId,
      };
    }),
  selectCluster: (clusterId) =>
    set((s) => ({
      // Click the already-open sub-theme to close it (toggle).
      selectedClusterId: s.selectedClusterId === clusterId ? null : clusterId,
    })),
  focusNode: (macroId, clusterId) =>
    set({ expandedMacroId: macroId, selectedClusterId: clusterId }),
  hover: (nodeId) => set({ hoveredNodeId: nodeId }),
}));
