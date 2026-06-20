import { create } from 'zustand';

/**
 * Minimal UI state for the batch swarm view. Kept deliberately generic (no
 * business-domain coupling) so the Phase 2 live wiring can extend it freely.
 *
 * - `selectedClusterId` drives the theme drill-down panel + swarm emphasis.
 * - `hoveredNodeId` is the node under the cursor (tooltip / soft highlight).
 */
interface GraphUIState {
  selectedClusterId: number | null;
  hoveredNodeId: string | null;
  selectCluster: (clusterId: number | null) => void;
  hover: (nodeId: string | null) => void;
}

export const useGraphStore = create<GraphUIState>((set) => ({
  selectedClusterId: null,
  hoveredNodeId: null,
  selectCluster: (clusterId) =>
    set((s) => ({
      // Click the already-open theme to close it (toggle).
      selectedClusterId: s.selectedClusterId === clusterId ? null : clusterId,
    })),
  hover: (nodeId) => set({ hoveredNodeId: nodeId }),
}));
