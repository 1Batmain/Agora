import { useEffect, useState } from 'react';
import { fetchDatasets } from '../api';
import type { Dataset } from '../types';
import { Landing } from './Landing';
import { Participate } from './Participate';
import RedesignApp from './RedesignApp';

/** App-level route (no react-router needed): a flat state machine + active id. */
type Route = 'landing' | 'analysis' | 'participate';

/**
 * Shell d'Agora. La vue d'accueil est la LANDING (grille de consultations). Au
 * clic sur une carte :
 *  - consultation CLOSE  → vue d'analyse (carte/synthèses/avis) du dataset,
 *  - consultation OUVERTE → vue PARTICIPER (placeholder sujet + formulaire).
 * Un « ← Consultations » ramène à la landing dans les deux cas.
 */
export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [route, setRoute] = useState<Route>('landing');
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchDatasets()
      .catch(() => [] as Dataset[])
      .then((ds) => {
        if (cancelled) return;
        setDatasets(ds);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const active = datasets.find((d) => d.id === activeId) ?? null;

  function openConsultation(d: Dataset) {
    setActiveId(d.id);
    setRoute(d.status === 'open' ? 'participate' : 'analysis');
  }
  function backToLanding() {
    setRoute('landing');
    setActiveId(null);
  }

  if (route === 'analysis' && active) {
    return <RedesignApp initialDataset={active.id} onBack={backToLanding} />;
  }
  if (route === 'participate' && active) {
    return <Participate dataset={active} onBack={backToLanding} />;
  }
  return <Landing datasets={datasets} loading={loading} onOpen={openConsultation} />;
}
