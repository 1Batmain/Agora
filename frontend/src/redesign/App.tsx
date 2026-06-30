import { useCallback, useEffect, useState } from 'react';
import { fetchDatasets } from '../api';
import type { Consultation } from './contract';
import { Landing } from './Landing';
import { Participate } from './Participate';
import { ConsultationOverview } from './ConsultationOverview';
import { AvisExplorer } from './AvisExplorer';
import RedesignApp from './RedesignApp';
import { Console } from './Console';
import { TodoPage } from './TodoPage';

/** App-level route (no react-router needed): a flat state machine + active id. */
type Route = 'landing' | 'overview' | 'analysis' | 'participate' | 'console' | 'avis' | 'todo';
type HistState = { route: Route; activeId: string | null; focus?: string | null };

/**
 * Shell d'Agora. La vue d'accueil est la LANDING (grille de consultations). Au clic
 * sur une carte : consultation CLOSE → vue d'analyse ; OUVERTE → vue PARTICIPER.
 *
 * Navigation câblée sur l'History API : chaque ouverture/retour fait un `pushState`
 * (URL `?c=<id>`), et le bouton RETOUR du navigateur (`popstate`) restaure la vue
 * précédente AU LIEU de quitter le site. Deep-link `?c=<id>` au chargement.
 */
export default function App() {
  const [datasets, setDatasets] = useState<Consultation[]>([]);
  const [loading, setLoading] = useState(true);
  const [route, setRoute] = useState<Route>('landing');
  const [activeId, setActiveId] = useState<string | null>(null);
  // Avis focalisé sur la page d'exploration (`view=avis&focus=`), sinon null.
  const [focusAvis, setFocusAvis] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchDatasets()
      .catch(() => [] as Consultation[])
      .then((ds) => {
        if (cancelled) return;
        setDatasets(ds);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Retour/avance NAVIGATEUR → restaure la vue depuis l'état d'historique (pas de push).
  useEffect(() => {
    const onPop = (e: PopStateEvent) => {
      const st = (e.state ?? null) as HistState | null;
      setRoute(st?.route ?? 'landing');
      setActiveId(st?.activeId ?? null);
      setFocusAvis(st?.focus ?? null);
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  // Deep-link initial : `?c=<id>` ouvre la consultation (une fois les datasets chargés).
  useEffect(() => {
    if (loading) return;
    const params = new URLSearchParams(window.location.search);
    // Feuille de route : `?view=todo` (sans dataset) → page /todo directe.
    if (params.get('view') === 'todo' && !params.get('c')) {
      setRoute('todo');
      setActiveId(null);
      setFocusAvis(null);
      window.history.replaceState({ route: 'todo', activeId: null } as HistState, '', '?view=todo');
      return;
    }
    const cid = params.get('c');
    const d = cid ? datasets.find((x) => x.id === cid) : null;
    if (d) {
      // `?view=console` → Console (closed datasets only) ; `?view=avis(&focus=)` → exploration des avis.
      const view = params.get('view');
      const wantConsole = view === 'console' && d.status !== 'open';
      const wantAvis = view === 'avis' && d.status !== 'open';
      const focus = wantAvis ? params.get('focus') : null;
      const r: Route = wantConsole
        ? 'console'
        : wantAvis
          ? 'avis'
          : d.status === 'open'
            ? 'participate'
            : 'overview';
      setRoute(r);
      setActiveId(d.id);
      setFocusAvis(focus);
      const url = wantConsole
        ? `?c=${d.id}&view=console`
        : wantAvis
          ? `?c=${d.id}&view=avis${focus ? `&focus=${encodeURIComponent(focus)}` : ''}`
          : `?c=${d.id}`;
      window.history.replaceState({ route: r, activeId: d.id, focus } as HistState, '', url);
    } else {
      window.history.replaceState({ route: 'landing', activeId: null } as HistState, '', window.location.pathname);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading]);

  const active = datasets.find((d) => d.id === activeId) ?? null;

  const openConsultation = useCallback((d: Consultation) => {
    const r: Route = d.status === 'open' ? 'participate' : 'overview';
    setActiveId(d.id);
    setRoute(r);
    window.history.pushState({ route: r, activeId: d.id } as HistState, '', `?c=${d.id}`);
  }, []);

  const viewGraph = useCallback((id: string) => {
    setActiveId(id);
    setRoute('analysis');
    window.history.pushState({ route: 'analysis', activeId: id } as HistState, '', `?c=${id}&g=1`);
  }, []);

  // Entrée depuis une citation de la synthèse : page d'exploration FOCALISÉE sur l'avis.
  const exploreAvis = useCallback((datasetId: string, avisId: string | null) => {
    setActiveId(datasetId);
    setRoute('avis');
    setFocusAvis(avisId);
    const url = `?c=${datasetId}&view=avis${avisId ? `&focus=${encodeURIComponent(avisId)}` : ''}`;
    window.history.pushState(
      { route: 'avis', activeId: datasetId, focus: avisId } as HistState,
      '',
      url,
    );
  }, []);

  const openConsole = useCallback((id: string) => {
    setActiveId(id);
    setRoute('console');
    window.history.pushState({ route: 'console', activeId: id } as HistState, '', `?c=${id}&view=console`);
  }, []);

  const backToLanding = useCallback(() => {
    setRoute('landing');
    setActiveId(null);
    window.history.pushState({ route: 'landing', activeId: null } as HistState, '', window.location.pathname);
  }, []);

  // Feuille de route IN-APP (`?view=todo`) : ouverte depuis la landing (« Collaborer »).
  const openTodo = useCallback(() => {
    setRoute('todo');
    setActiveId(null);
    window.history.pushState({ route: 'todo', activeId: null } as HistState, '', '?view=todo');
  }, []);

  if (route === 'overview' && active) {
    return (
      <ConsultationOverview
        dataset={active}
        onHome={backToLanding}
        onViewGraph={() => viewGraph(active.id)}
        onExploreAvis={(avisId) => exploreAvis(active.id, avisId)}
      />
    );
  }
  if (route === 'avis' && active) {
    return (
      <AvisExplorer dataset={active} focusAvisId={focusAvis} onHome={backToLanding} />
    );
  }
  if (route === 'analysis' && active) {
    return (
      <RedesignApp
        initialDataset={active.id}
        onBack={backToLanding}
        onConsole={() => openConsole(active.id)}
      />
    );
  }
  if (route === 'console' && active) {
    return <Console dataset={active.id} label={active.label} onHome={backToLanding} />;
  }
  if (route === 'participate' && active) {
    return <Participate dataset={active} onBack={backToLanding} />;
  }
  if (route === 'todo') {
    return <TodoPage onHome={backToLanding} />;
  }
  return (
    <Landing datasets={datasets} loading={loading} onOpen={openConsultation} onTodo={openTodo} />
  );
}
