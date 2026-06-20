import type { PackNode } from './hierarchy';

/**
 * Detail panel. A sub-theme selection lists its source avis (`node.props.text`);
 * a single avis selection shows it in full.
 */
export function AvisPanel({ selected }: { selected: PackNode | null }) {
  if (!selected) {
    return (
      <section className="panel avis">
        <header className="panel__head">
          <h2>Avis</h2>
        </header>
        <p className="avis__empty">Clique une bulle pour zoomer ; un sous-thème affiche ses avis sources.</p>
      </section>
    );
  }

  if (selected.data.kind === 'avis') {
    const n = selected.data.node!;
    return (
      <section className="panel avis">
        <header className="panel__head">
          <h2>Avis</h2>
          <span className="avis__count">{n.props.lang ?? ''}</span>
        </header>
        <article className="avis__one">
          <p>{n.props.text}</p>
          <footer className="avis__meta">
            {n.props.ts ?? ''} · poids {n.props.weight ?? 1} · {n.id}
          </footer>
        </article>
      </section>
    );
  }

  // sub-theme: list member avis
  const theme = selected.data.theme!;
  const avis = (selected.children ?? []).map((c) => c.data.node!).filter(Boolean);
  return (
    <section className="panel avis">
      <header className="panel__head">
        <h2 title={theme.label}>{theme.label}</h2>
        <span className="avis__count">{avis.length}</span>
      </header>
      {theme.keywords && theme.keywords.length > 0 && (
        <div className="avis__kw">
          {theme.keywords.map((k) => (
            <span className="kw" key={k}>
              {k}
            </span>
          ))}
        </div>
      )}
      <ul className="avis__list">
        {avis.map((n) => (
          <li className="avis__item" key={n.id}>
            <p>{n.props.text}</p>
            <span className="avis__sub">
              {n.props.lang ?? ''} {n.props.ts ?? ''}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
