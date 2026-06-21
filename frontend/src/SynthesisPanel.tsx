import { useState } from 'react';
import { synthesize } from './api';
import type { ClusterMethod, NamingMethod, SynthesisResult } from './types';

/**
 * "Synthèse" panel — discreet & collapsible. A single button asks the backend
 * (`POST /api/synthesize`) for a short Markdown report (synthesis of the citizen
 * voice + a critique of the clustering's pertinence), written by Mistral in the
 * corpus' dominant language. Shows a loading state (the call takes a few s) and
 * renders the returned Markdown. If the backend has no Mistral key it returns a
 * graceful notice (meta.fallback) which we surface as-is.
 *
 * Always reflects the CURRENT dataset/method/naming selection.
 */
export function SynthesisPanel({
  dataset,
  method,
  naming,
  disabled,
}: {
  dataset: string | null;
  method: ClusterMethod;
  naming: NamingMethod;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<SynthesisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function onGenerate() {
    setBusy(true);
    setError(null);
    try {
      const r = await synthesize(dataset ?? undefined, method, naming);
      setResult(r);
    } catch (e) {
      setError(`Synthèse impossible : ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="synth">
      <div className="panel__head">
        <h2>Synthèse</h2>
        <button
          type="button"
          className="btn"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? '▾' : '▸'}
        </button>
      </div>

      {open && (
        <div className="synth__body">
          <p className="synth__intro">
            Rapport court généré par l'IA (Mistral) : grands thèmes de la parole citoyenne +
            regard critique sur la pertinence du découpage.
          </p>
          <button
            type="button"
            className="btn synth__go"
            disabled={disabled || busy}
            onClick={onGenerate}
          >
            {busy ? 'Génération…' : result ? 'Régénérer la synthèse' : 'Générer la synthèse'}
          </button>

          {busy && <p className="synth__busy">Analyse des thèmes par Mistral… (quelques secondes)</p>}
          {error && <p className="app__error">{error}</p>}

          {result && !busy && (
            <article className="synth__report">
              {result.meta?.fallback && (
                <p className="synth__warn">Synthèse indisponible (repli) — voir le message ci-dessous.</p>
              )}
              <div
                className="md"
                dangerouslySetInnerHTML={{ __html: renderMarkdown(result.report_markdown) }}
              />
              {!result.meta?.fallback && (
                <footer className="synth__meta">
                  {result.meta?.model ?? ''}
                  {typeof result.meta?.n_clusters === 'number' ? ` · ${result.meta.n_clusters} thèmes` : ''}
                  {typeof result.meta?.took_ms === 'number' ? ` · ${(result.meta.took_ms / 1000).toFixed(1)} s` : ''}
                  {result.meta?.truncated ? ' · (tronqué)' : ''}
                </footer>
              )}
            </article>
          )}
        </div>
      )}
    </section>
  );
}

/**
 * Minimal, dependency-free Markdown → HTML for the report. Escapes HTML first
 * (the text comes from an LLM — treat as untrusted), then handles the small
 * subset the report uses: ## / ### headings, - / * bullet lists, **bold**,
 * *italic*, and paragraphs. Anything else renders as plain text.
 */
function renderMarkdown(md: string): string {
  const esc = (s: string) =>
    s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const inline = (s: string) =>
    esc(s)
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>');

  const lines = (md || '').replace(/\r\n/g, '\n').split('\n');
  const out: string[] = [];
  let inList = false;
  const closeList = () => {
    if (inList) {
      out.push('</ul>');
      inList = false;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) {
      closeList();
      continue;
    }
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      closeList();
      const level = Math.min(h[1].length, 6);
      out.push(`<h${level}>${inline(h[2])}</h${level}>`);
      continue;
    }
    const li = /^[-*]\s+(.*)$/.exec(line);
    if (li) {
      if (!inList) {
        out.push('<ul>');
        inList = true;
      }
      out.push(`<li>${inline(li[1])}</li>`);
      continue;
    }
    closeList();
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return out.join('\n');
}
