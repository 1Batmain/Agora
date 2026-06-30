import { useCallback, useEffect, useMemo, useState } from 'react';
import { Header } from './Header';
import { fetchTodo, patchTodo, postTodo } from './analysisApi';
import type { TodoItem, TodoStatus } from './contract';

/**
 * Page `/todo` — feuille de route COLLABORATIVE in-app. Outil de coordination du
 * hackathon : on LIT `GET /todo` (qui sert `todo.json`), et on ÉCRIT en direct —
 * AJOUTER une tâche (formulaire en tête), la RÉCLAMER (« Je prends » → `assignee` +
 * `wip`), et faire AVANCER son statut (todo→wip→done). Update optimiste + refetch.
 *
 * DA Agora : mono / Bleu France. Groupé par statut (À faire / En cours / Fait).
 */

/** Base GitHub pour transformer `pr: 41` en lien cliquable. */
const PR_BASE = 'https://github.com/1Batmain/Analyse-des-consultations-citoyennes/pull/';

/** Statuts dans l'ordre d'affichage + leur libellé FR. */
const GROUPS: { status: TodoStatus; label: string }[] = [
  { status: 'todo', label: 'À faire' },
  { status: 'wip', label: 'En cours' },
  { status: 'done', label: 'Fait' },
];

/** Libellé FR court d'un badge de statut. */
const STATUS_LABEL: Record<TodoStatus, string> = {
  todo: 'à faire',
  wip: 'en cours',
  done: 'fait',
};

/** Lanes de repli quand la feuille de route est encore vide (rare : todo.json est seedé). */
const FALLBACK_LANES = ['backend', 'frontend', 'pipeline', 'research', 'cross-lane'];

function TodoCard({
  item,
  onClaim,
  onStatus,
}: {
  item: TodoItem;
  onClaim: (item: TodoItem) => void;
  onStatus: (item: TodoItem, status: TodoStatus) => void;
}) {
  return (
    <li className="todo__card">
      <div className="todo__cardhead">
        <span className={`todo__badge todo__badge--${item.status}`}>
          {STATUS_LABEL[item.status]}
        </span>
        <span className="todo__lane">{item.lane}</span>
        {item.pr != null && (
          <a
            className="todo__pr"
            href={`${PR_BASE}${item.pr}`}
            target="_blank"
            rel="noreferrer"
          >
            PR&nbsp;#{item.pr}
          </a>
        )}
      </div>
      <strong className="todo__title">{item.title}</strong>
      {item.note && <p className="todo__note">{item.note}</p>}

      <div className="todo__cardactions">
        {item.assignee ? (
          <span className="todo__assignee" title="Personne qui a réclamé la tâche">
            ✋ {item.assignee}
          </span>
        ) : (
          <button type="button" className="todo__claim" onClick={() => onClaim(item)}>
            Je prends
          </button>
        )}
        <label className="todo__statussel">
          <span className="todo__statussel-lbl">statut</span>
          <select
            value={item.status}
            onChange={(e) => onStatus(item, e.target.value as TodoStatus)}
          >
            {GROUPS.map((g) => (
              <option key={g.status} value={g.status}>
                {g.label}
              </option>
            ))}
          </select>
        </label>
      </div>
    </li>
  );
}

function AddForm({
  lanes,
  onAdd,
}: {
  lanes: string[];
  onAdd: (input: { title: string; lane: string; note?: string }) => Promise<void>;
}) {
  const [title, setTitle] = useState('');
  const [lane, setLane] = useState(lanes[0] ?? '');
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);

  // Garde la lane sélectionnée valide quand la liste des lanes arrive/évolue.
  useEffect(() => {
    if (!lanes.includes(lane)) setLane(lanes[0] ?? '');
  }, [lanes, lane]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const t = title.trim();
    if (!t || !lane || busy) return;
    setBusy(true);
    await onAdd({ title: t, lane, note: note.trim() || undefined });
    setBusy(false);
    setTitle('');
    setNote('');
  };

  return (
    <form className="todo__add" onSubmit={submit}>
      <input
        className="todo__add-title"
        type="text"
        placeholder="Nouvelle tâche…"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        aria-label="Titre de la tâche"
      />
      <select
        className="todo__add-lane"
        value={lane}
        onChange={(e) => setLane(e.target.value)}
        aria-label="Lane"
      >
        {lanes.map((l) => (
          <option key={l} value={l}>
            {l}
          </option>
        ))}
      </select>
      <input
        className="todo__add-note"
        type="text"
        placeholder="Note (optionnelle)"
        value={note}
        onChange={(e) => setNote(e.target.value)}
        aria-label="Note"
      />
      <button type="submit" className="todo__add-btn" disabled={busy || !title.trim()}>
        Ajouter
      </button>
    </form>
  );
}

export function TodoPage({ onHome }: { onHome: () => void }) {
  const [items, setItems] = useState<TodoItem[]>([]);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refetch = useCallback(async () => {
    const payload = await fetchTodo();
    setItems(payload.items);
    setUpdatedAt(payload.updated_at ?? null);
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetchTodo().then((payload) => {
      if (cancelled) return;
      setItems(payload.items);
      setUpdatedAt(payload.updated_at ?? null);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Lanes proposées à l'ajout : celles déjà présentes ∪ repli (toujours non vide).
  const lanes = useMemo(() => {
    const set = new Set<string>(FALLBACK_LANES);
    items.forEach((it) => it.lane && set.add(it.lane));
    return [...set].sort();
  }, [items]);

  const addTodo = useCallback(
    async (input: { title: string; lane: string; note?: string }) => {
      const created = await postTodo(input);
      if (created) setItems((cur) => [...cur, created]); // optimiste
      await refetch(); // source de vérité
    },
    [refetch],
  );

  // Update OPTIMISTE local d'un item par id, puis refetch (source de vérité).
  const patch = useCallback(
    async (id: string, p: { status?: TodoStatus; assignee?: string }) => {
      setItems((cur) => cur.map((it) => (it.id === id ? { ...it, ...p } : it)));
      await patchTodo(id, p);
      await refetch();
    },
    [refetch],
  );

  const claim = useCallback(
    (item: TodoItem) => {
      const who = window.prompt('Ton prénom / pseudo court :', item.assignee ?? '');
      if (who == null) return;
      const name = who.trim();
      if (!name) return;
      void patch(item.id, { assignee: name, status: 'wip' });
    },
    [patch],
  );

  const setStatus = useCallback(
    (item: TodoItem, status: TodoStatus) => {
      if (status === item.status) return;
      void patch(item.id, { status });
    },
    [patch],
  );

  return (
    <div className="todo">
      <Header onHome={onHome} right={<span className="todo__crumb">Feuille de route</span>} />
      <main className="todo__main">
        <div className="todo__intro">
          <h1 className="todo__h1">Feuille de route</h1>
          <p className="todo__lede">
            L'outil de coordination du hackathon — ajoute une tâche, réclame-en une, fais
            avancer son statut. Lane par lane.
          </p>
          {updatedAt && <p className="todo__updated">Mise à jour&nbsp;: {updatedAt}</p>}
        </div>

        <AddForm lanes={lanes} onAdd={addTodo} />

        {loading ? (
          <p className="todo__empty">Chargement de la feuille de route…</p>
        ) : items.length === 0 ? (
          <p className="todo__empty">Aucune tâche pour le moment.</p>
        ) : (
          <div className="todo__board">
            {GROUPS.map(({ status, label }) => {
              const group = items.filter((it) => it.status === status);
              return (
                <section key={status} className="todo__col">
                  <h2 className="todo__coltitle">
                    {label} <span className="todo__count">{group.length}</span>
                  </h2>
                  {group.length === 0 ? (
                    <p className="todo__colempty">—</p>
                  ) : (
                    <ul className="todo__list">
                      {group.map((it) => (
                        <TodoCard
                          key={it.id}
                          item={it}
                          onClaim={claim}
                          onStatus={setStatus}
                        />
                      ))}
                    </ul>
                  )}
                </section>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}
