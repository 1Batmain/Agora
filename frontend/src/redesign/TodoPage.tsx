import { useEffect, useState } from 'react';
import { Header } from './Header';
import { fetchTodo } from './analysisApi';
import type { TodoItem, TodoStatus } from './contract';

/**
 * Page `/todo` — feuille de route collaborative in-app. Lit `GET /todo` (qui sert
 * `todo.json`, édité à la main au merge des PR) et affiche les tâches GROUPÉES par
 * statut (À faire / En cours / Fait), chacune badgée et reliée à sa PR si présente.
 *
 * DA Agora : mono / Bleu France. Lecture seule — aucune mutation côté front.
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

function TodoCard({ item }: { item: TodoItem }) {
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
    </li>
  );
}

export function TodoPage({ onHome }: { onHome: () => void }) {
  const [items, setItems] = useState<TodoItem[]>([]);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

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

  return (
    <div className="todo">
      <Header onHome={onHome} right={<span className="todo__crumb">Feuille de route</span>} />
      <main className="todo__main">
        <div className="todo__intro">
          <h1 className="todo__h1">Feuille de route</h1>
          <p className="todo__lede">
            L'état d'avancement d'Agora, lane par lane — édité au fil des merges.
          </p>
          {updatedAt && <p className="todo__updated">Mise à jour&nbsp;: {updatedAt}</p>}
        </div>

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
                        <TodoCard key={it.id} item={it} />
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
