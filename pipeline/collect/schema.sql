-- Schéma DuckDB du collecteur — appliqué à CHAQUE connexion (idempotent).
-- Pattern "schema-on-connect" : la base est jetable et auto-réparante.

CREATE TABLE IF NOT EXISTS consultations (
  slug             TEXT PRIMARY KEY,      -- issu du href de l'index, jamais en dur
  title            TEXT,                  -- texte du lien sur l'index
  page_url         TEXT NOT NULL,
  scraped_at       TIMESTAMP NOT NULL,
  n_files          INTEGER NOT NULL DEFAULT 0,
  n_files_ingested INTEGER NOT NULL DEFAULT 0,
  n_responses      BIGINT  NOT NULL DEFAULT 0,  -- enregistrements vus dans les fichiers ingérés
  n_answers        BIGINT  NOT NULL DEFAULT 0,  -- lignes écrites dans responses
  status           TEXT NOT NULL,  -- ok | partial | no_data_files | empty | skipped | error
  status_detail    TEXT
);

CREATE TABLE IF NOT EXISTS files (
  consultation_slug TEXT NOT NULL,
  filename         TEXT NOT NULL,          -- basename de l'URL
  url              TEXT NOT NULL,
  format           TEXT NOT NULL,          -- csv | json_zip | xml_zip | zip | json | xml | other
  size_bytes       BIGINT,
  downloaded_at    TIMESTAMP,
  status           TEXT NOT NULL,  -- ok | listed | empty | redundant | unsupported_format | too_large | error
  status_detail    TEXT,
  n_rows           BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (consultation_slug, filename)
);

CREATE TABLE IF NOT EXISTS questions (
  consultation_slug TEXT NOT NULL,
  source_file      TEXT NOT NULL,
  question_index   INTEGER NOT NULL,       -- ordinal de la colonne/clé dans la source
  question         TEXT NOT NULL,          -- libellé (header CSV ou clé JSON)
  n_answers        BIGINT NOT NULL,        -- cellules non vides
  n_distinct       BIGINT NOT NULL,        -- cappé (cf. classify.DISTINCT_CAP)
  distinct_ratio   DOUBLE,
  avg_len          DOUBLE,
  max_len          BIGINT,
  kind             TEXT NOT NULL,          -- open_text | closed | date | numeric | empty
  PRIMARY KEY (consultation_slug, source_file, question_index)
);

CREATE TABLE IF NOT EXISTS responses (
  consultation_slug TEXT NOT NULL,
  source_file      TEXT NOT NULL,
  row_num          BIGINT NOT NULL,        -- ordinal (base 0) de l'enregistrement dans la source
  submitted_at     TEXT,                   -- valeur brute de la 1re colonne kind=date, si présente
  question_index   INTEGER NOT NULL,
  question         TEXT NOT NULL,          -- dénormalisé pour l'ergonomie des requêtes
  answer           TEXT NOT NULL,          -- cellules non vides uniquement
  PRIMARY KEY (consultation_slug, source_file, row_num, question_index)
);

-- Interface stable : une ligne = une réponse à une question ouverte.
CREATE OR REPLACE VIEW contributions AS
SELECT r.consultation_slug, r.source_file, r.row_num, r.submitted_at,
       r.question_index, r.question, r.answer
FROM responses r
JOIN questions q USING (consultation_slug, source_file, question_index)
WHERE q.kind = 'open_text';
