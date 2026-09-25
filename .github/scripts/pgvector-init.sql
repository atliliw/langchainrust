-- T2 services CI: mounted into pgvector/pgvector:pg16's /docker-entrypoint-initdb.d/.
-- Runs once at first container boot (DB `vectors` creation), as the postgres superuser,
-- so `CREATE EXTENSION` succeeds without an admin DBA step. The pgvector tests
-- (`PGVectorStore::connect`) require the extension to already exist — this is the
-- documented admin prerequisite automated for CI.
CREATE EXTENSION IF NOT EXISTS vector;