# 5. Storage, history and live updates

**Status:** accepted, 2026-10-10

## Decision

- Postgres in production (Neon free tier), SQLite for tests and local runs, one SQLAlchemy schema for both.
- Events are matched across runs by object pair and a 30-minute TCA window, so a rerun updates an event and appends a row to its history instead of creating a duplicate.
- Reviews and the audit log are append-only.
- Element history keeps three weeks of payload element sets only; debris and rocket bodies cannot manoeuvre.
- Live updates reach browsers as server-sent events read from an `events` table. No message broker to host.
- The heavy work (screening) runs in GitHub Actions every four hours. The free API instance only serves.
