# Deployment playbook

How to put Ashvini in front of a new operator in one week.

## Day 1: Understand the operation

- Which satellites, which orbits, who decides on manoeuvres and how fast.
- Where warnings arrive today (Space-Track email, commercial provider, in-house).
- Which formats they can export: CCSDS OEM, OMM, CDM, CSV.
- Where they want alerts: Slack, email, webhook into their own tools.

## Day 2: Load their world

```bash
# create the organisation, an admin key and their thresholds
python -m ashvini.admin create-org "Example Orbital" --action-pc 1e-4 --monitor-pc 1e-6
python -m ashvini.admin create-key "Example Orbital" --role admin

# their satellites, by NORAD id
python -m ashvini.admin watch "Example Orbital" 25544 43013

# their own ephemeris and any CDMs they already have
curl -H "X-API-Key: $KEY" -F file=@sat1.oem https://<host>/api/upload/oem
curl -H "X-API-Key: $KEY" -F file=@event.cdm https://<host>/api/upload/cdm
```

## Day 3: Alerts

```bash
python -m ashvini.admin add-channel "Example Orbital" slack https://hooks.slack.com/...
python -m ashvini.admin add-channel "Example Orbital" webhook https://ops.example.com/ashvini
```

Send a test alert and confirm someone saw it.

## Day 4: Walk through real events

Sit with the person who decides. For three live events, let them decide first, then compare with Ashvini's recommendation. Write down every disagreement in `product/decision-log.md`.

## Day 5: Hand over

- Runbook: `docs/runbooks/operations.md`.
- Who to call when data goes stale (the health endpoint shows the age of every source).
- What Ashvini will not do (`product/prd.md`, non-goals).

## On their own servers

```bash
git clone https://github.com/Mahendr99ar/ashvini.git && cd ashvini
cp .env.example .env    # set ASHVINI_DATABASE_URL, ASHVINI_SECRET, sources to enable
docker compose up -d
```

Nothing calls out except the data sources they enable in `.env`.
