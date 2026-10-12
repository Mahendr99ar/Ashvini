# Ashvini PRD

Status: v0.1, Stage 1 in build. Owner: Mahendra Meena.

## Goal

Help an operator decide, for each predicted close approach, whether to manoeuvre, keep monitoring, or dismiss, and make that decision faster and with fewer false alarms than reading raw warnings.

## Non-goals

- Issuing official collision warnings.
- Commanding a spacecraft or planning the burn itself.
- Tracking people, ground stations or military assets.

## Users

See `personas.md`. Stage 1 targets Priya (small-operator flight dynamics). Daniel and Meera get read-only views.

## User stories

| # | As | I want | So that | Stage |
|---|---|---|---|---|
| 1 | Operator | a ranked list of close approaches for my satellites over the next 7 days | I look at the dangerous ones first | 1 |
| 2 | Operator | every event to show miss distance, Pc, relative speed and the data age behind them | I can judge how far to trust the number | 1 |
| 3 | Operator | a recommendation (manoeuvre, monitor, dismiss) with its reasons | I can agree or disagree quickly | 1 |
| 4 | Operator | to record my decision and a note | the team knows what was decided and why | 1 |
| 5 | Operator | to upload my own ephemeris (CCSDS OEM) | my satellites are screened with better data than public TLEs | 1 |
| 6 | Operator | to upload CDMs I already receive | Ashvini re-scores them and keeps the history per event | 1 |
| 7 | Operator | an alert in Slack or by webhook only above my threshold | I am not woken up for noise | 1 |
| 8 | Operator | the predicted final risk, with a range, days before TCA | I can plan a manoeuvre window early | 2 |
| 9 | Operator | a storm warning that tells me which of my events become less certain | I re-check the right events | 3 |
| 10 | Analyst | to replay a past event or break-up | I can explain what happened | 5 |
| 11 | Anyone | to ask the copilot why an event is high risk | I get a plain answer that cites the numbers | 1 |

## Requirements

- Every number on screen links to its inputs: which element set, which epoch, which run.
- Recommendations are never shown without the threshold that produced them.
- Decisions are append-only. A rerun of the pipeline never changes a decision.
- Alerts are deduplicated per event and channel. A worsening event re-alerts; a stable one does not.
- The copilot can read; it cannot record decisions, change thresholds or send alerts.

## Success metrics

See `metrics.md`.

## Risks

| Risk | Mitigation |
|---|---|
| Public TLE data is too coarse for real decisions | Say so on every event; accept operator ephemerides |
| Model looks good offline and drifts live | Drift job on live error; model cards with known gaps |
| Users over-trust the recommendation | Show uncertainty and the rule; require a note for every manoeuvre decision |
| Data source terms change | Source health on `/api/health`; Space-Track data never redistributed |
