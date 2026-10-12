# Metrics

## North star

**Correct decisions per operator-hour.** A decision is correct when a dismissed or monitored event does not end high risk, and a manoeuvre was taken on an event that the final data shows was high risk.

## Tree

```
Correct decisions per operator-hour
├── Decision quality
│   ├── High-risk events missed (target: 0 on the test set)          guardrail
│   ├── False alarms dismissed per week                               up is good
│   └── Model: challenge score and F2 vs the carried-forward baseline
├── Decision speed
│   ├── Median time from alert to recorded decision
│   └── Events needing a second look after the first decision
└── Trust
    ├── Share of recommendations the operator agreed with
    ├── Share of overrides that came with a note                      target 100%
    └── Calibration: predicted risk band vs final outcome
```

## Data health (shown on /api/health)

- Age of the newest element set per source.
- Objects screened in the last run, events created, events updated.
- Model drift: live error over the last 30 days divided by test error. Above 1.3 fails the drift check.

## What we will not count

- Page views and time on site. Longer time in a triage tool is usually worse.
- Number of alerts sent. More alerts is not more value.
