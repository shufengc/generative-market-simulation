# Documentation Naming Convention

All documentation files in `docs/` and `references/` follow this format:

```
MM-DD-<topic>-<kind>.<ext>
```

## Fields

| Field | Description | Examples |
|-------|-------------|---------|
| `MM-DD` | Month-day the document was created | `04-15`, `04-22` |
| `<topic>` | 2-4 hyphenated words describing the most important subject | `stride-eval-discrepancy`, `ddpm-vs-normflow`, `data-pipeline` |
| `<kind>` | Document type (see table below) | `analysis`, `experiment-results`, `status-report` |
| `<ext>` | File extension | `.md`, `.csv`, `.json`, `.html` |

## Kind Values

| Kind | Use for |
|------|---------|
| `meeting-transcript` | Raw notes/recording transcript from a meeting |
| `meeting-summary` | Condensed summary of a meeting |
| `meeting-report` | Formal report derived from meeting outcomes |
| `experiment-results` | Raw output, tables, or CSVs from a completed experiment |
| `campaign-report` | Report covering a full experiment campaign (multiple seeds/configs) |
| `campaign-analysis` | Analysis interpreting campaign results |
| `iteration-report` | Report from a single improvement iteration |
| `analysis` | Stand-alone analytical writeup (not tied to a specific experiment run) |
| `progress-walkthrough` | Step-by-step walkthrough of current progress state |
| `status-report` | Point-in-time team or project status |
| `policy` | Conventions, guidelines, or agreed-upon rules (like this file) |
| `literature-search` | Research/literature notes from external sources |

## Examples

```
04-15-team-sync-meeting-summary.md
04-15-ddpm-vs-normflow-iteration-report.md
04-15-ddpm-literature-search.md
04-16-data-pipeline-experiment-results.md
04-16-experiment-audit-status-report.md
04-16-phase5-innovation-experiment-results.md
04-22-stride-eval-discrepancy-analysis.md
04-22-cross-model-comparison-analysis.md
04-22-presentation-script-progress-walkthrough.md
04-24-ddpm-phase7-improvement-ideas.md
```

## Notes

- Use lowercase and hyphens only (no underscores, no spaces, no CamelCase)
- Keep `<topic>` concise -- 2-4 words is the target
- `PROJECT_STATUS.md` and `README.md` in root are exempt (they are living documents, not dated artifacts)
- `NAMING_CONVENTION.md` (this file) is also exempt
