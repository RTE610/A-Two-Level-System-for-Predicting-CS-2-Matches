# Request to the department (PhD students): data for CS2 Match Predictor v2.0

Status: draft — to be sent by PO

1. **Backfill of game statistics.** In the snapshot of 28.09.2026 the collections `matchStats` and `playerStats` end on 05.01.2026, so `summaryStats.allAverages` is empty for matches from 02.2026. Could you backfill `matchStats` for 01.2026–09.2026?
2. **Refresh schedule.** How often can the snapshot be updated (e.g. every 2 weeks), and who is our contact person for it?
3. **Stale schedule entries.** 2 970 matches in `matches` are still `not_started` with dates in 2024. Is this expected, or should these be cleaned up on your side?
4. **Feature-set version.** We use `powerRankingCalculatorVersion = 2.4`, `coefficientsVersion = 1`. Is this version still maintained, or should we move to 3.02?
