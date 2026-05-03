"""End-to-end cascade wiring: Tier1 -> Tier2 -> Tier3.

Single-entry-point module. `cascade.py` exposes one `process()` method that
takes a batch of CloudTrail events and returns incident reports for the
sessions that survived all three tiers. Per-tier intermediate artefacts are
persisted so the evaluation step can compute per-tier P/R/F1 separately
(required for the ablation table in the thesis evaluation chapter).

This is also where the cost-awareness contract is enforced operationally:
Tier 2 must never see events Tier 1 dismissed; Tier 3 must never see events
Tier 2 dismissed. Violating that ordering is a thesis-defining bug.
"""
