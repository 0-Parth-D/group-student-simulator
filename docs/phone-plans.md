# Phone plans — facilitator reference (not shown to students)

**Task id:** `phone_plans_linear_01`  
**Audience:** coach / facilitator only

## Problem

Plan A: $20 monthly fee + $0.10 per text  
Plan B: no monthly fee + $0.30 per text  
Question: When is each plan better? Represent and compare.

## Solution

| Representation | Formula |
|----------------|---------|
| Plan A | \(C = 20 + 0.1x\) |
| Plan B | \(C = 0.3x\) |

Set equal: \(20 + 0.1x = 0.3x\) → \(0.2x = 20\) → **\(x = 100\) texts** (both cost **$30**).

- **Fewer than 100 texts:** Plan B is cheaper
- **More than 100 texts:** Plan A is cheaper

Keep the crossover out of student prompts so struggling students stay genuinely confused.

## Typical wrong ideas (catalog)

| Id | Pattern |
|----|---------|
| `ur_rate_as_flat_add` | Treats $0.10/text as adding 10¢ once |
| `ur_cents_vs_dollars_slope` | Writes `10x` instead of `0.1x` |
| `ur_per_month_not_per_text` | Treats 0.10/0.30 as monthly, not per text |
| `pr_ignores_monthly_fee` | Drops the $20 fee from Plan A |
| `pr_lower_rate_always_wins` | Cheaper per-text rate ⇒ always better |
| `pr_swaps_plan_rates` | Swaps fee/rate between plans |
| `pr_compare_one_x_only` | Checks one/two values; misses crossover |
| `pr_slope_coefficient_only` | Writes equation; can’t say what 0.1 means |
| `pr_algebra_crossover_slip` | Sets equal but solves to wrong \(x\) |
| `pr_fee_added_to_both` | Puts $20 on both plans |

Retrieval utterances for these ids live in `MISCONCEPTION_INSTANCES` (`src/app/knowledge/misconception_store.py`).

## Offline battery scaffolds (`eval/fixtures/teacher_scripts.yaml`)

- Primary: `what does 0.1 mean in Plan A, and when are the two plans equal?`
- Alt: `when are they equal`
