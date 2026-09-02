# LPF Source Mapping — where every construct in the knowledge graph comes from

The knowledge graph in `backend/config/knowledge_graph.yaml` takes its **nodes and
edges** from one published framework:

> Hess, K. K. (Ed.), December 2010 (updated 2/24/2011).
> *Learning Progressions Frameworks Designed for Use with the Common Core State
> Standards in Mathematics K-12.* National Alternate Assessment Center at the
> University of Kentucky and the National Center for the Improvement of Educational
> Assessment, Dover, NH.
> <https://www.nciea.org/wp-content/uploads/2022/07/Math_LPF_KH11.pdf>

The LPF organises mathematics into six strands. Each strand has grade-span
**learning targets** (`M.` prefix = middle school, grades 5-8), and under each target
a numbered sequence of **progress indicators** (PIs) such as `M.PRF.1c`. LPF states
that "descriptions of earlier skills build the foundation for later skills at the next
grade level or grade span" (p. 10), which is what the `prerequisites` edges encode.

Three strands are in scope for our scenarios:

| Strand | Middle-school targets used |
|---|---|
| NO — The Nature of Numbers and Operations | M.NO-1, M.NO-2, M.NO-3 |
| PRF — Patterns, Relations, and Functions | M.PRF-1, M.PRF-2 |
| SE — Symbolic Expression | M.SE-1 |

Two other sources appear in the file but do **not** determine structure:

- **DLM Essential Elements** supply the `dlm_linkage_levels` (IP → DP → PP → T → S)
  wording for the fraction and ratio nodes, which is where our 0-3 mastery scale came
  from. Constructs added for linear change have no vetted DLM alignment yet and use
  `lpf_stack` instead, written from LPF PI wording.
- **PISA competency attributes** (PLOS ONE 2025) are a separate overlay in
  `pisa_links`, not part of the progression.

---

## Construct → LPF citation

| Construct | Strand / target | Progress indicators | CCSS in LPF |
|---|---|---|---|
| `part_whole` | NO / M.NO-1, M.NO-2 | M.NO.2b, M.NO.1c | 5.NF-3; 5.NBT-3a; 5.NF-1 |
| `fraction_equivalence` | NO / M.NO-1, M.NO-2 | M.NO.1f, M.NO.2b | 6.RP-1, 3; 5.NF-3 |
| `fraction_operations` | NO / M.NO-2 | M.NO.2c, M.NO.2a | 5.NBT-5,6,7; 5.NF-1,2,4,7; 6.NS-1,3 |
| `ratio_concept` | NO / M.NO-1 | M.NO.1f, M.SE.1d | 6.RP-1, 3c; 6.EE-1, 6 |
| `ratio_equivalence` | NO + PRF / M.NO-1, M.PRF-2 | M.NO.1f, M.PRF.2a | 6.RP-1, 3; 6.EE-6, 9 |
| `unit_rate` | PRF / M.PRF-1 | M.PRF.1e, M.PRF.1h | 7.RP-1,2,3; 8.EE-5 |
| `proportional_reasoning` | NO / M.NO-2, M.NO-3 | M.NO.2f, M.NO.3c | 7.RP-1,2,3; 8.EE-6; 8.F-5 |
| `additive_vs_multiplicative` | NO / M.NO-2 | M.NO.2d, M.PRF.1a | none — LPF lists M.NO.2d with no CCSS link |
| `linear_relationship` | PRF / M.PRF-1, M.PRF-2 | M.PRF.1b, M.PRF.2b, M.PRF.2c, M.PRF.2e | 6.RP-3a; 8.F-3; 8.EE-5,7 |
| `rate_comparison` | PRF / M.PRF-1 | M.PRF.1c, M.NO.3b | 6.RP-1,2,3b; 6.EE-4, 6 |
| `symbolic_modeling` | PRF + SE / M.PRF-1, M.PRF-2, M.SE-1 | M.PRF.1d, 1f, 1g, M.PRF.2d, M.SE.1f | 6.EE-4,7,9; 7.EE-2,4; 8.EE-6,7 |

`additive_vs_multiplicative` is worth noting: LPF includes M.NO.2d *"contrasting
situations as additive or multiplicative"* with no Common Core standard attached. LPF
explains that PIs with no CCSS link are kept when "the research review identified
critical learning needed at certain stages" (p. 10). It is the construct that decides
whether a student can separate the $20 monthly fee from the $0.10-per-text rate.

---

## Which constructs each scenario exercises

**Pizza / fractions** (`frac_word_maria_pizza_01`) — NO strand:
`part_whole` → `fraction_equivalence` → `fraction_operations`.

**Phone plans** (`phone_plans_linear_01`) — PRF strand:

| Step a student must take | Construct | PI |
|---|---|---|
| Read $0.10 as a rate, not a one-off charge | `unit_rate` | M.PRF.1e |
| Separate the $20 fee from the per-text rate | `additive_vs_multiplicative` | M.NO.2d |
| See Plan A as linear but not proportional | `linear_relationship` | M.PRF.1b, M.PRF.2b |
| Decide which plan wins over which range | `rate_comparison` | M.PRF.1c |
| Write 20 + 0.1x and 0.3x, set equal, solve | `symbolic_modeling` | M.PRF.1d/1f/1g, M.PRF.2d |

`proportional_reasoning` (M.NO.2f) is not listed as required for phone plans — it
enters through the prerequisite edges, since Plan B alone *is* proportional and Plan A
is the contrast case.

---

## Open questions for content review

1. **Edge M.PRF.1b before M.PRF.1c.** We read the LPF numbering as an ordering claim
   (distinguish linear/nonlinear before comparing two rates for best value). Is that
   how it should be taught, or are they parallel?
2. **`additive_vs_multiplicative` as its own node.** LPF gives M.NO.2d one line and no
   standard. Should it be a node, or an attribute of `proportional_reasoning`?
3. **`symbolic_modeling` scope.** It currently merges M.PRF.1d/1f/1g, M.PRF.2d, and
   M.SE.1f. Should writing the model and solving the equation be split into two nodes?
4. **Grade band.** LPF puts M.PRF.1c in grades 5-6 but M.PRF.1f/1g in 7-8. Phone plans
   therefore spans both. Is that the intended grade level for this scenario?
5. **Misconceptions.** `config/misconception_catalog.json` now keys each phone-plan
   misconception to one of these constructs. Are those the misconceptions worth
   simulating, and are any missing?
