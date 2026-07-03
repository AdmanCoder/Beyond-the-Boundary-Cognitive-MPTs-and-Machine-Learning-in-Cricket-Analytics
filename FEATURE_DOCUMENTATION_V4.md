# Feature Documentation — IPL Ball-by-Ball Prediction Dataset (V4)

## Project Goal

This dataset supports a **ball-by-ball outcome prediction model** for IPL (Indian Premier League) T20 cricket. Each row represents a single delivery (ball). The model predicts `mpt_outcome`, a 7-class categorical variable representing what happens on each ball: wicket, dot, single, double, triple, four, or six.

## Dataset Overview

| Property | Value |
|---|---|
| **File** | `ipl_training_data_v4.csv` |
| **Rows** | 278,205 (one per delivery across ~1,169 IPL matches) |
| **Columns** | 66 |
| **Target** | `mpt_outcome` (1–7 categorical) |
| **Source** | Ball-by-ball JSON files from [Cricsheet](https://cricsheet.org/) |
| **NaN** | 0 (after targeted imputation) |

## How to Read This Document

- **"Over" numbering** uses Cricsheet's 0-indexed convention: over 0 = the 1st over, over 19 = the 20th over.  
- **"Pre-ball state"** means the feature captures the situation *before* the current delivery is bowled (no future leakage).  
- **"Full career"** means the feature uses all matches in the dataset, not just prior matches. This is an approved design choice for static skill profiles.
- **Volume filters** are minimum career thresholds below which a feature is set to 0 to avoid noisy estimates.

---

## Target Variable: `mpt_outcome`

Derived from `runs_batter` (runs scored by the batter, excluding extras) and `is_wicket`:

| Value | Outcome | How Derived |
|---|---|---|
| 1 | Wicket | `is_wicket == 1` (batter dismissed) |
| 2 | Dot ball | `runs_batter == 0` AND not a wide |
| 3 | Single / Wide | `runs_batter == 1`, OR `is_wide == 1` with `runs_batter == 0` (wide = 1 run to team) |
| 4 | Double | `runs_batter == 2` |
| 5 | Triple | `runs_batter == 3` |
| 6 | Four | `runs_batter == 4` |
| 7 | Five+ / Six | `runs_batter >= 5` |

**Important:** Wide deliveries (where the batter does not face the ball) are classified as outcome 3 (single-equivalent) rather than outcome 2 (dot ball), because the batting team gains 1 run despite no batter action. Use `is_wide == 1` to filter these out during training if desired.

---

## Phase Boundaries

All scripts use consistent 0-indexed over boundaries:

| Phase | Over Range (0-indexed) | Real Overs (1-indexed) | Overs Count |
|---|---|---|---|
| **Powerplay** | 0–5 | 1–6 | 6 overs |
| **Middle** | 6–14 | 7–15 | 9 overs |
| **Death** | 15–19 | 16–20 | 5 overs |

---

## Feature Categories

### A. Match & Innings Context (6 features)

These identify the match situation. All are known before the ball is bowled.

| Feature | Type | Description | Leakage-Safe? |
|---|---|---|---|
| `innings` | int | 1 (batting first) or 2 (chasing) | ✅ Known before match |
| `is_playoff` | binary | 1 if qualifier/eliminator/final, 0 if league stage | ✅ Known before match |
| `match_number` | int | Sequential match number in the season (1, 2, 3, ...) | ✅ Known before match |
| `over` | int | Current over, 0-indexed (0 = 1st over, 19 = 20th) | ✅ Pre-ball |
| `ball` | int | Ball number within the over (1–6, can exceed 6 with extras) | ✅ Pre-ball |
| `current_score` | int | Team score **before** this delivery is bowled | ✅ Pre-ball |

### B. Situational Context (6 features)

| Feature | Type | Formula / Description | Leakage-Safe? |
|---|---|---|---|
| `wickets_lost` | int | Team wickets fallen before this ball | ✅ Pre-ball |
| `batter_balls_faced` | int | Balls this batter has faced so far in this innings | ✅ Pre-ball |
| `target_score` | float | **Innings 2 only:** innings-1 total + 1 (chase target). **Set to 0 for innings 1** (the final innings-1 score is unknown while innings 1 is in progress). | ✅ No future leakage |
| `balls_remaining` | int | `120 - (over × 6 + min(ball, 6))`, clipped to minimum 1. Uses `min(ball, 6)` because extra deliveries (wides/no-balls) do not consume the 120-ball quota. | ✅ Deterministic |
| `required_run_rate` | float | `(target_score - current_score) / balls_remaining × 6`. **Innings 2 only** (0 for innings 1). Clipped to [0, 36]. | ✅ Pre-ball |
| `is_wide` / `is_noball` | binary | 1 if this delivery is a wide or no-ball, 0 otherwise | ⚠️ **Concurrent** — must be **excluded** from model features (see note below) |

### C. Batter Career Skill Features (16 features)

All computed from **full career** data across all 1,169 JSON files using `extract_features_v2.py`. These are **static per player** — the same values for every delivery that player faces. Only players with **≥60 career balls faced** receive computed features; the remaining 350 out of 703 unique batters have **median-imputed** values. **Wide deliveries are excluded** from all batter ball/dot counts — only legitimate batter-faced deliveries are counted.

Each feature uses roughly this structure: `Raw_Metric × (1 + α × log₁₀(volume + 1))` where `α` is a weighting constant (0.3 or 0.4) and volume is a sample-size measure (career runs in that context). This ensures players with more data get higher weight.

| Feature | Formula | What It Measures | Volume Filter |
|---|---|---|---|
| `Hand_Eye_Pace` | `(SR_vs_pace × Contact%) / 10 × (1 + 0.3×log₁₀(runs_vs_pace+1))` | Scoring ability vs pace bowling. `Contact% = 1 - dot%_vs_pace`. | min 30 pace balls |
| `Head_Stability_Spin` | `(Six%_vs_spin × Control%_vs_spin) × 1000 × (1 + 0.3×log₁₀(runs_vs_spin+1))` | Power + control vs spin. `Control% = 1 - dot%_vs_spin`. | min 30 spin balls |
| `Pressure_Absorb` | `max(0, 20 - avg_balls_to_boundary_after_wicket) × (1 + 0.3×log₁₀(events+1))` | How quickly a batter finds a boundary after a wicket falls in overs 0–15. Lower delay = higher score. | None |
| `Chase_IQ` | `chase_average × (1 + 0.3×log₁₀(chase_runs+1))` | Batting average in **successful** chases only. | None |
| `Clutch_Index` | `(Death_SR × 0.6 + Playoff_Avg × 4 × 0.4) × (1 + 0.3×log₁₀(death_runs + playoff_runs + 1))` | Combined death-overs strike rate and playoff batting average. | None |
| `Hard_Hitting_Power` | `Overs_7_to_20_SR × (1 + 0.3×log₁₀(mid_runs+1))` | Strike rate in overs 7–20 (0-indexed: 6–19). Overlaps with death overs — captures general non-powerplay aggression. | None |
| `Technical_Adeptness` | `(Non_boundary_SR × 0.6 + Spin_Four% × 4 × 0.4) × (1 + 0.3×log₁₀(non_boundary_runs + fours_vs_spin×4 + 1))` | Ability to score without boundaries + finding fours against spin. Volume = non-boundary runs + spin four runs. | None |
| `Tough_Pitch_Performance` | `tough_runs / tough_balls × 100 × (1 + 0.4×log₁₀(tough_runs+1))` | Strike rate on pitches where match run rate < 7.5. Denominator is balls on tough pitches only (not total career balls). Penalized for low total career runs (<150 → 0, <200 → ×0.7, <350 → ×0.85). | 150 career runs |
| `Consistency_Gini` | `(1 - Gini_coefficient) × (1 + 0.5×log₁₀(avg_innings_score+1))` | Evenness of innings scores (1 = perfectly consistent). Weighted by average score to reward consistent high scorers. | None |
| `Lone_Wolf` | `Avg_when_team_collapses × (1 + 0.3×log₁₀(collapse_runs+1))` | Batting average when rest of team scores < 140 runs. Measures ability to carry a struggling team. | None |
| `Counter_Attack` | `SR_in_6_balls_after_wicket × (1 + 0.3×log₁₀(runs+1))` | Strike rate in the first 6 **team** balls immediately after any wicket. Measures team-level counterattacking response. Both the surviving batter and the new batter contribute. | **>300 career runs** |
| `Bowler_Reading` | `avg(late_SR - early_SR per bowler) × (1 + dot_recovery) × (1 + 0.3×log₁₀(runs+1))` | How much a batter's SR improves against the same bowler (balls 1–3 vs 4+). `dot_recovery` = proportion of dots followed by scoring. | **≥150 career runs** |
| `Perception_Skills` | `First_Ball_SR × 0.4 + Clean_Ratio × 100 × 0.3 + min(Death_Avg, 100) × 0.3` | Ball reading: first-ball scoring ability + avoiding bowled/lbw dismissals + death-overs composure. `Clean_Ratio = 1 - (bowled+lbw outs / total outs)`. Death_Avg capped at 100 to prevent outlier inflation. | **≥150 career runs** |
| `Shot_Inventory` | `(Phase_Spread + Bowler_Spread) × Consistency_Gini` | Versatility: evenness of SR across PP/Mid/Death phases and vs Pace/Spin. Higher = performs equally well in all contexts. | **≥150 career runs** |
| `Death_Specialist_SR` | `death_runs / death_balls × 100` | Career strike rate in death overs (0-indexed: ≥15, i.e. overs 16–20). | min 10 death balls |
| `Powerplay_Specialist_SR` | `pp_runs / pp_balls × 100` | Career strike rate in powerplay (0-indexed: 0–5, i.e. overs 1–6). | min 10 PP balls |

### D. Batter Form Features (4 features)

Computed from the batter's **last 3 completed innings before this match**. Varies per match. Generated by `extract_batter_form.py` using chronologically-sorted match data to prevent temporal leakage.

| Feature | Formula | Description |
|---|---|---|
| `batter_form_runs_last_3` | `sum(runs in last 3 innings)` | Total runs in last 3 innings |
| `batter_form_avg_last_3` | `total_runs / n_innings` | **Runs per innings** (not runs/outs — avoids inflation for not-out batters) |
| `batter_form_sr_last_3` | `runs / legal_balls × 100` | Strike rate in last 3 innings |
| `batter_form_bp_last_3` | `boundaries / legal_balls` | Boundary percentage in last 3 innings |

### E. Bowler Features (4 features)

Career + recent form stats for the bowler on this delivery. Generated by an upstream script.

| Feature | Description |
|---|---|
| `bowler_career_wickets` | Cumulative career wickets **up to this match** (rolling per-match, no temporal leakage) |
| `bowler_career_economy` | Cumulative career economy rate **up to this match** (rolling per-match) |
| `bowler_form_economy_last_3` | Economy rate in last 3 matches |
| `bowler_form_wickets_last_3` | Wickets taken in last 3 matches |

### F. Venue Features (3 features)

Historical pitch conditions. Generated by `extract_venue_features.py`. Uses **1st innings totals only** (2nd innings is constrained by target, so 1st innings better reflects pitch nature). All computed using matches **before** the current match (no temporal leakage).

| Feature | Description | Default |
|---|---|---|
| `venue_avg_runs_all` | Historical average 1st-innings total at this venue | Mean of all venues |
| `venue_avg_runs_last5` | Average 1st-innings total in last 5 matches at this venue | Mean of all venues |
| `venue_matches_played` | Number of matches completed at this venue before this match | 0 |

### G. Sequence / Momentum Features (4 features)

Per-ball momentum indicators. All use **`shift(1)`** (previous delivery's data) to prevent leakage. These are grouped by `(match_id, innings, batter_name)` — they track the batter's own sequence, not the team's.

**Important:** These features use `runs_batter` (batter's runs only, excluding extras like wides/no-balls) to be consistent with the `mpt_outcome` target.

| Feature | Formula | Description |
|---|---|---|
| `prev_ball_runs` | `shift(1)` of `runs_batter` | Batter runs scored on this batter's previous delivery (0 for first ball) |
| `avg_runs_last_3` | `shift(1).rolling(3).mean()` of `runs_batter` | Rolling 3-ball average for this batter (0 for first ball) |
| `prev_ball_boundary` | `1 if prev_ball_runs >= 4 else 0` | Whether the batter hit a boundary on their previous delivery |
| `feat_audacity` | Cumulative audacity score | Exponentially-weighted boundary streak metric from base pipeline |

### H. Ball-Level Contextual Features (8 features)

Computed directly from the JSON source files in a two-pass process within `create_training_data_v4.py`. All use **before-this-ball** state (features are recorded, then state is updated).

| Feature | Description | Leakage-Safe? |
|---|---|---|
| `dot_ball_pressure` | Count of consecutive dot balls this batter has faced (resets when they score) | ✅ Pre-ball |
| `partnership_runs` | Running total of the current batting partnership (resets on wicket) | ✅ Pre-ball |
| `batting_position` | Entry order 1–11, derived from actual batting appearance order in the JSON | ✅ Pre-ball |
| `wickets_last_12_balls` | Number of team wickets in the last 12 deliveries (collapse detector) | ✅ Pre-ball |
| `run_rate_ratio` | `Current_RR / Required_RR` (innings 2 only; 0 for innings 1). Values >1 = ahead of rate. Capped at 10. Uses **legal balls only** (excludes wides/no-balls from ball count). | ✅ Pre-ball |
| `batter_vs_bowler_career_sr` | Career SR of this batter vs this specific bowler across all matches (full career H2H lookup) | ✅ Static career |
| `batter_phase_career_sr` | Career SR of this batter in the current phase (PP/Mid/Death) across all matches | ✅ Static career |
| `bowler_death_economy` | This bowler's career economy rate in death overs (0-indexed: ≥15, i.e. overs 16–20) | ✅ Static career |

### I. Other (2 features)

| Feature | Description |
|---|---|
| `feat_unpredictability` | Trigram-based unpredictability score: how surprising this batter's shot sequence is compared to league averages (higher = less predictable) |
| `field_phase` | Categorical phase label: `Powerplay` (overs 0–5) / `Middle` (overs 6–14) / `Death` (overs 15–19) |

---

## Outcome Columns (NOT model features)

These columns are retained in the CSV for **audit and debugging** but must be **excluded from the model's feature set** — they contain the delivery's outcome, which is what `mpt_outcome` is derived from.

| Column | What It Contains | Why Kept |
|---|---|---|
| `total_runs` | Total runs scored on this delivery (batter + extras) | Audit: verify outcome mapping |
| `runs_batter` | Runs scored by the batter only (excludes wides/no-balls) | Audit: `mpt_outcome` is derived from this |
| `is_wicket` | 1 if a wicket fell on this delivery | Audit: `mpt_outcome=1` is derived from this |

## Categorical / Identifier Columns (not numeric features)

| Column | Role | Values |
|---|---|---|
| `match_id` | Unique match identifier | e.g. `1082591.json` |
| `venue` | Ground name | e.g. `M Chinnaswamy Stadium` |
| `batting_team` | Team batting on this delivery | e.g. `Mumbai Indians` |
| `batter_name` | Batter's name | e.g. `V Kohli` |
| `bowler_name` | Bowler's name | e.g. `JJ Bumrah` |
| `batter_hand` | Batter's stance | `Left` / `Right` |
| `bowler_hand` | Bowler's hand | `Left` / `Right` |
| `bowler_type` | Bowler's style | `Pace` / `Spin` |

---

## Data Integrity & Leakage Prevention

### Leakage Checks (all pass ✅)

| Check | Result |
|---|---|
| `target_score == 0` for all innings-1 deliveries | ✅ (innings-1 batters cannot see the final score) |
| `required_run_rate == 0` for all innings-1 deliveries | ✅ |
| `run_rate_ratio == 0` for all innings-1 deliveries | ✅ |
| `prev_ball_runs == 0` for each batter's first delivery | ✅ (shift(1) ensures no self-leakage) |
| `current_score` is pre-ball (before this delivery's runs) | ✅ (verified manually) |
| No `target_runs` column in final dataset | ✅ (dropped; was outcome data) |
| Wide deliveries not classified as dot balls | ✅ (mapped to outcome 3) |

### Imputation Strategy

All NaN values are handled with targeted, feature-appropriate imputation (not a blanket `fillna(0)`):

| Feature Group | Imputation | Rationale |
|---|---|---|
| Career skill features (16 columns) | **Median** of known players | Represents a "generic average player" for the 350 batters with <60 career balls |
| Venue features | **Mean** of all venues | Default pitch behavior |
| Bowler form features | **Median** | Avoids extreme zeros for bowlers with no recent matches |
| Ball-level features (dot pressure, partnership, etc.) | **0** | Natural default: no pressure, no partnership, no wickets |
| Sequence features (prev_ball_runs, avg, boundary) | **0** | First delivery defaults |
| Batting position | **7** | Lower-middle order default for unknowns |
| Categorical columns | **'Unknown'** | Explicit unknown category |

### Known Limitations

1. **Career feature coverage:** 353/703 unique batters (50.3%) have real career features. The remaining 350 have median-imputed values. These are generally low-volume players (<60 balls faced). Consider adding a binary `has_career_stats` flag for the model to learn different behaviors.
2. **`Hard_Hitting_Power` overlap:** Covers overs 7–20, which includes death overs (16–20). By design — captures general non-powerplay aggression, while `Death_Specialist_SR` isolates the final 5 overs.
3. **Full-career features include the current match:** `batter_vs_bowler_career_sr`, `batter_phase_career_sr`, and `bowler_death_economy` use all matches including the one being predicted. This is an **approved design choice** — these serve as "who is this player" descriptors, not temporal predictions.
4. **Bowler type classification:** Uses a mix of name-matching (for known bowlers) and middle-over bowling proportion (for unknown bowlers). Some fringe players may be misclassified.
5. **`is_wide` and `is_noball` are concurrent features:** These are known at delivery time, not before the ball is bowled. They must be **excluded from model features** to avoid target leakage — `is_wide=1` deterministically prevents `mpt_outcome=2` (dot ball).
