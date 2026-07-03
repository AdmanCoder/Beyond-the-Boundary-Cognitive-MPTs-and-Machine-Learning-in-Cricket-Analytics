# MPT Cricket Decision Tree — Feature Mapping (Final 7-Node Architecture)

## Tree Structure Recap

Our final Bayesian model (`v8_mpt_advi.stan`) uses a streamlined **7-node cognitive architecture** to map decision-making under uncertainty.

```
START
├── m (Premeditation)
│   ├── phi (Prediction Success)
│   │   ├── kappa (Contact Quality) → Boundary (4/6)
│   │   └── (1-kappa) → Error Node (epsilon) → Rotation (1/2/3)
│   │
│   └── (1-phi) (Prediction Failed)
│       └── kappa (Contact Quality) → Rotation (1/2/3)
│
└── (1-m) (Reactive)
    ├── r (Read Delivery Success)
    │   ├── a (Aggressive Intent)
    │   │   ├── kappa (Contact Quality) → Boundary (4/6)
    │   │   └── (1-kappa) → Error Node (epsilon) → Rotation (1/2/3)
    │   │
    │   └── (1-a) (Defensive/Rotational Intent)
    │       ├── s (Shot Execution)
    │       │   ├── kappa (Contact Quality) → Rotation (1/2/3)
    │       │   └── (1-kappa) → Fail (0/W)
    │       └── (1-s) → Fail (0/W)
    │
    └── (1-r) (Misread Delivery)
        ├── zeta (Improvisation)
        │   ├── kappa (Contact Quality) → Boundary (4/6)
        │   └── (1-kappa) → Error Node (epsilon) → Rotation (1/2/3)
        │
        └── (1-zeta) (Failed to Improvise)
            └── kappa (Contact Quality) → Rotation (1/2/3)
```

Outcome categories: **Boundary (4/6)** | **Rotation (1/2/3)** | **Fail (0/W)**

---

## Parameter Feature Mapping (Based on v8_mpt_advi.stan)

### 1. `m` — Premeditation (Latent Node)
*"Did the batter decide on a shot before the ball was bowled?"*
**Driving Features:**
*   `bsb` (Balls Since Boundary): Urgency triggers premeditation.
*   `current_score`: Settled batters feel freer to premeditate.
*   `over_num`: Death overs spike premeditation.
*   `batter_career_sr`: Intrinsic aggression baseline.

### 2. `phi` — Premeditated Prediction Success
*"If premeditated, did they correctly predict the bowler's length/line?"*
**Driving Features:**
*   `h2h_sr` (Head-to-Head Strike Rate): Familiarity improves prediction.
*   `balls_faced`: More time at the crease = better read of the pitch.
*   `bowler_career_economy`: Highly economical bowlers are harder to predict.
*   `batter_career_sr`: Proxies overall cognitive sharpness.

### 3. `r` — Reactive Read Success
*"If waiting reactively, did the batter successfully read the delivery?"*
**Driving Features:**
*   `balls_faced`: Eye-in factor drastically improves reactive reading.
*   `momentum_ewm`: Team momentum provides psychological clarity.
*   `batter_career_sr`: Baseline reaction speed and skill.

### 4. `a` — Aggressive Intent (Post-Read)
*"Having read the ball, did the batter choose to attack?"*
**Driving Features:**
*   `bsb`: Boundary drought forces aggressive intent.
*   `over_num`: Later overs demand attack.
*   `required_run_rate` (rrr): Scoreboard pressure forces aggression.
*   `pressure_index`: General match pressure index.

### 5. `s` — Defensive/Rotational Shot Execution
*"If choosing not to attack, did the batter successfully execute a rotational shot?"*
**Driving Features:**
*   `partnership_balls`: Established partnerships improve rotational execution.
*   `dot_pressure`: High dot pressure makes safe rotation harder.
*   `required_run_rate` (rrr): High RRR forces risky rotation.

### 6. `zeta` — Improvisation
*"If the batter misread the ball, were they able to instinctively improvise?"*
**Driving Features:**
*   `h2h_sr`: Familiarity allows for better last-second adjustments.
*   `pressure_index`: High pressure limits creative improvisation.
*   `dot_pressure`: Suffocation reduces improvisational success.

### 7. `kappa` — Contact Quality / Power
*"Regardless of intent, how clean was the physical contact with the ball?"*
**Driving Features:**
*   `current_score`: Settled batters time the ball significantly better.
*   `momentum_ewm`: Riding momentum leads to cleaner strikes.
*   `batter_career_sr`: Intrinsic power and timing baseline.
*   `pressure_index`: Handling pressure well mitigates contact degradation.
