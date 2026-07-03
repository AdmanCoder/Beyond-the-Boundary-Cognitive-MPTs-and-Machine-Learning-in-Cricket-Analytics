import pandas as pd
import numpy as np
import json
import os
from glob import glob
from collections import defaultdict

# CONFIG
BASE_TRAINING_FILE = "ipl_training_data.csv"
RAW_BALLS_FILE = "ipl_balls_raw_v2.csv"
BAT_FORM_FILE = "batter_form_features_v3.csv"
BOWL_FORM_FILE = "bowler_features_v2.csv"
VENUE_FEATS_FILE = "venue_features_v3.csv"
PLAYER_STATS_FILE = "IPL_Player_Stats_Advanced.csv"
JSON_DIR = "ipl_json"
OUTPUT_FILE = "ipl_training_data_v4.csv"


def compute_ball_level_features_from_json(json_dir):
    """
    Compute ball-level contextual features directly from JSON source of truth.
    Returns a DataFrame with one row per delivery, keyed by (match_id, innings, over, ball, batter).
    
    Features computed:
    - dot_ball_pressure: consecutive dots faced by this batter before this ball
    - partnership_runs: running total of current partnership
    - batting_position: ordinal position (1-11) based on entry order
    - wickets_last_12: number of team wickets in the last 12 balls
    - run_rate_ratio: current_run_rate / required_run_rate (innings 2 only)
    - batter_vs_bowler_career_sr: career SR of this batter vs this specific bowler
    - batter_phase_career_sr: career SR of this batter in current phase (PP/Mid/Death)
    - bowler_death_economy: this bowler's career economy in death overs (16-20)
    """
    files = sorted(glob(os.path.join(json_dir, '*.json')))
    print(f"Computing ball-level features from {len(files)} JSON files...")
    
    # === PASS 1: Pre-compute career aggregates across ALL matches ===
    # These are career-level (user said career features can use full career)
    batter_vs_bowler = defaultdict(lambda: {'runs': 0, 'balls': 0})  # (batter, bowler) -> stats
    batter_phase = defaultdict(lambda: {'runs': 0, 'balls': 0})     # (batter, phase) -> stats
    bowler_death = defaultdict(lambda: {'runs_conceded': 0, 'balls': 0})  # bowler -> death stats
    
    for filename in files:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except:
            continue
        
        for inning in data.get('innings', []):
            for over_data in inning.get('overs', []):
                over_num = over_data.get('over', 0)
                # Determine phase
                if over_num <= 5:
                    phase = 'Powerplay'
                elif over_num <= 14:
                    phase = 'Middle'
                else:
                    phase = 'Death'
                    
                for delivery in over_data.get('deliveries', []):
                    batter = delivery['batter']
                    bowler = delivery['bowler']
                    runs = delivery['runs']['batter']
                    total_runs = delivery['runs']['total']
                    
                    # Batter vs Bowler H2H
                    batter_vs_bowler[(batter, bowler)]['runs'] += runs
                    batter_vs_bowler[(batter, bowler)]['balls'] += 1
                    
                    # Batter Phase stats
                    batter_phase[(batter, phase)]['runs'] += runs
                    batter_phase[(batter, phase)]['balls'] += 1
                    
                    # Bowler Death Economy (overs 16-20, 0-indexed >= 15)
                    if over_num >= 15:
                        bowler_death[bowler]['runs_conceded'] += total_runs
                        bowler_death[bowler]['balls'] += 1
    
    # Convert to lookup dicts
    h2h_sr = {}
    for (bat, bowl), s in batter_vs_bowler.items():
        h2h_sr[(bat, bowl)] = (s['runs'] / s['balls'] * 100) if s['balls'] > 0 else 0
        
    phase_sr = {}
    for (bat, ph), s in batter_phase.items():
        phase_sr[(bat, ph)] = (s['runs'] / s['balls'] * 100) if s['balls'] > 0 else 0
        
    death_econ = {}
    for bowl, s in bowler_death.items():
        death_econ[bowl] = (s['runs_conceded'] / s['balls'] * 6) if s['balls'] > 0 else 0
    
    print(f"  Career aggregates: {len(h2h_sr)} H2H pairs, {len(phase_sr)} phase entries, {len(death_econ)} bowlers")
    
    # === PASS 2: Compute per-ball contextual features ===
    all_rows = []
    
    for filename in files:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except:
            continue
        
        match_id = os.path.basename(filename)
        info = data.get('info', {})
        
        # Get first innings total for target
        inn1_total = 0
        if len(data.get('innings', [])) >= 1:
            for ov in data['innings'][0].get('overs', []):
                for d in ov.get('deliveries', []):
                    inn1_total += d['runs']['total']
        target = inn1_total + 1
        
        for inning_idx, inning in enumerate(data.get('innings', [])):
            innings_num = inning_idx + 1
            
            # Tracking state for this innings
            batter_consec_dots = defaultdict(int)  # batter -> consecutive dots
            partnership_runs_counter = 0
            batter_entry_order = {}
            entry_counter = 0
            recent_wicket_balls = []  # list of ball_global indices where wickets fell
            current_score = 0
            ball_global = 0
            legal_balls_global = 0  # FIX BUG 2: only count legal deliveries
            total_wickets = 0
            
            for over_data in inning.get('overs', []):
                over_num = over_data.get('over', 0)
                
                if over_num <= 5:
                    phase = 'Powerplay'
                elif over_num <= 14:
                    phase = 'Middle'
                else:
                    phase = 'Death'
                
                for ball_idx, delivery in enumerate(over_data.get('deliveries', [])):
                    batter = delivery['batter']
                    bowler = delivery['bowler']
                    runs_batter = delivery['runs']['batter']
                    total_ball_runs = delivery['runs']['total']
                    ball_num = ball_idx + 1  # Approximate; JSON doesn't always have ball num
                    
                    # FIX BUG 2: Track legal vs total deliveries separately
                    # Wides and no-balls do NOT consume from the 120-ball quota
                    extras = delivery.get('extras', {})
                    is_wide = 'wides' in extras
                    is_noball = 'noballs' in extras
                    is_legal_delivery = not is_wide and not is_noball
                    
                    ball_global += 1
                    if is_legal_delivery:
                        legal_balls_global += 1
                    
                    # === Batting Position ===
                    if batter not in batter_entry_order:
                        entry_counter += 1
                        batter_entry_order[batter] = entry_counter
                    batting_position = batter_entry_order[batter]
                    
                    # === Dot Ball Pressure (BEFORE this ball) ===
                    dot_pressure = batter_consec_dots[batter]
                    
                    # === Partnership Runs (BEFORE this ball) ===
                    part_runs = partnership_runs_counter
                    
                    # === Wickets in Last 12 Balls (BEFORE this ball) ===
                    wickets_last_12 = sum(1 for wb in recent_wicket_balls if ball_global - wb <= 12)
                    
                    # === Run Rate Ratio (innings 2 only) ===
                    # FIX BUG 2: Use legal_balls_global (excludes wides/no-balls)
                    if innings_num == 2:
                        balls_bowled = legal_balls_global
                        current_rr = (current_score / balls_bowled * 6) if balls_bowled > 0 else 0
                        balls_left = max(1, 120 - legal_balls_global)
                        runs_needed = max(0, target - current_score)
                        required_rr = runs_needed / balls_left * 6
                        rr_ratio = current_rr / required_rr if required_rr > 0 else 1.0
                        rr_ratio = min(rr_ratio, 10.0)  # Cap extreme values
                    else:
                        rr_ratio = 0.0
                    
                    # === Career lookups ===
                    h2h = h2h_sr.get((batter, bowler), 0)
                    phase_career = phase_sr.get((batter, phase), 0)
                    bowler_de = death_econ.get(bowler, 0)
                    
                    all_rows.append({
                        'match_id': match_id,
                        'innings': innings_num,
                        'over': over_num,
                        'batter': batter,
                        'bowler': bowler,
                        'ball_global_json': ball_global,
                        'dot_ball_pressure': dot_pressure,
                        'partnership_runs': part_runs,
                        'batting_position': batting_position,
                        'wickets_last_12_balls': wickets_last_12,
                        'run_rate_ratio': round(rr_ratio, 4),
                        'batter_vs_bowler_career_sr': round(h2h, 2),
                        'batter_phase_career_sr': round(phase_career, 2),
                        'bowler_death_economy': round(bowler_de, 2),
                    })
                    
                    # === Update state AFTER recording (features use state BEFORE this ball) ===
                    current_score += total_ball_runs
                    partnership_runs_counter += total_ball_runs
                    
                    # Update consecutive dots
                    if runs_batter == 0:
                        batter_consec_dots[batter] += 1
                    else:
                        batter_consec_dots[batter] = 0
                    
                    # Wicket handling
                    if 'wickets' in delivery:
                        recent_wicket_balls.append(ball_global)
                        partnership_runs_counter = 0  # Reset partnership
                        total_wickets += 1
                        # New batter arrives — will be tracked on next delivery
    
    df_ball = pd.DataFrame(all_rows)
    print(f"  Generated {len(df_ball)} ball-level feature rows from JSON")
    return df_ball


def create_training_data_v4():
    print(f"Loading base data from {BASE_TRAINING_FILE}...")
    df_base = pd.read_csv(BASE_TRAINING_FILE)
    
    print(f"Loading raw ball data from {RAW_BALLS_FILE}...")
    df_raw = pd.read_csv(RAW_BALLS_FILE, usecols=[
        'match_id', 'innings', 'over', 'ball', 'batter', 'bowler',
        'is_wicket', 'total_runs', 'runs_batter', 'current_score', 'is_legal', 'is_wide', 'is_noball'
    ])
    if not str(df_raw['match_id'].iloc[0]).endswith('.json'):
        df_raw['match_id'] = df_raw['match_id'].astype(str) + '.json'
        
    print("Merging Raw context (Sequence-Safe Alignment)...")
    # Step 1: Standardize Match IDs
    if not str(df_raw['match_id'].iloc[0]).endswith('.json'):
        df_raw['match_id'] = df_raw['match_id'].astype(str) + '.json'
    if not str(df_base['match_id'].iloc[0]).endswith('.json'):
        df_base['match_id'] = df_base['match_id'].astype(str) + '.json'

    # Step 2: Batter-Sequence Alignment (Including Extras)
    df_raw['batter_ball_seq'] = df_raw.groupby(['match_id', 'innings', 'batter']).cumcount()
    df_base['batter_ball_seq'] = df_base.groupby(['match_id', 'innings', 'batter_name']).cumcount()
    
    # Create the join
    df = pd.merge(
        df_base, 
        df_raw[['match_id', 'innings', 'batter', 'batter_ball_seq', 'is_wicket', 'total_runs', 'runs_batter', 'over', 'ball', 'is_wide', 'is_noball']], 
        left_on=['match_id', 'innings', 'batter_name', 'batter_ball_seq'],
        right_on=['match_id', 'innings', 'batter', 'batter_ball_seq'],
        how='left'
    )
    
    # Validation Check
    mask = df['total_runs'].notna()
    mismatch = (df[mask]['total_runs'] != df[mask]['target_runs']).mean()
    print(f"Alignment Audit: Outcome Mismatch Rate = {mismatch:.2%}")

    # === DROP DUPLICATE / LEAKAGE FEATURES ===
    print("Dropping duplicate/redundant/leakage columns...")
    drop_cols = ['feat_clutch', 'feat_head_stability', 'feat_pressure', 'batter_form',
                 'overs_done', 'ball_no_global', 'prev_ball_boundary',
                 'target_runs']  # target_runs is per-delivery outcome data (leakage)
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            print(f"  Dropped: {col}")

    # 1. Joins for Form and Venue Features
    print("Joining Form and Venue Features...")
    df_bat_form = pd.read_csv(BAT_FORM_FILE)
    df_bowl_form = pd.read_csv(BOWL_FORM_FILE)
    df_venue = pd.read_csv(VENUE_FEATS_FILE)
    
    for d in [df_bat_form, df_bowl_form, df_venue]:
        if not str(d['match_id'].iloc[0]).endswith('.json'):
            d['match_id'] = d['match_id'].astype(str) + '.json'
            
    df = pd.merge(df, df_bat_form, left_on=['match_id', 'batter_name'], right_on=['match_id', 'batter'], how='left', suffixes=('', '_form'))
    df = pd.merge(df, df_bowl_form, left_on=['match_id', 'bowler_name'], right_on=['match_id', 'bowler'], how='left', suffixes=('', '_bowl'))
    df = pd.merge(df, df_venue, on=['match_id', 'venue'], how='left')
    
    # 2. Join Advanced Career Skills (now includes 6 new features)
    print("Merging Advanced Career Skills...")
    df_skills = pd.read_csv(PLAYER_STATS_FILE)
    df_skills = df_skills.rename(columns={'Name': 'batter_name'})
    df = pd.merge(df, df_skills.drop(columns=['Total_Runs', 'Audacity', 'Unpredictability'], errors='ignore'), 
                  on='batter_name', how='left')

    # 3. Calculated Momentum and Pressure
    print("Calculating Pressure and Momentum...")
    df = df.sort_values(['match_id', 'innings', 'over', 'ball'])
    
    # Override inherited field_phase with correct boundaries (PP: 0-5, Mid: 6-14, Death: 15-19)
    df['field_phase'] = np.where(df['over'] <= 5, 'Powerplay',
                        np.where(df['over'] <= 14, 'Middle', 'Death'))
    
    inn1 = df[df['innings'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    inn1.rename(columns={'total_runs': 'target_score_calculated'}, inplace=True)
    inn1['target_score_calculated'] += 1
    df = pd.merge(df, inn1, on='match_id', how='left')
    
    # FIX Error 1: target_score only applies to innings 2 (chase target)
    # Innings 1 batters should NOT see the final innings total (future leakage)
    df['target_score'] = np.where(
        df['innings'] == 2,
        df['target_score_calculated'],
        0
    )
    
    # FIX BUG 1: Only count legal deliveries for balls_remaining
    # ball can exceed 6 due to extras (wides/no-balls), which don't consume quota
    df['balls_remaining'] = 120 - (df['over'] * 6 + df['ball'].clip(upper=6))
    df['balls_remaining'] = df['balls_remaining'].clip(lower=1)
    df['required_run_rate'] = np.where(
        df['innings'] == 2,
        (df['target_score'] - df['current_score']) / df['balls_remaining'] * 6,
        0
    ).clip(0, 36)
    
    # FIX Error 6: Use runs_batter (not total_runs) for batter sequence features
    # total_runs includes extras (wides/noballs) which aren't the batter's contribution
    group_batter_final = df.groupby(['match_id', 'innings', 'batter_name'])
    df['prev_ball_runs'] = group_batter_final['runs_batter'].shift(1).fillna(0)
    df['avg_runs_last_3'] = group_batter_final['runs_batter'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).fillna(0)
    # prev_ball_boundary: 1 if the batter's previous ball was a boundary (4+)
    df['prev_ball_boundary'] = np.where(df['prev_ball_runs'] >= 4, 1, 0)

    # 4. Merge Ball-Level Features from JSON (source of truth)
    print("Computing and merging ball-level features from JSON...")
    json_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), JSON_DIR)
    df_ball_feats = compute_ball_level_features_from_json(json_dir)
    
    # Create sequence key for merge (same batter-ball-seq approach)
    df_ball_feats['batter_ball_seq'] = df_ball_feats.groupby(['match_id', 'innings', 'batter']).cumcount()
    
    ball_feat_cols = [
        'dot_ball_pressure', 'partnership_runs', 'batting_position',
        'wickets_last_12_balls', 'run_rate_ratio',
        'batter_vs_bowler_career_sr', 'batter_phase_career_sr', 'bowler_death_economy'
    ]
    
    df = pd.merge(
        df,
        df_ball_feats[['match_id', 'innings', 'batter', 'batter_ball_seq'] + ball_feat_cols],
        left_on=['match_id', 'innings', 'batter_name', 'batter_ball_seq'],
        right_on=['match_id', 'innings', 'batter', 'batter_ball_seq'],
        how='left',
        suffixes=('', '_json')
    )
    
    # Drop the extra batter column from JSON merge
    if 'batter_json' in df.columns:
        df.drop(columns=['batter_json'], inplace=True)
    
    # 5. Outcome Mapping (uses runs_batter, handles wides/noballs)
    # FIX Error 12: Wide deliveries are not dot balls — exclude from batter outcomes
    # Wides: batter doesn't face the ball, so runs_batter=0 but it's NOT a dot ball.
    # We filter out wides before model training (they don't represent batter decisions)
    # but we still map them correctly here for transparency.
    def map_outcome(row):
        if row['is_wicket'] == 1: return 1
        # Wide balls: not a batter-faced delivery — mark as special case
        # A wide with runs_batter=0 is NOT a dot ball the batter played
        if row.get('is_wide', 0) == 1:
            # Map by total extras runs for wides (1 wide = 1 run minimum)
            r = int(row['runs_batter']) if not pd.isna(row['runs_batter']) else 0
            if r == 0: return 3  # Wide = 1 run to team, classify as single-equivalent
            # If batter ran on a wide, use batter runs
        r = int(row['runs_batter']) if not pd.isna(row['runs_batter']) else 0
        if r == 0: return 2
        if r == 1: return 3
        if r == 2: return 4
        if r == 3: return 5
        if r == 4: return 6
        if r >= 5: return 7
        return 3
    df['mpt_outcome'] = df.apply(map_outcome, axis=1)

    # 6. Smart Statistical Imputation
    print("Applying Statistical Imputation...")
    venue_metrics = ['venue_avg_runs_all', 'venue_avg_runs_last5']
    player_metrics = [
        'Hand_Eye_Pace', 'Head_Stability_Spin', 'Technical_Adeptness', 
        'Hard_Hitting_Power', 'Clutch_Index', 'Chase_IQ',
        # New career features
        'Counter_Attack', 'Bowler_Reading', 'Perception_Skills',
        'Shot_Inventory', 'Death_Specialist_SR', 'Powerplay_Specialist_SR',
        'Pressure_Absorb', 'Tough_Pitch_Performance', 'Consistency_Gini', 'Lone_Wolf'
    ]
    
    for col in venue_metrics:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
            
    for col in player_metrics:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Drop merge artifact columns (duplicate names from joins)
    merge_artifacts = ['batter_form', 'batter', 'bowler', 'batter_ball_seq', 'target_score_calculated']
    for col in merge_artifacts:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            print(f"  Dropped merge artifact: {col}")
    
    # Outcome columns: must NOT be model features (they contain the outcome)
    # is_wide and is_noball: concurrent delivery data, also must NOT be model features
    outcome_cols = ['total_runs', 'runs_batter', 'is_wicket']
    print("  NOTE: 'total_runs', 'runs_batter', 'is_wicket' retained for audit but are outcome columns (exclude from features)")
    print("  NOTE: 'is_wide', 'is_noball' retained for audit but are concurrent features (exclude from features)")
    
    # FIX Error 8: Targeted imputation instead of blanket fillna(0)
    # Ball-level JSON features: 0 is a valid default (no pressure, no partnership, etc.)
    ball_level_zero_fill = ['dot_ball_pressure', 'partnership_runs', 'wickets_last_12_balls',
                            'run_rate_ratio', 'batter_vs_bowler_career_sr', 'batter_phase_career_sr',
                            'bowler_death_economy']
    for col in ball_level_zero_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Sequence features: 0 means no prior data (valid)
    sequence_zero_fill = ['prev_ball_runs', 'avg_runs_last_3', 'prev_ball_boundary']
    for col in sequence_zero_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Bowler form features: use median (better than 0 for rates/counts)
    bowler_form_cols = ['bowler_career_wickets', 'bowler_career_economy',
                        'bowler_form_economy_last_3', 'bowler_form_wickets_last_3']
    for col in bowler_form_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Batting position: default 7 (lower-middle order) for unknowns
    if 'batting_position' in df.columns:
        df['batting_position'] = df['batting_position'].fillna(7)
    
    # Categorical/identifier NaNs: fill with 'Unknown'
    cat_cols = ['batter_hand', 'bowler_hand', 'bowler_type', 'field_phase']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Final safety: any remaining NaN -> 0 (with warning)
    remaining_nans = df.isna().sum()
    if remaining_nans.sum() > 0:
        print(f"  WARNING: {remaining_nans.sum()} remaining NaNs being filled with 0:")
        for col_name, count in remaining_nans[remaining_nans > 0].items():
            print(f"    {col_name}: {count}")
    df.fillna(0, inplace=True)
    
    print(f"Saving final dataset to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    print(f"\n=== V4 Pipeline Complete ===")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"New ball-level features: {ball_feat_cols}")
    print(f"Dropped duplicates: {drop_cols}")
    print(f"NaN remaining: {df.isna().sum().sum()}")

if __name__ == "__main__":
    create_training_data_v4()
