import json
import os
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import difflib
import re

# --- 1. Math Helper Functions ---
def calculate_entropy(prob_dist):
    if not prob_dist: return 0
    return -sum(p * math.log2(p) for p in prob_dist if p > 0)

def calculate_gini(incomes):
    if incomes.size == 0: return 0
    incomes = np.sort(incomes)
    n = len(incomes)
    index = np.arange(1, n + 1)
    return ((2 * index - n - 1) * incomes).sum() / (n * incomes.sum()) if incomes.sum() > 0 else 0

def parse_height(height_str):
    """Converts '6 ft 2 in' to cm."""
    if not isinstance(height_str, str): return None
    try:
        # Regex for 'X ft Y in' or 'X ft'
        match = re.search(r'(\d+)\s*ft\s*(\d*)\s*in', height_str)
        if match:
            ft = int(match.group(1))
            inch = int(match.group(2)) if match.group(2) else 0
            return int((ft * 30.48) + (inch * 2.54))
        
        match_ft = re.search(r'(\d+)\s*ft', height_str)
        if match_ft:
            return int(int(match_ft.group(1)) * 30.48)
    except:
        return None
    return None

def classify_bowler_style(style_str):
    """Returns 'Pace' or 'Spin' based on string."""
    if not isinstance(style_str, str): return 'Unknown'
    s = style_str.lower()
    if any(x in s for x in ['fast', 'medium', 'pace', 'seam']):
        return 'Pace'
    if any(x in s for x in ['spin', 'break', 'orthodox', 'chinaman', 'googly']):
        return 'Spin'
    return 'Unknown'

# --- 2. Data Loading & Merging ---
def normalize_name(name):
    """Removes punctuation and extra spaces."""
    return re.sub(r'[^a-zA-Z\s]', '', name).lower().strip()

def load_auxiliary_data(csv_bio_path, csv_cricket_data_path):
    """
    Loads TWO datasets into a SMART Lookup DB.
    Structure:
    {
        'lastname_map': { 'kohli': [obj, obj], 'dhoni': [obj] },
        'fullname_map': { 'virat kohli': obj }
    }
    """
    player_db = {'lastname_map': defaultdict(list), 'fullname_map': {}}
    
    def add_to_db(name, data):
        norm = normalize_name(name)
        data['full_norm'] = norm
        player_db['fullname_map'][norm] = data
        
        # Index by EVERY part of the name to catch "Mohammed Shami Ahmed" via "Shami"
        parts = norm.split()
        for part in parts:
            if len(part) > 2: # Skip initials like 'm'
                player_db['lastname_map'][part].append(data)

    # A. Load Age from Bio CSV
    print("Loading Age Data...")
    try:
        df_bio = pd.read_csv(csv_bio_path)
        for _, row in df_bio.iterrows():
            if pd.isna(row['fullname']): continue
            dob_year = None
            try:
                dob_year = datetime.strptime(str(row['dateofbirth']), "%d-%m-%Y").year
            except: pass
            
            data = {'dob_year': dob_year, 'height': None, 'bowling_style': 'Unknown', 'playing_role': 'Unknown'}
            add_to_db(str(row['fullname']), data)
    except Exception as e:
        print(f"Bio Load Error: {e}")

    # B. Load Height/Style from Cricket Data CSV
    print("Loading Height & Style Data...")
    try:
        df_cric = pd.read_csv(csv_cricket_data_path, low_memory=False)
        for _, row in df_cric.iterrows():
            name = str(row['Full name']) if pd.notna(row['Full name']) else str(row['NAME'])
            if pd.isna(name): continue
            
            ht_cm = None
            if pd.notna(row['Height']): ht_cm = parse_height(str(row['Height']))
            
            style = 'Unknown'
            if pd.notna(row['Bowling style']): style = classify_bowler_style(str(row['Bowling style']))
            
            role = 'Unknown'
            if pd.notna(row['Playing role']): role = str(row['Playing role']).strip()
            
            # Update existing or create new
            norm = normalize_name(name)
            if norm in player_db['fullname_map']:
                # Update existing
                if ht_cm: player_db['fullname_map'][norm]['height'] = ht_cm
                if style != 'Unknown': player_db['fullname_map'][norm]['bowling_style'] = style
                if role != 'Unknown': player_db['fullname_map'][norm]['playing_role'] = role
            else:
                data = {'dob_year': None, 'height': ht_cm, 'bowling_style': style, 'playing_role': role}
                add_to_db(name, data)
                 
    except Exception as e:
        print(f"CricData Load Error: {e}")
        
    return player_db

def get_player_meta(json_name, db):
    """
    Smart Matcher:
    1. Exact Fullname Match
    2. 'Initial Lastname' Match (e.g. 'V Kohli' -> 'Virat Kohli')
    """
    norm = normalize_name(json_name)
    
    best_cand = None
    
    # 1. Exact Match
    if norm in db['fullname_map']:
        cand = db['fullname_map'][norm]
        # If exact match has Role, it's the gold standard. Return.
        if cand.get('playing_role', 'Unknown') != 'Unknown':
            return cand
        # Else, keep it as fallback but keep searching for a richer entry
        best_cand = cand
    
    # 2. Initials Match (e.g. 'R Ashwin' -> 'Ravichandran Ashwin')
    parts = norm.split()
    if len(parts) >= 2 and len(parts[0]) == 1:
        initial = parts[0]
        lastname = parts[-1] 
        
        # Look for candidates with same last name
        candidates = db['lastname_map'].get(lastname, [])
        
        # Filter by first initial
        valid_cands = []
        for c in candidates:
            # Check if this candidate's fullname starts with the initial
            c_norm = c['full_norm']
            if c_norm.startswith(initial):
                valid_cands.append(c)
        
        # Heuristic: If we found exactly 1 valid candidate (or multiple but they are the same person), use it
        if valid_cands:
            # Prioritize candidates with known Role/Style (Bio Data)
            bio_cands = [c for c in valid_cands if c.get('bowling_style', 'Unknown') != 'Unknown']
            
            if len(bio_cands) >= 1:
                cand = bio_cands[0]
                # If we already have an exact match candidate, only override if new one has better data
                if not best_cand or (best_cand.get('bowling_style', 'Unknown') == 'Unknown'):
                    best_cand = cand
            elif not best_cand: # No Bio Data, just take the name match
                 best_cand = valid_cands[0]
                 
    # 3. Lastname Fallback (Risky, only if unique and we are desperate)
    if not best_cand:
        lastname = norm.split()[-1]
        cands = db['lastname_map'].get(lastname, [])
        if len(cands) == 1:
            best_cand = cands[0]
    
    # 2. Initials / Fuzzy Logic
    parts = norm.split()
    if len(parts) >= 2:
        lname = parts[-1]
        fname_initial = parts[0][0]
        
        candidates = db['lastname_map'].get(lname, [])
        if "shami" in norm:
            print(f"DEBUG SHAMI: Candidates for '{norm}': {[(c['full_norm'] + ':' + str(c.get('playing_role'))) for c in candidates]}")
            
        for cand in candidates:
            # Check First Name Initial
            cand_parts = cand['full_norm'].split()
            # Ensure we don't re-check the exact match if we already have it
            if cand['full_norm'] == norm:
                continue
                
                
            if len(cand_parts) > 0 and cand_parts[0].startswith(fname_initial):
                # We found a potential match.
                if not best_cand:
                    best_cand = cand
                else:
                    # Upgrade if we find Role
                    curr_role = best_cand.get('playing_role', 'Unknown')
                    new_role = cand.get('playing_role', 'Unknown')
                    
                    new_role = cand.get('playing_role', 'Unknown')
                    
                    if curr_role == 'Unknown' and new_role != 'Unknown':
                        best_cand = cand
                    # Break tie with Height
                    elif best_cand.get('height') is None and cand.get('height') is not None:
                        best_cand = cand
                        
        if best_cand: return best_cand
                
    return {'dob_year': None, 'height': None, 'bowling_style': 'Unknown', 'playing_role': 'Unknown'}


# --- 3. Main Extraction ---
def extract_features(json_dir, bio_csv, cric_csv):
    meta_db = load_auxiliary_data(bio_csv, cric_csv)
    
    # Store aggregation stats
    player_stats = defaultdict(lambda: {
        'matches': set(),
        'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0, 'dots': 0,
        'outs': 0, 
        # Context Specific
        'balls_vs_pace': 0, 'runs_vs_pace': 0, 'dots_vs_pace': 0, 'outs_vs_pace': 0,
        'balls_vs_spin': 0, 'runs_vs_spin': 0, 'dots_vs_spin': 0, 'outs_vs_spin': 0,
        # Clutch
        'death_runs': 0, 'death_balls': 0,
        'playoff_runs': 0, 'playoff_outs': 0,
        # Chase
        'chase_runs': 0, 'chase_outs': 0, 'chase_matches_won': 0,
        # Cognitive: Pressure
        'pressure_accum_balls': 0, 'pressure_events': 0, # Balls to boundary after wicket
        # Cognitive: Adaptability
        'markov_dots': 0, 'markov_bound_after_dot': 0,
        # Hard Hitting
        'middle_over_runs': 0, 'middle_over_balls': 0,
        # Technical (Wrist/Rotation)
        'runs_123': 0, 
        'non_boundary_runs': 0, 'boundary_balls': 0,
        'fours_vs_spin': 0, 
        'tough_runs': 0, # Low Scoring Matches
        # Entropy & Gini
        'run_distribution': defaultdict(int),
        'innings_scores': []
    })

    files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    print(f"Processing {len(files)} matches...")

    for filename in files:
        try:
            with open(os.path.join(json_dir, filename)) as f: data = json.load(f)
        except: continue

        info = data.get('info', {})
        dates = info.get('dates', ['2008-01-01'])
        match_year = int(dates[0][:4])

        # --- PRE-CALCULATE MATCH DIFFICULTY ---
        total_match_runs = 0
        total_match_balls = 0
        for inning in data.get('innings', []):
            for over in inning.get('overs', []):
                for ball in over.get('deliveries', []):
                    total_match_runs += ball['runs']['total']
                    total_match_balls += 1
        
        match_rr = (total_match_runs / total_match_balls) * 6 if total_match_balls > 0 else 10
        is_tough_pitch = match_rr < 7.5
        
        # Determine if Playoff
        stage = info.get('stage', '').lower() # Qualifier, Final, Eliminator
        is_playoff = any(x in stage for x in ['final', 'qualifier', 'eliminator', 'semi'])
        
        winner = info.get('outcome', {}).get('winner')
        
        # Track active partnership wickets for Pressure Absorption
        # We need to know when a wicket fell in the INNING context
        
        for inning_idx, inning in enumerate(data.get('innings', [])):
            inning_num = inning_idx + 1
            batting_team = inning.get('team')
            is_chase = (inning_num == 2)
            team_won = (winner == batting_team)
            
            # Per-Inning Helper to track "Balls since last wicket" for each batter?
            # Actually, "Balls to boundary after PARTNER wicket".
            # Simplification: Global "Wicket Just Happened" flag for the team.
            wicket_just_fell = False
            balls_since_wicket = 0
            
            current_inn_scores = defaultdict(int)

            for over_data in inning.get('overs', []):
                over_num = over_data.get('over') # 0-19
                
                # Check Bowler Type
                # JSON doesn't list bowler type per ball, we must look it up in OUR DB
                # Note: This is an approximation if multiple bowlers used. Usually 1 per over.
                # Actually, JSON has 'bowler' name per ball. Perfect.
                
                for ball in over_data.get('deliveries', []):
                    batter = ball.get('batter')
                    bowler = ball.get('bowler')
                    runs = ball.get('runs', {}).get('batter', 0)
                    extras = ball.get('runs', {}).get('extras', 0)
                    total_ball_runs = runs + extras # For boundary check? usually boundary is batter runs
                    
                    # Update Stats
                    p = player_stats[batter]
                    p['matches'].add(filename)
                    p['runs'] += runs
                    p['balls'] += 1
                    p['run_distribution'][runs] += 1
                    current_inn_scores[batter] += runs
                    
                    # 1. Bowler Type Stats
                    b_meta = get_player_meta(bowler, meta_db)
                    b_style = b_meta['bowling_style']
                    
                    if b_style == 'Pace':
                        p['balls_vs_pace'] += 1
                        p['runs_vs_pace'] += runs
                        if runs == 0: p['dots_vs_pace'] += 1
                    elif b_style == 'Spin':
                        p['balls_vs_spin'] += 1
                        p['runs_vs_spin'] += runs
                        if runs == 0: p['dots_vs_spin'] += 1
                        
                    # 2. Clutch (Death Overs) & Playoffs
                    if over_num >= 15: # 16-20th over
                        p['death_runs'] += runs
                        p['death_balls'] += 1
                    
                    if is_playoff:
                        p['playoff_runs'] += runs
                        
                    # 3. Chase IQ
                    if is_chase and team_won:
                        p['chase_runs'] += runs
                        # Outs handled later
                        
                    # 4. Wrist & Hard Hitting & Technical
                    # Tough Pitch
                    if is_tough_pitch:
                        p['tough_runs'] += runs
                        
                    # Technical (Rotation)
                    if runs in [1, 2, 3]:
                        p['runs_123'] += runs
                        p['non_boundary_runs'] += runs
                    elif runs >= 4:
                        p['boundary_balls'] += 1

                    if runs == 4: 
                        p['fours'] += 1
                        if b_style == 'Spin': p['fours_vs_spin'] += 1 # Technical Boundary
                    if runs == 6: p['sixes'] += 1
                    if runs == 0: p['dots'] += 1
                    
                    if 6 <= over_num <= 19: # Overs 7-20 (Power Phase)
                        p['middle_over_runs'] += runs
                        p['middle_over_balls'] += 1
                        
                    # 5. Pressure Absorption (Counter-Attack Delay)
                    if wicket_just_fell:
                        # If this batter hits a boundary, record the "delay"
                        if runs >= 4:
                            p['pressure_accum_balls'] += balls_since_wicket
                            p['pressure_events'] += 1
                            wicket_just_fell = False # Reset flag, pressure released
                            balls_since_wicket = 0
                        else:
                            # He defended/took single. Pressure continues? 
                            # We increment the counter for THIS batter? 
                            # Logic Check: This needs per-batter tracking.
                            # Simplified: We just increment a global counter? No.
                            # Let's Skip complex logic for now and use "Avg balls to get off mark" as pressure?
                            # User asked for "Balls to boundary after wicket".
                            # Let's count balls faced by EVERYONE until a 4/6 matches.
                            balls_since_wicket += 1
                            
                    # 6. Wicket Event
                    if 'wickets' in ball:
                        wicket_just_fell = True
                        balls_since_wicket = 0 # Reset counter for new pressure phase
                        for w in ball['wickets']:
                            out_player = w['player_out']
                            player_stats[out_player]['outs'] += 1
                            
                            # Specific Outs
                            if b_style == 'Pace': player_stats[out_player]['outs_vs_pace'] += 1
                            if b_style == 'Spin': player_stats[out_player]['outs_vs_spin'] += 1
                            
                            if is_playoff: player_stats[out_player]['playoff_outs'] += 1
                            if is_chase and team_won: player_stats[out_player]['chase_outs'] += 1

            # End of Inning: Record scores for Gini
            for bat, score in current_inn_scores.items():
                player_stats[bat]['innings_scores'].append(score)
                if is_chase and team_won and bat not in [w['player_out'] for o in inning.get('overs', []) for d in o.get('deliveries', []) for w in d.get('wickets', [])]:
                     # Not out in successful chase
                     pass # handled by total outs being 0 for this innings


    
    print("Calculating Derived Features...")
    final_data = []
    
    # DEBUG: Track filter counts
    kept_count = 0
    dropped_count = 0
    
    for p, stats in player_stats.items():
        # Confidence Weighting Alpha
        # Raw Score * (1 + (ALPHA * log10(Volume + 1)))
        WEIGHT_ALPHA = 0.2
        
        if stats['balls'] < 60: continue # Basic filter
        
        meta = get_player_meta(p, meta_db)
        role = str(meta.get('playing_role', 'Unknown')).lower().strip()
        
        # DEBUG: Print specific players to see why they persist
        if "shami" in p.lower() or "bumrah" in p.lower() or "siraj" in p.lower():
            print(f"DEBUG: {p} -> Role: '{role}' (Meta Found: {meta.get('height') is not None or meta.get('dob_year') is not None})")

        # User Filter: Remove purely "Bowler"
        # 1. Explicit Role Check
        if 'bowler' in role and 'allrounder' not in role:
            dropped_count += 1
            continue
        if '12th man' in role:
            dropped_count += 1
            continue
            
        # 2. Hardcoded Exclusion (Safety Net)
        # Explicitly remove known bowlers if role matching failed
        known_bowlers = ["shami", "bumrah", "siraj", "chahal", "mustafizur", "trent boult", "kagiso rabada", "anrich nortje"]
        if any(b in p.lower() for b in known_bowlers):
            dropped_count += 1
            continue

        # 3. Statistical Heuristic for 'Unknown' Roles
        # If Role is Unknown, filter out players with low batting stats (likely bowlers)
        if role == 'unknown':
            sr = (stats['runs'] / stats['balls']) * 100
            avg = stats['runs'] / stats['outs'] if stats['outs'] > 0 else stats['runs']
            
            # Threshold: Avg < 15 AND SR < 120 (Generous for lower order batsmen, strict for bowlers)
            # Only apply if sample size is decent (>60 balls is already filtered)
            if avg < 15 and sr < 125:
                dropped_count += 1
                continue
            
        # Basic Rates
        sr = (stats['runs'] / stats['balls']) * 100
        avg = stats['runs'] / stats['outs'] if stats['outs'] > 0 else stats['runs']
        
        # 1. Hand-Eye (Pace Destroyer)
        # SR_Pace * Contact_Pace
        pace_balls = stats['balls_vs_pace']
        if pace_balls > 30:
            sr_pace = (stats['runs_vs_pace'] / pace_balls) * 100
            contact_pace = (pace_balls - stats['dots_vs_pace']) / pace_balls
            raw_hand_eye = (sr_pace * contact_pace) / 10 # Normalize
            # Weighting
            vol_pace_runs = stats['runs_vs_pace']
            hand_eye = raw_hand_eye * (1 + (WEIGHT_ALPHA * math.log10(vol_pace_runs + 1)))
        else:
            hand_eye = 0
            
        # 2. Stability (Spin Dominance)
        # SR_Spin * Control_Spin
        spin_balls = stats['balls_vs_spin']
        if spin_balls > 30:
            sr_spin = (stats['runs_vs_spin'] / spin_balls) * 100
            control_spin = (spin_balls - stats['dots_vs_spin']) / spin_balls # Using contact as control proxy
            raw_stability = (sr_spin * control_spin) / 10
            # Weighting
            vol_spin_runs = stats['runs_vs_spin']
            stability = raw_stability * (1 + (WEIGHT_ALPHA * math.log10(vol_spin_runs + 1)))
        else:
            stability = 0
            
        # 3. Pressure Absorption
        # Lower is better (fewer balls to hit back).
        # Invert: 100 / Avg_Balls
        avg_delay = stats['pressure_accum_balls'] / stats['pressure_events'] if stats['pressure_events'] > 0 else 20
        raw_pressure = (20 - avg_delay) if avg_delay < 20 else 0
        # Weighting
        pressure_score = raw_pressure * (1 + (WEIGHT_ALPHA * math.log10(stats['pressure_events'] + 1)))
        
        # 4. Chase IQ
        chase_avg = stats['chase_runs'] / stats['chase_outs'] if stats['chase_outs'] > 0 else stats['chase_runs']
        # Weighting
        chase_iq = chase_avg * (1 + (WEIGHT_ALPHA * math.log10(stats['chase_runs'] + 1)))
        
        # 5. Clutch
        death_sr = (stats['death_runs'] / stats['death_balls'] * 100) if stats['death_balls'] > 0 else 0
        playoff_avg = stats['playoff_runs'] / stats['playoff_outs'] if stats['playoff_outs'] > 0 else stats['playoff_runs']
        
        # Scale Average to match SR dimensions (Avg 30 ~ SR 120 -> 4x multiplier)
        playoff_score = playoff_avg * 4.0
        
        raw_clutch = (death_sr * 0.6) + (playoff_score * 0.4)
        
        # Weighting: Volume of Clutch Runs
        clutch_vol = stats['death_runs'] + stats['playoff_runs']
        clutch_idx = raw_clutch * (1 + (WEIGHT_ALPHA * math.log10(clutch_vol + 1)))
        
        # 6. Hard Hitting (Power)
        # Powerplay/Middle Overs Boundary %
        # Let's use Middle Overs (pure power, no field restrictions)
        mid_balls = stats['middle_over_balls']
        raw_power = (stats['middle_over_runs'] / mid_balls * 100) if mid_balls > 0 else 0
        # Weighting: Volume of Middle Over Runs
        hard_hit_idx = raw_power * (1 + (WEIGHT_ALPHA * math.log10(stats['middle_over_runs'] + 1)))
        
        # 7. Consistency (Gini) - Experience weighted
        # Lower Gini = More consistent. But we want to reward players who are consistent over MANY matches.
        # So we INVERT: Consistency Score = (1 - Gini) * Experience_Weight
        GINI_ALPHA = 0.4  # Experience matters more for consistency
        raw_gini = calculate_gini(np.array(stats['innings_scores']))
        num_matches = len(stats['matches'])
        # We still output raw Gini for interpretability, but the comparison should weight experience
        gini = raw_gini  # Keep raw for output (lower = better)
        
        # 8. Entropy
        total_events = stats['balls']
        probs = [c/total_events for c in stats['run_distribution'].values()]
        entropy = calculate_entropy(probs)
        
        # 9. Technical Adeptness (Wrist Work)
        # Components:
        # A. Strike Rotation (Non-Boundary SR)
        # B. Spin Manipulation (Fours vs Spin %)
        
        nb_balls = stats['balls'] - stats['boundary_balls']
        nb_sr = (stats['non_boundary_runs'] / nb_balls * 100) if nb_balls > 0 else 0
        
        spin_balls = stats['balls_vs_spin']
        spin_fours_pct = (stats['fours_vs_spin'] / spin_balls * 100) if spin_balls > 0 else 0
        
        # Composite Score (Equal Weighting after scale adjustment)
        # nb_sr is usually ~60-80. spin_fours_pct is ~8-15.
        # Scale spin_pct by 4x to match influence.
        raw_tech = (nb_sr * 0.6) + (spin_fours_pct * 4.0 * 0.4) 
        
        # Weighting: Volume
        # We value consistency.
        unique_tech_runs = stats['non_boundary_runs'] + (stats['fours_vs_spin'] * 4)
        tech_score = raw_tech * (1 + (WEIGHT_ALPHA * math.log10(unique_tech_runs + 1)))

        # 10. Tough Pitch Warrior
        # % of runs scored in tough conditions - Higher experience weight (0.4)
        TOUGH_ALPHA = 0.4  # Experience matters more for tough conditions
        tough_run_rate = (stats['tough_runs'] / stats['balls']) * 100 
        tough_vol_weight = tough_run_rate * (1 + (TOUGH_ALPHA * math.log10(stats['tough_runs'] + 1)))

        final_data.append({
            'Name': p,
            # Age Removed
            'Hand_Eye_Pace': round(hand_eye, 2),
            'Stability_Spin': round(stability, 2),
            'Pressure_Absorb': round(pressure_score, 2),
            'Chase_IQ': round(chase_iq, 2),
            'Clutch_Index': round(clutch_idx, 2),
            'Hard_Hitting_Power': round(hard_hit_idx, 1),
            'Technical_Adeptness': round(tech_score, 1),
            'Tough_Pitch_Performance': round(tough_vol_weight, 1),
            'Consistency_Gini': round(gini, 3),
            'Audacity_Entropy': round(entropy, 3),
            'Total_Runs': stats['runs']
        })
        kept_count += 1
        
    print(f"Dropped {dropped_count} players based on Role Filter.")
    print(f"Kept {kept_count} players.")
        
    df = pd.DataFrame(final_data)
    # Sort for best viewing
    df.sort_values(by='Total_Runs', ascending=False, inplace=True)
    return df

if __name__ == "__main__":
    base = "/Users/aadityamukherjee/Documents/ipl analysis"
    json_dir = os.path.join(base, "ipl_json")
    bio_csv = os.path.join(base, "players_data_with_all_info.csv")
    cric_csv = os.path.join(base, "cricket_data.csv")
    
    df = extract_features(json_dir, bio_csv, cric_csv)
    out_path = os.path.join(base, "IPL_Player_Stats_Advanced.csv")
    df.to_csv(out_path, index=False)
    print(f"Done. Saved to {out_path}")
    
    # Authenticate / Validate
    print("\n--- AUTHENTICATION: Top 5 by Chase IQ (Expect Dhoni/Kohli) ---")
    print(df.sort_values(by='Chase_IQ', ascending=False)[['Name', 'Chase_IQ', 'Total_Runs']].head(5))
    
    print("\n--- AUTHENTICATION: Top 5 by Hand-Eye (Pace) ---")
    print(df.sort_values(by='Hand_Eye_Pace', ascending=False)[['Name', 'Hand_Eye_Pace']].head(5))
    
    # Removed Height Check

