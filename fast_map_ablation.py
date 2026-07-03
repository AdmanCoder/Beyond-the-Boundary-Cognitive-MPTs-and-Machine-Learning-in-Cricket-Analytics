import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cmdstanpy import CmdStanModel
import json

print("Loading test dataset...")
df = pd.read_parquet('v7_cached_dataset.parquet')
df['y_stan'] = df['y'] + 1 
df = df[df['season'].between(2019, 2024)].copy().reset_index(drop=True)
TARGET_BATTERS = {'V Kohli', 'KL Rahul', 'DA Warner', 'JC Buttler', 'HH Pandya'}
test = df[(df['season'] == 2022) & (df['batter'].isin(TARGET_BATTERS))].copy().reset_index(drop=True)
train_core = df[~((df['season'] == 2022) & (df['batter'].isin(TARGET_BATTERS)))].copy().reset_index(drop=True)

all_batters = sorted(train_core['batter'].unique().tolist())
b_idx = {b: i+1 for i, b in enumerate(all_batters)}
for b in test['batter'].unique():
    if b not in b_idx:
        all_batters.append(b)
        b_idx[b] = len(all_batters)
J = len(all_batters)

# Build alphas exactly like v8_bayesian_advi_eval.py
raw_a = np.zeros(J)
raw_s = np.zeros(J)
raw_kappa = np.zeros(J)
raw_m = np.zeros(J)
gb = train_core.groupby('batter')
for b, group in gb:
    idx = b_idx[b] - 1
    total = len(group)
    bound = (group['y'] == 0).sum()
    rot = (group['y'] == 1).sum()
    a = bound / total if total > 0 else 0.01
    s_denom = total - bound
    s = rot / s_denom if s_denom > 0 else 0.01
    k_denom = bound + rot
    kappa = bound / k_denom if k_denom > 0 else 0.01
    m = group['batter_boundary_pct'].mean()
    if pd.isna(m) or m <= 0: m = 0.01
    raw_a[idx] = a
    raw_s[idx] = s
    raw_kappa[idx] = kappa
    raw_m[idx] = m
raw_a[raw_a == 0] = np.mean(raw_a[raw_a > 0])
raw_s[raw_s == 0] = np.mean(raw_s[raw_s > 0])
raw_kappa[raw_kappa == 0] = np.mean(raw_kappa[raw_kappa > 0])
raw_m[raw_m == 0] = np.mean(raw_m[raw_m > 0])

def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))

def zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

alpha_m_fixed = zscore(logit(raw_m))
alpha_a_fixed = zscore(logit(raw_a))
alpha_s_fixed = zscore(logit(raw_s))
alpha_kappa_fixed = zscore(logit(raw_kappa))

def standardize(train_series, test_series):
    m = train_series.mean()
    s = train_series.std() + 1e-8
    return ((train_series - m) / s).values.tolist(), ((test_series - m) / s).values.tolist()

features = [
    'bsb', 'current_score', 'over', 'batter_career_sr', 'h2h_sr', 
    'balls_faced', 'bowler_career_economy', 'momentum_ewm', 
    'required_run_rate', 'pressure_index', 'partnership_balls', 'dot_pressure'
]
stan_feature_names = [
    'bsb', 'current_score', 'over_num', 'batter_career_sr', 'h2h_sr', 
    'balls_faced', 'bowler_career_economy', 'momentum_ewm', 
    'required_run_rate', 'pressure_index', 'partnership_balls', 'dot_pressure'
]

train_data_dict = {
    'N': len(train_core), 'J': J,
    'batter_idx': train_core['batter'].map(b_idx).values.tolist(),
    'y': train_core['y_stan'].values.tolist(),
    'alpha_m_fixed': alpha_m_fixed.tolist(), 'alpha_a_fixed': alpha_a_fixed.tolist(),
    'alpha_s_fixed': alpha_s_fixed.tolist(), 'alpha_kappa_fixed': alpha_kappa_fixed.tolist()
}

test_data_dict = {
    'N': len(test), 'J': J,
    'batter_idx': test['batter'].map(b_idx).values.tolist(),
    'y': test['y_stan'].values.tolist(),
    'alpha_m_fixed': alpha_m_fixed.tolist(), 'alpha_a_fixed': alpha_a_fixed.tolist(),
    'alpha_s_fixed': alpha_s_fixed.tolist(), 'alpha_kappa_fixed': alpha_kappa_fixed.tolist()
}

for f, sf in zip(features, stan_feature_names):
    tr_vals, te_vals = standardize(train_core[f], test[f])
    train_data_dict[sf] = tr_vals
    test_data_dict[sf] = te_vals

# The Stan model takes about 1-2 minutes to optimize for MAP
print("Running MAP Estimation (Fast Mode)...")
model = CmdStanModel(stan_file='v8_mpt_advi.stan')
fit = model.optimize(data=train_data_dict, seed=42)

# Extract optimized params
print("Extracting parameters...")
params = fit.optimized_params_dict

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def calc_ll(test_data, p_dict, ablate_node=None):
    total_ll = 0
    # ensure p_dict has scalar values for betas
    for i in range(len(test)):
        j = test_data['batter_idx'][i] - 1 
        
        m_val = sigmoid(test_data['alpha_m_fixed'][j] + p_dict['beta_m_bsb']*test_data['bsb'][i] + p_dict['beta_m_curr_score']*test_data['current_score'][i] + p_dict['beta_m_over']*test_data['over_num'][i] + p_dict['beta_m_career_sr']*test_data['batter_career_sr'][i])
        phi_val = sigmoid(p_dict['alpha_phi'][j] + p_dict['beta_phi_h2h']*test_data['h2h_sr'][i] + p_dict['beta_phi_bf']*test_data['balls_faced'][i] + p_dict['beta_phi_bowl_eco']*test_data['bowler_career_economy'][i] + p_dict['beta_phi_career_sr']*test_data['batter_career_sr'][i])
        r_val = sigmoid(p_dict['alpha_r'][j] + p_dict['beta_r_bf']*test_data['balls_faced'][i] + p_dict['beta_r_mom']*test_data['momentum_ewm'][i] + p_dict['beta_r_career_sr']*test_data['batter_career_sr'][i])
        a_val = sigmoid(test_data['alpha_a_fixed'][j] + p_dict['beta_a_bsb']*test_data['bsb'][i] + p_dict['beta_a_over']*test_data['over_num'][i] + p_dict['beta_a_rrr']*test_data['required_run_rate'][i] + p_dict['beta_a_press']*test_data['pressure_index'][i])
        s_val = sigmoid(test_data['alpha_s_fixed'][j] + p_dict['beta_s_pb']*test_data['partnership_balls'][i] + p_dict['beta_s_dot']*test_data['dot_pressure'][i] + p_dict['beta_s_rrr']*test_data['required_run_rate'][i])
        zeta_val = sigmoid(p_dict['alpha_zeta'][j] + p_dict['beta_zeta_h2h']*test_data['h2h_sr'][i] + p_dict['beta_zeta_press']*test_data['pressure_index'][i] + p_dict['beta_zeta_dot']*test_data['dot_pressure'][i])
        kappa_val = sigmoid(test_data['alpha_kappa_fixed'][j] + p_dict['beta_kappa_curr_score']*test_data['current_score'][i] + p_dict['beta_kappa_mom']*test_data['momentum_ewm'][i] + p_dict['beta_kappa_career_sr']*test_data['batter_career_sr'][i] + p_dict['beta_kappa_press']*test_data['pressure_index'][i])
        
        # Ablation
        if ablate_node == 'M (Match Reading)': m_val = sigmoid(test_data['alpha_m_fixed'][j])
        elif ablate_node == 'Phi (Bowler Matchup)': phi_val = sigmoid(p_dict['alpha_phi'][j])
        elif ablate_node == 'R (Form/Rhythm)': r_val = sigmoid(p_dict['alpha_r'][j])
        elif ablate_node == 'A (Aggression)': a_val = sigmoid(test_data['alpha_a_fixed'][j])
        elif ablate_node == 'S (Situation)': s_val = sigmoid(test_data['alpha_s_fixed'][j])
        elif ablate_node == 'Zeta (Clutch)': zeta_val = sigmoid(p_dict['alpha_zeta'][j])
        elif ablate_node == 'Kappa (Consistency)': kappa_val = sigmoid(test_data['alpha_kappa_fixed'][j])

        eps = sigmoid(p_dict['alpha_epsilon'])

        p_bound = (m_val * phi_val * kappa_val) + ((1-m_val) * r_val * a_val * kappa_val) + ((1-m_val) * (1-r_val) * zeta_val * kappa_val)
        p_rot = (m_val * phi_val * (1-kappa_val) * eps) + ((1-m_val) * r_val * a_val * (1-kappa_val) * eps) + ((1-m_val) * (1-r_val) * zeta_val * (1-kappa_val) * eps) + ((1-m_val) * r_val * (1-a_val) * s_val * kappa_val) + (m_val * (1-phi_val) * kappa_val) + ((1-m_val) * (1-r_val) * (1-zeta_val) * kappa_val)
        p_fail = 1.0 - p_bound - p_rot
        
        p_bound = max(1e-5, min(1-1e-5, p_bound))
        p_rot = max(1e-5, min(1-1e-5, p_rot))
        p_fail = max(1e-5, min(1-1e-5, p_fail))
        
        total = p_bound + p_rot + p_fail
        
        true_y = test_data['y'][i]
        if true_y == 1: total_ll += np.log(p_bound/total)
        elif true_y == 2: total_ll += np.log(p_rot/total)
        else: total_ll += np.log(p_fail/total)
        
    return total_ll

print("Calculating Baseline Log-Likelihood...")
baseline_ll = calc_ll(test_data_dict, params)

nodes = {
    'M (Match Reading)': 'm',
    'Phi (Bowler Matchup)': 'phi',
    'R (Form/Rhythm)': 'r',
    'A (Aggression)': 'a',
    'S (Situation)': 's',
    'Zeta (Clutch)': 'zeta',
    'Kappa (Consistency)': 'kappa'
}

drops = {}
for name in nodes.keys():
    print(f"Ablating {name}...")
    ablate_ll = calc_ll(test_data_dict, params, ablate_node=name)
    drops[name] = baseline_ll - ablate_ll

plt.figure(figsize=(10, 6))
sns.barplot(x=list(drops.values()), y=list(drops.keys()), palette="Reds_r")
plt.title('Node Feature Importance (MAP Ablation Log-Likelihood Drop)', fontsize=16, fontweight='bold')
plt.xlabel('Drop in Predictive Log-Likelihood (Higher = More Important)')
plt.tight_layout()
plt.savefig('node_importance.png', dpi=150)
print("Saved node_importance.png")
