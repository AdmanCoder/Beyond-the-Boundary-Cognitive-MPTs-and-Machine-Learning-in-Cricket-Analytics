import pandas as pd
import numpy as np
import cmdstanpy
from sklearn.metrics import log_loss, accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

TARGET_BATTERS = {'V Kohli', 'KL Rahul', 'DA Warner', 'JC Buttler', 'HH Pandya'}
CLASS_ORDER = [0, 1, 2] # Boundary, Rotation, Fail

print('Loading dataset...')
df = pd.read_parquet('v7_cached_dataset.parquet')

# Stan requires y to be 1-indexed (1, 2, 3)
df['y_stan'] = df['y'] + 1 

# The parquet file contains 2008-2024 (268,000 balls).
# We must filter it to 2019-2024 (approx 100,000 balls) to exactly match your non-linear models.
df = df[df['season'].between(2019, 2024)].copy().reset_index(drop=True)

# Split exactly like v8_sota_sweep.py
test = df[(df['season'] == 2022) & (df['batter'].isin(TARGET_BATTERS))].copy().reset_index(drop=True)
train_core = df[~((df['season'] == 2022) & (df['batter'].isin(TARGET_BATTERS)))].copy().reset_index(drop=True)

print(f'Train size (2019-2024): {len(train_core)} | Test size (2022): {len(test)}')

# Calculate Career Baseline Alphas for ALL batters in Train set
print('Calculating Career Alphas...')
all_batters = sorted(train_core['batter'].unique().tolist())
b_idx = {b: i+1 for i, b in enumerate(all_batters)}

# We must ensure test batters are in the train index (they should be, as we only excluded 2022)
for b in test['batter'].unique():
    if b not in b_idx:
        all_batters.append(b)
        b_idx[b] = len(all_batters)

J = len(all_batters)

# Compute raw ratios
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
    
    # Use batter_boundary_pct as proxy for m (premeditation/aggression baseline)
    m = group['batter_boundary_pct'].mean()
    if pd.isna(m) or m <= 0: m = 0.01
    
    raw_a[idx] = a
    raw_s[idx] = s
    raw_kappa[idx] = kappa
    raw_m[idx] = m

# Impute any missing (batters only in test set, shouldn't happen)
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

# Standardize predictors function
def standardize(train_series, test_series):
    m = train_series.mean()
    s = train_series.std() + 1e-8
    return ((train_series - m) / s).values.tolist(), ((test_series - m) / s).values.tolist()

print('Formatting Stan Data...')
features = [
    'bsb', 'current_score', 'over', 'batter_career_sr', 'h2h_sr', 
    'balls_faced', 'bowler_career_economy', 'momentum_ewm', 
    'required_run_rate', 'pressure_index', 'partnership_balls', 'dot_pressure'
]

train_data_dict = {
    'N': len(train_core),
    'J': J,
    'batter_idx': train_core['batter'].map(b_idx).values.tolist(),
    'y': train_core['y_stan'].values.tolist(),
    'alpha_m_fixed': alpha_m_fixed.tolist(),
    'alpha_a_fixed': alpha_a_fixed.tolist(),
    'alpha_s_fixed': alpha_s_fixed.tolist(),
    'alpha_kappa_fixed': alpha_kappa_fixed.tolist()
}

test_data_dict = {
    'N': len(test),
    'J': J,
    'batter_idx': test['batter'].map(b_idx).values.tolist(),
    'y': test['y_stan'].values.tolist(),
    'alpha_m_fixed': alpha_m_fixed.tolist(),
    'alpha_a_fixed': alpha_a_fixed.tolist(),
    'alpha_s_fixed': alpha_s_fixed.tolist(),
    'alpha_kappa_fixed': alpha_kappa_fixed.tolist()
}

# Standardize all features and assign names
stan_feature_names = [
    'bsb', 'current_score', 'over_num', 'batter_career_sr', 'h2h_sr', 
    'balls_faced', 'bowler_career_economy', 'momentum_ewm', 
    'required_run_rate', 'pressure_index', 'partnership_balls', 'dot_pressure'
]

for f, sf in zip(features, stan_feature_names):
    tr_vals, te_vals = standardize(train_core[f], test[f])
    train_data_dict[sf] = tr_vals
    test_data_dict[sf] = te_vals

print('Compiling Model...')
model = cmdstanpy.CmdStanModel(stan_file='v8_mpt_advi.stan')

print('Running Variational Inference (ADVI)...')
fit = model.variational(data=train_data_dict, seed=42, output_samples=1000)

print('Inference complete. Extracting posterior parameters...')
# Get posterior means for betas and intercepts
posteriors = fit.variational_sample
df_post = pd.DataFrame(posteriors, columns=fit.column_names)

print('\\n--- Key Learned Betas (Posterior Means) ---')
params = ['beta_kappa_mom', 'beta_phi_h2h', 'beta_a_rrr', 'beta_s_dot', 'beta_kappa_career_sr']
for p in params:
    if p in df_post.columns:
        print(f'{p:<25}: {df_post[p].mean():.3f}')

# Now we need to manually reconstruct the tree logic for the Test Set 
# using the posterior means to get point predictions.
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def get_mean(param_name):
    return df_post[param_name].mean()

def get_mean_vec(param_name, J):
    return np.array([get_mean(f'{param_name}[{i+1}]') for i in range(J)])

print('\\nRunning Test Set Point Predictions...')
# Intercepts
a_phi = get_mean_vec('alpha_phi', J)
a_r = get_mean_vec('alpha_r', J)
a_z = get_mean_vec('alpha_zeta', J)
a_eps = get_mean('alpha_epsilon')

# Compute node probabilities manually for test set
preds = []
for i in range(len(test)):
    j = test_data_dict['batter_idx'][i] - 1
    
    m_val = sigmoid(alpha_m_fixed[j] + 
                    get_mean('beta_m_bsb')*test_data_dict['bsb'][i] + 
                    get_mean('beta_m_curr_score')*test_data_dict['current_score'][i] + 
                    get_mean('beta_m_over')*test_data_dict['over_num'][i] + 
                    get_mean('beta_m_career_sr')*test_data_dict['batter_career_sr'][i])
                    
    phi_val = sigmoid(a_phi[j] + 
                      get_mean('beta_phi_h2h')*test_data_dict['h2h_sr'][i] + 
                      get_mean('beta_phi_bf')*test_data_dict['balls_faced'][i] + 
                      get_mean('beta_phi_bowl_eco')*test_data_dict['bowler_career_economy'][i] + 
                      get_mean('beta_phi_career_sr')*test_data_dict['batter_career_sr'][i])
                      
    r_val = sigmoid(a_r[j] + 
                    get_mean('beta_r_bf')*test_data_dict['balls_faced'][i] + 
                    get_mean('beta_r_mom')*test_data_dict['momentum_ewm'][i] + 
                    get_mean('beta_r_career_sr')*test_data_dict['batter_career_sr'][i])
                    
    a_val = sigmoid(alpha_a_fixed[j] + 
                    get_mean('beta_a_bsb')*test_data_dict['bsb'][i] + 
                    get_mean('beta_a_over')*test_data_dict['over_num'][i] + 
                    get_mean('beta_a_rrr')*test_data_dict['required_run_rate'][i] + 
                    get_mean('beta_a_press')*test_data_dict['pressure_index'][i])
                    
    s_val = sigmoid(alpha_s_fixed[j] + 
                    get_mean('beta_s_pb')*test_data_dict['partnership_balls'][i] + 
                    get_mean('beta_s_dot')*test_data_dict['dot_pressure'][i] + 
                    get_mean('beta_s_rrr')*test_data_dict['required_run_rate'][i])
                    
    z_val = sigmoid(a_z[j] + 
                    get_mean('beta_zeta_h2h')*test_data_dict['h2h_sr'][i] + 
                    get_mean('beta_zeta_press')*test_data_dict['pressure_index'][i] + 
                    get_mean('beta_zeta_dot')*test_data_dict['dot_pressure'][i])
                    
    k_val = sigmoid(alpha_kappa_fixed[j] + 
                    get_mean('beta_kappa_curr_score')*test_data_dict['current_score'][i] + 
                    get_mean('beta_kappa_mom')*test_data_dict['momentum_ewm'][i] + 
                    get_mean('beta_kappa_career_sr')*test_data_dict['batter_career_sr'][i] + 
                    get_mean('beta_kappa_press')*test_data_dict['pressure_index'][i])
                    
    eps = sigmoid(a_eps)
    
    # Tree math (Point prediction using means)
    t_bound = (m_val * phi_val * k_val) + ((1-m_val) * r_val * a_val * k_val) + ((1-m_val) * (1-r_val) * z_val * k_val)
    t_rot = (m_val * phi_val * (1-k_val) * eps) + ((1-m_val) * r_val * a_val * (1-k_val) * eps) + ((1-m_val) * (1-r_val) * z_val * (1-k_val) * eps) + ((1-m_val) * r_val * (1-a_val) * s_val * k_val) + (m_val * (1-phi_val) * k_val) + ((1-m_val) * (1-r_val) * (1-z_val) * k_val)
    
    t_fail = 1.0 - t_bound - t_rot
    t_bound = max(1e-5, t_bound)
    t_rot = max(1e-5, t_rot)
    t_fail = max(1e-5, t_fail)
    
    total = t_bound + t_rot + t_fail
    preds.append([t_bound/total, t_rot/total, t_fail/total])

preds = np.array(preds)
y_test = test['y'].values

acc = accuracy_score(y_test, np.argmax(preds, axis=1))
ll = log_loss(y_test, preds)

from sklearn.metrics import classification_report, confusion_matrix
y_pred = np.argmax(preds, axis=1)

print(f'\\n--- Frequentist View (Point Predictions) ---')
print(f'Log-Loss (Cross-Entropy) : {ll:.4f}')
print(f'Accuracy                 : {acc:.4f}')

print('\\n--- Classification Report ---')
# Classes are 0: Boundary, 1: Rotation, 2: Fail
print(classification_report(y_test, y_pred, target_names=['Boundary', 'Rotation', 'Fail']))

print('\\n--- Confusion Matrix ---')
print(confusion_matrix(y_test, y_pred))

print('\\nCalculating Distributional Distance (Bayesian Log Predictive Density)...')
# For the 'Prob View', we evaluate the true probability under every single posterior draw
# and take the log of the mean probability.
from scipy.special import logsumexp

S = len(df_post) # Number of draws (1000)
test_log_likelihoods = [] # Will hold shape (N_test, S)

for i in range(len(test)):
    j = test_data_dict['batter_idx'][i] - 1
    
    # Vectorized calculation over all S draws for this single ball
    m_val = sigmoid(alpha_m_fixed[j] + df_post['beta_m_bsb']*test_data_dict['bsb'][i] + df_post['beta_m_curr_score']*test_data_dict['current_score'][i] + df_post['beta_m_over']*test_data_dict['over_num'][i] + df_post['beta_m_career_sr']*test_data_dict['batter_career_sr'][i])
    phi_val = sigmoid(df_post[f'alpha_phi[{j+1}]'] + df_post['beta_phi_h2h']*test_data_dict['h2h_sr'][i] + df_post['beta_phi_bf']*test_data_dict['balls_faced'][i] + df_post['beta_phi_bowl_eco']*test_data_dict['bowler_career_economy'][i] + df_post['beta_phi_career_sr']*test_data_dict['batter_career_sr'][i])
    r_val = sigmoid(df_post[f'alpha_r[{j+1}]'] + df_post['beta_r_bf']*test_data_dict['balls_faced'][i] + df_post['beta_r_mom']*test_data_dict['momentum_ewm'][i] + df_post['beta_r_career_sr']*test_data_dict['batter_career_sr'][i])
    a_val = sigmoid(alpha_a_fixed[j] + df_post['beta_a_bsb']*test_data_dict['bsb'][i] + df_post['beta_a_over']*test_data_dict['over_num'][i] + df_post['beta_a_rrr']*test_data_dict['required_run_rate'][i] + df_post['beta_a_press']*test_data_dict['pressure_index'][i])
    s_val = sigmoid(alpha_s_fixed[j] + df_post['beta_s_pb']*test_data_dict['partnership_balls'][i] + df_post['beta_s_dot']*test_data_dict['dot_pressure'][i] + df_post['beta_s_rrr']*test_data_dict['required_run_rate'][i])
    z_val = sigmoid(df_post[f'alpha_zeta[{j+1}]'] + df_post['beta_zeta_h2h']*test_data_dict['h2h_sr'][i] + df_post['beta_zeta_press']*test_data_dict['pressure_index'][i] + df_post['beta_zeta_dot']*test_data_dict['dot_pressure'][i])
    k_val = sigmoid(alpha_kappa_fixed[j] + df_post['beta_kappa_curr_score']*test_data_dict['current_score'][i] + df_post['beta_kappa_mom']*test_data_dict['momentum_ewm'][i] + df_post['beta_kappa_career_sr']*test_data_dict['batter_career_sr'][i] + df_post['beta_kappa_press']*test_data_dict['pressure_index'][i])
    eps = sigmoid(df_post['alpha_epsilon'])
    
    t_bound = (m_val * phi_val * k_val) + ((1-m_val) * r_val * a_val * k_val) + ((1-m_val) * (1-r_val) * z_val * k_val)
    t_rot = (m_val * phi_val * (1-k_val) * eps) + ((1-m_val) * r_val * a_val * (1-k_val) * eps) + ((1-m_val) * (1-r_val) * z_val * (1-k_val) * eps) + ((1-m_val) * r_val * (1-a_val) * s_val * k_val) + (m_val * (1-phi_val) * k_val) + ((1-m_val) * (1-r_val) * (1-z_val) * k_val)
    t_fail = 1.0 - t_bound - t_rot
    
    t_bound = np.maximum(1e-5, t_bound)
    t_rot = np.maximum(1e-5, t_rot)
    t_fail = np.maximum(1e-5, t_fail)
    total = t_bound + t_rot + t_fail
    
    prob_bound = t_bound / total
    prob_rot = t_rot / total
    prob_fail = t_fail / total
    
    # We want the probability of the ACTUAL observed outcome
    actual_y = y_test[i]
    if actual_y == 0:
        true_probs = prob_bound
    elif actual_y == 1:
        true_probs = prob_rot
    else:
        true_probs = prob_fail
        
    test_log_likelihoods.append(np.log(true_probs).values)

test_log_likelihoods = np.array(test_log_likelihoods) # Shape (N_test, S)

# Expected Log Predictive Density (ELPD) across all test balls
# For each ball, we log-average the probabilities over the S draws
log_pred_density = logsumexp(test_log_likelihoods, axis=1) - np.log(S)
total_elpd = np.sum(log_pred_density)
mean_elpd = np.mean(log_pred_density)

print(f'\\n--- Probabilistic View (Bayesian Distributions) ---')
print(f'Total Expected Log Predictive Density (ELPD) : {total_elpd:.2f}')
print(f'Mean ELPD per ball                           : {mean_elpd:.4f}')
print(f'Higher ELPD (closer to 0) means the distributions matched reality perfectly.')

