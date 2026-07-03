data {
  int<lower=1> N; // Total number of deliveries analyzed
  int<lower=1> J; // Total number of batters
  
  array[N] int<lower=1, upper=J> batter_idx; // Which batter is facing the ball
  array[N] int<lower=1, upper=3> y;          // Outcome: 1=Boundary, 2=Rotation, 3=Fail
  
  // Predictors from the Parquet dataset (Standardized in Python)
  vector[N] bsb;
  vector[N] current_score;
  vector[N] over_num;
  vector[N] batter_career_sr;
  vector[N] h2h_sr;
  vector[N] balls_faced;
  vector[N] bowler_career_economy;
  vector[N] momentum_ewm;
  vector[N] required_run_rate;
  vector[N] pressure_index;
  vector[N] partnership_balls;
  vector[N] dot_pressure;
  
  // Fixed Alphas
  vector[J] alpha_m_fixed;
  vector[J] alpha_a_fixed;
  vector[J] alpha_s_fixed;
  vector[J] alpha_kappa_fixed;
}

parameters {
  // Free Intercepts
  vector[J] alpha_phi;
  vector[J] alpha_r;
  vector[J] alpha_zeta;
  
  // Beta Weights
  // Node m
  real beta_m_bsb;
  real beta_m_curr_score;
  real beta_m_over;
  real beta_m_career_sr;
  
  // Node phi
  real<lower=0> beta_phi_h2h;
  real beta_phi_bf;
  real<upper=0> beta_phi_bowl_eco;
  real<lower=0> beta_phi_career_sr;
  
  // Node r
  real beta_r_bf;
  real beta_r_mom;
  real<lower=0> beta_r_career_sr;
  
  // Node a
  real beta_a_bsb;
  real beta_a_over;
  real beta_a_rrr;
  real beta_a_press;
  
  // Node s
  real beta_s_pb;
  real beta_s_dot;
  real beta_s_rrr;
  
  // Node zeta
  real<lower=0> beta_zeta_h2h;
  real beta_zeta_press;
  real beta_zeta_dot;
  
  // Node kappa
  real beta_kappa_curr_score;
  real<lower=0> beta_kappa_mom;
  real<lower=0> beta_kappa_career_sr;
  real<lower=0> beta_kappa_press; // Handling pressure well leads to better contact
  
  real alpha_epsilon; 
}

transformed parameters {
  array[N] simplex[3] p; 
  
  for (n in 1:N) {
    int j = batter_idx[n];
    
    real m = inv_logit(alpha_m_fixed[j] + beta_m_bsb*bsb[n] + beta_m_curr_score*current_score[n] + beta_m_over*over_num[n] + beta_m_career_sr*batter_career_sr[n]);
    
    real phi = inv_logit(alpha_phi[j] + beta_phi_h2h*h2h_sr[n] + beta_phi_bf*balls_faced[n] + beta_phi_bowl_eco*bowler_career_economy[n] + beta_phi_career_sr*batter_career_sr[n]);
    
    real r = inv_logit(alpha_r[j] + beta_r_bf*balls_faced[n] + beta_r_mom*momentum_ewm[n] + beta_r_career_sr*batter_career_sr[n]);
    
    real a = inv_logit(alpha_a_fixed[j] + beta_a_bsb*bsb[n] + beta_a_over*over_num[n] + beta_a_rrr*required_run_rate[n] + beta_a_press*pressure_index[n]);
    
    real s = inv_logit(alpha_s_fixed[j] + beta_s_pb*partnership_balls[n] + beta_s_dot*dot_pressure[n] + beta_s_rrr*required_run_rate[n]);
    
    real zeta = inv_logit(alpha_zeta[j] + beta_zeta_h2h*h2h_sr[n] + beta_zeta_press*pressure_index[n] + beta_zeta_dot*dot_pressure[n]);
    
    real kappa = inv_logit(alpha_kappa_fixed[j] + beta_kappa_curr_score*current_score[n] + beta_kappa_mom*momentum_ewm[n] + beta_kappa_career_sr*batter_career_sr[n] + beta_kappa_press*pressure_index[n]);
    
    real epsilon = inv_logit(alpha_epsilon);
    
    real t_bound = (m * phi * kappa) + 
                   ((1-m) * r * a * kappa) + 
                   ((1-m) * (1-r) * zeta * kappa);
                   
    real t_rot = (m * phi * (1-kappa) * epsilon) + 
                 ((1-m) * r * a * (1-kappa) * epsilon) + 
                 ((1-m) * (1-r) * zeta * (1-kappa) * epsilon) + 
                 ((1-m) * r * (1-a) * s * kappa) + 
                 (m * (1-phi) * kappa) + 
                 ((1-m) * (1-r) * (1-zeta) * kappa);
                 
    real t_fail = 1.0 - t_bound - t_rot;
    
    if (t_bound < 1e-5) t_bound = 1e-5;
    if (t_rot < 1e-5) t_rot = 1e-5;
    if (t_fail < 1e-5) t_fail = 1e-5;
    
    real total = t_bound + t_rot + t_fail;
    p[n, 1] = t_bound / total;
    p[n, 2] = t_rot / total;
    p[n, 3] = t_fail / total;
  }
}

model {
  alpha_phi ~ normal(0, 1.5);
  alpha_r ~ normal(0, 1.5);
  alpha_zeta ~ normal(0, 1.5);
  alpha_epsilon ~ normal(-1.38, 0.5);
  
  beta_m_bsb ~ normal(0, 0.5);
  beta_m_curr_score ~ normal(0, 0.5);
  beta_m_over ~ normal(0, 0.5);
  beta_m_career_sr ~ normal(0, 0.5);
  
  beta_phi_h2h ~ normal(0, 0.5);
  beta_phi_bf ~ normal(0, 0.5);
  beta_phi_bowl_eco ~ normal(0, 0.5);
  beta_phi_career_sr ~ normal(0, 0.5);
  
  beta_r_bf ~ normal(0, 0.5);
  beta_r_mom ~ normal(0, 0.5);
  beta_r_career_sr ~ normal(0, 0.5);
  
  beta_a_bsb ~ normal(0, 0.5);
  beta_a_over ~ normal(0, 0.5);
  beta_a_rrr ~ normal(0, 0.5);
  beta_a_press ~ normal(0, 0.5);
  
  beta_s_pb ~ normal(0, 0.5);
  beta_s_dot ~ normal(0, 0.5);
  beta_s_rrr ~ normal(0, 0.5);
  
  beta_zeta_h2h ~ normal(0, 0.5);
  beta_zeta_press ~ normal(0, 0.5);
  beta_zeta_dot ~ normal(0, 0.5);
  
  beta_kappa_curr_score ~ normal(0, 0.5);
  beta_kappa_mom ~ normal(0, 0.5);
  beta_kappa_career_sr ~ normal(0, 0.5);
  beta_kappa_press ~ normal(0, 0.5);
  
  for (n in 1:N) {
    y[n] ~ categorical(p[n]);
  }
}
