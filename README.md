
Welcome to the repository for my research project on modeling batter decision-making and predicting T20 cricket outcomes.

Traditional sports analytics often rely entirely on black-box machine learning models to predict what will happen next. While accurate, these models can't explain why a player made a specific decision. This project bridges the gap between pure prediction and cognitive understanding by utilizing an Explainable AI framework that combines structural Bayesian modeling with state-of-the-art machine learning ensembles.

Project Highlights 

Objective: 
To predict ball-by-ball T20 outcomes through an Explainable AI framework that models decision-making.

Architecture:
Engineered a 7-node Bayesian MPT across 90k+ deliveries to mathematically isolate match context from intrinsic traits.

Data Engineering: 
Engineered 66 career and contextual features across a 90k-row dataset, preventing temporal data leakage.

Machine Learning: 
Architected a parallel 4-model weighted ensemble (EBM, XGBoost, RF, LightGBM) classifying outcomes.

Optimization: 
Tuned hyperparameters via Optuna, optimizing a custom 0.6 Acc+0.4 F1 score to find the optimal ensemble.

Results:
Outperformed traditional model baselines by 20%, achieving a 53.7% ball-by-ball peak prediction accuracy. Built an interpretable Bayesian model retaining 95.1% of the predictive accuracy of the black-box ensemble.

Repository Structure

This repository contains the core scripts necessary to replicate the data pipeline, the machine learning models, and the Bayesian cognitive models.

Data Pipeline (/data_pipeline)

create_training_data_v4.py: The master script that joins heterogeneous data sources while strictly enforcing chronological temporal ordering to prevent data leakage.
extract_features.py: Contains the logic and mathematical formulas (including volume-weighting) used to engineer the career and contextual features.
FEATURE_DOCUMENTATION_V4.md: Detailed documentation explaining the logic behind all 66 engineered features.

Machine Learning Pipeline (/machine_learning)

v16_sota_experiments.py: The core ML script. Contains the Optuna tuning logic, the custom 0.6 Acc + 0.4 F1 objective function, and the parallel weighted ensemble architecture.
train_ebm.py: Script detailing the training and hyperparameter selection for the Explainable Boosting Machine (EBM).
v16_best_metrics.json: Output logs proving the final 53.7% prediction accuracy.

Bayesian MPT (/bayesian_mpt)

v8_mpt_advi.stan: The structural probabilistic code defining the 7-node cognitive tree.
v8_bayesian_advi_eval.py: Executes the Mean-Field ADVI inference and evaluates the model's ELPD.
fast_map_ablation.py: Executes the node ablation study used to isolate match context from intrinsic athletic traits.
MPT_FEATURE_MAPPING.md: Documentation explaining how the data features map to the 7 cognitive nodes.

Web Application (/dashboard)

dashboard.py: A 900+ line Streamlit application utilizing Plotly to visualize player cognitive DNA profiles and interactive radar charts.


Tech Stack

Data Processing: Python, Pandas, NumPy
Machine Learning: Scikit-Learn, XGBoost, LightGBM, InterpretML (EBM), Optuna
Probabilistic Modeling: Stan, PyStan
