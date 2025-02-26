# nba_prediction_model
This project predicts the outcome of NBA games by leveraging historical game data retrieved via the nba_api. It computes rolling averages and rest-day features from past games, then uses an XGBoost classifier (integrated into a scikit‑learn pipeline with StandardScaler) to forecast win probabilities.
