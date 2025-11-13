import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- Configuration ---
MODEL_DIR = "models"
CDT_MODEL_PATH = os.path.join(MODEL_DIR, "cdt_model.pkl")
DATA_SIZE = 1000

def generate_consequence_data(size=DATA_SIZE):
    """
    Generates synthetic data linking risk inputs (GNN Score, Amount) 
    to an outcome (Regulatory Penalty).
    """
    np.random.seed(42)
    
    gnn_score = np.random.beta(a=2, b=5, size=size)
    
    amount = np.random.uniform(low=5000, high=10000000, size=size)
    
    
    penalty_base = (2 * gnn_score + 0.5) * np.log(amount)
    penalty_noise = penalty_base + np.random.normal(0, 5)
    
    regulatory_penalty = np.clip(penalty_noise * 1000, a_min=500, a_max=100000)
    
    df = pd.DataFrame({
        'gnn_score': gnn_score,
        'amount': amount,
        'regulatory_penalty': regulatory_penalty
    })
    
    return df

def train_and_save_cdt_model():
    print("--- 1. Generating Synthetic Consequence Data ---")
    df = generate_consequence_data()
    
    X = df[['gnn_score', 'amount']]
    y = df['regulatory_penalty']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"--- 2. Training Linear Regression CDT Model on {len(X_train)} samples ---")
    
    cdt_model = LinearRegression()
    cdt_model.fit(X_train, y_train)
    
    score = cdt_model.score(X_test, y_test)
    print(f"   Model R-squared on test data: {score:.4f}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(cdt_model, CDT_MODEL_PATH)
    
    print(f"--- 3. CDT Model saved successfully to {CDT_MODEL_PATH} ---")
    
    return cdt_model

if __name__ == "__main__":
    train_and_save_cdt_model()