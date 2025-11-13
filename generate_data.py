# generate_data.py (Nigerian Banking System Context)
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_nigerian_fraud_data(num_transactions=10000, num_accounts=500, num_devices=60, num_merchants=130):
    """Generates a synthetic transaction dataset tailored to Nigerian banking fraud patterns."""

    np.random.seed(42)
    
    # 1. Base Data Generation with Nigerian Context
    df = pd.DataFrame()
    df['txn_id'] = range(1, num_transactions + 1)
    
    # IDs: Using 10-digit Nigerian bank account structure for realism
    # ACCT_ prefix is maintained for internal script compatibility
    rng = np.random.default_rng(seed=42) # Use a new generator instance for large integers
    
    # IDs: Using 10-digit Nigerian bank account structure for realism
    # Generate 10-digit integers using the safe 'integers' function
    account_numbers = rng.integers(low=1000000000, high=9999999999, size=num_accounts, dtype=np.int64) 
    accounts = [f'ACCT_{num}' for num in account_numbers]
    devices = [f'DEVICE_{i:02d}' for i in range(num_devices)]
    merchants = [f'VNDR_{i:02d}' for i in range(num_merchants)] # Vendor/Merchant ID

    # Skewed distribution for senders/devices (some accounts/devices are higher volume)
    df['sender_id'] = np.random.choice(accounts, num_transactions, p=np.random.dirichlet(np.ones(num_accounts)*10))
    df['receiver_id'] = np.random.choice(accounts, num_transactions)
    df['device_id'] = np.random.choice(devices, num_transactions, p=np.random.dirichlet(np.ones(num_devices)*5))
    df['merchant_id'] = np.random.choice(merchants, num_transactions)
    
    # Time and Amount (Scaled to NGN context)
    start_time = datetime(2025, 11, 10, 8, 0, 0)
    df['timestamp'] = [start_time + timedelta(minutes=i*5) for i in range(num_transactions)]
    # Base amounts (small to moderate, typical lognormal distribution)
    df['amount'] = np.random.lognormal(mean=9, sigma=1.2, size=num_transactions).astype(int) + 100 

    # 2. Introduce Fraud Patterns (Targeting ~8% fraud rate)
    df['is_fraud'] = 0
    num_fraud = int(num_transactions * 0.15)
    fraud_indices = np.random.choice(df.index, size=num_fraud, replace=False)
    
    # Identify small, compromised groups
    fraud_senders = np.random.choice(df['sender_id'].unique(), size=10, replace=False) # Accounts used for draining
    fraud_merchants = np.random.choice(df['merchant_id'].unique(), size=5, replace=False) # Scam vendor accounts
    
    # Apply enhanced fraud features
    for idx in fraud_indices:
        
        # Pattern A: High-Value, Round Number Scams (Social Engineering)
        if np.random.rand() < 0.4:
            df.loc[idx, 'amount'] = np.random.choice([1000000, 2500000, 5000000, 10000000]) # Large NGN transfers
            df.loc[idx, 'sender_id'] = np.random.choice(fraud_senders) # From a known compromised account
            df.loc[idx, 'device_id'] = np.random.choice(devices) # Often a new, suspicious device
            
        # Pattern B: Frequent, Sequential Small Drains (Account Takeover)
        elif np.random.rand() < 0.7:
            df.loc[idx, 'amount'] = np.random.choice([5000, 10000, 25000, 50000]) # Small NGN amounts
            df.loc[idx, 'sender_id'] = np.random.choice(fraud_senders)
            df.loc[idx, 'timestamp'] = df.loc[idx, 'timestamp'] + timedelta(seconds=np.random.randint(10, 60)) # Sequential timestamps

        # Pattern C: Merchant Fraud (Scam Vendor)
        else:
            df.loc[idx, 'amount'] = np.random.choice([150000, 300000, 750000])
            df.loc[idx, 'merchant_id'] = np.random.choice(fraud_merchants)

        df.loc[idx, 'is_fraud'] = 1 # Mark as fraud

    # Final cleanup and structure
    df['amount'] = df['amount'].round(0).astype(int)
    
    # Rename columns to match the EXPECTED names in prepare_data.py
    df = df.rename(columns={
        'sender_id': 'sender_id', # Keep for compatibility
        'receiver_id': 'receiver_id', # Keep for compatibility
        'merchant_id': 'merchant_id' # Keep for compatibility
    })
    
    return df[['txn_id', 'sender_id', 'receiver_id', 'amount', 'device_id', 'merchant_id', 'timestamp', 'is_fraud']]


# --- Execution ---
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    print("Generating 10,000 synthetic Nigerian-context transactions...")
    data_df = generate_synthetic_nigerian_fraud_data()
    
    output_path = os.path.join("data", "transactions.csv")
    data_df.to_csv(output_path, index=False)
    
    print(f"✅ Data generated successfully.")
    print(f"   Saved to: {output_path}")
    print(f"   Total transactions: {len(data_df):,}")
    print(f"   Fraud rate: {data_df['is_fraud'].mean() * 100:.2f}%")