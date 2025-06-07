# filename: generate_simulated_data.py

import pandas as pd
import numpy as np
import os

# Set random seeds for reproducibility
np.random.seed(42)

def generate_and_save_synthetic_data(num_customers=5000, num_months=12, tampering_rate=0.015, filename='simulated_disco_data.csv'):
    """
    Generates synthetic meter reading data simulating DISCO operations,
    including normal consumption, tampering, and missing values, then saves it to a CSV.
    """
    data = []
    customer_ids = [f'CUST_{i:05d}' for i in range(num_customers)]
    
    # Randomly select a subset of customers for tampering
    tampering_customers = np.random.choice(
        customer_ids,
        size=int(num_customers * tampering_rate),
        replace=False
    )

    for cust_id in customer_ids:
        # Generate base consumption and variability for each customer
        base_consumption = np.random.uniform(50, 500)
        std_dev_consumption = base_consumption * 0.05 + np.random.uniform(5, 20)
        
        # Assign payment history and customer category
        payment_history = np.random.choice(['On-time', 'Late', 'Missed'], p=[0.7, 0.2, 0.1])
        customer_category = np.random.choice(['Residential', 'Commercial'], p=[0.8, 0.2])

        is_tampering_customer = cust_id in tampering_customers
        
        # Define tampering characteristics if customer is a tampering customer
        tamper_duration = np.random.randint(2, 6) if is_tampering_customer else 0
        tamper_start_month = np.random.randint(3, num_months - tamper_duration + 1) if is_tampering_customer else -1

        for month in range(1, num_months + 1):
            consumption = max(0, np.random.normal(base_consumption, std_dev_consumption))
            billed_amount = consumption * np.random.uniform(20, 30)

            is_tampering_month = 0
            if is_tampering_customer and month >= tamper_start_month and month < tamper_start_month + tamper_duration:
                reduction_factor = np.random.uniform(0.1, 0.4) # Consumption reduction due to tampering
                consumption *= reduction_factor
                billed_amount = consumption * np.random.uniform(20, 30)
                if np.random.rand() < 0.1: # 10% chance of missing data during tampering
                    consumption = np.nan
                    billed_amount = np.nan
                is_tampering_month = 1

            data.append({
                'customer_id': cust_id,
                'month': month,
                'consumption_kwh': consumption,
                'billed_amount_ngn': billed_amount,
                'payment_history': payment_history,
                'customer_category': customer_category,
                'is_tampering_month': is_tampering_month,
                'is_tampering_customer': int(is_tampering_customer)
            })

    df = pd.DataFrame(data)

    # Introduce some random missing data for non-tampering months as well (0.5% chance)
    for col in ['consumption_kwh', 'billed_amount_ngn']:
        mask = np.random.rand(len(df)) < 0.005
        df.loc[mask & (df['is_tampering_month'] == 0), col] = np.nan
    
    df = df.sort_values(by=['customer_id', 'month']).reset_index(drop=True)

    df.to_csv(filename, index=False)
    print(f"Synthetic data generated for {num_customers} customers over {num_months} months and saved to '{os.path.abspath(filename)}'.")
    print(f"Actual tampering rate (by customer): {df.drop_duplicates(subset=['customer_id'])['is_tampering_customer'].mean():.4f}")

if __name__ == "__main__":
    generate_and_save_synthetic_data()
