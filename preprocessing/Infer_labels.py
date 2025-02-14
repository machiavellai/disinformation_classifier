import pandas as pd

def assign_label(row):
    """Infers labels based on dataset features like fact_check_url."""
    if isinstance(row['fact_check_url'], str) and len(row['fact_check_url']) > 0:
        return 1  # Fake News (fact-checked as false)
    else:
        return 0  # Real News (no fact-checking evidence)

def infer_labels(df):
    """Applies label inference to the dataset and returns updated DataFrame."""
    if 'fact_check_url' not in df.columns:
        raise ValueError("Dataset is missing the 'fact_check_url' column.")
    
    df['label'] = df.apply(assign_label, axis=1)
    
    # Verify label distribution
    print("Label Distribution:")
    print(df['label'].value_counts())
    
    return df
