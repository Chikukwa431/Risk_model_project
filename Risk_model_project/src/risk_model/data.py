import pandas as pd
from sklearn.preprocessing import LabelEncoder

NUMERIC_FEATURES = ['Age', 'Youth_Activity_Level']
CATEGORICAL_FEATURES = [
    'Gender', 'Peer_Influence', 'Family_Environment', 
    'Psychological_Status', 'Social_Environment', 
    'Drug_Exposure', 'Received_Counselling'
]

def load_data(file_path: str):
    """Loads dataset and extracts feature space and target variable."""
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Youth_ID', 'Substance_Abuse_Risk'], errors='ignore')
    y = df['Substance_Abuse_Risk']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return df, X, y_encoded, le
