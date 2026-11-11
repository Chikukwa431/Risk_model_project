from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from risk_model.data import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def create_pipeline() -> Pipeline:
    """Constructs the column transformer and random forest pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            class_weight='balanced'
        ))
    ])
    return pipeline

def train_and_eval(X, y_encoded):
    """Splits data (80/20 stratified) and trains the model pipeline."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    
    return pipeline, X_train, X_test, y_train, y_test
