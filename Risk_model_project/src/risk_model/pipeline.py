import sys
from risk_model.data import load_data
from risk_model.model import train_and_eval
from sklearn.metrics import classification_report

def main():
    csv_path = r'C:\Users\hp\Desktop\ZIM_Survey_Youth_Substance_Abuse_Dataset.csv'
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
        
    print(f"Loading data from: {csv_path}")
    
    df, X, y_encoded, le = load_data(csv_path)
    pipeline, X_train, X_test, y_train, y_test = train_and_eval(X, y_encoded)
    
    y_pred = pipeline.predict(X_test)
    print("\n=========================================================")
    print("           GLOBAL STANDARD CONSOLE LOG SUMMARY           ")
    print("=========================================================")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    main()
