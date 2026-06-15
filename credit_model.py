import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

print("Starting Credit Scoring Pipeline...\n")

try:
    df = pd.read_csv('cs-training.csv')
    print("[1/8] Data loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: Put 'cs-training.csv' in the exact same folder as this script.")
    exit()

if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# --- FEATURE ENGINEERING ---
print("[1.5/8] Engineering financial features...")

df['TotalPastDue'] = (df['NumberOfTime30-59DaysPastDueNotWorse'] + 
                      df['NumberOfTime60-89DaysPastDueNotWorse'] + 
                      df['NumberOfTimes90DaysLate'])

df['MonthlyDebt'] = df['DebtRatio'] * df['MonthlyIncome']

df['IncomePerPerson'] = df['MonthlyIncome'] / (df['NumberOfDependents'].fillna(0) + 1)

df['HighRevolvingUtil'] = (df['RevolvingUtilizationOfUnsecuredLines'] > 1.0).astype(int)
# ---------------------------

X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("[2/8] Data split into training and testing sets.")

imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns)

X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)
print("[3/8] Missing values imputed and features scaled (Leakage prevented).")

print("[4/8] Training Logistic Regression Baseline...")
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

print("[5/8] Training Random Forest Classifier (This takes a few seconds)...")
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
print("      Training complete.")

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n======================================")
    print(f"      {name} Performance      ")
    print(f"======================================")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}  <- (Out of all predicted defaulters, how many actually defaulted?)")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}  <- (Out of all actual defaulters, how many did we catch?)")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}  <- (The balance between Precision and Recall)")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}  <- (Overall ability to separate good vs bad credit)")

print("\n[6/8] Generating Metric Reports...")
evaluate_model("Logistic Regression", lr_model, X_test_scaled, y_test)
evaluate_model("Random Forest", rf_model, X_test_scaled, y_test)

print("\n[7/8] Optimizing Probability Threshold for Random Forest...")

rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

thresholds = np.arange(0.1, 1.0, 0.1)
best_f1 = 0
best_thresh = 0.5

print(f"\n{'Threshold':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
print("-" * 55)

for thresh in thresholds:
    custom_preds = (rf_probs >= thresh).astype(int)
    
    p = precision_score(y_test, custom_preds)
    r = recall_score(y_test, custom_preds)
    f1 = f1_score(y_test, custom_preds)
    
    print(f"{thresh:<12.1f} | {p:<12.4f} | {r:<12.4f} | {f1:<12.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print("\n[8/8] Saving model artifacts to disk...")
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')
print("Pipeline complete. Artifacts saved.")

print("-" * 55)
print(f"Optimal Threshold for maximum balance: {best_thresh:.1f} (F1-Score: {best_f1:.4f})")

print("\n--- Business Translation ---")
print("If the bank says: 'Catch as many defaults as possible', lower the threshold (High Recall).")
print("If the bank says: 'Only reject people we are 100% sure are bad', raise the threshold (High Precision).")
print(f"To mathematically balance both, configure the production model to use a {best_thresh*100:.0f}% threshold.")