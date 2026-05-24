"""
ChurnZero – Full ML Pipeline
Team: SuperNinja
Deliverable 3: Reproducible Python Code
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings, os, json
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

os.makedirs("charts", exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

train_df = pd.read_csv("ChurnZero_dataset_v1.csv")
test_df  = pd.read_csv("ChurnZero_test_v1.csv")

print(f"Train shape : {train_df.shape}")
print(f"Test  shape : {test_df.shape}")
print(f"Churn rate  : {train_df['churn'].mean()*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EDA  (save key charts)
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 2: EDA & charts")

# 2a – Churn distribution
fig, ax = plt.subplots(figsize=(5, 4))
counts = train_df['churn'].value_counts()
colors = ['#2196F3', '#F44336']
bars = ax.bar(['No Churn (0)', 'Churn (1)'], counts.values, color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{val:,}\n({val/len(train_df)*100:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Churn Distribution', fontsize=14, fontweight='bold', pad=10)
ax.set_ylabel('Count')
ax.set_ylim(0, counts.max() * 1.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("charts/churn_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# 2b – Age distribution by churn
fig, ax = plt.subplots(figsize=(7, 4))
for churn_val, color, label in [(0, '#2196F3', 'No Churn'), (1, '#F44336', 'Churn')]:
    subset = train_df[train_df['churn'] == churn_val]['age']
    ax.hist(subset, bins=25, alpha=0.6, color=color, label=label, edgecolor='white')
ax.set_title('Age Distribution by Churn', fontsize=13, fontweight='bold')
ax.set_xlabel('Age'); ax.set_ylabel('Count')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("charts/age_by_churn.png", dpi=150, bbox_inches='tight')
plt.close()

# 2c – Satisfaction score vs churn
fig, ax = plt.subplots(figsize=(6, 4))
train_df.boxplot(column='satisfaction_score', by='churn', ax=ax,
                 boxprops=dict(color='#2196F3'),
                 medianprops=dict(color='#F44336', linewidth=2))
ax.set_title('Satisfaction Score by Churn', fontsize=13, fontweight='bold')
ax.set_xlabel('Churn (0=No, 1=Yes)'); ax.set_ylabel('Satisfaction Score')
plt.suptitle('')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("charts/satisfaction_by_churn.png", dpi=150, bbox_inches='tight')
plt.close()

# 2d – Missing values heatmap
missing = train_df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) == 0:
    # No missing – create a placeholder chart
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, 'No Missing Values ✓', ha='center', va='center',
            fontsize=16, color='green', fontweight='bold')
    ax.axis('off')
    ax.set_title('Missing Value Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("charts/missing_values.png", dpi=150, bbox_inches='tight')
    plt.close()
else:
    fig, ax = plt.subplots(figsize=(8, 4))
    missing.head(20).plot(kind='bar', ax=ax, color='#FF7043')
    ax.set_title('Top Missing Value Columns', fontsize=13, fontweight='bold')
    ax.set_ylabel('Missing Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("charts/missing_values.png", dpi=150, bbox_inches='tight')
    plt.close()

# 2e – Correlation heatmap (numeric only, top 12 cols)
numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
corr_with_churn = train_df[numeric_cols].corr()['churn'].abs().sort_values(ascending=False)
top_cols = corr_with_churn.head(13).index.tolist()  # includes 'churn' itself
fig, ax = plt.subplots(figsize=(10, 8))
corr_mat = train_df[top_cols].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            ax=ax, annot_kws={'size': 8}, linewidths=0.5)
ax.set_title('Correlation Heatmap – Top Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("charts/correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

print("  Charts saved to charts/")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 3: Preprocessing")

def preprocess(df, fit_encoders=None, reference_columns=None):
    df = df.copy()

    # Drop id column for features
    id_col = df['customer_id'].copy() if 'customer_id' in df.columns else None
    drop_cols = ['customer_id']
    if 'churn' in df.columns:
        y = df['churn'].copy()
        drop_cols.append('churn')
    else:
        y = None

    df = df.drop(columns=drop_cols, errors='ignore')

    # Identify column types
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Fill missing values
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0] if len(df[c].mode()) > 0 else 'Unknown')

    # Label encode categoricals
    if fit_encoders is None:
        encoders = {}
        for c in cat_cols:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
    else:
        encoders = fit_encoders
        for c in cat_cols:
            le = encoders.get(c)
            if le:
                known = set(le.classes_)
                df[c] = df[c].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                df[c] = le.transform(df[c])

    # Align columns to reference
    if reference_columns is not None:
        for col in reference_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[reference_columns]

    return df, y, id_col, encoders

# Fit on train
X_train_raw, y_train, _, encoders = preprocess(train_df)
feature_cols = X_train_raw.columns.tolist()

# Transform test using same encoders & column order
X_test_raw, _, test_ids, _ = preprocess(test_df, fit_encoders=encoders, reference_columns=feature_cols)

print(f"  Train features: {X_train_raw.shape}")
print(f"  Test  features: {X_test_raw.shape}")
print(f"  Churn rate    : {y_train.mean()*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 4: Feature Engineering")

def engineer_features(X_raw, original_df):
    """Add derived features without data leakage."""
    df = X_raw.copy()
    orig = original_df.copy()

    # Recompute from original (or use already-encoded proxies)
    # Balance stress index
    if 'balance_decline_percentage' in df and 'account_inactive_days' in df:
        df['balance_stress_index'] = df['balance_decline_percentage'] * df['account_inactive_days']

    # Digital engagement score
    if 'mobile_app_login_count' in df and 'digital_transaction_ratio' in df:
        df['digital_engagement_score'] = df['mobile_app_login_count'] * df['digital_transaction_ratio']

    # Complaint severity
    if 'total_complaints' in df and 'unresolved_complaint_count' in df:
        df['complaint_severity'] = df['total_complaints'] + 2 * df['unresolved_complaint_count']

    # Product diversification
    product_flags = [c for c in df.columns if c.endswith('_flag')]
    if product_flags:
        df['product_count'] = df[product_flags].sum(axis=1)

    # Credit pressure
    if 'credit_utilization_ratio' in df and 'loan_default_risk_score' in df:
        df['credit_pressure'] = df['credit_utilization_ratio'] * df['loan_default_risk_score']

    # Recency risk
    if 'last_login_days' in df and 'last_contacted_days' in df:
        df['recency_risk'] = df['last_login_days'] + df['last_contacted_days']

    return df

X_train = engineer_features(X_train_raw, train_df)
X_test  = engineer_features(X_test_raw,  test_df)

# Re-align columns
all_cols = X_train.columns.tolist()
for c in all_cols:
    if c not in X_test.columns:
        X_test[c] = 0
X_test = X_test[all_cols]

print(f"  Features after engineering: {X_train.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. HANDLE CLASS IMBALANCE WITH SMOTE
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 5: Handling class imbalance (SMOTE)")
sm = SMOTE(random_state=SEED)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
print(f"  Before SMOTE: {dict(pd.Series(y_train).value_counts())}")
print(f"  After  SMOTE: {dict(pd.Series(y_resampled).value_counts())}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL TRAINING – XGBoost (primary) + cross-validation
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 6: Model Training (XGBoost)")

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=SEED,
    eval_metric='aucpr',
    use_label_encoder=False,
    verbosity=0
)

# Cross-validation on original (imbalanced) train data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_prauc = cross_val_score(xgb_model, X_train, y_train, cv=cv,
                           scoring='average_precision', n_jobs=-1)
cv_f1    = cross_val_score(xgb_model, X_train, y_train, cv=cv,
                           scoring='f1', n_jobs=-1)

print(f"  CV PR-AUC : {cv_prauc.mean():.4f} ± {cv_prauc.std():.4f}")
print(f"  CV F1     : {cv_f1.mean():.4f}   ± {cv_f1.std():.4f}")

# Fit on SMOTE-resampled data
xgb_model.fit(X_resampled, y_resampled)

# Evaluate on original train
y_train_prob  = xgb_model.predict_proba(X_train)[:, 1]
y_train_pred  = (y_train_prob >= 0.5).astype(int)

train_prauc = average_precision_score(y_train, y_train_prob)
train_f1    = f1_score(y_train, y_train_pred)
train_roc   = roc_auc_score(y_train, y_train_prob)
cm          = confusion_matrix(y_train, y_train_pred)

print(f"\n  Train PR-AUC : {train_prauc:.4f}")
print(f"  Train F1     : {train_f1:.4f}")
print(f"  Train ROC-AUC: {train_roc:.4f}")
print(f"\n  Confusion Matrix:\n{cm}")
print(f"\n{classification_report(y_train, y_train_pred)}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE PERFORMANCE CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 7: Saving performance charts")

# PR Curve
precision_arr, recall_arr, _ = precision_recall_curve(y_train, y_train_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall_arr, precision_arr, color='#1976D2', lw=2,
        label=f'XGBoost (PR-AUC = {train_prauc:.3f})')
ax.axhline(y=y_train.mean(), color='grey', linestyle='--', lw=1.5, label='Baseline')
ax.fill_between(recall_arr, precision_arr, alpha=0.15, color='#1976D2')
ax.set_xlabel('Recall', fontsize=12); ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("charts/pr_curve.png", dpi=150, bbox_inches='tight')
plt.close()

# Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'],
            annot_kws={'size': 14})
ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("charts/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

# Feature Importance
importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 6))
colors_bar = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_features)))
bars = ax.barh(top_features.index[::-1], top_features.values[::-1],
               color=colors_bar, edgecolor='white')
ax.set_title('Top 15 Feature Importances (XGBoost)', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("charts/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()

# CV Scores bar chart
fig, ax = plt.subplots(figsize=(6, 4))
folds = [f'Fold {i+1}' for i in range(5)]
x = np.arange(5)
ax.bar(x - 0.2, cv_prauc, 0.35, label='PR-AUC', color='#1976D2')
ax.bar(x + 0.2, cv_f1,    0.35, label='F1',     color='#43A047')
ax.set_xticks(x); ax.set_xticklabels(folds)
ax.set_ylim(0, 1); ax.set_ylabel('Score')
ax.set_title('5-Fold Cross-Validation Scores', fontsize=13, fontweight='bold')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.axhline(cv_prauc.mean(), color='#1976D2', ls='--', lw=1)
ax.axhline(cv_f1.mean(),    color='#43A047', ls='--', lw=1)
plt.tight_layout()
plt.savefig("charts/cv_scores.png", dpi=150, bbox_inches='tight')
plt.close()

# Business Cost Analysis
TN, FP, FN, TP = cm.ravel()
cost_fn = 40000  # cost of missed churn (FN) in ₹
cost_fp = 500    # cost of false alarm (FP) in ₹
total_cost = FN * cost_fn + FP * cost_fp

# Show business cost comparison
baseline_cost = y_train.sum() * cost_fn  # if we predict everyone stays
fig, ax = plt.subplots(figsize=(6, 4))
scenarios = ['No Model\n(All Predicted 0)', 'XGBoost Model']
costs = [baseline_cost, total_cost]
colors_cost = ['#EF5350', '#66BB6A']
bars = ax.bar(scenarios, costs, color=colors_cost, edgecolor='white', linewidth=1.5, width=0.4)
for bar, cost in zip(bars, costs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + baseline_cost*0.01,
            f'₹{cost:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_title('Business Cost Comparison (₹)', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Cost (₹)')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
savings = baseline_cost - total_cost
ax.text(0.5, 0.92, f'Savings: ₹{savings:,.0f}  ({savings/baseline_cost*100:.1f}%)',
        transform=ax.transAxes, ha='center', fontsize=11, color='green', fontweight='bold')
plt.tight_layout()
plt.savefig("charts/business_cost.png", dpi=150, bbox_inches='tight')
plt.close()

print("  All charts saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. GENERATE PREDICTIONS ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 8: Generating test predictions")

test_prob = xgb_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= 0.5).astype(int)

predictions_df = pd.DataFrame({
    'customer_id'       : test_ids.values,
    'churn_prediction'  : test_pred,
    'churn_probability' : np.round(test_prob, 6)
})

assert len(predictions_df) == 2026, "Row count mismatch!"
assert predictions_df['churn_prediction'].isnull().sum() == 0
assert predictions_df['churn_probability'].isnull().sum() == 0

predictions_df.to_csv("ChurnZero_SuperNinja_Predictions.csv", index=False)
print(f"  Predictions saved: {len(predictions_df)} rows")
print(f"  Predicted churn rate: {test_pred.mean()*100:.2f}%")
print(f"  Probability range: {test_prob.min():.4f} – {test_prob.max():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE METRICS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
metrics = {
    "cv_prauc_mean"  : round(float(cv_prauc.mean()), 4),
    "cv_prauc_std"   : round(float(cv_prauc.std()), 4),
    "cv_f1_mean"     : round(float(cv_f1.mean()), 4),
    "cv_f1_std"      : round(float(cv_f1.std()), 4),
    "train_prauc"    : round(train_prauc, 4),
    "train_f1"       : round(train_f1, 4),
    "train_roc_auc"  : round(train_roc, 4),
    "confusion_matrix": cm.tolist(),
    "business_cost_model"   : int(total_cost),
    "business_cost_baseline": int(baseline_cost),
    "savings"               : int(savings),
    "top_features"          : top_features.head(10).index.tolist()
}

with open("metrics_summary.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "="*60)
print("ALL STEPS COMPLETE")
print("="*60)
print(f"  ✓ ChurnZero_SuperNinja_Predictions.csv")
print(f"  ✓ charts/  (8 chart PNGs)")
print(f"  ✓ metrics_summary.json")
print(f"\n  CV PR-AUC : {cv_prauc.mean():.4f}")
print(f"  CV F1     : {cv_f1.mean():.4f}")
print(f"  Cost saved: ₹{savings:,.0f}")
