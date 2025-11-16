# FME-UPC Datathon 2025: Revenue Prediction Pipeline

**Challenge**: Predict 7-day in-app purchase (IAP) revenue (`iap_revenue_d7`) for mobile game install events.

**Approach**: Teacher-Student Distillation with Two-Stage Revenue Prediction

---

## 📊 Dataset Insights

### Key Findings from EDA

The exploratory data analysis revealed five critical insights that drive our modeling approach:

#### 1. **Hour-of-Day Effect** - User behavior is NOT random
- **Revenue spikes** at specific hours (18h, 00-02h, 16h)
- **Install hour** is highly predictive of buyer probability and revenue
- **Model impact**: 
  - Teachers: CatBoost/LightGBM capture non-linear interactions (`hour × country`, `hour × category`, `hour × device`)
  - Students: Use `sin(hour)`, `cos(hour)`, and `hour_bin` features for speed

#### 2. **Category-Based Segmentation** - Some categories are gold mines
- Categories like `finance`, `real money casino`, `shopping`, `travel` have massive outliers (whales)
- **Advertiser category** is one of the strongest revenue predictors
- **Model impact**:
  - Teachers: Use target encoding (mean revenue per category) + HistOS sampling
  - Students: Precomputed lookup tables (`category_whale_rate`, `category_avg_rev_d7`, `category_buyer_rate`)

#### 3. **Geographic Differences** - Markets vary drastically
- High-value countries: US, IN, ID, MX, BR
- Low-value countries: PK, BD
- **Country determines** buyer probability and expected revenue
- **Model impact**:
  - Teachers: Capture `country × dev_os`, `country × category`, `country × hour` interactions
  - Students: Lookup tables (`country_avg_rev`, `country_buyer_rate`, `country_whale_rate`)

#### 4. **Bimodal Distribution** - Two-stage structure is natural
- **Buyers**: `log1p(revenue)` around 1-3, with whale clusters
- **Non-buyers**: All zeros
- **This is a mixture distribution**: `p(buyer) × E[revenue | buyer]`
- **Model impact**: Two-stage pipeline is the statistically correct structure

#### 5. **Feature Correlations** - Redundancies exist
- Buyer metrics at different horizons (`buyer_d1`, `buyer_d7`, `buyer_d14`, `buyer_d28`) are highly correlated
- Retention metrics form strong clusters
- **Model impact**:
  - Teachers: Robust to redundancy, add derived features (time between events, recency)
  - Students: Keep only best features (`retentions`, `buyer_d1`, `weeks_since_first_seen`, `avg_act_days`)

---

## 🏗️ Project Structure

```text
FME-UPC-datathon-2025/UlmXH/
├── data/
│   ├── raw/
│   │   ├── train/train/         # Hive-partitioned parquet (datetime=YYYY-MM-DD-HH-mm/)
│   │   └── test/test/           # Test data (similar structure)
│   ├── processed/
│   │   ├── encoders/            # Saved label encoders for inference
│   │   ├── lookup_tables/       # Precomputed statistics (country, category, bundle)
│   │   └── teacher_outputs/     # Soft labels from teacher models
│   └── submissions/             # Generated submission files
├── models/
│   ├── teachers/                # Large offline models (CatBoost, LightGBM)
│   │   ├── teacher_classifier_catboost.cbm
│   │   └── teacher_regressor_lgb_d7.txt
│   └── students/                # Small fast models for inference
│       ├── student_classifier_lgb.txt
│       └── student_regressor_lgb.txt
├── src/
│   ├── data/
│   │   ├── loader.py           # ✅ DONE: Dask-based loading with time splits
│   │   └── preprocessor.py     # Feature engineering logic
│   ├── features/
│   │   ├── __init__.py
│   │   ├── online.py           # Fast features for students (< 100μs)
│   │   ├── offline.py          # Rich features for teachers (slow OK)
│   │   └── lookup_tables.py    # Precomputed statistics generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── teacher_classifier.py
│   │   ├── teacher_regressor.py
│   │   ├── student_trainer.py
│   │   └── histos_sampling.py   # HistOS-like sampling for whale modeling
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py        # Fast inference pipeline
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # ✅ DONE: Structured logging
│       └── metrics.py          # Evaluation metrics (MSLE, AUC, etc.)
├── scripts/
│   ├── train_teachers.py       # Phase 3-4: Train CatBoost + LightGBM
│   ├── train_students.py       # Phase 5-6: Distillation
│   └── make_submission.py      # Phase 7: Generate submission.csv
├── research/
│   ├── explore_train_data.ipynb         # ✅ DONE: Dataset exploration
│   └── pol-andreu-research.ipynb        # ✅ DONE: Revenue analysis
├── tests/                       # Unit tests
├── docs/
│   └── DASK_BEST_PRACTICES.md  # ✅ DONE: Dask guidelines
├── pyproject.toml              # uv dependency management
└── README.md                   # This file
```

---

## 🚀 Training Pipeline

### Design Philosophy

**Teacher-Student Distillation**:
1. **Teachers** (offline): Large, slow models with rich features → high accuracy
2. **Students** (online): Small, fast models with cheap features → production-ready
3. **Distillation**: Students learn from teacher soft labels, not ground truth

**Two-Stage Revenue Prediction**:
- **Stage 1**: Classifier predicts `p(buyer_d7)`
- **Stage 2**: Regressor predicts `E[log1p(revenue_d7) | buyer]`
- **Final prediction**: `revenue_pred = p_buyer × exp(log_rev) - 1`

---

## 📋 Implementation Phases

### ✅ Phase 0: Environment Setup (DONE)

**Status**: Complete - using `uv` with Python 3.13

**Current dependencies**:
```toml
[project]
dependencies = [
    "dask[dataframe,distributed]>=2025.11.0",
    "jupyter>=1.1.1",
    "pandas>=2.3.3",
    "pyarrow>=22.0.0",
    "matplotlib>=3.9.0",
    "seaborn>=0.13.0",
]
```

**Next**: Add ML dependencies
```bash
uv add lightgbm catboost scikit-learn numpy
```

---

### ✅ Phase 1: Data Loading & Time-Based Splits (DONE)

**Status**: Complete - `src/data/loader.py` implements Dask-based loading

**What exists**:
- ✅ `DataLoader` class with time-based train/val/test splits
- ✅ Partition pruning for efficient filtering by `datetime`
- ✅ Best practices: lazy evaluation, single `compute()` calls, `persist()` for caching
- ✅ Dashboard monitoring for distributed computation

**Data splits** (6 days of hourly data):
- **Train**: Oct 1-5 (120 hours) → ~17M rows
- **Val**: Oct 6 (24 hours) → ~3.5M rows  
- **Test**: Separate parquet (no labels)

**Usage example**:
```python
from src.data.loader import DataLoader

loader = DataLoader(
    train_path="data/raw/train/train",
    test_path="data/raw/test/test"
)

# Load with time-based split
ddf_train, ddf_val = loader.load_train_val_split(
    val_start="2025-10-06-00-00",
    val_end="2025-10-06-23-00"
)

# For fast iteration: use a subset
ddf_train_subset = loader.load_train_val_split(
    val_start="2025-10-06-00-00",
    val_end="2025-10-06-23-00",
    train_time_filter="datetime < '2025-10-02-00-00'"  # Oct 1 only → ~3M rows
)
```

**Quick iteration note**: The full dataset (~20M rows) takes ~5 minutes to load. For training iteration:
- **Use Oct 1 only** (~3M rows, ~30-60 seconds load time)
- **Validate approach**, then scale to full data for final models

---

### ✅ Phase 2: Feature Engineering (COMPLETE)

**Goal**: Define feature sets for teachers (offline) vs students (online)

#### 2.1. Online Features (Students) - `src/features/online.py`

**Constraints**: 
- ⚡ **< 100μs per prediction**
- ❌ No groupby, no histogram rebuilds, no heavy aggregations
- ✅ Simple scalars, lookups, sum/len of lists

**Feature categories**:

1. **Request Features** (direct columns):
   - Categorical: `country`, `region`, `dev_os`, `dev_osv`, `dev_make`, `dev_model`, `carrier`
   - Advertiser: `advertiser_bundle`, `advertiser_category`, `advertiser_subcategory`, `advertiser_bottom_taxonomy_level`
   - Temporal: `hour`, `release_date` (convert to age)

2. **User Scalar Features** (direct columns):
   - Activity: `avg_act_days`, `avg_daily_sessions`, `avg_days_ins`, `avg_duration`
   - Ratios: `weekend_ratio`, `wifi_ratio`
   - Recency: `weeks_since_first_seen`

3. **Simple Aggregates** (from complex columns):
   - From `user_bundles`, `user_bundles_l28d`: `len(list)` → num_installed_apps
   - From `bundles_ins`: `len(array)` → num_bundles_ins
   - From `num_buys_bundle` (dict): `sum(values)`, `len(keys)` → total_buys, num_bought_bundles
   - From `iap_revenue_usd_bundle`: `sum(values)` → total_past_revenue

4. **Precomputed Lookup Tables** (O(1) dict lookups):
   - Bundle stats: `bundle_buyer_rate`, `bundle_avg_rev_d7`, `bundle_whale_rate`
   - Category stats: `category_buyer_rate`, `category_avg_rev_d7`, `category_whale_rate`
   - Country stats: `country_buyer_rate`, `country_avg_rev_d7`, `country_whale_rate`
   - Segment stats: `(country, dev_os)_buyer_rate`, `(country, dev_os)_avg_rev_d7`

**Implementation**:
```python
def build_online_features(
    ddf: dd.DataFrame,
    lookup_tables: Dict[str, Dict],
    encoders: Optional[Dict[str, Any]] = None
) -> Tuple[dd.DataFrame, Dict[str, Any]]:
    """Build fast features for student models.
    
    Returns:
        X_online: DataFrame with encoded features
        encoders: Dict of encoders (save for inference)
    """
    # 1. Encode categoricals (OrdinalEncoder or LabelEncoder)
    # 2. Extract simple aggregates from complex columns
    # 3. Add lookup table features
    # 4. Return X_online + encoders
```

#### 2.2. Offline Features (Teachers) - `src/features/offline.py`

**Constraints**: 
- 🐌 **Slow OK** (offline training only)
- ✅ Heavy aggregations, histograms, complex derived features

**Feature categories** (includes all online features + these):

1. **Histogram Features** (from `*_hist` columns):
   - Columns: `city_hist`, `country_hist`, `region_hist`, `dev_language_hist`, `dev_osv_hist`, `hour_ratio`
   - Extracted features per histogram:
     - `n_unique_keys`: diversity of behavior
     - `total_count`: activity level
     - `entropy`: `- sum(p * log(p))` where `p = count / total`
     - `top1_fraction`: `max(counts) / sum(counts)` → loyalty to one value

2. **Revenue/Buy Map Features**:
   - Columns: `num_buys_bundle`, `num_buys_category`, `iap_revenue_usd_bundle`, `iap_revenue_usd_category`, etc.
   - Extracted features per map:
     - `sum_values`, `max_values`, `mean_values` (sum/count)
     - `n_nonzero_keys`: number of distinct bundles/categories with activity
     - `advertiser_specific_value`: value for current `advertiser_bundle` if exists

3. **Whale Features**:
   - From `whale_users_bundle_*` columns:
     - `max_revenue_prank`, `max_num_buys_prank`
     - `current_bundle_revenue_prank`: whale rank for this advertiser

4. **Recency Features**:
   - From `last_buy_ts_bundle`, `last_install_ts_bundle`, `first_request_ts_*`:
     - `days_since_last_buy`, `days_since_last_install`, `days_since_first_request`
     - `recency_for_current_bundle` vs `min_recency_other_bundles`

5. **Cross-Feature Interactions** (CatBoost/LightGBM handle these):
   - `hour × country`, `hour × category`, `category × dev_os`, etc.

**Implementation**:
```python
def build_offline_features(
    ddf: dd.DataFrame,
    encoders_from_online: Dict[str, Any]
) -> dd.DataFrame:
    """Build rich features for teacher models.
    
    Uses same encoders as online features for consistency.
    Returns X_offline with all heavy features computed.
    """
    # 1. Start with online features
    # 2. Add histogram aggregates
    # 3. Add revenue/buy map features
    # 4. Add whale features
    # 5. Add recency features
    # 6. Return X_offline
```

#### 2.3. Lookup Table Generation - `src/features/lookup_tables.py`

**Goal**: Precompute statistics for fast inference

**Tables to generate** (from training data only):

1. **Bundle-level**:
   ```python
   bundle_stats = {
       'com.example.game': {
           'buyer_rate': 0.15,
           'avg_rev_d7': 2.34,
           'whale_rate': 0.02
       },
       ...
   }
   ```

2. **Category-level** (same structure as bundle)

3. **Country-level** (same structure)

4. **Segment-level** (`(country, dev_os)` tuples):
   ```python
   segment_stats = {
       ('US', 'ios'): {'buyer_rate': 0.20, 'avg_rev_d7': 4.5},
       ('US', 'android'): {'buyer_rate': 0.12, 'avg_rev_d7': 2.1},
       ...
   }
   ```

**Implementation**:
```python
def generate_lookup_tables(ddf_train: dd.DataFrame) -> Dict[str, Dict]:
    """Precompute statistics from training data.
    
    Returns:
        lookup_tables: Dict with keys ['bundle', 'category', 'country', 'segment']
    """
    # Compute aggregates using Dask groupby
    # Save to data/processed/lookup_tables/
```

**Usage in online features**: Simple dict lookups with defaults for unseen keys.

---

### 🎯 Phase 3: Teacher A - CatBoost Buyer Classifier

**Goal**: Strong `p(buyer_d7)` predictor using rich offline features

**File**: `src/models/teacher_classifier.py`

#### 3.1. Data Preparation

```python
from src.data.loader import DataLoader
from src.features.offline import build_offline_features

# Load data (subset for fast iteration)
loader = DataLoader(train_path="data/raw/train/train")
ddf_train, ddf_val = loader.load_train_val_split(
    val_start="2025-10-06-00-00",
    val_end="2025-10-06-23-00",
    train_time_filter="datetime < '2025-10-02-00-00'"  # Oct 1 only
)

# Build offline features
X_train = build_offline_features(ddf_train)
X_val = build_offline_features(ddf_val)

# Target
y_train = ddf_train['buyer_d7'].compute()
y_val = ddf_val['buyer_d7'].compute()
```

#### 3.2. Handle Class Imbalance

**Option A**: Class weights
```python
from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
sample_weights = np.where(y_train == 1, weights[1], weights[0])
```

**Option B**: Oversample minority class to ~20-30%
```python
from sklearn.utils import resample

# Separate classes
X_neg = X_train[y_train == 0]
X_pos = X_train[y_train == 1]

# Oversample positive
X_pos_resampled = resample(X_pos, n_samples=int(len(X_neg) * 0.3), random_state=42)

# Combine
X_train_balanced = pd.concat([X_neg, X_pos_resampled])
y_train_balanced = np.concatenate([np.zeros(len(X_neg)), np.ones(len(X_pos_resampled))])
```

#### 3.3. Train CatBoost

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    loss_function='Logloss',
    eval_metric='AUC',
    depth=8,
    learning_rate=0.05,
    iterations=2000,
    early_stopping_rounds=100,
    verbose=200,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    sample_weight=sample_weights  # if using weights
)

# Save model
model.save_model('models/teachers/teacher_classifier_catboost.cbm')
```

#### 3.4. Generate Soft Labels

```python
# Predict probabilities (these are "soft labels" for distillation)
p_buyer_train = model.predict_proba(X_train)[:, 1]
p_buyer_val = model.predict_proba(X_val)[:, 1]

# Save for distillation
np.savez(
    'data/processed/teacher_outputs/teacher_classifier_outputs.npz',
    p_buyer_train=p_buyer_train,
    p_buyer_val=p_buyer_val,
    y_train=y_train,
    y_val=y_val
)
```

#### 3.5. Evaluation

```python
from sklearn.metrics import roc_auc_score, average_precision_score

print(f"Val AUC: {roc_auc_score(y_val, p_buyer_val):.4f}")
print(f"Val AUC-PR: {average_precision_score(y_val, p_buyer_val):.4f}")
```

---

### 🎯 Phase 4: Teacher B - LightGBM Revenue Regressor with HistOS

**Goal**: Strong `E[log1p(revenue_d7)]` predictor with whale-aware sampling

**File**: `src/models/teacher_regressor.py`

#### 4.1. HistOS-like Sampling - `src/models/histos_sampling.py`

**Motivation**: Revenue distribution is heavily skewed:
- **Many zeros** (non-buyers)
- **Few whales** (high spenders)
- Standard training underrepresents whales → poor predictions for high-value users

**Solution**: Oversample high-revenue bins

```python
def histos_sample(
    df: pd.DataFrame,
    revenue_col: str = 'iap_revenue_d7',
    bins: List[float] = [0, 1, 3, 6, 10, np.inf],
    weights: List[float] = [0.3, 1.0, 2.0, 3.0, 10.0],
    random_state: int = 42
) -> pd.DataFrame:
    """HistOS-like sampling for revenue prediction.
    
    Args:
        df: Input DataFrame
        revenue_col: Revenue column name
        bins: Revenue bin edges (in original scale)
        weights: Sampling weight for each bin (higher = more samples)
        
    Returns:
        Sampled DataFrame with overrepresentation of whales
    """
    df = df.copy()
    df['log_rev'] = np.log1p(df[revenue_col])
    
    # Assign bins
    df['rev_bin'] = pd.cut(
        df[revenue_col],
        bins=bins,
        labels=range(len(bins) - 1),
        include_lowest=True
    )
    
    # Sample each bin according to weights
    sampled = []
    for bin_idx, weight in enumerate(weights):
        bin_df = df[df['rev_bin'] == bin_idx]
        if len(bin_df) == 0:
            continue
        
        # Sample with replacement if weight > 1
        n_samples = int(len(bin_df) * weight)
        sampled.append(bin_df.sample(n=n_samples, replace=(weight > 1), random_state=random_state))
    
    result = pd.concat(sampled, ignore_index=True)
    result = result.drop(columns=['rev_bin'])
    
    return result
```

#### 4.2. Train LightGBM Regressor

```python
import lightgbm as lgb
from src.models.histos_sampling import histos_sample

# Load data and features (same as classifier)
X_train = build_offline_features(ddf_train).compute()
X_val = build_offline_features(ddf_val).compute()

# Target: log1p(revenue)
y_train = np.log1p(ddf_train['iap_revenue_d7'].compute())
y_val = np.log1p(ddf_val['iap_revenue_d7'].compute())

# Apply HistOS sampling to training set
train_df = pd.concat([X_train, pd.Series(y_train, name='log_rev')], axis=1)
train_sampled = histos_sample(
    train_df,
    revenue_col='iap_revenue_d7',  # need original revenue for binning
    bins=[0, 1, 3, 6, 10, np.inf],
    weights=[0.3, 1.0, 2.0, 3.0, 10.0]
)

X_train_sampled = train_sampled.drop(columns=['log_rev'])
y_train_sampled = train_sampled['log_rev']

# Train
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 127,
    'max_depth': 10,
    'learning_rate': 0.03,
    'n_estimators': 2000,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

model = lgb.LGBMRegressor(**params)
model.fit(
    X_train_sampled, y_train_sampled,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
)

# Save
model.booster_.save_model('models/teachers/teacher_regressor_lgb_d7.txt')
```

#### 4.3. Generate Soft Labels

```python
# Predict log-revenue
log_rev_train = model.predict(X_train)
log_rev_val = model.predict(X_val)

# Save
np.savez(
    'data/processed/teacher_outputs/teacher_regressor_outputs.npz',
    log_rev_train=log_rev_train,
    log_rev_val=log_rev_val,
    y_train=y_train,  # ground truth log-revenue
    y_val=y_val
)
```

#### 4.4. Evaluation

```python
from sklearn.metrics import mean_squared_log_error, mean_squared_error

# RMSE on log-scale
rmse = np.sqrt(mean_squared_error(y_val, log_rev_val))
print(f"Val RMSE (log-scale): {rmse:.4f}")

# MSLE on original scale
rev_pred_val = np.expm1(log_rev_val)
rev_true_val = np.expm1(y_val)
msle = mean_squared_log_error(rev_true_val, np.maximum(0, rev_pred_val))
print(f"Val MSLE: {msle:.4f}")
```

---

### 🎓 Phase 5-6: Student Training (Distillation)

**Goal**: Compress teacher knowledge into tiny models with fast features

**File**: `src/models/student_trainer.py`

#### 5.1. Build Online Features

```python
from src.features.online import build_online_features
from src.features.lookup_tables import generate_lookup_tables

# Generate lookup tables from train data
lookup_tables = generate_lookup_tables(ddf_train)

# Build fast features
X_stu_train, encoders = build_online_features(ddf_train, lookup_tables)
X_stu_val, _ = build_online_features(ddf_val, lookup_tables, encoders=encoders)

# Compute (students work with small datasets)
X_stu_train = X_stu_train.compute()
X_stu_val = X_stu_val.compute()

# Save encoders for inference
import pickle
with open('data/processed/encoders/online_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
```

#### 5.2. Load Teacher Outputs

```python
# Load soft labels
teacher_cls = np.load('data/processed/teacher_outputs/teacher_classifier_outputs.npz')
teacher_reg = np.load('data/processed/teacher_outputs/teacher_regressor_outputs.npz')

p_teacher_train = teacher_cls['p_buyer_train']
p_teacher_val = teacher_cls['p_buyer_val']

log_rev_teacher_train = teacher_reg['log_rev_train']
log_rev_teacher_val = teacher_reg['log_rev_val']

# Ground truth (for blending)
y_cls_train = teacher_cls['y_train']
y_cls_val = teacher_cls['y_val']

y_reg_train = teacher_reg['y_train']
y_reg_val = teacher_reg['y_val']
```

#### 5.3. Train Student Classifier

**Soft labels**: Blend teacher predictions with ground truth

```python
alpha = 0.8  # 80% teacher, 20% ground truth

y_soft_cls_train = alpha * p_teacher_train + (1 - alpha) * y_cls_train
y_soft_cls_val = alpha * p_teacher_val + (1 - alpha) * y_cls_val

# Train tiny LightGBM
student_cls = lgb.LGBMRegressor(  # regression to soft probability
    num_leaves=31,
    max_depth=5,
    learning_rate=0.05,
    n_estimators=100,
    feature_fraction=0.8,
    verbose=-1,
    random_state=42
)

student_cls.fit(
    X_stu_train, y_soft_cls_train,
    eval_set=[(X_stu_val, y_soft_cls_val)],
    callbacks=[lgb.early_stopping(50)]
)

student_cls.booster_.save_model('models/students/student_classifier_lgb.txt')
```

#### 5.4. Train Student Regressor

```python
y_soft_reg_train = alpha * log_rev_teacher_train + (1 - alpha) * y_reg_train
y_soft_reg_val = alpha * log_rev_teacher_val + (1 - alpha) * y_reg_val

student_reg = lgb.LGBMRegressor(
    num_leaves=31,
    max_depth=5,
    learning_rate=0.05,
    n_estimators=100,
    feature_fraction=0.8,
    verbose=-1,
    random_state=42
)

student_reg.fit(
    X_stu_train, y_soft_reg_train,
    eval_set=[(X_stu_val, y_soft_reg_val)],
    callbacks=[lgb.early_stopping(50)]
)

student_reg.booster_.save_model('models/students/student_regressor_lgb.txt')
```

#### 5.5. Evaluation

```python
# Classifier
p_student_val = np.clip(student_cls.predict(X_stu_val), 0, 1)
print(f"Student Classifier AUC: {roc_auc_score(y_cls_val, p_student_val):.4f}")
print(f"  (Teacher AUC: {roc_auc_score(y_cls_val, p_teacher_val):.4f})")

# Regressor
log_rev_student_val = student_reg.predict(X_stu_val)
print(f"Student Regressor RMSE: {np.sqrt(mean_squared_error(y_reg_val, log_rev_student_val)):.4f}")
print(f"  (Teacher RMSE: {np.sqrt(mean_squared_error(y_reg_val, log_rev_teacher_val)):.4f})")

# Two-stage prediction
rev_student_val = np.expm1(log_rev_student_val)
final_pred_val = p_student_val * rev_student_val

rev_true_val = np.expm1(y_reg_val)
msle = mean_squared_log_error(rev_true_val, np.maximum(0, final_pred_val))
print(f"Student Two-Stage MSLE: {msle:.4f}")
```

---

### 🚀 Phase 7: Inference Pipeline

**Goal**: Fast, production-ready inference on test set

**File**: `src/inference/predictor.py`

#### 7.1. Load Models and Artifacts

```python
import lightgbm as lgb
import pickle

class RevenuePredictor:
    def __init__(
        self,
        student_cls_path: str = 'models/students/student_classifier_lgb.txt',
        student_reg_path: str = 'models/students/student_regressor_lgb.txt',
        encoders_path: str = 'data/processed/encoders/online_encoders.pkl',
        lookup_tables_path: str = 'data/processed/lookup_tables/'
    ):
        # Load models
        self.classifier = lgb.Booster(model_file=student_cls_path)
        self.regressor = lgb.Booster(model_file=student_reg_path)
        
        # Load encoders
        with open(encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)
        
        # Load lookup tables
        self.lookup_tables = self._load_lookup_tables(lookup_tables_path)
    
    def _load_lookup_tables(self, path: str) -> Dict:
        """Load precomputed statistics."""
        import json
        tables = {}
        for name in ['bundle', 'category', 'country', 'segment']:
            with open(f"{path}/{name}_stats.json", 'r') as f:
                tables[name] = json.load(f)
        return tables
    
    def _make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same transformations as build_online_features."""
        from src.features.online import build_online_features
        X, _ = build_online_features(df, self.lookup_tables, encoders=self.encoders)
        return X
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict revenue for batch of rows.
        
        Args:
            df: Raw test data (same schema as train)
            
        Returns:
            revenue_pred: Predicted iap_revenue_d7
        """
        X = self._make_features(df)
        
        # Stage 1: Predict p(buyer)
        p_buyer = self.classifier.predict(X)
        p_buyer = np.clip(p_buyer, 0.0, 1.0)
        
        # Stage 2: Predict log(revenue)
        log_rev = self.regressor.predict(X)
        rev = np.maximum(0.0, np.expm1(log_rev))
        
        # Combine
        revenue_pred = p_buyer * rev
        
        return revenue_pred
```

#### 7.2. Generate Submission

**Script**: `scripts/make_submission.py`

```python
import pandas as pd
from src.data.loader import DataLoader
from src.inference.predictor import RevenuePredictor

# Load test data
loader = DataLoader(test_path="data/raw/test/test")
ddf_test = loader.load_test()

# Compute (test set is small)
df_test = ddf_test.compute()

# Predict
predictor = RevenuePredictor()
predictions = predictor.predict(df_test)

# Create submission
submission = pd.DataFrame({
    'row_id': df_test['row_id'],
    'iap_revenue_d7': predictions.astype(float)
})

# Save
submission.to_csv('data/submissions/submission.csv', index=False)
print(f"✅ Submission saved: {len(submission)} rows")
```

---

## 🎯 Quick Start Guide

### Initial Setup

```bash
# Clone repo
git clone <repo-url>
cd FME-UPC-datathon-2025/UlmXH

# Install dependencies
uv sync
uv add lightgbm catboost scikit-learn numpy

# Extract data (if needed)
unzip data/raw/smadex-challenge-predict-the-revenue.zip -d data/raw/
```

### Fast Iteration (1 day, ~3M rows, low memory)

```bash
# Train teachers (1 day of data - default for subset mode)
uv run python scripts/train_teachers.py --subset

# Or specify number of days explicitly (1-7)
uv run python scripts/train_teachers.py --days 1

# Train students
uv run python scripts/train_students.py

# Make submission
uv run python scripts/make_submission.py
```

### Gradual Scaling (2-4 days, moderate memory)

```bash
# Train with 2 days of data (~6M rows)
uv run python scripts/train_teachers.py --days 2

# Train with 3 days of data (~9M rows)
uv run python scripts/train_teachers.py --days 3
```

### Full Training (5-7 days, high memory)

```bash
# Train teachers (5 days - default for full mode, ~15M rows)
uv run python scripts/train_teachers.py --full

# Or use all 7 days (~21M rows, requires 32GB+ RAM)
uv run python scripts/train_teachers.py --days 7

# Train students
uv run python scripts/train_students.py

# Make submission
uv run python scripts/make_submission.py
```

---

## 📊 Expected Results

### Teacher Models (Offline, Rich Features)

- **CatBoost Classifier**: Val AUC ≈ 0.78-0.82
- **LightGBM Regressor**: Val MSLE ≈ 0.30-0.40

### Student Models (Online, Fast Features)

- **Should match teachers** within 2-5% performance
- **Significantly beat** simple baselines (e.g., global mean, per-category mean)

### Inference Speed

- **Target**: < 1ms per prediction (single row)
- **Batch mode**: ~10-100μs per row for large batches

---

## 🧪 Testing & Validation

### Unit Tests

```bash
# Run all tests
uv run pytest tests/

# Test specific module
uv run pytest tests/test_features.py
```

### Notebooks for Debugging

- `research/explore_train_data.ipynb`: Data exploration
- `research/pol-andreu-research.ipynb`: Revenue analysis and insights

---

## 📚 Best Practices Followed

### Dask Best Practices (see `docs/DASK_BEST_PRACTICES.md`)
- ✅ Load data with Dask (not pandas → Dask)
- ✅ Avoid repeated `compute()` calls
- ✅ Use partition pruning for time-based filtering
- ✅ Keep strings as PyArrow (convert to numeric for ML via `map_partitions`)
- ✅ Use `persist()` strategically
- ✅ Pure functions with explicit parameters

### Python Best Practices
- ✅ Type hints for all public APIs
- ✅ Dataclasses for configuration
- ✅ Structured logging
- ✅ Unit tests for core logic
- ✅ Dependency management with `uv`

### ML Best Practices
- ✅ Time-based train/val split (no data leakage)
- ✅ Save encoders/artifacts for reproducible inference
- ✅ Separate feature logic for teachers vs students
- ✅ Handle class imbalance and skewed distributions
- ✅ Distillation with soft labels

---

## 🚨 Common Pitfalls to Avoid

1. **Data Leakage**: Always use time-based splits, not random splits
2. **Compute Overuse**: Build Dask graphs, compute once at the end
3. **String Explosion**: Label-encode categoricals before training
4. **Memory Issues**: Use Dask, not pandas, for full dataset
5. **Inference Mismatch**: Ensure test features match training exactly (save encoders!)
6. **Whale Underrepresentation**: Use HistOS sampling for regressor
7. **Forgetting Lookup Tables**: Precompute and save statistics for fast inference

---

## 📝 TODO: Implementation Checklist

### Phase 2: Features ✅ COMPLETE
- [x] `src/features/online.py`: Fast feature builder
- [x] `src/features/offline.py`: Rich feature builder
- [x] `src/features/lookup_tables.py`: Statistics generation

### Phase 3-4: Teachers ✅ COMPLETE
- [x] `src/models/teacher_classifier.py`: CatBoost training (in train_teachers.py)
- [x] `src/models/teacher_regressor.py`: LightGBM training (in train_teachers.py)
- [x] `src/models/histos_sampling.py`: Whale-aware sampling

### Phase 5-6: Students ✅ COMPLETE
- [x] `src/models/student_trainer.py`: Distillation logic (in train_students.py)

### Phase 7: Inference ✅ COMPLETE
- [x] `src/inference/predictor.py`: Fast prediction pipeline
- [x] `scripts/train_teachers.py`: Teacher training script
- [x] `scripts/train_students.py`: Student training script
- [x] `scripts/make_submission.py`: Submission generation

### Utilities ✅ COMPLETE
- [x] `src/utils/metrics.py`: MSLE, AUC, etc.
- [x] Create `models/teachers/` and `models/students/` directories
- [x] Create `data/processed/` subdirectories

---

## 🤝 Contributing

See `CONTRIBUTING.md` for git workflow and dependency management with `uv`.

---

## 📄 License

Datathon 2025 - FME-UPC



