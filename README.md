# Customer Churn Prediction 

This project predicts whether a telecom customer will **churn** (cancel their service) or not, based on their account and usage data. 

---

## 1. What problem are we solving?

We're predicting one of exactly two outcomes for each customer: **"Yes, they will churn"** or **"No, they won't."** This is called a **classification problem** in machine learning — as opposed to a **regression problem**, where you'd predict a continuous number (like predicting a house price). A common beginner confusion: "Logistic Regression" has the word *Regression* in it, but it is actually a **classification algorithm**, not a regression one — the name is historical/mathematical, not descriptive of its use case. It predicts a *probability* (a number between 0 and 1) and then converts that probability into a category (Yes/No) using a cutoff, usually 0.5.

---

## 2. The dataset

We used the **Telco Customer Churn dataset** — 7,043 customers, 21 columns, including:

- **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account info**: `tenure` (months as a customer), `Contract` (Month-to-month / One year / Two year), `PaymentMethod`, `PaperlessBilling`
- **Services subscribed**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Billing**: `MonthlyCharges`, `TotalCharges`
- **Target/label**: `Churn` (Yes/No) — the thing we're trying to predict

**Important distinction (a common point of confusion):** all the other columns are called **features** (inputs) — they describe the customer. `Churn` is the **target** (or **label**) — it's not "just another column," it's the answer we're trying to teach the model to predict using the other columns.

---

## 3. Step 1 — Exploratory Data Analysis (EDA)

**Why do this before touching any model?** Raw data is never perfectly clean or obviously useful. Before training anything, you inspect it like a detective: how big is it, what types of data does each column hold, are there errors or missing values, and — critically — does the data even *contain* real signal relating to the outcome you care about? If you skip this and jump straight to modeling, you risk feeding garbage into the model and getting garbage predictions out, without understanding why.

### What we checked and found:

- **`df.shape` → (7043, 21)** — 7043 customer rows, 21 columns.
- **`df.info()`** — lists each column's *dtype* (data type). Two matter most:
  - `object` = pandas' term for text/string data (e.g., `"Yes"`, `"Female"`, `"Month-to-month"`)
  - `int64` / `float64` = actual numeric types
- **`df['Churn'].value_counts(normalize=True)` → 73.5% No / 26.5% Yes.** This tells us the dataset is **imbalanced** — churners are the minority class. This one fact shapes almost everything later: it means plain "accuracy" as a scoring metric would be misleading (see the Evaluation Metrics section below for exactly why), so we deliberately chose not to rely on it.
- **`df.isnull().sum()` → all zeros.** No column reported any missing values — *but this turned out to be misleading*, explained next.

### The `TotalCharges` gotcha

`df.info()` showed `TotalCharges` as dtype `object` (text), even though it's clearly meant to be a money amount (a number). This is a classic real-world data quality issue: a column *looks* numeric to a human, but pandas loaded it as text, which means something in that column isn't a clean number.

We investigated with:
```python
df[df['TotalCharges'] == ' ']
```
This filters the dataframe down to only rows where `TotalCharges` is literally a blank space character `" "` — not a real missing value (`NaN`), which is why `isnull().sum()` didn't catch it; pandas only counts actual `NaN` as "null," not blank strings. This is the core confusion to understand: **a column can report "0 missing values" and still be broken**, if the "missing" values are disguised as something else (blank strings, placeholder text like `"N/A"`, etc.) instead of proper `NaN`.

We found **11 rows** with this blank value, and every single one had **`tenure == 0`** — brand new customers who had just signed up and hadn't been billed a full cycle yet, so they genuinely had no "total charges" recorded. All 11 also had `Churn == No` (you can't churn before your first billing cycle even starts).

### Manual relationship checks (building intuition before modeling)

We picked two columns to manually check against churn, using `groupby`:

```python
df.groupby('Contract')['Churn'].value_counts(normalize=True)
```
**What `groupby` does:** splits the dataframe into separate groups based on the values in one column (here, the 3 contract types), then lets you run further calculations *within each group separately*. Result:
- Month-to-month: **42.7%** churn
- One year: **11.3%** churn
- Two year: **2.8%** churn

This is a huge, intuitive signal: the shorter/more flexible a contract, the more likely a customer is to leave — makes sense, since month-to-month customers can leave anytime penalty-free, while 2-year customers are contractually locked in.

```python
df.groupby('Churn')['tenure'].mean()
```
Result: customers who stayed averaged **~37 months** of tenure; churners averaged only **~17 months** — roughly half. Newer customers are far more likely to leave; long-tenured customers are "stickier."

**Why do this manual check at all, if the model will figure out important features itself later?** Two reasons: (1) it builds your own intuition for whether the data makes real-world sense (a sanity check before you trust anything downstream), and (2) it gives you something to compare the model's own learned feature importances against later — if the model completely disagreed with something this obvious, that would be a red flag worth investigating (data leakage, a bug, etc.).

---

## 4. Step 2 — Data Cleaning

Based on what EDA found, we fixed the `TotalCharges` issue:

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
```

- **`pd.to_numeric(..., errors='coerce')`**: tries to convert every value in the column into a real number. Wherever it can't (our 11 blank-space rows), instead of crashing the whole program, `errors='coerce'` tells pandas to put `NaN` there instead — a proper, recognized "missing value" marker.
- **`.fillna(0)`**: replaces those `NaN`s with `0`, since we already determined (via the `tenure == 0` check) that these represent brand-new customers with genuinely zero charges so far — not truly "unknown" data, just zero.

**Why not just delete those 11 rows instead?** With only 11 rows out of 7043 (0.16%), it likely wouldn't matter much either way — but filling with a logically justified value (0) preserves more data and is generally preferred over deleting rows unless you have a strong reason to (e.g., if the missingness itself were suspicious or unexplainable).

---

## 5. Step 3 — Feature Engineering (turning text into numbers)

**The core problem:** ML algorithms are fundamentally doing math (weighted sums, distances, splits on numeric thresholds) — they cannot directly understand text like `"Yes"`, `"Female"`, or `"Month-to-month"`. Every text column has to be converted into numbers before any model can use it.

### Dropping useless columns
```python
df = df.drop('customerID', axis=1)
```
`customerID` is a unique random identifier per row (like `7590-VHVEG`) — it carries no real predictive information (a customer's ID string doesn't cause or prevent churn). Keeping it risks the model latching onto meaningless noise. `axis=1` tells pandas "drop a column" (as opposed to `axis=0`, which would drop a row).

### Encoding the target column
```python
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
```
`.map()` here is a simple find-and-replace: every `"Yes"` becomes `1`, every `"No"` becomes `0`. We do this for the target column specifically first because the model needs a numeric answer to learn to predict.

### Encoding the remaining feature columns — two different techniques, and why

There are two situations, handled differently:

1. **Binary columns (only 2 possible categories)**, e.g. `gender` (Male/Female), `Partner` (Yes/No) — these can simply become `0`/`1` directly, since there's no ordering issue with only two options.

2. **Multi-category columns (3+ options)**, e.g. `Contract` (Month-to-month / One year / Two year), `PaymentMethod`, `InternetService` — here we use **one-hot encoding** instead of just labeling them `0, 1, 2`. This is a critical concept to understand:

   **Why not just assign `Month-to-month=0, One year=1, Two year=2`?** Because that would falsely imply an *order and mathematical relationship* between categories that doesn't actually exist — the model would interpret "Two year" as literally "twice" the value of "One year," or assume "Two year" is somehow "more" than "Month-to-month" in a linear sense. That's a fabricated relationship the data never intended. One-hot encoding avoids this by creating a **separate 0/1 column for each category** instead, so each category is treated as independent and equally distinct, with no false ordering.

We used pandas' shortcut for this:
```python
df_encoded = pd.get_dummies(df, drop_first=True)
```
- `pd.get_dummies(df)` automatically finds every remaining text column and one-hot encodes it — e.g., `Contract` becomes two separate columns, `Contract_One year` and `Contract_Two year` (each 0 or 1).
- **`drop_first=True`** drops one category's column per original column. Why is this needed? If you know `Contract_One year = 0` AND `Contract_Two year = 0`, you already know for certain it must be `Month-to-month` — keeping a third `Contract_Month-to-month` column would be pure redundant information (this redundancy is called the **"dummy variable trap"**), and for some models (especially linear/logistic regression) that redundancy can cause instability in how weights are calculated. Dropping one category per group removes the redundancy without losing any actual information.

After this step, our dataframe grew from 20 columns (after dropping `customerID`) to **31 columns**, because each multi-category column expanded into several binary columns.

---

## 6. Step 4 — Train/Test Split

**The core idea, explained with an analogy:** Imagine a teacher has 100 practice questions with an answer key. She gives 80 (with answers) to students to study — that's **training**. She keeps 20 aside, hidden, and on exam day gives students only those 20 questions (no answers) — that's **testing**. Afterward, she compares student guesses against the answer key she'd kept hidden, to see how well they actually learned (versus just memorized the practice set).

```python
from sklearn.model_selection import train_test_split

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **`X`** = all the input features (every column except `Churn`)
- **`y`** = the target/label column we're trying to predict (`Churn`)
- **`test_size=0.2`** = hold back 20% of rows for testing, train on the remaining 80%. Result: **5634 training rows, 1409 test rows.** (80/20 is a common convention — not a strict rule, just a practical balance between "enough data to learn from" and "enough held-out data to trust the evaluation.")
- **`random_state=42`** = a fixed "seed" for the random split, so re-running this code produces the *exact same* split every time, instead of a different random one each run. This matters for reproducibility — if you or someone else reruns your notebook, you get identical results. (42 has no special meaning — it's just a very common convention in the ML community, a nod to *The Hitchhiker's Guide to the Galaxy*.)
- **`stratify=y`** — this is important and easy to overlook: since churn is imbalanced (73.5%/26.5%), a plain random split *could*, by chance, put too many or too few churners into either the train or test set. `stratify=y` forces **both** the train set and the test set to preserve that same ~73.5/26.5 ratio, so the test set is a fair, representative sample rather than a lucky or unlucky slice.

**A key point that's easy to miss:** after this split, both `X_test` and `y_test` sit "in storage" — the model is only ever shown `X_test` (features, no answers) when making predictions. `y_test` is kept aside purely for us to compare against afterward — it is never used to influence the model's training or predictions in any way. If it were, that would be a serious methodology error called **data leakage**.

---

## 7. Step 5 — Feature Scaling

Right after training the first Logistic Regression model, we hit this warning:
```
ConvergenceWarning: lbfgs failed to converge (status=1): TOTAL NO. OF ITERATIONS REACHED LIMIT.
```

**Why this happened:** Logistic Regression finds the best feature weights using an iterative optimization process (conceptually: repeatedly nudging the weights to reduce prediction error, like walking downhill step by step to find the lowest point). This works best when all input features are on a **similar numeric scale**. Our data had wildly different scales: `tenure` ranged roughly 0–72, `MonthlyCharges` roughly 18–118, `TotalCharges` up into the thousands, while our one-hot encoded columns were just 0 or 1. That mismatch makes the "downhill walk" lopsided and slow, so it hit the iteration limit before finishing.

**The fix — `StandardScaler`, not just raising `max_iter`:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

For every value in a numeric column, `StandardScaler` applies:
```
scaled_value = (original_value - mean) / standard_deviation
```
In plain terms: it shifts each column so its average becomes exactly 0, and rescales its spread so most values land roughly between -3 and +3 — regardless of whether the original column's range was 0–72 or 0–8000. This doesn't change the *relationships* in the data at all (a customer with above-average tenure still has above-average tenure after scaling) — it only changes the *units* the numbers are expressed in, purely so the optimization algorithm can converge efficiently.

**An important, easy-to-miss detail:** we used `.fit_transform()` on the training data, but only `.transform()` (no `fit`) on the test data. **Why the difference matters:** `fit` calculates the mean/standard deviation to use for scaling; `transform` actually applies the scaling using those numbers. If we called `fit_transform` on the test set too, the scaler would calculate a *different* mean/std from the test data itself, meaning we'd be "peeking" at test set statistics during preprocessing — a subtle form of **data leakage**. Instead, we scale the test set using the *training set's* mean/std, exactly mimicking the real-world scenario where you'd only ever know your historical (training) data's statistics when a new, unseen customer arrives.

**Note:** this scaling step is specifically needed for Logistic Regression (and other distance/gradient-based models like KNN, SVM, neural networks). Tree-based models like Random Forest and XGBoost don't require it, because they split data based on thresholds column-by-column, not on distances or gradients across combined scales.

---

## 8. Step 6 — Training the Model (Logistic Regression)

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

**Why start with Logistic Regression instead of something fancier?** Standard ML practice: always start with the simplest reasonable baseline model, measure its performance honestly, and only justify a more complex model (Random Forest, XGBoost, neural nets, etc.) if it actually beats that baseline by a meaningful margin. Jumping straight to the fanciest algorithm without a baseline makes it impossible to know if the complexity was actually worth it.

**What Logistic Regression does conceptually:** it assigns a numeric **weight** to every feature (e.g., `tenure` might get a weight like `-1.24`). It multiplies each feature's value by its weight, sums them all together into one number, then squashes that number into a probability between 0 and 1 using a mathematical function called the **sigmoid**. If that probability is above 0.5, it predicts "Yes churn" (1); otherwise "No" (0).

**What `.fit()` is actually doing under the hood:** it starts with initial (often near-zero or random) weights, checks how wrong its predictions are on the training data using a mathematical error measure, then adjusts the weights slightly in the direction that reduces that error — a process called **gradient descent**. It repeats this adjustment many times (`max_iter` limits how many attempts it gets) until the weights stop meaningfully improving. The end result is a fixed set of weights, one per feature, that best map the training features to the true training labels.

**A subtlety worth understanding:** `.fit()` only ever looks at the training data (`X_train_scaled`, `y_train`). It has no access to `X_test`/`y_test` at all during this step — those stay completely hidden until the separate `.predict()` call afterward.

---

## 9. Step 7 — Evaluating the Model (and why accuracy alone is misleading here)

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)
roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
```

Our actual results:
```
Accuracy: 0.807

Confusion Matrix:
 [[925 110]
  [162 212]]

              precision  recall  f1-score  support
0 (No churn)    0.85       0.89     0.87     1035
1 (Churn)       0.66       0.57     0.61     374

ROC-AUC: 0.842
```

### Why accuracy alone is a trap here
Remember: 73.5% of customers didn't churn. A "model" that just blindly predicts "No churn" for *every single customer*, without even looking at any features, would already score ~73.5% accuracy — while being completely useless, since it would never once correctly flag an actual churner (the entire point of building this system). This is exactly why, on an **imbalanced dataset**, accuracy alone can look deceptively good while the model is actually failing at the one thing that matters. Our model's 80.7% accuracy is genuinely better than that naive baseline, but we need the other metrics to know *how much* better, and specifically *where* it succeeds or fails.

### The Confusion Matrix, decoded in full
```
                  Predicted: No     Predicted: Yes
Actual: No             925 (TN)         110 (FP)
Actual: Yes            162 (FN)         212 (TP)
```
- **True Negative (925):** correctly predicted "won't churn" ✅
- **True Positive (212):** correctly predicted "will churn" ✅ — these are the churners we successfully caught
- **False Positive (110):** predicted churn, but they actually stayed — a "false alarm." Costly in a mild way (e.g., wasted retention offer/discount sent to someone who wasn't leaving anyway), but not the dangerous kind of error here.
- **False Negative (162): the most important error type in this specific business problem.** These are customers who *actually churned*, but the model told us "they'll stay" — meaning the business would lose them with zero warning and zero chance to intervene. Reducing this number is usually the top priority in churn prediction work, even more than raw accuracy.

### Precision, Recall, F1 — what they mean and why both matter (for the churn=1 class specifically)
- **Precision (0.66 for class 1):** "Of everyone the model predicted *would* churn, what fraction actually did?" Low precision means lots of false alarms — wasted retention effort on people who were never going to leave.
- **Recall (0.57 for class 1):** "Of everyone who *actually* churned, what fraction did the model successfully catch?" Low recall means the model is *missing* actual churners — these are our 162 false negatives. **This is usually the more business-critical number for churn prediction**, since missing a real churner (losing a customer with no warning) is typically costlier than a false alarm.
- **F1-score (0.61):** a single number balancing precision and recall together (their harmonic mean) — useful as a compact summary, but always worth checking the two underlying numbers separately too, since a model could reach a "decent" F1 while still being lopsided in a way that matters for your specific problem.

**A common beginner confusion:** precision and recall trade off against each other depending on where you set the decision threshold (we used the default 0.5 cutoff on the predicted probability). You could push recall up by lowering the threshold (predict "churn" more liberally), but that would drop precision (more false alarms) — there's no free lunch; you have to choose based on what mistake is more costly for the actual business use case.

### ROC-AUC — the "big picture" metric
**ROC-AUC (0.842)** measures how well the model *ranks* customers by churn risk, across every possible decision threshold — not just the default 0.5 cutoff. Concretely: if you randomly pick one customer who churned and one who didn't, ROC-AUC is the probability the model assigns a higher churn-risk score to the one who actually churned. It ranges from 0.5 (no better than random coin-flipping) to 1.0 (perfect ranking). **0.842 is a genuinely good score** — it tells us the model has learned real, useful signal about relative churn risk, even though its default-threshold recall (0.57) shows there's room to improve how that signal gets converted into a final Yes/No decision.

---

## 10. Step 8 — Interpreting the Model's Learned Feature Importances

```python
coefficients = pd.Series(model.coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
```

For Logistic Regression, `model.coef_[0]` holds the actual weight the trained model assigned to each feature. **Sign and magnitude both matter:** a positive weight pushes the prediction *toward* churn (1); a negative weight pushes *toward* staying (0); and larger magnitude (ignoring sign) means stronger influence either way.

Our top results, and what they mean:

| Feature | Weight | Interpretation |
|---|---|---|
| `tenure` | -1.24 | Longer-tenured customers churn less. Matches our manual EDA finding exactly. |
| `MonthlyCharges` | -0.92 | Counterintuitive at first glance — explained below. |
| `InternetService_Fiber optic` | +0.78 | Having fiber internet is a strong churn risk factor — a new signal we hadn't manually checked. |
| `Contract_Two year` | -0.59 | Being on a 2-year contract strongly reduces churn — matches manual EDA. |
| `TotalCharges` | +0.51 | Slightly pushes toward churn — somewhat conflicts with `tenure`'s direction; explained below. |
| `Contract_One year` | -0.29 | Same direction as two-year, smaller effect. |
| `StreamingMovies_Yes` / `StreamingTV_Yes` | ~+0.26 | Mild churn signal, possibly tied to a higher-spend, more price-sensitive customer segment. |
| `MultipleLines_Yes` | +0.22 | Mild churn signal. |
| `PaperlessBilling_Yes` | +0.18 | Mild churn signal. |

### The important nuance: why `MonthlyCharges` and `TotalCharges` seem to disagree, and why coefficients aren't always simple to read in isolation

You might expect "higher monthly charges → more churn" (expensive service, unhappy customers), but the model found the opposite direction for `MonthlyCharges` alone. This is very likely because `InternetService_Fiber optic` — which is strongly and separately linked to churn — is *also* linked to higher `MonthlyCharges` in the raw data. Once the model accounts for fiber-vs-not as its own separate feature, the *remaining* effect of "charges" alone (holding fiber status constant) tilts toward higher-paying, longer-term, more invested customers who tend to stay.

Similarly, `tenure` (negative) and `TotalCharges` (positive) look like they're pulling in opposite directions, even though `TotalCharges` is largely just "tenure × monthly charges" mathematically — more months as a customer naturally means more total billed. This overlap between correlated features is called **multicollinearity**, and it's a well-known limitation of interpreting linear/logistic regression coefficients directly: **when two or more features carry overlapping information, their individual weights can look unstable or counterintuitive, even though the model's overall predictions are still perfectly valid.** This is one reason tree-based models (Random Forest, XGBoost) are often preferred for feature importance interpretation — their importance measures tend to be more robust to this correlation problem, though they have their own caveats too.

**Beginner takeaway:** the two cleanest, most trustworthy signals here are **tenure** and **contract type** — both independently confirmed by our manual EDA *and* the model's learned weights, which is reassuring (the model isn't hallucinating something disconnected from reality). Treat any single coefficient with some caution when correlated features are present nearby.

---

## 11. Common points of confusion, addressed directly

- **"Splitting the data" doesn't mean copying files anywhere** — it just creates 4 in-memory variables (`X_train`, `X_test`, `y_train`, `y_test`) from the same original dataframe. Nothing is saved to disk as part of this step.
- **`isnull().sum()` showing all zeros does NOT guarantee your data is clean.** Missing data can hide in disguised forms (blank strings, placeholder text) that aren't technically `NaN`. Always sanity-check suspicious dtypes (like a money column loaded as `object`).
- **Scaling data does NOT mean "squishing everything into a circle around 0."** It's a per-column linear transformation (subtract mean, divide by standard deviation) that changes the *numeric scale/units* of each column while preserving all the actual relationships/order within that column.
- **One-hot encoding vs. plain numeric labeling (0,1,2...)** — always use one-hot encoding for unordered categories (like contract type or payment method), because plain integer labels falsely imply a mathematical order/distance between categories that doesn't exist.
- **Logistic Regression's coefficients aren't always simple to interpret one at a time**, especially when features are correlated with each other (multicollinearity) — a large or surprising weight on one feature doesn't always mean what it looks like in isolation.
- **Accuracy alone is not a reliable scorecard for imbalanced classification problems** — always check the confusion matrix, precision, recall, and ROC-AUC together, and think about which type of mistake (false positive vs. false negative) is actually more costly for the real-world problem you're solving.

---

## 12. What's next

- Train and compare **Random Forest** and **XGBoost** against this Logistic Regression baseline, using proper cross-validation (not just one train/test split) for a fair comparison.
- Compare their feature importances against what we found here — do they agree on `tenure` and `Contract`, and do they resolve the `MonthlyCharges`/`TotalCharges` ambiguity better?
- Add **SHAP** analysis for individual-prediction-level explainability — not just "which features matter overall," but "why did the model predict *this specific customer* will churn."
