# 📊 Telecom Customer Churn Prediction using Decision Tree

This project predicts whether a telecom customer is likely to **churn** (i.e., stop using the service) based on usage patterns and service-related features. It uses a **Decision Tree Classifier** trained on customer behavior data.

---

## 🧠 Problem Statement

Telecom companies face major revenue loss due to customer churn. Early identification of customers likely to churn enables better retention strategies. This ML model helps classify customers as **churners or non-churners** based on their behavior.

---

## 📁 Dataset

- **Name:** Telecom Customer Churn Dataset  
- **Size:** 667 rows × 11 columns  
- **Target:** `Churn` (1 = customer left, 0 = stayed)  
- **Features:**  
  - `AccountWeeks`, `ContractRenewal`, `DataPlan`, `DataUsage`  
  - `CustServCalls`, `DayMins`, `DayCalls`, `MonthlyCharge`  
  - `OverageFee`, `RoamMins`  

> ✅ Dataset Source: Local CSV file  
> ✅ No missing values  
> ✅ No scaling needed (Decision Tree is insensitive to feature scaling)

---

## 🔧 Technologies Used

- Python 3.x  
- pandas, NumPy, scikit-learn, matplotlib  
- Jupyter Notebook / VS Code  
- pickle (for model saving)

---

## 🚀 Workflow

1. **Data Preprocessing:**  
   - Loaded and cleaned dataset  
   - Verified data types and nulls  
   - No scaling or encoding needed

2. **Modeling:**  
   - Used `DecisionTreeClassifier`  
   - Split into training (80%) and test (20%)  
   - Evaluated using accuracy, confusion matrix, precision, recall, and F1-score

3. **Saving the Model:**  
   - Saved trained model using `pickle`

---

## 📈 Results

- **Accuracy:** 91.6%  
- **Confusion Matrix:**  
  ```
  [[558   8]
   [ 48  53]]
  ```

- **Precision (Churn Class = 1):** 87%  
- **Recall (Churn Class = 1):** 52%  
- **F1 Score (Churn Class = 1):** 65%

> 🔎 Model performs well overall, but recall for churners can be improved with:
> - Feature Engineering  
> - Ensemble methods (e.g., Random Forest, XGBoost)  
> - Hyperparameter tuning

---

## 💾 Model Usage

```python
import pickle

# Load model
with open("decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict
prediction = model.predict(X_test)
```

---

## 🧠 Future Improvements

- Try Random Forest or Gradient Boosting  
- Handle class imbalance (churn class is underrepresented)  
- Perform GridSearchCV for optimal depth, split criteria  
- Deploy as a Flask app or Streamlit dashboard

---

## 📌 Author

**Rushi** – Final Year B.Tech CSE Student, AI/ML Enthusiast  
_“Building intelligent systems to make life better.”_

---

## 📂 Project Structure

```
├── Dataset/
│   └── telecom_churn.csv
├── decision_tree_model.pkl
├── churn_decision_tree.ipynb
├── README.md
```

---

## ✅ Run This Project

```bash
# Step 1: Clone the repo
git clone https://github.com/yourusername/telecom-churn-decision-tree.git

# Step 2: Install requirements
pip install pandas scikit-learn matplotlib

# Step 3: Run the notebook
jupyter notebook churn_decision_tree.ipynb
```

---

## ⭐️ If you like this project, consider giving it a star on GitHub!
