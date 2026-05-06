# train_model.py
import logging
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1️⃣ Load your dataset
data = pd.read_csv("Testing.csv")

# Filter out rows with NaN in target variable
data = data.dropna(subset=['Suitable Fertilizer'])
logger.info("Training on %d rows with valid fertilizer data", len(data))

# 2️⃣ Encode text columns to numbers
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 3️⃣ Split into features (X) and label (y)
X = data.drop(["Suitable Fertilizer", "Crop Yield"], axis=1)
y = data["Suitable Fertilizer"]

# 4️⃣ Apply SMOTE to balance classes
smote = SMOTE(random_state=42, k_neighbors=1)  # k_neighbors=1 since minority class has only 4 samples
X_balanced, y_balanced = smote.fit_resample(X, y)

logger.info("After SMOTE: %d samples (was %d)", len(X_balanced), len(X))

# 5️⃣ Split for training/testing (optional)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# 6️⃣ Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 7️⃣ Save model and encoders
joblib.dump(model, "crop_Model1.joblib")
joblib.dump(label_encoders, "label_encoders.joblib")
logger.info("Training complete with SMOTE balancing. Saved crop_Model1.joblib and label_encoders.joblib")

print("✅ Model trained and saved successfully with SMOTE balancing")
