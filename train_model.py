import os
import numpy as np
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_features(file_path):
    """Extract multiple audio features"""
    try:
        x, sr = librosa.load(file_path, sr=22050, duration=3.0)  # Limit duration
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        mfcc_std = np.std(mfccs.T, axis=0)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(x)
        
        # Combine all features
        features = np.hstack([
            mfcc_mean,
            mfcc_std,
            np.mean(spectral_centroids),
            np.mean(spectral_rolloff),
            np.mean(zero_crossing_rate)
        ])
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Data collection
X, y = [], []
data_counts = {"female": 0, "male": 0}

for label, gender in enumerate(["female", "male"]):
    folder = f"data/{gender}"
    if not os.path.exists(folder):
        print(f"⚠️ Folder {folder} tidak ditemukan!")
        continue
        
    print(f"🔍 Memproses data {gender}...")
    
    files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3'))]
    data_counts[gender] = len(files)
    
    for filename in tqdm(files, desc=f"{gender.capitalize()} files"):
        file_path = os.path.join(folder, filename)
        features = extract_features(file_path)
        
        if features is not None:
            X.append(features)
            y.append(label)

# Check data balance
print(f"\n📊 Distribusi Data:")
print(f"Female: {data_counts['female']} files")
print(f"Male: {data_counts['male']} files")
print(f"Total features extracted: {len(X)}")

if len(X) == 0:
    print("❌ Tidak ada data yang berhasil diproses!")
    exit()

X = np.array(X)
y = np.array(y)

# Check for any NaN values
if np.isnan(X).any():
    print("⚠️ Ditemukan NaN values, menghapus...")
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train test split dengan stratify untuk menjaga balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📈 Split Data:")
print(f"Training: {X_train.shape[0]} samples")
print(f"Testing: {X_test.shape[0]} samples")

# Compute class weights untuk menangani imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"\n⚖️ Class Weights:")
print(f"Female (0): {class_weight_dict[0]:.3f}")
print(f"Male (1): {class_weight_dict[1]:.3f}")

# Model training dengan class weights
model = LogisticRegression(
    max_iter=1000, 
    random_state=42,
    class_weight=class_weight_dict  # Menangani imbalanced data
)

model.fit(X_train, y_train)

# Cross validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"\n🔄 Cross Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Evaluation
y_pred = model.predict(X_test)
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["female", "male"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n🔍 Confusion Matrix:")
print(f"          Predicted")
print(f"Actual    F   M")
print(f"Female   {cm[0,0]:2d}  {cm[0,1]:2d}")
print(f"Male     {cm[1,0]:2d}  {cm[1,1]:2d}")

# Prediction probabilities untuk analisis
y_proba = model.predict_proba(X_test)
print(f"\n🎯 Sample Predictions (first 10):")
for i in range(min(10, len(y_test))):
    actual = "Female" if y_test[i] == 0 else "Male"
    predicted = "Female" if y_pred[i] == 0 else "Male"
    confidence = max(y_proba[i]) * 100
    print(f"Actual: {actual:6} | Predicted: {predicted:6} | Confidence: {confidence:.1f}%")

# Save model dan scaler
joblib.dump(model, "gender_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save class weights untuk inference
joblib.dump(class_weight_dict, "class_weights.pkl")

print(f"\n✅ Model, scaler, dan class weights berhasil disimpan.")
print(f"📁 Files saved:")
print(f"  - gender_model.pkl")
print(f"  - scaler.pkl") 
print(f"  - class_weights.pkl")