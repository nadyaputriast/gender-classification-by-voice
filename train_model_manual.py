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

# --- Manual Feature Extraction Functions ---
def manual_trim_silence(audio_data, threshold=0.01):
    start_index = 0
    for i in range(len(audio_data)):
        if abs(audio_data[i]) > threshold:
            start_index = i
            break
    end_index = len(audio_data) - 1
    for i in range(len(audio_data) - 1, -1, -1):
        if abs(audio_data[i]) > threshold:
            end_index = i
            break
    return audio_data[start_index : end_index + 1]

def manual_normalize_audio(audio_data):
    max_abs_amplitude = np.max(np.abs(audio_data))
    if max_abs_amplitude == 0:
        return audio_data
    return audio_data / max_abs_amplitude

def manual_stft(audio_data, frame_size=2048, hop_length=512, window_fn=np.hanning):
    audio_data = np.asarray(audio_data)
    if len(audio_data) < frame_size:
        audio_data = np.pad(audio_data, (0, frame_size - len(audio_data)), mode='constant')
    num_frames = 1 + int((len(audio_data) - frame_size) / hop_length)
    stft_matrix = np.empty((frame_size // 2 + 1, num_frames), dtype=np.complex64)
    window = window_fn(frame_size)
    for n in range(num_frames):
        start = n * hop_length
        frame = audio_data[start:start + frame_size] * window
        fft_result = np.fft.rfft(frame)
        stft_matrix[:, n] = fft_result
    return stft_matrix

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def manual_mel_filter_bank(num_filters, n_fft, sr, fmin=0, fmax=None):
    if fmax is None:
        fmax = sr / 2
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)
    mel_points = np.linspace(min_mel, max_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    filter_bank = np.zeros((num_filters, int(n_fft // 2 + 1)))
    for i in range(num_filters):
        left = bin_points[i]
        center = bin_points[i+1]
        right = bin_points[i+2]
        if left < center:
            filter_bank[i, left:center] = (np.arange(left, center) - left) / (center - left)
        if center < right:
            filter_bank[i, center:right] = (right - np.arange(center, right)) / (right - center)
    return filter_bank

def manual_dct(x, type=2, norm='ortho'):
    N = x.shape[0]
    X = np.zeros_like(x)
    for k in range(N):
        X[k] = np.sum(x * np.cos(np.pi * k * (2 * np.arange(N) + 1) / (2 * N)))
    if norm == 'ortho':
        X[0] *= 1/np.sqrt(N)
        X[1:] *= np.sqrt(2/N)
    return X

def manual_mfcc(y, sr, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=128, fmin=0, fmax=None):
    stft_matrix = manual_stft(y, frame_size=n_fft, hop_length=hop_length)
    S = np.abs(stft_matrix) ** 2
    mel_fb = manual_mel_filter_bank(n_mels, n_fft, sr, fmin=fmin, fmax=fmax)
    mel_spec = np.dot(mel_fb, S)
    log_mel_spec = np.log(mel_spec + 1e-10)
    mfccs = np.apply_along_axis(lambda x: manual_dct(x, type=2, norm='ortho')[:n_mfcc], axis=0, arr=log_mel_spec)
    return mfccs

def manual_spectral_rolloff(y, sr, n_fft=2048, hop_length=512, roll_percent=0.85):
    stft = np.abs(manual_stft(y, frame_size=n_fft, hop_length=hop_length))**2
    energy = np.sum(stft, axis=0)
    cumulative_energy = np.cumsum(stft, axis=0)
    rolloff_bin = np.argmax(cumulative_energy >= roll_percent * energy, axis=0)
    freqs = np.linspace(0, sr/2, stft.shape[0])
    rolloff = freqs[rolloff_bin]
    return rolloff

def manual_zcr(y, frame_length=2048, hop_length=512):
    y = np.pad(y, (0, frame_length - len(y) % hop_length), mode='constant')
    n_frames = 1 + (len(y) - frame_length) // hop_length
    zcr = np.empty(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + frame_length]
        signs = np.sign(frame)
        zcr[i] = np.sum(np.abs(np.diff(signs))) / 2 / frame_length
    return zcr

def manual_spectral_centroid(stft_matrix, sr):
    freqs = np.linspace(0, sr/2, stft_matrix.shape[0])
    mag = np.abs(stft_matrix)
    centroid = np.sum(freqs[:, np.newaxis] * mag, axis=0) / (np.sum(mag, axis=0) + 1e-10)
    return centroid

# --- Manual Feature Extraction Pipeline ---
def extract_features_manual(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=3.0)
        y = manual_trim_silence(y, threshold=0.01)
        y = manual_normalize_audio(y)
        n_fft = 2048
        hop_length = 512

        mfcc = manual_mfcc(y, sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        rolloff = manual_spectral_rolloff(y, sr, n_fft=n_fft, hop_length=hop_length)
        rolloff_mean = np.mean(rolloff)

        zcr = manual_zcr(y, frame_length=n_fft, hop_length=hop_length)
        zcr_mean = np.mean(zcr)

        stft_matrix = manual_stft(y, frame_size=n_fft, hop_length=hop_length)
        centroid = manual_spectral_centroid(stft_matrix, sr)
        centroid_mean = np.mean(centroid)

        features = np.hstack([
            mfcc_mean,
            mfcc_std,
            centroid_mean,
            rolloff_mean,
            zcr_mean
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
        features = extract_features_manual(file_path)
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
joblib.dump(model, "gender_model_manual.pkl")
joblib.dump(scaler, "scaler_manual.pkl")

# Save class weights untuk inference
joblib.dump(class_weight_dict, "class_weights_manual.pkl")

print(f"\n✅ Model, scaler, dan class weights berhasil disimpan (manual version).")
print(f"📁 Files saved:")
print(f"  - gender_model_manual.pkl")
print(f"  - scaler_manual.pkl") 
print(f"  - class_weights_manual.pkl")