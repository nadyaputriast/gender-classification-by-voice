import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import soundfile as sf
import os
import tempfile

# Styling - HARUS PERTAMA!
st.set_page_config(page_title="Gender Voice Classifier", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    h1 { color: #f63366; }
    .stButton>button { background-color: #f63366; color: white; }
    </style>
""", unsafe_allow_html=True)

# Load model dan scaler
@st.cache_resource
def load_models():
    try:
        model = joblib.load("gender_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

model, scaler, error = load_models()

def extract_features(file_path):
    """Extract features sesuai dengan training - HARUS SAMA!"""
    try:
        x, sr = librosa.load(file_path, sr=22050, duration=3.0)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        mfcc_std = np.std(mfccs.T, axis=0)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(x)
        
        # Combine all features - SAMA seperti di training
        features = np.hstack([
            mfcc_mean,
            mfcc_std,
            np.mean(spectral_centroids),
            np.mean(spectral_rolloff),
            np.mean(zero_crossing_rate)
        ])
        
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

st.title("🎤 Gender Voice Classifier")
st.caption("Upload file audio (.wav) dan deteksi apakah suara laki-laki atau perempuan.")

# Check if models loaded successfully
if error:
    st.error(f"❌ Error loading model: {error}")
    st.stop()
else:
    st.success("✅ Model dan scaler berhasil dimuat")

# Upload file
uploaded_file = st.file_uploader("Upload file .wav", type=["wav", "mp3"])

if uploaded_file is not None:
    # Simpan ke temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        # Load audio
        x, sr = librosa.load(tmp_path, sr=22050, duration=3.0)  # Sama dengan training
        
        # Audio info
        st.subheader("🎵 Audio Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{len(x)/sr:.2f}s")
        with col2:
            st.metric("Sample Rate", f"{sr} Hz")
        with col3:
            st.metric("Samples", len(x))

        # Tampilkan waveform
        st.subheader("📈 Waveform")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(x, sr=sr, ax=ax1)
        ax1.set(title="Waveform")
        st.pyplot(fig1)

        # Spectrogram
        st.subheader("🌈 Spectrogram")
        X = librosa.stft(x)
        X_db = librosa.amplitude_to_db(np.abs(X))
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        img2 = librosa.display.specshow(X_db, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
        fig2.colorbar(img2, ax=ax2, format="%+2.0f dB")
        ax2.set(title="Spectrogram (dB)")
        st.pyplot(fig2)

        # MFCC
        st.subheader("🎹 MFCC (Mel Frequency Cepstral Coefficients)")
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        img3 = librosa.display.specshow(mfcc, x_axis='time', ax=ax3)
        fig3.colorbar(img3, ax=ax3)
        ax3.set(title="MFCC")
        st.pyplot(fig3)

        # Extract features untuk prediksi
        st.subheader("🤖 Prediksi Gender")
        
        with st.spinner("Menganalisis audio..."):
            features = extract_features(tmp_path)
            
            if features is not None:
                # Reshape dan scale features
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # Prediksi dengan probabilitas
                prediction = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
                
                # Hasil
                label = "👩 Female" if prediction == 0 else "👨 Male"
                confidence = max(probabilities) * 100
                
                # Display hasil
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Hasil Prediksi: {label}**")
                with col2:
                    st.info(f"**Confidence: {confidence:.1f}%**")
                
                # Probability bar
                st.subheader("📊 Probabilitas")
                prob_female = probabilities[0] * 100
                prob_male = probabilities[1] * 100
                
                st.write("👩 Female:")
                st.progress(prob_female / 100)
                st.write(f"{prob_female:.1f}%")
                
                st.write("👨 Male:")
                st.progress(prob_male / 100)
                st.write(f"{prob_male:.1f}%")
                
                # Feature analysis
                with st.expander("🔍 Feature Analysis"):
                    st.write("**Extracted Features:**")
                    st.write(f"- MFCC Mean: {features[:13].mean():.3f}")
                    st.write(f"- MFCC Std: {features[13:26].mean():.3f}")
                    st.write(f"- Spectral Centroid: {features[26]:.3f}")
                    st.write(f"- Spectral Rolloff: {features[27]:.3f}")
                    st.write(f"- Zero Crossing Rate: {features[28]:.3f}")
            else:
                st.error("❌ Gagal mengekstrak fitur audio")

    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
    
    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)