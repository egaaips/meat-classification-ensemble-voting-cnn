import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from collections import Counter
from PIL import Image
from tensorflow import keras

# Load label kelas
@st.cache_data
def load_labels():
    labels = np.load("labels.npy")
    return labels.tolist()

# Load all models
@st.cache_resource
def load_models():
    resnet = load_model("models/ResNet50V2.keras")
    inception = load_model("models/InceptionV3.keras")
    mobilenet = load_model("models/MobileNetV2_1.keras")
    return resnet, inception, mobilenet

class_names = load_labels()
resnet_model, inception_model, mobilenet_model = load_models()

# Fungsi preprocessing sesuai standar kamu
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    shape = np.array(img).shape

    if len(shape) == 2:
        img = img.convert(mode='RGB')
    elif shape[-1] == 4:
        img = Image.fromarray(np.array(img)[:, :, :3])

    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Pakai preprocessing dari ResNet karena kamu gunakan ensemble (umum pakai satu jenis preprocessing)
    return keras.applications.resnet_v2.preprocess_input(img_array)

# Fungsi prediksi
def predict(model, img_array):
    pred = model.predict(img_array)
    return np.argmax(pred)

# Streamlit App UI
st.title("🥩 Meat Image Classifier with CNN Ensemble (Hard Voting)")
st.write("Upload gambar daging untuk klasifikasi: Sapi, Kambing, atau Babi.")

uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Gambar yang diupload", use_container_width=True)

    img_array = preprocess_image(uploaded_file)

    st.write("⏳ Memproses...")

    pred1 = predict(resnet_model, img_array)
    pred2 = predict(inception_model, img_array)
    pred3 = predict(mobilenet_model, img_array)

    st.write("📊 Hasil Prediksi Model:")
    st.write(f"- ResNet50V2: **{class_names[pred1]}**")
    st.write(f"- InceptionV3: **{class_names[pred2]}**")
    st.write(f"- MobileNetV2: **{class_names[pred3]}**")

    final_vote = Counter([pred1, pred2, pred3]).most_common(1)[0][0]
    st.success(f"✅ Prediksi Akhir (Hard Voting): **{class_names[final_vote]}**")
