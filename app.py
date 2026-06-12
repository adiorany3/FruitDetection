import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="Fruit Detection CNN",
    page_icon="🍎",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_CANDIDATES = [
    os.path.join(MODEL_DIR, "fruit_cnn_best_model.h5"),
    os.path.join(MODEL_DIR, "fruit_cnn_model.h5"),
    os.path.join(MODEL_DIR, "fruit_cnn_best_model.keras"),
    os.path.join(MODEL_DIR, "fruit_cnn_model.keras"),
]

LABEL_PATH = os.path.join(MODEL_DIR, "class_labels.json")


@st.cache_resource
def load_cnn_model():
    last_error = None

    for model_path in MODEL_CANDIDATES:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                return model, model_path
            except Exception as e:
                last_error = e

    if last_error is not None:
        raise RuntimeError(f"Model ditemukan, tetapi gagal dimuat: {last_error}")

    raise FileNotFoundError(
        "File model tidak ditemukan. Pastikan file model berada di folder model/."
    )


@st.cache_data
def load_labels():
    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(
            "File class_labels.json tidak ditemukan di folder model/."
        )

    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # class_labels.json menggunakan key string: {"0": "Apple", "1": "..."}
    return {int(k): v for k, v in labels.items()}


def get_model_image_size(model):
    """
    Mengambil ukuran input dari model.
    Jika tidak terbaca, gunakan default 224x224 sesuai notebook training.
    """
    try:
        input_shape = model.input_shape
        height = int(input_shape[1])
        width = int(input_shape[2])
        if height > 0 and width > 0:
            return height, width
    except Exception:
        pass

    return 224, 224


def predict_image(model, image, labels, top_k=5):
    height, width = get_model_image_size(model)

    image = image.convert("RGB")
    image_resized = image.resize((width, height))

    img_array = np.array(image_resized).astype(np.float32)

    # Disamakan dengan preprocessing pada notebook training:
    # ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    prediction = np.array(prediction)

    if prediction.ndim == 2:
        prediction = prediction[0]

    top_indices = prediction.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        label = labels.get(int(idx), f"Kelas {idx}")
        confidence = float(prediction[idx]) * 100
        results.append({
            "index": int(idx),
            "label": label,
            "confidence": confidence
        })

    return results


st.title("🍎 Fruit Detection Using CNN")
st.write(
    "Aplikasi ini menggunakan model CNN hasil training Kaggle untuk mengenali jenis buah "
    "berdasarkan gambar yang diunggah."
)

with st.sidebar:
    st.header("Informasi Model")
    st.write("Model: CNN Conv2D + MaxPooling2D")
    st.write("Input gambar: 224 × 224 RGB")
    st.write("Output: 131 kelas")
    st.caption(
        "Catatan: model ini fokus mengenali jenis buah/sayur. "
        "Penentuan tingkat kematangan hanya bisa dilakukan jika kelas pada dataset memang memiliki label kematangan."
    )

try:
    model, loaded_model_path = load_cnn_model()
    labels = load_labels()

    st.success("Model berhasil dimuat.")
    st.caption(f"File model aktif: `{os.path.basename(loaded_model_path)}`")

    uploaded_file = st.file_uploader(
        "Upload gambar buah",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

        with st.spinner("Sedang melakukan prediksi..."):
            results = predict_image(model, image, labels, top_k=5)

        best = results[0]

        st.subheader("Hasil Prediksi")
        st.success(f"Prediksi utama: {best['label']}")
        st.metric("Confidence", f"{best['confidence']:.2f}%")

        if best["confidence"] < 50:
            st.warning(
                "Confidence masih rendah. Coba gunakan gambar yang lebih jelas, "
                "pencahayaan cukup, dan objek buah berada di tengah gambar."
            )

        st.subheader("Top 5 Prediksi")
        for item in results:
            st.write(f"**{item['label']}** — {item['confidence']:.2f}%")
            st.progress(min(item["confidence"] / 100, 1.0))

    else:
        st.info("Silakan upload gambar buah terlebih dahulu.")

    with st.expander("Daftar kelas yang dapat dikenali model"):
        ordered_labels = [labels[i] for i in sorted(labels.keys())]
        st.write(", ".join(ordered_labels))

except Exception as e:
    st.error("Aplikasi gagal memuat model atau label.")
    st.exception(e)
    st.info(
        "Pastikan struktur folder benar: app.py, requirements.txt, runtime.txt, "
        "dan folder model/ yang berisi file model serta class_labels.json."
    )
