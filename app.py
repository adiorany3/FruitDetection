import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ============================================================
# Konfigurasi dasar
# ============================================================
APP_TITLE = "Fruit Recognition CNN"
IMAGE_SIZE = (224, 224)  # mengikuti notebook training: img_size=(224, 224)
MODEL_DIR = Path("model")

MODEL_PATTERNS = [
    "*.keras",
    "*.h5",
    "*.hdf5",
]
LABEL_CANDIDATES = [
    MODEL_DIR / "class_labels.json",
    MODEL_DIR / "labels.json",
    MODEL_DIR / "label_map.json",
    Path("class_labels.json"),
    Path("labels.json"),
    Path("label_map.json"),
]
TXT_LABEL_CANDIDATES = [
    MODEL_DIR / "labels.txt",
    Path("labels.txt"),
]


# ============================================================
# Helper loading model dan label
# ============================================================
def find_model_file() -> Path | None:
    """Cari file model Keras di folder model/ atau root project."""
    candidates: List[Path] = []
    for base in [MODEL_DIR, Path(".")]:
        for pattern in MODEL_PATTERNS:
            candidates.extend(base.glob(pattern))

    # Prioritaskan format .keras, lalu .h5/.hdf5
    candidates = sorted(
        set(candidates),
        key=lambda p: (0 if p.suffix == ".keras" else 1, str(p).lower()),
    )
    return candidates[0] if candidates else None


@st.cache_resource(show_spinner="Memuat model...")
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def normalize_label_mapping(data) -> Dict[int, str]:
    """Terima berbagai format label dan ubah menjadi {index:int -> label:str}."""
    if isinstance(data, list):
        return {idx: str(label) for idx, label in enumerate(data)}

    if isinstance(data, dict):
        # Format yang disarankan: {"0": "Apple", "1": "Banana"}
        if all(str(k).isdigit() for k in data.keys()):
            return {int(k): str(v) for k, v in data.items()}

        # Format alternatif dari generator: {"Apple": 0, "Banana": 1}
        if all(str(v).isdigit() for v in data.values()):
            return {int(v): str(k) for k, v in data.items()}

    raise ValueError(
        "Format label tidak dikenali. Gunakan class_labels.json dengan format "
        '{"0": "nama_kelas", "1": "nama_kelas"} atau {"nama_kelas": 0}.'
    )


def load_labels() -> Dict[int, str] | None:
    for label_path in LABEL_CANDIDATES:
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return normalize_label_mapping(data)

    for label_path in TXT_LABEL_CANDIDATES:
        if label_path.exists():
            labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return {idx: label for idx, label in enumerate(labels)}

    return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocessing mengikuti notebook: MobileNetV2 preprocess_input + resize 224x224."""
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(image).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr


def predict_image(model, labels: Dict[int, str], image: Image.Image, top_k: int = 5) -> List[Tuple[str, float]]:
    arr = preprocess_image(image)
    preds = model.predict(arr, verbose=0)[0]

    top_indices = np.argsort(preds)[::-1][:top_k]
    results: List[Tuple[str, float]] = []
    for idx in top_indices:
        label = labels.get(int(idx), f"Kelas {idx}")
        confidence = float(preds[idx]) * 100
        results.append((label, confidence))
    return results


# ============================================================
# Tampilan Streamlit
# ============================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🍎",
    layout="centered",
)

st.title("🍎 Fruit Recognition Using CNN")
st.write(
    "Aplikasi ini menggunakan model CNN hasil training dari Kaggle untuk mengenali jenis buah "
    "berdasarkan gambar yang diunggah."
)

with st.expander("Panduan file model", expanded=False):
    st.markdown(
        """
        Letakkan file hasil output Kaggle ke folder `model/`:

        ```text
        model/
        ├── fruit_cnn_model.keras   # atau .h5/.hdf5
        └── class_labels.json       # daftar label kelas
        ```

        Contoh format `class_labels.json`:

        ```json
        {
          "0": "Apple",
          "1": "Banana",
          "2": "Orange"
        }
        ```

        Jika output Kaggle hanya berisi `test_submission.csv`, model belum bisa dipakai untuk prediksi online.
        Notebook training harus menyimpan model dengan `model.save(...)` terlebih dahulu.
        """
    )

model_path = find_model_file()
labels = load_labels()

if model_path is None:
    st.error("File model belum ditemukan. Letakkan file `.keras`, `.h5`, atau `.hdf5` di folder `model/`.")
    st.stop()

if labels is None:
    st.error("File label belum ditemukan. Letakkan `class_labels.json` atau `labels.txt` di folder `model/`.")
    st.stop()

try:
    model = load_model(str(model_path))
except Exception as e:
    st.error("Model gagal dimuat. Pastikan file model tidak rusak dan versi TensorFlow sesuai.")
    st.exception(e)
    st.stop()

st.success(f"Model berhasil dimuat: `{model_path}`")
st.caption(f"Jumlah label: {len(labels)} kelas")

uploaded_file = st.file_uploader(
    "Upload gambar buah",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    with st.spinner("Melakukan prediksi..."):
        results = predict_image(model, labels, image, top_k=min(5, len(labels)))

    best_label, best_confidence = results[0]
    st.subheader("Hasil Prediksi")
    st.success(f"{best_label} — {best_confidence:.2f}%")

    st.write("Top prediksi:")
    chart_data = {
        "Label": [label for label, _ in results],
        "Confidence (%)": [round(conf, 2) for _, conf in results],
    }
    st.dataframe(chart_data, use_container_width=True, hide_index=True)
    st.bar_chart(chart_data, x="Label", y="Confidence (%)")

st.info(
    "Catatan: aplikasi ini mengenali jenis buah. Untuk mendeteksi tingkat kematangan, "
    "model harus dilatih ulang memakai dataset berlabel kematangan seperti mentah, setengah matang, matang, dan busuk."
)
