import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Fruit Detection CNN",
    page_icon="🍎",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown(
    """
    <style>
        /* Hide Streamlit default branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden !important;}
        [data-testid="stDecoration"] {display: none !important;}
        [data-testid="stStatusWidget"] {visibility: hidden !important;}
        [data-testid="stHeader"] {display: none !important;}
        .viewerBadge_container__1QSob {display: none !important;}

        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
            max-width: 1180px;
        }

        .main {
            background: linear-gradient(135deg, #fff7ed 0%, #ffffff 45%, #f0fdf4 100%);
        }

        .hero-card {
            background: linear-gradient(135deg, #f97316 0%, #fb923c 45%, #22c55e 100%);
            border-radius: 28px;
            padding: 34px 36px;
            color: white;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.16);
            margin-bottom: 24px;
        }

        .hero-title {
            font-size: 42px;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 10px;
        }

        .hero-subtitle {
            font-size: 17px;
            opacity: 0.96;
            max-width: 760px;
            line-height: 1.6;
        }

        .soft-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #fed7aa;
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
            margin-bottom: 18px;
        }

        .result-card {
            background: linear-gradient(180deg, #ffffff 0%, #fff7ed 100%);
            border: 1px solid #fdba74;
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
            margin-bottom: 18px;
        }

        .prediction-label {
            font-size: 34px;
            font-weight: 800;
            color: #ea580c;
            margin-bottom: 4px;
        }

        .prediction-caption {
            color: #64748b;
            font-size: 15px;
            margin-bottom: 16px;
        }

        .confidence-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 16px;
            text-align: center;
        }

        .confidence-number {
            font-size: 32px;
            font-weight: 800;
            color: #16a34a;
            margin-bottom: 2px;
        }

        .confidence-text {
            color: #64748b;
            font-size: 14px;
        }

        .image-frame {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 24px;
            padding: 16px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            text-align: center;
        }

        .section-title {
            font-size: 22px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 8px;
        }

        .small-muted {
            font-size: 14px;
            color: #64748b;
            line-height: 1.6;
        }


        .tips-card {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 20px;
            padding: 18px 20px;
            margin-top: 16px;
            margin-bottom: 18px;
        }

        .tips-title {
            font-size: 18px;
            font-weight: 800;
            color: #166534;
            margin-bottom: 8px;
        }

        .tips-list {
            color: #334155;
            font-size: 14px;
            line-height: 1.75;
            margin-bottom: 0;
        }

        .quality-badge-good {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
            padding: 10px 14px;
            border-radius: 14px;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .quality-badge-medium {
            background: #fef9c3;
            color: #854d0e;
            border: 1px solid #fde68a;
            padding: 10px 14px;
            border-radius: 14px;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .quality-badge-low {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
            padding: 10px 14px;
            border-radius: 14px;
            font-weight: 700;
            margin-bottom: 12px;
        }




        /* Visible upload empty message */
        .visible-upload-message {
            background: #fff7ed !important;
            border: 2px solid #fb923c !important;
            border-radius: 18px !important;
            padding: 16px 18px !important;
            margin-top: 14px !important;
            color: #111827 !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            line-height: 1.6 !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08) !important;
        }

        .visible-upload-message span {
            color: #9a3412 !important;
            font-weight: 800 !important;
        }

        .visible-upload-caption {
            color: #334155 !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            margin-top: 8px !important;
            line-height: 1.6 !important;
        }

        div[data-testid="stAlert"] {
            background: #fff7ed !important;
            color: #111827 !important;
            border: 1px solid #fb923c !important;
        }

        div[data-testid="stAlert"] * {
            color: #111827 !important;
        }

        /* Upload component visibility fix */
        .upload-wrapper {
            background: #ffffff !important;
            border: 2px solid #ea580c !important;
            border-radius: 22px !important;
            padding: 20px !important;
            margin-top: 14px !important;
            margin-bottom: 18px !important;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08) !important;
        }

        .upload-title-visible {
            font-size: 18px !important;
            font-weight: 800 !important;
            color: #9a3412 !important;
            margin-bottom: 6px !important;
        }

        .upload-help-visible {
            font-size: 14px !important;
            color: #334155 !important;
            line-height: 1.6 !important;
            margin-bottom: 14px !important;
        }

        div[data-testid="stFileUploader"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            background: #fff7ed !important;
            color: #0f172a !important;
            border: 2px dashed #ea580c !important;
            border-radius: 18px !important;
            padding: 18px !important;
            min-height: 120px !important;
        }

        div[data-testid="stFileUploader"] * {
            visibility: visible !important;
            opacity: 1 !important;
            color: #0f172a !important;
        }

        div[data-testid="stFileUploader"] label {
            display: block !important;
            visibility: visible !important;
            color: #0f172a !important;
            font-weight: 700 !important;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: #ffffff !important;
            border: 1px solid #fdba74 !important;
            border-radius: 14px !important;
            color: #0f172a !important;
        }

        div[data-testid="stFileUploaderDropzone"] * {
            color: #0f172a !important;
        }

        div[data-testid="stFileUploader"] button {
            background: #ea580c !important;
            color: #ffffff !important;
            border: 1px solid #ea580c !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }

        div[data-testid="stFileUploader"] button * {
            color: #ffffff !important;
        }

        /* Readability and contrast improvement */
        html, body, [class*="css"] {
            color: #0f172a !important;
        }

        .stApp {
            background: linear-gradient(135deg, #fff7ed 0%, #ffffff 50%, #ecfdf5 100%) !important;
            color: #0f172a !important;
        }

        .main {
            background: transparent !important;
            color: #0f172a !important;
        }

        p, li, span, label, div {
            color: inherit;
        }

        .soft-card,
        .result-card,
        .image-frame {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.10) !important;
        }

        .hero-card {
            background: linear-gradient(135deg, #9a3412 0%, #ea580c 48%, #166534 100%) !important;
            color: #ffffff !important;
        }

        .hero-card,
        .hero-card div {
            color: #ffffff !important;
        }

        .hero-subtitle {
            color: #fff7ed !important;
        }

        .section-title {
            color: #111827 !important;
        }

        .small-muted,
        .prediction-caption {
            color: #475569 !important;
        }

        .prediction-label {
            color: #c2410c !important;
        }

        .confidence-box {
            background: #f8fafc !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
        }

        .confidence-number {
            color: #15803d !important;
        }

        .confidence-text {
            color: #475569 !important;
        }

        .tips-card {
            background: #ecfdf5 !important;
            color: #0f172a !important;
            border: 1px solid #86efac !important;
        }

        .tips-title {
            color: #14532d !important;
        }

        .tips-list,
        .tips-list li {
            color: #1f2937 !important;
        }

        .quality-badge-good {
            background: #dcfce7 !important;
            color: #14532d !important;
            border: 1px solid #22c55e !important;
        }

        .quality-badge-medium {
            background: #fef3c7 !important;
            color: #78350f !important;
            border: 1px solid #f59e0b !important;
        }

        .quality-badge-low {
            background: #fee2e2 !important;
            color: #7f1d1d !important;
            border: 1px solid #ef4444 !important;
        }

        div[data-testid="stFileUploader"] {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 2px dashed #ea580c !important;
        }

        div[data-testid="stFileUploader"] * {
            color: #0f172a !important;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff !important;
            color: #0f172a !important;
            border-right: 1px solid #e2e8f0 !important;
        }

        section[data-testid="stSidebar"] * {
            color: #0f172a !important;
        }

        .custom-footer {
            background-color: #ffffff !important;
            color: #334155 !important;
            border-top: 1px solid #cbd5e1 !important;
        }

        .custom-footer strong {
            color: #c2410c !important;
        }

        .stAlert {
            color: #0f172a !important;
        }

        .custom-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #ffffff;
            color: #475569;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            border-top: 1px solid #e2e8f0;
            z-index: 9999;
        }

        .custom-footer strong {
            color: #f97316;
        }

        div[data-testid="stFileUploader"] {
            background: #fff7ed;
            border: 1.5px dashed #fb923c;
            padding: 18px;
            border-radius: 20px;
        }

        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #f97316, #22c55e);
        }
    </style>

    <div class="custom-footer">
        Developed by <strong>Galuh Adi Insani</strong>
    </div>
    """,
    unsafe_allow_html=True
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

    return {int(k): v for k, v in labels.items()}


def get_model_image_size(model):
    try:
        input_shape = model.input_shape
        height = int(input_shape[1])
        width = int(input_shape[2])
        if height > 0 and width > 0:
            return height, width
    except Exception:
        pass

    return 224, 224


def make_display_image(image, max_size=(420, 420)):
    """
    Resize gambar hanya untuk tampilan.
    Gambar prediksi tetap diproses sesuai input model.
    """
    display_image = image.convert("RGB").copy()
    display_image.thumbnail(max_size)
    return display_image


def predict_image(model, image, labels, top_k=5):
    height, width = get_model_image_size(model)

    image = image.convert("RGB")
    image_resized = image.resize((width, height))

    img_array = np.array(image_resized).astype(np.float32)

    # Disamakan dengan notebook training:
    # ImageDataGenerator(rescale=1./255)
    img_array = img_array / 255.0
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


def get_confidence_message(confidence):
    if confidence >= 80:
        return "good", "Prediksi sangat baik. Gambar cukup sesuai dengan pola data training."
    elif confidence >= 50:
        return "medium", "Prediksi cukup, tetapi masih bisa ditingkatkan dengan gambar yang lebih jelas."
    else:
        return "low", "Prediksi kurang yakin. Coba gunakan gambar yang lebih terang, jelas, dan background lebih polos."

# =========================
# UI
# =========================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🍎 Fruit Detection Using CNN</div>
        <div class="hero-subtitle">
            Upload gambar buah, lalu sistem akan mengenali jenis buah menggunakan model CNN
            hasil training Kaggle. Tampilan gambar dibuat lebih rapi dengan ukuran preview yang konsisten.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

try:
    model, loaded_model_path = load_cnn_model()
    labels = load_labels()

    with st.sidebar:
        st.markdown("### 🍊 Informasi Model")
        st.write("**Model:** CNN Conv2D + MaxPooling2D")
        st.write("**Input:** 224 × 224 RGB")
        st.write("**Preprocessing:** resize + rescale 1/255")
        st.write("**Jumlah kelas:** 131")
        st.caption(f"Model aktif: {os.path.basename(loaded_model_path)}")
        st.caption("Tampilan menggunakan warna kontras agar teks mudah dibaca.")
        st.divider()
        st.caption(
            "Catatan: model ini fokus mengenali jenis buah/sayur. "
            "Tingkat kematangan hanya dapat dikenali jika label tersebut tersedia pada dataset."
        )

    left_col, right_col = st.columns([1.05, 1], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="soft-card">
                <div class="section-title">📤 Upload Gambar</div>
                <div class="small-muted">
                    Gunakan gambar yang jelas, pencahayaan cukup, dan objek buah berada di tengah gambar.
                    Format yang didukung: JPG, JPEG, PNG, dan WEBP.
                </div>
            </div>

            <div class="tips-card">
                <div class="tips-title">✅ Tips agar prediksi lebih akurat</div>
                <ul class="tips-list">
                    <li>Gunakan <b>satu buah utama</b> dalam satu gambar.</li>
                    <li>Pastikan buah berada di <b>tengah gambar</b> dan tidak terpotong.</li>
                    <li>Gunakan <b>pencahayaan terang</b>, tetapi hindari bayangan terlalu gelap.</li>
                    <li>Gunakan <b>background polos</b> atau tidak terlalu ramai.</li>
                    <li>Hindari gambar buram, terlalu jauh, terlalu dekat, atau tertutup objek lain.</li>
                    <li>Gunakan gambar buah yang termasuk dalam daftar kelas model.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="upload-wrapper">
                <div class="upload-title-visible">Pilih atau drag gambar buah ke sini</div>
                <div class="upload-help-visible">
                    Klik tombol browse/upload di bawah ini, lalu pilih file gambar buah dari perangkat kamu.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload gambar buah",
            type=["jpg", "jpeg", "png", "webp"],
            help="Format yang didukung: JPG, JPEG, PNG, dan WEBP."
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            display_image = make_display_image(image, max_size=(420, 420))

            st.markdown('<div class="image-frame">', unsafe_allow_html=True)
            st.image(display_image, caption="Preview gambar", width=420)
            st.markdown('</div>', unsafe_allow_html=True)

            st.caption(f"Ukuran asli gambar: {image.size[0]} × {image.size[1]} px")
        else:
            st.markdown(
                """
                <div class="visible-upload-message">
                    📌 <span>Silakan upload gambar buah terlebih dahulu.</span>
                    <div class="visible-upload-caption">
                        Pilih file gambar melalui tombol upload di atas agar sistem dapat melakukan prediksi.
                        Jika tombol upload belum terlihat setelah deploy, lakukan refresh halaman atau klik
                        <b>Manage App → Clear cache → Reboot app</b>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            image = None

    with right_col:
        st.markdown(
            """
            <div class="soft-card">
                <div class="section-title">🔍 Hasil Analisis</div>
                <div class="small-muted">
                    Hasil prediksi utama dan lima kemungkinan kelas tertinggi akan muncul di bagian ini.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if uploaded_file is not None and image is not None:
            with st.spinner("Sedang melakukan prediksi..."):
                results = predict_image(model, image, labels, top_k=5)

            best = results[0]
            quality_level, quality_message = get_confidence_message(best["confidence"])
            badge_class = f"quality-badge-{quality_level}"

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="prediction-caption">Prediksi utama</div>
                    <div class="prediction-label">{best['label']}</div>
                    <div class="{badge_class}">{quality_message}</div>
                    <div class="confidence-box">
                        <div class="confidence-number">{best['confidence']:.2f}%</div>
                        <div class="confidence-text">Confidence</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if best["confidence"] < 50:
                st.warning(
                    "Confidence masih rendah. Coba foto ulang dengan pencahayaan lebih terang, "
                    "buah berada di tengah, background polos, dan gambar tidak buram."
                )
            elif best["confidence"] < 80:
                st.info(
                    "Prediksi sudah cukup, tetapi hasil bisa lebih baik jika gambar lebih dekat, "
                    "buah tidak tertutup objek lain, dan background lebih bersih."
                )

            st.markdown("### Top 5 Prediksi")
            for item in results:
                st.write(f"**{item['label']}**")
                st.progress(min(item["confidence"] / 100, 1.0))
                st.caption(f"{item['confidence']:.2f}%")

        else:
            st.markdown(
                """
                <div class="result-card">
                    <div class="prediction-caption">Belum ada gambar</div>
                    <div class="prediction-label">Upload gambar dulu</div>
                    <div class="small-muted">
                        Setelah gambar diupload, hasil prediksi akan tampil otomatis di sini.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with st.expander("Panduan membaca hasil prediksi"):
        st.write(
            """
            **Confidence** menunjukkan tingkat keyakinan model terhadap hasil prediksi.
            Nilai confidence yang tinggi biasanya menunjukkan gambar lebih sesuai dengan pola data training.

            Agar hasil lebih baik, gunakan gambar yang mirip dengan data training, yaitu buah terlihat jelas,
            satu objek utama, pencahayaan cukup, dan background tidak terlalu ramai.
            Jika confidence rendah, jangan langsung dianggap salah total. Coba upload gambar lain dengan sudut,
            jarak, atau pencahayaan yang lebih baik.
            """
        )

    with st.expander("Lihat daftar 131 kelas yang dapat dikenali"):
        ordered_labels = [labels[i] for i in sorted(labels.keys())]
        st.write(", ".join(ordered_labels))

except Exception as e:
    st.error("Aplikasi gagal memuat model atau label.")
    st.exception(e)
    st.info(
        "Pastikan struktur folder benar: app.py, requirements.txt, runtime.txt, "
        "dan folder model/ yang berisi file model serta class_labels.json."
    )
