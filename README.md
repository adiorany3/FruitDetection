# Fruit Recognition CNN - Streamlit Online

Project ini adalah aplikasi Streamlit untuk menjalankan model CNN pengenalan buah hasil training dari Kaggle.

## 1. Download output model dari Kaggle

Jalankan perintah berikut di terminal lokal:

```bash
kaggle kernels output adioranye/fruits-recognition-using-cnn-by-galuh-adi-insani -p model
```

Perintah tersebut akan mengunduh output kernel Kaggle ke folder `model/`.

## 2. File yang wajib ada

Agar aplikasi bisa melakukan prediksi, folder `model/` harus berisi minimal:

```text
model/
├── fruit_cnn_model.keras   # atau file .h5 / .hdf5
└── class_labels.json       # label kelas
```

Jika output Kaggle hanya berisi `test_submission.csv`, maka model belum bisa dipakai untuk Streamlit. Notebook training harus menyimpan model terlebih dahulu.

Tambahkan cell ini di akhir notebook training Kaggle:

```python
import json

# Simpan model
model.save('/kaggle/working/fruit_cnn_model.keras')

# Simpan label kelas dari generator training
label_map = {v: k for k, v in x_train.class_indices.items()}
with open('/kaggle/working/class_labels.json', 'w') as f:
    json.dump(label_map, f)
```

Setelah itu jalankan ulang notebook, lalu download ulang output Kaggle.

## 3. Jalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Deploy ke Streamlit Community Cloud

1. Upload semua file project ini ke GitHub.
2. Pastikan file model dan `class_labels.json` ikut masuk ke repository atau disediakan melalui mekanisme hosting lain.
3. Buka Streamlit Community Cloud.
4. Pilih repository GitHub.
5. Set main file path menjadi:

```text
app.py
```

6. Klik Deploy.

## 5. Catatan penting

Aplikasi ini mengikuti preprocessing dari notebook training, yaitu:

```python
tf.keras.applications.mobilenet_v2.preprocess_input
```

dengan ukuran gambar:

```text
224 x 224 pixel
```

Untuk mendeteksi tingkat kematangan buah, dataset dan model harus dilatih ulang dengan label seperti:

```text
mentah
setengah_matang
matang
busuk
```

atau label gabungan seperti:

```text
pisang_mentah
pisang_matang
apel_mentah
apel_matang
```
