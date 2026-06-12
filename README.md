# Fruit Detection Using CNN - Streamlit Online

Project ini berisi aplikasi Streamlit untuk mengenali jenis buah menggunakan model CNN hasil training Kaggle.

## Struktur Folder

```text
fruitdetection/
├── app.py
├── requirements.txt
├── runtime.txt
├── .streamlit/
│   └── config.toml
└── model/
    ├── fruit_cnn_best_model.h5
    ├── fruit_cnn_model.h5
    └── class_labels.json
```

## Cara Deploy ke Streamlit Community Cloud

1. Upload semua file ke repository GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih repository.
4. Main file path: `app.py`.
5. Deploy.

Jika sebelumnya pernah error dependency, buka **Manage App**, lalu pilih **Clear cache** dan **Reboot app**.

## Catatan Penting

File model dari Kaggle awalnya bernama `.keras`, tetapi format internalnya adalah HDF5. 
Agar lebih kompatibel dengan `tf.keras.models.load_model`, file model disimpan sebagai `.h5`.

Model ini mengenali 131 kelas berdasarkan `class_labels.json`.
Model belum dirancang khusus untuk tingkat kematangan semua buah, kecuali kelas yang memang sudah mengandung label seperti `Avocado ripe` atau `Tomato not Ripened`.
