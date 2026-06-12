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


## Customisasi Tampilan

Aplikasi ini sudah menyembunyikan emblem/menu/footer bawaan Streamlit dan menambahkan footer custom:

```text
Developed by Galuh Adi Insani
```

Jika ingin mengganti nama developer, ubah bagian berikut di `app.py`:

```html
Developed by <strong>Galuh Adi Insani</strong>
```


## Perbaikan Kompatibilitas Streamlit

Parameter `use_container_width=True` pada `st.image()` sudah diganti menjadi `use_column_width=True` agar kompatibel dengan versi Streamlit yang digunakan pada `requirements.txt`.


## Perbaikan Prediksi Papaya

Jika aplikasi selalu memprediksi `Papaya`, penyebab utamanya adalah preprocessing gambar yang tidak sama dengan notebook training.

Notebook training disempurnakan memakai:

```python
ImageDataGenerator(rescale=1./255)
```

Maka pada `app.py`, preprocessing prediksi juga harus:

```python
img_array = np.array(image_resized).astype(np.float32)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)
```

Versi ini sudah diperbaiki agar sesuai dengan preprocessing training.


## Perbaikan Desain dan Resize Gambar

Versi ini memperbaiki tampilan aplikasi dengan:
- Layout `wide`
- Header gradient
- Card untuk upload dan hasil prediksi
- Preview gambar dengan ukuran maksimal 420 × 420 px
- Footer custom `Developed by Galuh Adi Insani`
- Penyembunyian menu/emblem bawaan Streamlit
- Tampilan top 5 prediksi menggunakan progress bar


## Keterangan agar Prediksi Lebih Baik

Versi ini menambahkan panduan pada aplikasi:
- Tips mengambil gambar buah agar prediksi lebih akurat.
- Penjelasan confidence.
- Pesan otomatis ketika confidence rendah, sedang, atau tinggi.
- Panduan membaca hasil prediksi.
