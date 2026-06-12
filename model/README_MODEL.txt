Letakkan output Kaggle di folder ini.

Contoh perintah:

kaggle kernels output adioranye/fruits-recognition-using-cnn-by-galuh-adi-insani -p model

File minimal yang dibutuhkan:
1. Model: .keras / .h5 / .hdf5
2. Label: class_labels.json atau labels.txt

Jika Kaggle output hanya berisi test_submission.csv, tambahkan cell berikut ke notebook training:

import json
model.save('/kaggle/working/fruit_cnn_model.keras')
label_map = {v: k for k, v in x_train.class_indices.items()}
with open('/kaggle/working/class_labels.json', 'w') as f:
    json.dump(label_map, f)
