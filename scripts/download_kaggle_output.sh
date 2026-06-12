#!/usr/bin/env bash
set -e
mkdir -p model
kaggle kernels output adioranye/fruits-recognition-using-cnn-by-galuh-adi-insani -p model
printf '\nOutput Kaggle berhasil diunduh ke folder model/.\n'
