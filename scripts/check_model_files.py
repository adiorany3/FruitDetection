from pathlib import Path

model_dir = Path('model')
models = list(model_dir.glob('*.keras')) + list(model_dir.glob('*.h5')) + list(model_dir.glob('*.hdf5'))
labels = [p for p in [model_dir / 'class_labels.json', model_dir / 'labels.json', model_dir / 'labels.txt'] if p.exists()]

print('Model files:')
for p in models:
    print('-', p)

print('\nLabel files:')
for p in labels:
    print('-', p)

if not models:
    print('\nPERINGATAN: file model belum ditemukan.')
if not labels:
    print('\nPERINGATAN: file label belum ditemukan.')
