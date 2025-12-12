# gem_identifier_model

This project trains a gemstone classifier via transfer learning in the notebook `gemora.ipynb`. The current workflow downloads a Kaggle dataset, builds an EfficientNetB0 model in TensorFlow, and fine-tunes it on training images.

---

## Notebook Walkthrough (`gemora.ipynb`)
- Installs dependencies: `opendatasets`, `tensorflow`, `matplotlib`, `seaborn`, `scikit-learn`.
- Downloads the Kaggle dataset `gemstones-images` with `opendatasets` (expects Kaggle API credentials).
- Uses dataset paths under `/content/gemstones-images` (Colab-style paths). Adjust `base_path`, `train_dir`, and `test_dir` if you run locally.
- Builds data loaders with `tf.keras.preprocessing.image_dataset_from_directory`, using an 80/20 train/validation split from `train_dir` and a separate `test` directory for test data.
- Defines a transfer learning model:
	- Base: `EfficientNetB0` pre-trained on ImageNet, initially frozen.
	- Head: global average pooling → dropout → dense softmax sized to `num_classes`.
- Training flow:
	- Stage 1: train head for 10 epochs (`optimizer='adam'`, `sparse_categorical_crossentropy`).
	- Stage 2: unfreeze base, fine-tune for 8 epochs with a low LR (`Adam(1e-5)`).
- The notebook currently does not save the trained model or run explicit test/evaluation steps; add those if needed.

---

## Requirements
- Python 3.9+ (tested with TensorFlow 2.x-compatible setups)
- Packages: `tensorflow`, `matplotlib`, `seaborn`, `scikit-learn`, `opendatasets` (plus their transitive deps)

Install locally:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow matplotlib seaborn scikit-learn opendatasets
```
---

## Dataset
- Source: Kaggle dataset [lsind18/gemstones-images](https://www.kaggle.com/datasets/lsind18/gemstones-images)
- `opendatasets` downloads after Kaggle auth. Set env vars before running the download cell:
	- `KAGGLE_USERNAME=your_username`
	- `KAGGLE_KEY=your_api_key`
- Default paths in the notebook assume the files land at `/content/gemstones-images/{train,test}`. If you run outside Colab, change those paths to match your local download location.

---

## Running the Notebook
1) Open `gemora.ipynb` in Jupyter, VS Code, or Colab.
2) Ensure Kaggle credentials are set so `od.download(...)` can fetch the dataset.
3) Update `base_path`, `train_dir`, and `test_dir` if not using `/content` paths.
4) Run cells in order to install deps, download data, build datasets, train, and fine-tune.
5) (Optional) Add evaluation on `test_ds` and save the trained model (`model.save(...)`).

---

## Quick Repo Setup
```bash
git clone https://github.com/binojmadhuranga/gemora-gem-identification-ml.git
cd gemora-gem-identification-ml
```

