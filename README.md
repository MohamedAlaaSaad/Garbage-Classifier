# Garbage Classifier

Image classifier that sorts photos of waste into 12 classes using transfer learning (Xception, Keras / TensorFlow).

## Credit and license

This project is **based on** the Kaggle notebook
[Garbage Classification using Keras and Transfer Learning](https://www.kaggle.com/code/mostafaabla/garbage-classification-keras-transfer-learning)
by **Mostafa Mohamed** (Kaggle: `mostafaabla`), which is published under the **Apache License 2.0**.
The original notebook was adapted and then updated here. All credit for the original approach and structure goes to the original author.
This repository keeps the Apache 2.0 terms for the derived work; see the changes listed below.

## Problem

Given a photo of a single waste item, predict its material/type so it can be sorted for recycling.

## Dataset

12 classes, 15,515 images: `battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass`.
This is the public "Garbage Classification (12 classes)" dataset on Kaggle. The dataset and the Xception weights are not stored in this repo; see the file `Data and Xception` for the Drive link the author used.
Split used: 80% train / 10% validation / 10% test (12,412 / 1,551 / 1,552 images), `random_state=42`.

## Model

- Xception pretrained on ImageNet (frozen), input 320x320x3
- Global average pooling, then a Dense(12, softmax) layer
- Adam, categorical cross-entropy, batch size 64, early stopping on validation accuracy (patience 2)

## Results (measured, original notebook)

Run on the author's machine with the original settings (no augmentation, frozen backbone):

- Test accuracy: **94.91%** (1,552 test images)
- Macro-average F1: 0.92, weighted F1: 0.95
- Weakest classes: plastic (F1 0.84) and white-glass (F1 0.84)

See `Accuracy.png` and the output cells of `garbage-classification-keras-transfer-learning.ipynb`.

**The updated notebook (`garbage-classification-keras-transfer-learning-v2.ipynb`) has not been run yet, so there are no new accuracy numbers.** Do not assume it beats 94.91% until you run it and record the result here.

## What was changed compared to the original

The original notebook is left untouched. The v2 notebook changes:

1. Cleaner structure: all settings (paths, image size, batch size, epochs, seed) in one config cell.
2. Fixed seeds for reproducible splits.
3. Removed unused imports and the hard-coded GPU memory limit.
4. Replaced deprecated `fit_generator`, `evaluate_generator`, `predict_generator` with `fit`, `evaluate`, `predict`.
5. Data augmentation turned on (flip, small rotation, zoom, shift). In the original it was commented out.
6. Fixed a print bug (it printed the validation size as the train size) and typos in headings.
7. Added a confusion matrix and per-class report on the test set.
8. Added an optional fine-tuning stage (unfreeze the top of Xception with a low learning rate).
9. Saves the full model, not only the weights.
10. Added `requirements.txt` and this README.

Status: items 1 to 10 are **untested** until the notebook is run end to end.

## How to run

1. `pip install -r requirements.txt`
2. Download the dataset and put the class folders under `garbage_classification/`.
3. Open the v2 notebook in Jupyter or Kaggle and run all cells. A GPU is strongly recommended (one epoch took about 25 minutes on the author's machine).
4. Copy the new test accuracy and report into the Results section.

## License

Derived from an Apache 2.0 notebook. Keep the attribution above when reusing or redistributing.
