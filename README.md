# Lung Cancer Image Classification

## 1. Project Overview

This repository contains exploratory deep-learning workflows for classifying lung-image datasets using TensorFlow and Keras.

The project includes:

1. **Binary classification** for normal versus cancer images.
2. **Three-class classification** for benign, malignant, and normal images.
3. **Model testing notebooks** for single-image inference.
4. **PCA experiments** for image-feature visualisation.
5. **Autoencoder experiments** for image reconstruction and feature learning.
6. A **Gemini notebook** that combines model prediction with generated image-related explanations.

> **Important:** This is an educational deep-learning project. It is not a clinical diagnostic system and must not be used to diagnose cancer or make treatment decisions.

---

## 2. Project Objectives

The project aims to:

* Load labelled lung-image datasets from directory-based class folders.
* Train convolutional neural networks for binary and multi-class image classification.
* Apply image rescaling and augmentation.
* Evaluate predictions using classification reports and confusion matrices.
* Visualise training and validation accuracy.
* Explore image-feature variation with Principal Component Analysis.
* Test saved models on individual image files.
* Experiment with autoencoders for image reconstruction and learned feature representations.

---

## 3. Repository Contents

```text
Lung-cancer-classification/
│
├── normal vs cancer.ipynb
├── THREE CLASS classification.ipynb
├── testing normal vs cancer.ipynb
├── testing 3class classification.ipynb
├── autoencoder.ipynb
├── GEMINI.ipynb
├── README.md
└── SECURITY.md
```

### File Description

| File                                  | Purpose                                                                                               |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `normal vs cancer.ipynb`              | Binary CNN workflow for normal versus cancer classification                                           |
| `THREE CLASS classification.ipynb`    | Three-class CNN workflow and PCA exploration                                                          |
| `testing normal vs cancer.ipynb`      | Loads the saved binary model and predicts individual images                                           |
| `testing 3class classification.ipynb` | Loads the saved three-class model and predicts individual images                                      |
| `autoencoder.ipynb`                   | Autoencoder and feature-learning experiments                                                          |
| `GEMINI.ipynb`                        | Loads the three-class model, predicts image class, and experiments with Gemini-generated explanations |

---

## 4. High-Level Workflow

```text
Labelled lung-image folders
        │
        ├── Train images
        └── Validation images
        │
        ▼
Image loading and resizing
        │
        ├── RGB image conversion
        ├── Image size: 256 × 256
        └── Batch size: 32
        │
        ▼
Image rescaling and augmentation
        │
        ▼
Convolutional Neural Network training
        │
        ├── Binary classifier
        └── Three-class classifier
        │
        ▼
Validation and evaluation
        │
        ├── Accuracy curves
        ├── Classification report
        └── Confusion matrix
        │
        ▼
Saved Keras model
        │
        ├── pranay.keras
        └── 3class.keras
        │
        ▼
Single-image prediction notebooks
```

---

## 5. Dataset Format

The notebooks load images using TensorFlow’s directory-based image loader.

Expected structure:

```text
project-folder/
│
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── class_3/
│
└── valid/
    ├── class_1/
    ├── class_2/
    └── class_3/
```

The binary workflow expects two class folders, while the three-class workflow expects three class folders.

### Image-Loading Configuration

| Setting              | Value       |
| -------------------- | ----------- |
| Image size           | 256 × 256   |
| Colour mode          | RGB         |
| Batch size           | 32          |
| Label mode           | Categorical |
| Training directory   | `train`     |
| Validation directory | `valid`     |

Example code:

```python
training_set = tf.keras.utils.image_dataset_from_directory(
    "train",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256)
)
```

---

## 6. Image Preprocessing and Augmentation

The notebooks create an image-data generator for training images.

```python
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode="nearest"
)
```

### Augmentation Operations

| Transformation    | Purpose                                          |
| ----------------- | ------------------------------------------------ |
| Rescaling         | Converts pixel values into a smaller range       |
| Rotation          | Makes the model less sensitive to image rotation |
| Width shift       | Shifts images horizontally                       |
| Height shift      | Shifts images vertically                         |
| Horizontal flip   | Creates flipped image variations                 |
| Zoom              | Creates zoomed variations of images              |
| Nearest fill mode | Fills missing pixels after transformations       |

These operations increase variation in the training data and aim to improve generalisation.

---

## 7. Binary Classification: Normal vs Cancer

The binary-classification notebook trains a CNN to classify images into:

1. Normal
2. Cancer

### Model Architecture

The model uses multiple convolution and pooling blocks.

```text
Input image: 256 × 256 × 3
        │
        ▼
Convolution layers: 32 filters
        │
        ▼
Max pooling
        │
        ▼
Convolution layers: 64 filters
        │
        ▼
Max pooling
        │
        ▼
Convolution layers: 128 filters
        │
        ▼
Max pooling
        │
        ▼
Convolution layers: 256 filters
        │
        ▼
Max pooling
        │
        ▼
Convolution layers: 512 filters
        │
        ▼
Max pooling
        │
        ▼
Dropout
        │
        ▼
Dense layer: 1000 units
        │
        ▼
Output layer: 2 units with softmax activation
```

### Training Configuration

| Setting                 | Value                     |
| ----------------------- | ------------------------- |
| Optimiser               | Adam                      |
| Learning rate           | 0.0001                    |
| Loss function           | Categorical cross-entropy |
| Evaluation metric       | Accuracy                  |
| Maximum epochs          | 10                        |
| Early stopping monitor  | Validation loss           |
| Early stopping patience | 10                        |

The trained binary model is saved as:

```text
pranay.keras
```

---

## 8. Three-Class Classification

The three-class notebook trains a CNN for the following classes:

| Class ID | Class label |
| -------: | ----------- |
|        0 | BENIGN      |
|        1 | MALIGNANT   |
|        2 | NORMAL      |

### Model Architecture

The three-class CNN follows a similar convolutional architecture to the binary model, with the final output layer changed to three units.

```python
model.add(Dense(units=3, activation="softmax"))
```

### Training Configuration

| Setting                 | Value                     |
| ----------------------- | ------------------------- |
| Optimiser               | Adam                      |
| Learning rate           | 0.0001                    |
| Loss function           | Categorical cross-entropy |
| Evaluation metric       | Accuracy                  |
| Maximum epochs          | 10                        |
| Early stopping monitor  | Validation loss           |
| Early stopping patience | 10                        |

The trained three-class model is saved as:

```text
3class.keras
```

---

## 9. PCA Feature Exploration

The three-class classification notebook also performs Principal Component Analysis.

### PCA Workflow

1. Collect image data and labels.
2. Flatten image arrays.
3. Standardise the flattened image data.
4. Apply PCA with 50 components.
5. Calculate explained variance.
6. Plot cumulative explained variance.
7. Reduce the data to two principal components.
8. Visualise the first two components in a scatter plot.

### Purpose of PCA

PCA is used to:

* Reduce the dimensionality of flattened image data.
* Examine how much variation is captured by principal components.
* Visualise whether image classes show separation in a lower-dimensional space.

> PCA is used for exploration in this notebook. The CNN remains the main image-classification model.

---

## 10. Model Evaluation

The notebooks evaluate model predictions using:

* Training accuracy
* Validation accuracy
* Classification report
* Confusion matrix

### Accuracy Curves

The training workflow plots training and validation accuracy across epochs.

```python
plt.plot(epochs, training_history.history["accuracy"])
plt.plot(epochs, training_history.history["val_accuracy"])
```

### Classification Report

```python
print(
    classification_report(
        Y_true,
        Predicted_categories,
        target_names=class_name
    )
)
```

The classification report provides:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

```python
cm = confusion_matrix(Y_true, Predicted_categories)
sns.heatmap(cm, annot=True)
```

The confusion matrix compares actual classes with predicted classes.

> The repository does not include a final saved evaluation report or verified model-performance metrics. Do not claim an accuracy score unless it is reproduced from the notebook using the original dataset.

---

## 11. Testing Saved Models

Two notebooks load the saved Keras models and perform single-image prediction.

| Notebook                              | Saved model loaded |
| ------------------------------------- | ------------------ |
| `testing normal vs cancer.ipynb`      | `pranay.keras`     |
| `testing 3class classification.ipynb` | `3class.keras`     |

### Single-Image Prediction Workflow

```text
Image path
        │
        ▼
Load image
        │
        ▼
Resize to 256 × 256
        │
        ▼
Convert image to array
        │
        ▼
Add batch dimension
        │
        ▼
Run model prediction
        │
        ▼
Display predicted class
```

Example preprocessing:

```python
image = tf.keras.preprocessing.image.load_img(
    image_path,
    target_size=(256, 256)
)

image_arr = tf.keras.preprocessing.image.img_to_array(image)
image_arr = np.expand_dims(image_arr, axis=0)
```

---

## 12. Autoencoder Experiments

The `autoencoder.ipynb` notebook contains experiments with autoencoders.

It includes:

* Dense autoencoder architecture.
* Convolutional autoencoder architecture.
* Image reconstruction with mean squared error loss.
* Image loading with `ImageDataGenerator`.
* Feature-learning experiments.
* Classification-related experiments using encoded features.
* Evaluation functions using classification metrics and confusion matrices.

### Autoencoder Objective

An autoencoder learns to:

1. Encode an input image into a compressed representation.
2. Decode the compressed representation.
3. Reconstruct the original image.

This can be useful for representation learning, image reconstruction, and potential feature extraction before downstream classification.

---

## 13. Gemini Experiment

The `GEMINI.ipynb` notebook combines three-class model prediction with Gemini experimentation.

The workflow is:

1. Load the saved three-class model.
2. Preprocess an image as RGB and resize it to 256 × 256.
3. Predict the image class.
4. Map the predicted class ID to BENIGN, MALIGNANT, or NORMAL.
5. Send an explanatory prompt to a Gemini model.

> **Security note:** Do not hard-code API keys in a notebook or GitHub repository. Store them in environment variables or a local `.env` file, and revoke any key that has already been exposed.

---

## 14. Tools and Libraries

| Category                 | Tools                                                   |
| ------------------------ | ------------------------------------------------------- |
| Programming language     | Python                                                  |
| Deep learning            | TensorFlow, Keras                                       |
| Data handling            | NumPy, Pandas                                           |
| Image processing         | Pillow, OpenCV                                          |
| Visualisation            | Matplotlib, Seaborn                                     |
| Dimensionality reduction | scikit-learn PCA                                        |
| Evaluation               | scikit-learn classification report and confusion matrix |
| Generative AI experiment | Google Generative AI                                    |
| Development environment  | Google Colab / Jupyter Notebook                         |

---

## 15. Installation

### 15.1 Clone the Repository

```bash
git clone https://github.com/Pranaybannu/Lung-cancer-classification.git
cd Lung-cancer-classification
```

### 15.2 Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow opencv-python google-generativeai
```

### 15.3 Start Jupyter Notebook

```bash
jupyter notebook
```

### 15.4 Run a Training Notebook

For binary classification:

```text
normal vs cancer.ipynb
```

For three-class classification:

```text
THREE CLASS classification.ipynb
```

Before running, create the required `train` and `valid` image-directory structure and update any notebook paths if necessary.

---

## 16. Limitations

1. The image dataset is not included in this repository.
2. The trained `.keras` model files are not included in the repository.
3. The original dataset split, image-source information, and class distribution are not documented in the repository.
4. Verified final performance metrics are not available as standalone repository artifacts.
5. The notebook should not be interpreted as clinical validation.
6. Any Gemini API key should be removed from the notebook and replaced with a secure environment variable.
7. Medical-image models require rigorous external validation, bias analysis, and clinical governance before any real-world use.

---

## 17. Future Improvements

1. Add dataset source, image counts, class distribution, and licence information.
2. Add a reproducible training configuration file.
3. Save trained models and preprocessing details using versioned releases.
4. Add precision, recall, F1-score, ROC-AUC, and class-specific metrics.
5. Add Grad-CAM or similar explainability visualisations.
6. Compare transfer-learning architectures such as EfficientNet, ResNet, and MobileNet.
7. Add data-leakage checks and external validation.
8. Remove all hard-coded secrets and use environment variables.
9. Build a clearly labelled educational demo interface only after robust validation.

---

## 18. Disclaimer

This repository is for educational and experimental purposes only.

It is not approved for medical diagnosis, cancer screening, treatment planning, or clinical decision-making. Always consult qualified healthcare professionals for medical concerns.
