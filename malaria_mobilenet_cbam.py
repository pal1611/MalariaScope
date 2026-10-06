"""
Malaria Risk Detection — Hybrid MobileNetV2 + CBAM
----------------------------------------------------
Implements the patch-level classifier recommended for the app (VectorSight /
MalariaScope), adapted to a dataset of paired monthly NDVI / NDWI Sentinel-2
tiles (the format: "<idx>_<start>_<end>_Sentinel-2_L2A_<NDVI|NDWI|Scene_classification>").

Design choices, and why:
  - Reports REAL metrics only. No artificial capping (max_metric), no label
    noise injection. If accuracy is 0.99, it is reported as 0.99.
  - Filters out near-black / near-blank tiles (cloud cover or no data), since
    the provided dataset has some fully black months.
  - Uses NDVI + NDWI together as a 2-index risk signal (not NDWI alone),
    stacked into a 3-channel pseudo-RGB input so a pretrained MobileNetV2
    (ImageNet weights) can be used directly.
  - CBAM (channel + spatial attention) is implemented manually and inserted
    on top of the MobileNetV2 feature map, matching the architecture
    described in the base research paper.
  - Two-phase training: Phase 1 trains only the classifier head (base
    frozen), Phase 2 fine-tunes the top 40 layers of MobileNetV2 at a lower
    learning rate.

Run with:  python malaria_mobilenet_cbam.py
(Recommended: Google Colab with a GPU runtime)
"""

import os
import re
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
)
import matplotlib.pyplot as plt

# ====================== CONFIG ======================
DATASET_PATH = r"./dataset"          # folder containing the monthly TIFF/PNG tiles
PATCH_SIZE = 64
BATCH_SIZE = 32
PHASE1_EPOCHS = 10                   # matches paper: frozen base, lr=1e-3
PHASE2_EPOCHS = 10                   # matches paper: fine-tune top-40 layers, lr=1e-5
BLACK_TILE_THRESHOLD = 2.0           # mean pixel value below this = treat as no-data/cloud
NDWI_RISK_THRESHOLD = 0.0            # paper's water-presence threshold
NDVI_RISK_RANGE = (0.3, 0.7)         # paper's vegetation-density risk band
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ====================== STEP 1: FILE DISCOVERY ======================
def discover_paired_tiles(base_path):
    """
    Matches files by their date range so each date has one NDVI tile and one
    NDWI tile. Scene_classification tiles are read too, used only to help
    detect no-data areas if present; they are not required.

    Expected filename pattern (case-insensitive), e.g.:
      0_2025-04-30_23_59_Sentinel-2_L2A_NDVI.tif
      0_2025-04-30_23_59_Sentinel-2_L2A_NDWI.tif
    """
    all_files = glob.glob(os.path.join(base_path, "*"))
    pattern = re.compile(r"(.+?Sentinel-2_L2A)_(NDVI|NDWI|Scene_classification)", re.IGNORECASE)

    groups = {}
    for f in all_files:
        name = os.path.basename(f)
        match = pattern.search(name)
        if not match:
            continue
        key, band = match.group(1), match.group(2).upper()
        groups.setdefault(key, {})[band] = f

    pairs = []
    for key, bands in groups.items():
        if "NDVI" in bands and "NDWI" in bands:
            pairs.append((bands["NDVI"], bands["NDWI"]))

    return pairs


# ====================== STEP 2: LOADING + CLOUD/BLANK FILTERING ======================
def load_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img.astype('float32'), None, 0, 255, cv2.NORM_MINMAX)
    return img


def is_blank_tile(img):
    """Flags fully black / near-blank tiles (cloud cover or missing coverage)."""
    return np.mean(img) < BLACK_TILE_THRESHOLD or np.std(img) < 1.0


def load_dataset(base_path):
    pairs = discover_paired_tiles(base_path)
    ndvi_imgs, ndwi_imgs = [], []

    for ndvi_path, ndwi_path in pairs:
        ndvi = load_grayscale(ndvi_path)
        ndwi = load_grayscale(ndwi_path)
        if ndvi is None or ndwi is None:
            continue
        if is_blank_tile(ndvi) or is_blank_tile(ndwi):
            continue  # skip cloud-covered / no-data months
        ndvi_imgs.append(ndvi)
        ndwi_imgs.append(ndwi)

    print(f"Loaded {len(ndvi_imgs)} usable NDVI/NDWI tile pairs "
          f"(blank/cloudy months filtered out).")
    return ndvi_imgs, ndwi_imgs


# ====================== STEP 3: PATCH GENERATION + RISK LABELING ======================
def assign_risk_label(ndvi_patch_norm, ndwi_patch_norm):
    """
    Threshold-based proxy labeling, matching the base paper's Cohort 2 rule:
    high risk if water is present (NDWI >= 0) OR vegetation density falls in
    the mosquito-resting-habitat band (NDVI in [0.3, 0.7]).
    NOTE: these are environmental proxy labels, not clinical ground truth.
    """
    mean_ndvi = np.mean(ndvi_patch_norm)   # already rescaled to ~[-1, 1] range below
    mean_ndwi = np.mean(ndwi_patch_norm)
    water_risk = mean_ndwi >= NDWI_RISK_THRESHOLD
    veg_risk = NDVI_RISK_RANGE[0] <= mean_ndvi <= NDVI_RISK_RANGE[1]
    return 1 if (water_risk or veg_risk) else 0


def create_patches(ndvi_imgs, ndwi_imgs, patch_size=PATCH_SIZE):
    X, y = [], []
    for ndvi_img, ndwi_img in zip(ndvi_imgs, ndwi_imgs):
        h, w = ndvi_img.shape[:2]
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                ndvi_patch = ndvi_img[i:i + patch_size, j:j + patch_size]
                ndwi_patch = ndwi_img[i:i + patch_size, j:j + patch_size]
                if ndvi_patch.shape != (patch_size, patch_size):
                    continue

                # normalize each patch to roughly [-1, 1] before label check
                ndvi_norm = (ndvi_patch / 127.5) - 1.0
                ndwi_norm = (ndwi_patch / 127.5) - 1.0
                label = assign_risk_label(ndvi_norm, ndwi_norm)

                # 3-channel pseudo-RGB: NDVI, NDWI, and their average
                combined = ((ndvi_patch.astype('float32') + ndwi_patch.astype('float32')) / 2.0)
                stacked = np.stack([ndvi_patch, ndwi_patch, combined], axis=-1)

                X.append(stacked)
                y.append(label)

    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    return X, y


# ====================== STEP 4: CBAM ATTENTION MODULE ======================
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_dense_one = layers.Dense(channel // ratio, activation='relu',
                                     kernel_initializer='he_normal', use_bias=True)
    shared_dense_two = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True)

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    feature = layers.Add()([avg_pool, max_pool])
    feature = layers.Activation('sigmoid')(feature)

    return layers.Multiply()([input_feature, feature])


def spatial_attention(input_feature, kernel_size=7):
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = layers.Conv2D(1, kernel_size=kernel_size, strides=1, padding='same',
                               activation='sigmoid', kernel_initializer='he_normal',
                               use_bias=False)(concat)
    return layers.Multiply()([input_feature, attention])


def cbam_block(input_feature, ratio=8, kernel_size=7):
    x = channel_attention(input_feature, ratio)
    x = spatial_attention(x, kernel_size)
    return x


# ====================== STEP 5: BUILD HYBRID MODEL ======================
def build_hybrid_mobilenet_cbam(input_shape):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False  # Phase 1: fully frozen

    x = base_model.output
    x = cbam_block(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model, base_model


# ====================== STEP 6: TRAIN (TWO PHASES) ======================
def train_model(model, base_model, X_train, y_train, X_test, y_test):
    train_datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.15, height_shift_range=0.15,
        zoom_range=0.15, horizontal_flip=True, vertical_flip=True, fill_mode='nearest',
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint('best_hybrid_cbam.h5', monitor='val_loss', save_best_only=True, verbose=1),
    ]

    print("\n=== PHASE 1: Training classifier head (MobileNetV2 frozen) ===")
    history1 = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
        validation_data=(X_test, y_test),
        epochs=PHASE1_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n=== PHASE 2: Fine-tuning top 40 layers of MobileNetV2 ===")
    for layer in base_model.layers[-40:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    total_epochs = len(history1.history['loss']) + PHASE2_EPOCHS
    history2 = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
        validation_data=(X_test, y_test),
        epochs=total_epochs,
        initial_epoch=len(history1.history['loss']),
        callbacks=callbacks,
        verbose=1,
    )

    combined_history = {k: history1.history[k] + history2.history[k] for k in history1.history}
    return model, combined_history


# ====================== STEP 7: HONEST EVALUATION ======================
def evaluate_model(model, X_test, y_test):
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # NOTE: these are the model's real, uncapped, unmodified results.
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No Malaria", "Malaria"])

    print("\n" + "=" * 60)
    print("HYBRID MOBILENETV2+CBAM — EVALUATION REPORT (real results)")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("=" * 60)

    return acc, prec, rec, f1, cm


# ====================== STEP 8: RISK HEATMAP GENERATION ======================
def generate_heatmap(model, ndvi_img, ndwi_img, patch_size=PATCH_SIZE):
    h, w = ndvi_img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.int32)
    stride = patch_size // 2

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            ndvi_patch = ndvi_img[i:i + patch_size, j:j + patch_size]
            ndwi_patch = ndwi_img[i:i + patch_size, j:j + patch_size]
            if ndvi_patch.shape != (patch_size, patch_size):
                continue

            combined = (ndvi_patch.astype('float32') + ndwi_patch.astype('float32')) / 2.0
            stacked = np.stack([ndvi_patch, ndwi_patch, combined], axis=-1)
            patch_input = np.expand_dims(stacked.astype('float32') / 255.0, axis=0)

            prob = model.predict(patch_input, verbose=0)[0][0]
            heatmap[i:i + patch_size, j:j + patch_size] += prob
            counts[i:i + patch_size, j:j + patch_size] += 1

    heatmap = np.divide(heatmap, counts, where=counts > 0)
    return heatmap


def save_heatmap_figure(ndvi_img, heatmap, out_path="results/heatmap.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(ndvi_img, cmap='gray')
    axes[0].set_title("Original NDVI Tile")
    axes[0].axis('off')

    im = axes[1].imshow(heatmap, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1].set_title("Malaria Risk Heatmap")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(ndvi_img, cmap='gray', alpha=0.6)
    axes[2].imshow(heatmap, cmap='RdYlGn_r', alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {out_path}")


# ====================== MAIN ======================
if __name__ == "__main__":
    ndvi_imgs, ndwi_imgs = load_dataset(DATASET_PATH)
    if len(ndvi_imgs) == 0:
        raise ValueError("No usable NDVI/NDWI pairs found. Check DATASET_PATH and filenames.")

    X, y = create_patches(ndvi_imgs, ndwi_imgs)
    print(f"Total patches: {len(X)} | Risk={np.sum(y == 1)} | No-risk={np.sum(y == 0)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model, base_model = build_hybrid_mobilenet_cbam(X_train.shape[1:])
    print(f"Total params: {model.count_params():,}")

    model, history = train_model(model, base_model, X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)

    model.save("results/hybrid_mobilenet_cbam_malaria.h5")
    print("Model saved to results/hybrid_mobilenet_cbam_malaria.h5")

    # Generate a sample heatmap on the first available tile pair
    sample_heatmap = generate_heatmap(model, ndvi_imgs[0], ndwi_imgs[0])
    save_heatmap_figure(ndvi_imgs[0], sample_heatmap)
