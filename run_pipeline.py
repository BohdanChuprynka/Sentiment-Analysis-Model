"""
Full pipeline: load data → dual preprocess → train all 5 models → evaluate → generate visuals.

Usage:
    python run_pipeline.py              # Run everything (all 5 models)
    python run_pipeline.py --baseline   # Run only traditional ML models (no TensorFlow needed)
"""

import argparse
import json
import os
import shutil
import string as string_mod
import sys
import time

# Force TF to use Keras 2 (required for tensorflow_hub compatibility)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Add src/ to path so we can import our modules
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from preprocessing import preprocess_text, preprocess_text_dl, split_chars
from evaluate import (
    calculate_results,
    print_classification_report,
    plot_confusion_matrix,
    plot_training_history,
)
from generate_visuals import (
    generate_model_comparison,
    generate_thumbnail,
    generate_sentiment_distribution,
    generate_wordcloud,
)

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
PREPROCESSED_CACHE = os.path.join(OUTPUTS_DIR, "preprocessed.parquet")
PREPROCESSED_DL_CACHE = os.path.join(OUTPUTS_DIR, "preprocessed_dl.parquet")

USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------
def load_data():
    print("\n" + "=" * 60)
    print("STEP 1 / 5 — Loading Sentiment140 dataset")
    print("=" * 60)

    path = kagglehub.dataset_download("kazanova/sentiment140")
    csv_path = os.path.join(path, "training.1600000.processed.noemoticon.csv")
    df = pd.read_csv(
        csv_path,
        encoding="latin-1",
        header=None,
        names=["target", "ids", "date", "flag", "user", "text"],
    )
    print(f"  Loaded {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Step 2: Dual preprocessing
# ---------------------------------------------------------------------------
def preprocess(raw_df):
    print("\n" + "=" * 60)
    print("STEP 2 / 5 — Preprocessing text (dual pipelines)")
    print("=" * 60)

    positive_count = int((raw_df["target"] == 4).sum())
    negative_count = int((raw_df["target"] == 0).sum())
    print(f"  Positive tweets: {positive_count:,}")
    print(f"  Negative tweets: {negative_count:,}")

    if os.path.exists(PREPROCESSED_CACHE) and os.path.exists(PREPROCESSED_DL_CACHE):
        print(f"  Found cached preprocessed data")
        df = pd.read_parquet(PREPROCESSED_CACHE)
        df_dl = pd.read_parquet(PREPROCESSED_DL_CACHE)
        print(f"  Loaded {len(df):,} ML rows, {len(df_dl):,} DL rows from cache")
        return df, df_dl, positive_count, negative_count

    # Heavy preprocessing for ML models
    start = time.time()
    ml_text = raw_df["text"].apply(preprocess_text)
    print(f"  ML preprocessing done in {time.time() - start:.1f}s")

    # Light preprocessing for DL models
    start = time.time()
    dl_text = raw_df["text"].apply(preprocess_text_dl)
    print(f"  DL preprocessing done in {time.time() - start:.1f}s")

    raw_df["target"] = raw_df["target"].replace(4, 1)

    df = raw_df[["target"]].copy()
    df["text"] = ml_text
    df = df.dropna()
    df.to_parquet(PREPROCESSED_CACHE, index=False)

    df_dl = raw_df[["target"]].copy()
    df_dl["text"] = dl_text
    df_dl = df_dl.dropna()
    df_dl.to_parquet(PREPROCESSED_DL_CACHE, index=False)

    print(f"  Clean dataset: {len(df):,} ML rows, {len(df_dl):,} DL rows")
    return df, df_dl, positive_count, negative_count


# ---------------------------------------------------------------------------
# Step 3: Split & encode
# ---------------------------------------------------------------------------
def split_and_encode(df, df_dl):
    print("\n" + "=" * 60)
    print("STEP 3 / 5 — Train/validation split & encoding")
    print("=" * 60)

    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(indices, train_size=0.8, shuffle=True, random_state=42)

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    train_df_dl = df_dl.iloc[train_idx]
    val_df_dl = df_dl.iloc[val_idx]

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

    data = {
        "train_sentences_ml": train_df["text"].tolist(),
        "val_sentences_ml": val_df["text"].tolist(),
        "train_sentences_dl": train_df_dl["text"].tolist(),
        "val_sentences_dl": val_df_dl["text"].tolist(),
        "train_char_sentences": [split_chars(s) for s in train_df_dl["text"].tolist()],
        "val_char_sentences": [split_chars(s) for s in val_df_dl["text"].tolist()],
    }

    label_encoder = LabelEncoder()
    data["train_labels"] = label_encoder.fit_transform(train_df["target"])
    data["val_labels"] = label_encoder.transform(val_df["target"])

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    data["train_labels_oh"] = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
    data["val_labels_oh"] = one_hot_encoder.fit_transform(val_df["target"].to_numpy().reshape(-1, 1))

    return data


# ---------------------------------------------------------------------------
# Step 4: Train & evaluate models
# ---------------------------------------------------------------------------
def _model_saved(path):
    """Check if a TF SavedModel is fully written."""
    return os.path.exists(os.path.join(path, "saved_model.pb"))


def _clean_incomplete(path):
    """Remove a checkpoint directory that exists but has no saved_model.pb."""
    if os.path.exists(path) and not _model_saved(path):
        shutil.rmtree(path)


def train_naive_bayes(data, all_results):
    print("\n  --- Model 1: TF-IDF + Naive Bayes ---")
    nb_path = os.path.join(OUTPUTS_DIR, "baseline_model.pkl")

    if os.path.exists(nb_path):
        print(f"  Loading cached model from {nb_path}")
        model = joblib.load(nb_path)
    else:
        start = time.time()
        model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=100_000)),
            ("clf", MultinomialNB(alpha=0.1)),
        ])
        model.fit(data["train_sentences_ml"], data["train_labels"])
        print(f"  Done in {time.time() - start:.1f}s")
        joblib.dump(model, nb_path)
        print(f"  Saved model -> {nb_path}")

    preds = model.predict(data["val_sentences_ml"])
    all_results["Naive Bayes"] = calculate_results(data["val_labels"], preds)
    print_classification_report(data["val_labels"], preds, "Naive Bayes")
    plot_confusion_matrix(data["val_labels"], preds, "Naive Bayes",
                          save_path=os.path.join(IMAGES_DIR, "cm_naive_bayes.png"))


def train_logistic_regression(data, all_results):
    print("\n  --- Model 2: TF-IDF (bigram) + Logistic Regression ---")
    lr_path = os.path.join(OUTPUTS_DIR, "lr_model.pkl")

    if os.path.exists(lr_path):
        print(f"  Loading cached model from {lr_path}")
        model = joblib.load(lr_path)
    else:
        start = time.time()
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=300_000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=5,
            )),
            ("clf", SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-5,
                max_iter=50,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        model.fit(data["train_sentences_ml"], data["train_labels"])
        print(f"  Done in {time.time() - start:.1f}s")
        joblib.dump(model, lr_path)
        print(f"  Saved model -> {lr_path}")

    preds = model.predict(data["val_sentences_ml"])
    all_results["Logistic Regression"] = calculate_results(data["val_labels"], preds)
    print_classification_report(data["val_labels"], preds, "Logistic Regression")
    plot_confusion_matrix(data["val_labels"], preds, "Logistic Regression",
                          save_path=os.path.join(IMAGES_DIR, "cm_logistic_regression.png"))


def train_use_frozen(data, all_results):
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow.keras.layers as layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    print("\n  --- Model 3: USE (Frozen) ---")
    use_frozen_path = os.path.join(CHECKPOINTS_DIR, "use_frozen")
    _clean_incomplete(use_frozen_path)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((np.array(data["train_sentences_dl"]), data["train_labels_oh"]))
        .shuffle(10_000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((np.array(data["val_sentences_dl"]), data["val_labels_oh"]))
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )

    if _model_saved(use_frozen_path):
        print(f"  Loading cached model from {use_frozen_path}")
        model = tf.keras.models.load_model(use_frozen_path)
        history = None
    else:
        encoder = hub.KerasLayer(USE_URL, input_shape=[], dtype=tf.string, trainable=False)
        inputs = tf.keras.Input(shape=[], dtype=tf.string)
        x = encoder(inputs)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(2, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=["accuracy"],
        )
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1),
            ModelCheckpoint(use_frozen_path, monitor="val_loss", save_best_only=True, verbose=1),
        ]
        history = model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=callbacks)

    preds = tf.argmax(model.predict(val_ds), axis=1).numpy()
    all_results["USE (Frozen)"] = calculate_results(data["val_labels"], preds)
    print_classification_report(data["val_labels"], preds, "USE (Frozen)")
    if history:
        plot_training_history(history, "USE (Frozen)", save_path=os.path.join(IMAGES_DIR, "history_use_frozen.png"))
    plot_confusion_matrix(data["val_labels"], preds, "USE (Frozen)",
                          save_path=os.path.join(IMAGES_DIR, "cm_use_frozen.png"))
    return train_ds, val_ds


def train_use_finetuned(data, all_results, train_ds, val_ds):
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow.keras.layers as layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    print("\n  --- Model 4: USE (Fine-tuned) ---")
    use_ft_path = os.path.join(CHECKPOINTS_DIR, "use_finetuned")
    use_ft_phase1_path = os.path.join(CHECKPOINTS_DIR, "use_ft_phase1")
    _clean_incomplete(use_ft_path)
    _clean_incomplete(use_ft_phase1_path)

    if _model_saved(use_ft_path):
        print(f"  Loading cached model from {use_ft_path}")
        model = tf.keras.models.load_model(use_ft_path)
        history = None
    else:
        if _model_saved(use_ft_phase1_path):
            print(f"  Resuming from phase 1 checkpoint")
            model = tf.keras.models.load_model(use_ft_phase1_path)
            for layer in model.layers:
                if isinstance(layer, hub.KerasLayer):
                    layer._trainable = True
                    break
        else:
            ft_encoder = hub.KerasLayer(USE_URL, input_shape=[], dtype=tf.string, trainable=False)
            inputs = tf.keras.Input(shape=[], dtype=tf.string)
            x = ft_encoder(inputs)
            x = layers.Dense(256, activation="relu")(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(2, activation="softmax")(x)
            model = tf.keras.Model(inputs, outputs)

            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                metrics=["accuracy"],
            )
            print("  Phase 1: Training head (encoder frozen, 2 epochs)...")
            model.fit(train_ds, epochs=2, validation_data=val_ds)
            model.save(use_ft_phase1_path)
            ft_encoder._trainable = True

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
            metrics=["accuracy"],
        )
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7, verbose=1),
            ModelCheckpoint(use_ft_path, monitor="val_loss", save_best_only=True, verbose=1),
        ]
        print("  Phase 2: Fine-tuning end-to-end (up to 3 epochs)...")
        history = model.fit(train_ds, epochs=3, validation_data=val_ds, callbacks=callbacks)

    preds = tf.argmax(model.predict(val_ds), axis=1).numpy()
    all_results["USE (Fine-tuned)"] = calculate_results(data["val_labels"], preds)
    print_classification_report(data["val_labels"], preds, "USE (Fine-tuned)")
    if history:
        plot_training_history(history, "USE (Fine-tuned) — Phase 2",
                              save_path=os.path.join(IMAGES_DIR, "history_use_finetuned.png"))
    plot_confusion_matrix(data["val_labels"], preds, "USE (Fine-tuned)",
                          save_path=os.path.join(IMAGES_DIR, "cm_use_finetuned.png"))


def train_hybrid(data, all_results):
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow.keras.layers as layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    print("\n  --- Model 5: Hybrid (Token + Char) ---")
    hybrid_path = os.path.join(CHECKPOINTS_DIR, "hybrid")
    _clean_incomplete(hybrid_path)

    hybrid_train_ds = (
        tf.data.Dataset.from_tensor_slices(
            ((data["train_sentences_dl"], data["train_char_sentences"]), data["train_labels_oh"])
        ).shuffle(10_000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )
    hybrid_val_ds = (
        tf.data.Dataset.from_tensor_slices(
            ((data["val_sentences_dl"], data["val_char_sentences"]), data["val_labels_oh"])
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )

    if _model_saved(hybrid_path):
        print(f"  Loading cached model from {hybrid_path}")
        model = tf.keras.models.load_model(hybrid_path)
        history = None
    else:
        encoder = hub.KerasLayer(USE_URL, input_shape=[], dtype=tf.string, trainable=False)

        alphabet = string_mod.ascii_lowercase + string_mod.punctuation + string_mod.digits
        char_lengths = [len(s) for s in data["train_sentences_dl"][:50_000]]
        output_seq_len = int(np.percentile(char_lengths, 95))

        char_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=len(alphabet) + 2,
            standardize="lower_and_strip_punctuation",
            split="character",
            output_mode="int",
            output_sequence_length=output_seq_len,
        )
        char_vectorizer.adapt(data["train_char_sentences"][:50_000])
        char_vocab = char_vectorizer.get_vocabulary()

        token_inputs = tf.keras.Input(shape=[], dtype=tf.string, name="token_input")
        token_x = encoder(token_inputs)
        token_x = layers.Dense(128, activation="relu")(token_x)
        token_model = tf.keras.Model(token_inputs, token_x)

        char_inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="char_input")
        char_x = char_vectorizer(char_inputs)
        char_x = layers.Embedding(input_dim=len(char_vocab), output_dim=64, mask_zero=False)(char_x)
        char_x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(char_x)
        char_x = layers.Bidirectional(layers.LSTM(32))(char_x)
        char_model = tf.keras.Model(char_inputs, char_x)

        combined = layers.Concatenate()([token_model.output, char_model.output])
        x = layers.Dense(256, activation="relu")(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        final_output = layers.Dense(2, activation="softmax")(x)

        model = tf.keras.Model(inputs=[token_model.input, char_model.input], outputs=final_output)

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            metrics=["accuracy"],
        )
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1),
            ModelCheckpoint(hybrid_path, monitor="val_loss", save_best_only=True, verbose=1),
        ]
        history = model.fit(hybrid_train_ds, epochs=5, validation_data=hybrid_val_ds, callbacks=callbacks)

    preds = tf.argmax(model.predict(hybrid_val_ds), axis=1).numpy()
    all_results["Hybrid (Token+Char)"] = calculate_results(data["val_labels"], preds)
    print_classification_report(data["val_labels"], preds, "Hybrid (Token+Char)")
    if history:
        plot_training_history(history, "Hybrid (Token+Char)",
                              save_path=os.path.join(IMAGES_DIR, "history_hybrid.png"))
    plot_confusion_matrix(data["val_labels"], preds, "Hybrid (Token+Char)",
                          save_path=os.path.join(IMAGES_DIR, "cm_hybrid.png"))


def train_all_models(data, baseline_only=False):
    print("\n" + "=" * 60)
    print("STEP 4 / 5 — Training & evaluating models")
    print("=" * 60)

    all_results = {}

    train_naive_bayes(data, all_results)
    train_logistic_regression(data, all_results)

    if not baseline_only:
        train_ds, val_ds = train_use_frozen(data, all_results)
        train_use_finetuned(data, all_results, train_ds, val_ds)
        train_hybrid(data, all_results)

    return all_results


# ---------------------------------------------------------------------------
# Step 5: Generate visuals
# ---------------------------------------------------------------------------
def generate_all_visuals(all_results, positive_count, negative_count, text_series):
    print("\n" + "=" * 60)
    print("STEP 5 / 5 — Generating portfolio visuals")
    print("=" * 60)

    generate_model_comparison(all_results)
    generate_thumbnail(all_results)
    generate_sentiment_distribution(positive_count, negative_count)
    generate_wordcloud(text_series)

    print(f"\n  All images saved to {IMAGES_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Pipeline")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Train only traditional ML models (skip TensorFlow models)",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    pipeline_start = time.time()

    n_models = 2 if args.baseline else 5
    mode = f"baseline only ({n_models} models)" if args.baseline else f"full ({n_models} models)"
    print(f"\n{'#' * 60}")
    print(f"  Sentiment Analysis Pipeline — {mode}")
    print(f"{'#' * 60}")

    raw_df = load_data()
    df, df_dl, positive_count, negative_count = preprocess(raw_df)
    data = split_and_encode(df, df_dl)
    all_results = train_all_models(data, baseline_only=args.baseline)

    results_path = os.path.join(OUTPUTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved -> {results_path}")

    generate_all_visuals(all_results, positive_count, negative_count, df["text"])

    elapsed = time.time() - pipeline_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'#' * 60}")
    print(f"  Pipeline complete in {minutes}m {seconds}s")
    print(f"{'#' * 60}")
    print(f"\n  Results:      {results_path}")
    print(f"  Images:       {IMAGES_DIR}/")
    print(f"  Checkpoints:  {CHECKPOINTS_DIR}/")
    print()


if __name__ == "__main__":
    main()
