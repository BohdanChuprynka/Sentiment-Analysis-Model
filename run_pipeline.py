"""
Full pipeline: load data → preprocess → train all 3 models → evaluate → generate visuals.

Usage:
    python run_pipeline.py              # Run everything
    python run_pipeline.py --baseline   # Run only the Naive Bayes baseline (no TensorFlow needed)
"""

import argparse
import json
import os
import sys
import time

# Force TF to use Keras 2 (required for tensorflow_hub compatibility)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Add src/ to path so we can import our modules
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from preprocessing import preprocess_text, split_chars
from evaluate import calculate_results
from generate_visuals import (
    generate_model_comparison,
    generate_thumbnail,
    generate_sentiment_distribution,
    generate_wordcloud,
)

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
PREPROCESSED_CACHE = os.path.join(OUTPUTS_DIR, "preprocessed.parquet")

USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"


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
# Step 2: Preprocess
# ---------------------------------------------------------------------------
def preprocess(df):
    print("\n" + "=" * 60)
    print("STEP 2 / 5 — Preprocessing text")
    print("=" * 60)

    positive_count = int((df["target"] == 4).sum())
    negative_count = int((df["target"] == 0).sum())
    print(f"  Positive tweets: {positive_count:,}")
    print(f"  Negative tweets: {negative_count:,}")

    # Check for cached preprocessed data
    if os.path.exists(PREPROCESSED_CACHE):
        print(f"  Found cached preprocessed data at {PREPROCESSED_CACHE}")
        print("  Skipping preprocessing — loading from cache...")
        df = pd.read_parquet(PREPROCESSED_CACHE)
        print(f"  Loaded {len(df):,} rows from cache")
        return df, positive_count, negative_count

    start = time.time()
    df["text"] = df["text"].apply(preprocess_text)
    elapsed = time.time() - start
    print(f"  Preprocessing done in {elapsed:.1f}s")

    df["target"] = df["target"].replace(4, 1)
    df = df[["target", "text"]].dropna()
    print(f"  Clean dataset: {len(df):,} rows")

    # Cache for future runs
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df.to_parquet(PREPROCESSED_CACHE, index=False)
    print(f"  Cached preprocessed data → {PREPROCESSED_CACHE}")

    return df, positive_count, negative_count


# ---------------------------------------------------------------------------
# Step 3: Split & encode
# ---------------------------------------------------------------------------
def split_and_encode(df):
    print("\n" + "=" * 60)
    print("STEP 3 / 5 — Train/validation split & encoding")
    print("=" * 60)

    train_df, val_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=42)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

    train_sentences = train_df["text"].tolist()
    val_sentences = val_df["text"].tolist()
    train_char_sentences = [split_chars(s) for s in train_sentences]
    val_char_sentences = [split_chars(s) for s in val_sentences]

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df["target"])
    val_labels = label_encoder.transform(val_df["target"])

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    train_labels_oh = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
    val_labels_oh = one_hot_encoder.fit_transform(val_df["target"].to_numpy().reshape(-1, 1))

    return {
        "train_sentences": train_sentences,
        "val_sentences": val_sentences,
        "train_char_sentences": train_char_sentences,
        "val_char_sentences": val_char_sentences,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "train_labels_oh": train_labels_oh,
        "val_labels_oh": val_labels_oh,
    }


# ---------------------------------------------------------------------------
# Step 4: Train & evaluate models
# ---------------------------------------------------------------------------
def train_baseline(data):
    print("\n  --- Model 1: TF-IDF + Naive Bayes (Baseline) ---")
    model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])
    model.fit(data["train_sentences"], data["train_labels"])
    preds = model.predict(data["val_sentences"])
    results = calculate_results(data["val_labels"], preds)

    model_path = os.path.join(OUTPUTS_DIR, "baseline_model.pkl")
    joblib.dump(model, model_path)
    print(f"  Saved model → {model_path}")

    return results


def train_use_model(data):
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow.keras.layers as layers

    print("\n  --- Model 2: Universal Sentence Encoder ---")
    sentence_encoder = hub.KerasLayer(
        USE_URL,
        input_shape=[],
        dtype=tf.string,
    )

    inputs = tf.keras.Input(shape=[], dtype=tf.string)
    x = sentence_encoder(inputs)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (np.array(data["train_sentences"]), data["train_labels_oh"])
        )
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            (np.array(data["val_sentences"]), data["val_labels_oh"])
        )
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        epochs=5,
        validation_data=val_ds,
        validation_steps=int(0.1 * len(val_ds)),
    )

    preds_probs = model.predict(val_ds)
    preds = tf.argmax(preds_probs, axis=1).numpy()
    results = calculate_results(data["val_labels"], preds)
    return results


def train_hybrid_model(data):
    import string as string_mod
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow.keras.layers as layers

    print("\n  --- Model 3: Hybrid Token + Character Embeddings ---")

    sentence_encoder = hub.KerasLayer(
        USE_URL,
        input_shape=[],
        dtype=tf.string,
    )

    # Character vectorization
    alphabet = string_mod.ascii_lowercase + string_mod.punctuation + string_mod.digits
    char_lengths = [len(s) for s in data["train_sentences"]]
    output_seq_len = int(np.percentile(char_lengths, 97))

    char_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=len(alphabet) + 2,
        standardize="lower_and_strip_punctuation",
        split="character",
        output_mode="int",
        output_sequence_length=output_seq_len,
    )
    char_vectorizer.adapt(data["train_char_sentences"])
    char_vocab = char_vectorizer.get_vocabulary()

    # Token branch
    token_inputs = tf.keras.Input(shape=[], dtype=tf.string, name="token_input")
    token_x = sentence_encoder(token_inputs)
    token_x = layers.Dense(128, activation="relu")(token_x)
    token_model = tf.keras.Model(token_inputs, token_x)

    # Char branch
    char_inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="char_input")
    char_x = char_vectorizer(char_inputs)
    char_x = layers.Embedding(input_dim=len(char_vocab), output_dim=56, mask_zero=False)(char_x)
    char_x = layers.Bidirectional(layers.LSTM(32))(char_x)
    char_model = tf.keras.Model(char_inputs, char_x)

    # Combine
    combined = layers.Concatenate()([token_model.output, char_model.output])
    x = layers.Dense(128, activation="relu")(combined)
    x = layers.Dropout(0.3)(x)
    final_output = layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=[token_model.input, char_model.input], outputs=final_output)

    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            ((data["train_sentences"], data["train_char_sentences"]), data["train_labels_oh"])
        )
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            ((data["val_sentences"], data["val_char_sentences"]), data["val_labels_oh"])
        )
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        epochs=5,
        validation_data=val_ds,
        validation_steps=int(0.1 * len(val_ds)),
    )

    preds_probs = model.predict(val_ds)
    preds = tf.argmax(preds_probs, axis=1).numpy()
    results = calculate_results(data["val_labels"], preds)
    return results


def train_all_models(data, baseline_only=False):
    print("\n" + "=" * 60)
    print("STEP 4 / 5 — Training & evaluating models")
    print("=" * 60)

    all_results = {}

    all_results["Naive Bayes (Baseline)"] = train_baseline(data)
    print_results("Naive Bayes (Baseline)", all_results["Naive Bayes (Baseline)"])

    if not baseline_only:
        all_results["USE Embeddings"] = train_use_model(data)
        print_results("USE Embeddings", all_results["USE Embeddings"])

        all_results["Hybrid (Token + Char)"] = train_hybrid_model(data)
        print_results("Hybrid (Token + Char)", all_results["Hybrid (Token + Char)"])

    return all_results


def print_results(name, results):
    print(f"\n  {name}:")
    for metric, value in results.items():
        print(f"    {metric:>10}: {value}")


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
        help="Train only the Naive Bayes baseline (skip TensorFlow models)",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    pipeline_start = time.time()

    mode = "baseline only" if args.baseline else "full (all 3 models)"
    print(f"\n{'#' * 60}")
    print(f"  Sentiment Analysis Pipeline — {mode}")
    print(f"{'#' * 60}")

    # Step 1
    raw_df = load_data()

    # Step 2
    df, positive_count, negative_count = preprocess(raw_df)

    # Step 3
    data = split_and_encode(df)

    # Step 4
    all_results = train_all_models(data, baseline_only=args.baseline)

    # Save results to JSON
    results_path = os.path.join(OUTPUTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {results_path}")

    # Step 5
    generate_all_visuals(all_results, positive_count, negative_count, df["text"])

    elapsed = time.time() - pipeline_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'#' * 60}")
    print(f"  Pipeline complete in {minutes}m {seconds}s")
    print(f"{'#' * 60}")
    print(f"\n  Results:  {results_path}")
    print(f"  Images:   {IMAGES_DIR}/")
    print(f"  Model:    {OUTPUTS_DIR}/baseline_model.pkl")
    print()


if __name__ == "__main__":
    main()
