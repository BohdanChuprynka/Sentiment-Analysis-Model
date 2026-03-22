# Dataset

This project uses the **Sentiment140** dataset, which contains 1.6 million tweets labeled for binary sentiment (positive/negative).

## Source

- **Kaggle:** [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Original paper:** Go, A., Bhayani, R., & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision.*

## Download

The dataset is downloaded automatically via `kagglehub` when running the training script:

```bash
python src/train_baseline.py
```

To download manually:

```python
import kagglehub
path = kagglehub.dataset_download("kazanova/sentiment140")
```

## Schema

| Column   | Description                              |
|----------|------------------------------------------|
| `target` | Sentiment label (0 = negative, 4 = positive) |
| `ids`    | Tweet ID                                 |
| `date`   | Timestamp                                |
| `flag`   | Query flag (unused)                      |
| `user`   | Username                                 |
| `text`   | Tweet text                               |

## Notes

- The dataset is not included in this repository due to its size (~230 MB).
- Labels are remapped during preprocessing: `4 → 1` (positive), `0` stays as negative.
- The dataset is balanced: ~800K positive and ~800K negative tweets.
