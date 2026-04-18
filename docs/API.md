# API Reference

## Python package exports

From `src` package:

- `engineer_features(df)`
- `build_preprocessor(X)`
- `main()`

## CLI (`predict.py`)

```bash
python predict.py --input <file.csv> [--output <scores.csv>] [--threshold 0.5]
python predict.py --json '{"duration":24,...}' [--threshold 0.5]
```

Output fields:

- `default_probability` (float)
- `risk_label` (`Low|Medium|High`)
- `predicted_default` (`0|1`)

## REST API (`app.py`)

### `GET /health`

Response:

```json
{"status":"ok"}
```

### `POST /predict`

Request body: single JSON object or JSON array of loan records.

Response for single record:

```json
{
  "default_probability": 0.23,
  "risk_label": "Low",
  "predicted_default": 0
}
```
