## Random Forest Training

Навчальний пакет будує класичний класифікатор емоцій на основі таблиці
ландмарок, яку генерує `model_prepare/extract_face_landmarks.py`.

### Швидкий старт

```bash
cd tg-bot
conda activate moodbot
python -m model_train.train_random_forest \
  --input model_prepare/face_landmarks.csv \
  --model-output artifacts/random_forest.joblib \
  --label-output artifacts/label_encoder.joblib \
  --metrics-output artifacts/metrics.json
```

Параметри:

- `--input` – шлях до CSV з колонками `file_name`, `class_name`, `x_i`, `y_i`, `z_i`.
- `--model-output` – куди зберегти модель `RandomForestClassifier` (joblib).
- `--label-output` – файл із закодованими мітками `LabelEncoder`.
- `--metrics-output` – необовʼязковий шлях для метрик (`accuracy`, `classification_report`).
- `--test-size`, `--n-estimators`, `--max-depth`, `--random-state` – стандартні
  гіперпараметри sklearn, мають розумні значення за замовчуванням.

Скрипт друкує базові метрики та записує їх у файл, якщо вказати
`--metrics-output`. Після тренування модель та енкодер можна завантажити у
будь-який інший сервіс (наприклад, Telegram-бот) та використовувати для
прогнозування настрою.

