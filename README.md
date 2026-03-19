# CS2 Match Predictor (refactored)

Цей проєкт переносить пайплайн з ноутбука в модульну структуру Python:
`JSON -> (header) -> DataFrame -> preprocessing -> distance features -> PCA sweep -> training`.

## Встановлення

1. Створіть віртуальне середовище та встановіть залежності:
   - `pip install -r requirements.txt`

## Підготовка даних

1. Покладіть JSON з матчами у зручне місце.
2. (Опційно) Підготуйте `headers.txt` — список header-рядків, по одному на рядок.
   Якщо `--header-path` не вказати, header згенерується автоматично з JSON.

## Запуск тренування

```bash
python scripts/train.py --json-path "./data/matches.json" --distance-set orig
```

Додатково можна:

- `--header-path "./data/headers.txt"`
- `--distance-set aug` (використовує augmented distance-датасет)
- `--output-dir "./models"`

За замовчуванням використовується метрика відстані `difference_distance` (як у ноутбуці).

## Результати

У `models/<run_id>/dist_<orig|aug>/` збережуться:

- `scaler.pkl`
- `best_setting_result.txt`
- `*_best_model.pkl`
- `*_best_pca.pkl`

## Оцінювання

Для генерації таблиць і рисунків:

```bash
python scripts/evaluate.py --model-dir "./models/<run_id>/dist_orig" --output-dir "./results"
```

