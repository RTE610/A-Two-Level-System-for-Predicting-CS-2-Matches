# Дипломна робота: план коду та документа

## Поточний стан

**Код (готово):**
- Модульна структура: `config.py`, `src/data_loader.py`, `src/preprocessing.py`, `src/features.py`, `src/training.py`
- Pipeline: JSON → header/DF → preprocessing → distance features → PCA sweep → training 11 classifiers
- Збереження моделей в `models/<run_id>/`

**Текст (готово):**
- Аналітичний розділ (~50 джерел): стан кіберспорту, методи прогнозування, дворівневі архітектури, метрики

**Що потрібно додати** — і в код, і в документ:

---

## Частина 1: Код — що додати

### 1.1 `src/evaluation.py` — модуль оцінювання

Зараз training.py рахує тільки accuracy. Для диплому потрібно більше метрик і їх аналіз.

**Функції:**

```
evaluate_model(y_true, y_pred, y_proba=None) → dict
```
Повертає словник з:
- `accuracy`
- `precision` (для кожного класу + macro)
- `recall` (для кожного класу + macro)
- `f1` (для кожного класу + macro)
- `roc_auc` (якщо є y_proba)
- `log_loss` (якщо є y_proba)
- `confusion_matrix`

```
compare_models(results: dict[str, dict]) → pd.DataFrame
```
Зводить результати всіх моделей в одну таблицю для порівняння.

```
run_cross_validation(X, y, classifiers, n_splits=5) → dict
```
Stratified K-Fold CV для кожного класифікатора. Повертає mean ± std для кожної метрики.

**Навіщо для диплому:** розділ "Результати" має містити не просто "accuracy 63%", а розгорнутий аналіз — де модель помиляється, наскільки стабільна, як порівнюється з іншими.

---

### 1.2 `src/visualization.py` — модуль візуалізацій

Кожна функція зберігає фігуру у файл (для вставки в документ).

**Функції:**

```
plot_pca_accuracy_curves(pca_results, save_path)
```
Графік accuracy vs n_components для кожного класифікатора. Показує як PCA впливає на якість — ключова візуалізація для диплому.

```
plot_confusion_matrix(cm, class_names, save_path)
```
Heatmap confusion matrix для найкращої моделі.

```
plot_roc_curves(models_results, save_path)
```
ROC-криві кількох моделей на одному графіку з AUC у легенді.

```
plot_feature_importance(model, feature_names, save_path, top_n=20)
```
Горизонтальний bar chart top-20 фіч за важливістю (для tree-based моделей: RF, XGB, LightGBM, CatBoost).

```
plot_class_distribution(y, save_path)
```
Розподіл target-змінної (баланс класів).

```
plot_correlation_heatmap(X, save_path, top_n=30)
```
Кореляційна матриця найважливіших фіч.

```
plot_cv_results(cv_results, save_path)
```
Box-plot або bar chart з error bars для CV результатів кожної моделі.

**Навіщо для диплому:** кожен з цих графіків — це рисунок у розділі "Експериментальна частина". Чим більше якісних візуалізацій — тим переконливіше виглядає робота.

---

### 1.3 Експерименти для запуску

Це конкретні запуски, які дадуть дані для документа.

#### Експеримент 1: Порівняння distance-метрик
- Запустити pipeline для кожної distance-функції: `difference`, `euclidean`, `manhattan`, `bray_curtis`, `cosine`
- Зібрати best accuracy для кожної метрики
- **Результат для документа:** таблиця "вплив метрики відстані на якість прогнозування"

#### Експеримент 2: orig vs augmented
- Запустити pipeline для `orig` та `aug` datasets
- Порівняти accuracy / F1 / AUC
- **Результат для документа:** таблиця + аналіз "чи допомагає swap augmentation"

#### Експеримент 3: PCA sweep аналіз
- Зберегти повні графіки accuracy vs PCA components для всіх 11 класифікаторів
- Визначити оптимальний діапазон PCA для кожного
- **Результат для документа:** графік + таблиця оптимальних PCA

#### Експеримент 4: Cross-Validation
- 5-fold stratified CV для всіх класифікаторів з їх оптимальним PCA
- Зібрати mean ± std для accuracy, F1, AUC
- **Результат для документа:** таблиця стабільності моделей

#### Експеримент 5: Confusion matrix аналіз
- Для top-3 моделей побудувати confusion matrices
- Порівняти precision/recall для кожного класу
- **Результат для документа:** 3 рисунки + аналіз де які моделі помиляються

#### Експеримент 6: Feature importance
- Для tree-based моделей (RF, XGB, LightGBM, CatBoost) дістати feature importance
- Порівняти які фічі найважливіші для різних моделей
- **Результат для документа:** bar chart + інтерпретація ("ключову роль відіграє win percentage та power rank")

#### Експеримент 7: ROC-криві
- Побудувати ROC для всіх моделей які підтримують `predict_proba`
- **Результат для документа:** 1 графік з кривими + AUC в легенді

---

### 1.4 `scripts/evaluate.py` — скрипт оцінювання

Окремий скрипт який:
1. Завантажує збережені моделі з `models/<run_id>/`
2. Запускає evaluation + CV
3. Генерує всі візуалізації
4. Зберігає summary таблицю в CSV

```bash
python scripts/evaluate.py --model-dir models/<run_id>/dist_orig/ --output-dir results/
```

---

### 1.5 Структура outputs для документа

```
results/
├── tables/
│   ├── distance_metrics_comparison.csv
│   ├── orig_vs_aug_comparison.csv
│   ├── pca_optimal_components.csv
│   ├── cv_results.csv
│   └── final_model_comparison.csv
└── figures/
    ├── pca_accuracy_curves.png
    ├── confusion_matrix_best_model.png
    ├── confusion_matrix_rf.png
    ├── confusion_matrix_xgb.png
    ├── roc_curves.png
    ├── feature_importance_rf.png
    ├── feature_importance_xgb.png
    ├── feature_importance_lgbm.png
    ├── class_distribution.png
    ├── correlation_heatmap.png
    └── cv_boxplot.png
```

---

## Частина 2: Документ — структура дипломної роботи

### ВСТУП (~2–3 сторінки)
- Актуальність: кіберспорт росте, CS2 — найпопулярніша тактична гра, прогнозування матчів — відкрита задача
- Мета: розробити систему прогнозування результатів матчів CS2 на основі ML
- Задачі:
  1. Провести аналіз існуючих методів прогнозування в кіберспорті
  2. Зібрати та обробити дані матчів (PandaScore API)
  3. Розробити pipeline feature engineering з distance-based features
  4. Порівняти 11 класифікаторів з PCA dimensionality reduction
  5. Оцінити вплив data augmentation (swap teams)
  6. Провести cross-validation та аналіз стабільності моделей
- Об'єкт: процес прогнозування результатів кіберспортивних матчів
- Предмет: методи ML для класифікації результатів матчів CS2
- Наукова новизна: комбінація distance-based feature engineering + PCA sweep + swap augmentation для CS2

---

### РОЗДІЛ 1: АНАЛІТИЧНИЙ ОГЛЯД (~15–20 сторінок)
*Вже написано з 50 джерелами.*

- 1.1 Сучасний стан кіберспорту та CS2
- 1.2 Методи прогнозування в спорті та кіберспорті
- 1.3 Дворівневі архітектури
- 1.4 Обробка даних у спортивній аналітиці
- 1.5 Метрики оцінювання
- 1.6 Висновки до розділу 1

---

### РОЗДІЛ 2: ОПИС ДАНИХ ТА МЕТОДОЛОГІЯ (~15–20 сторінок)

#### 2.1 Джерело даних
- PandaScore API: що це, які дані надає
- Структура JSON: match → summaryStats → team1/team2 → ~160 фіч на команду
- Часовий діапазон вибірки (90 днів, endpoint date)
- Обсяг: скільки матчів, скільки фіч
- Обмеження даних: відсутні значення, Infinity, шум

#### 2.2 Preprocessing pipeline
- Парсинг JSON → DataFrame (header-based підхід, чому саме так)
- Обробка аномалій: Infinity → NaN → mean imputation
- Видалення ID-колонок (actualWinnerId, pandaScoreTeamId — щоб модель не "підглядала")
- Target variable: y=1 (team1 wins), y=0 (team2 wins)
- Статистика після очистки: розмір датасету, баланс класів

**Рисунок:** розподіл target-змінної (class_distribution.png)

#### 2.3 Feature engineering: distance-based approach
- Ідея: замість raw stats team1 і team2 → обчислити distance між ними
- Навіщо: зменшити розмірність, зробити фічі інваріантними до "який team — team1"
- Реалізація: для кожної пари (team1_stat, team2_stat) → distance(stat1, stat2)
- Протестовані метрики відстані (перелік і формули):
  - Euclidean: $d = \sqrt{(x-y)^2}$
  - Manhattan: $d = |x-y|$
  - Difference: $d = x - y$ (зберігає знак!)
  - Bray-Curtis: $d = \frac{|x-y|}{|x|+|y|}$
  - Cosine, Log-Euclidean, Exponential та інші
- Чому difference_distance обрано як основну (якщо так — результати експерименту 1)

**Таблиця:** порівняння distance-метрик (distance_metrics_comparison.csv)

#### 2.4 Data augmentation: swap teams
- Ідея: для кожного матчу (team1 vs team2, y=1) створити дзеркальний (team2 vs team1, y=0)
- Це подвоює розмір датасету
- Для difference_distance: swap просто змінює знак
- Обґрунтування: усуває bias моделі до "team1 завжди сильніша"

#### 2.5 Dimensionality reduction: PCA
- Навіщо: ~160 distance-фіч — багато для деяких моделей (curse of dimensionality)
- Підхід: sweep від 1 до max_components, для кожного n тренувати всі моделі
- StandardScaler перед PCA (обов'язково для коректної роботи)

#### 2.6 Класифікатори
Для кожного — 2–3 речення: суть методу, сильні/слабкі сторони, чому включений

| # | Класифікатор | Тип |
|---|---|---|
| 1 | Logistic Regression | Лінійний |
| 2 | Decision Tree | Дерево |
| 3 | Random Forest | Ансамбль (bagging) |
| 4 | Gradient Boosting | Ансамбль (boosting) |
| 5 | XGBoost | Ансамбль (boosting) |
| 6 | LightGBM | Ансамбль (boosting) |
| 7 | CatBoost | Ансамбль (boosting) |
| 8 | SVM | Kernel-based |
| 9 | k-NN | Instance-based |
| 10 | Naive Bayes | Ймовірнісний |
| 11 | MLP (Neural Network) | Нейронна мережа |

#### 2.7 Метрики оцінювання
- Accuracy — основна метрика (бо класи збалансовані)
- Precision / Recall / F1 — для аналізу помилок по класах
- ROC-AUC — для ймовірнісної оцінки
- Cross-Validation (5-fold stratified) — для оцінки стабільності

---

### РОЗДІЛ 3: ЕКСПЕРИМЕНТАЛЬНА ЧАСТИНА (~20–25 сторінок)

Це найважливіший розділ — тут кожен експеримент = підрозділ з таблицею/рисунком і аналізом.

#### 3.1 Середовище та налаштування
- Python 3.x, scikit-learn, XGBoost, LightGBM, CatBoost
- Hardware (Colab specs або локальна машина)
- Train/test split: 80/20, stratified, random_state=42
- Репозиторій: посилання на GitHub

#### 3.2 Експеримент 1: вплив метрики відстані
- Що робили: запустили pipeline для 5+ метрик
- Результати: таблиця best accuracy для кожної метрики
- Аналіз: чому difference працює краще (зберігає напрямок, sign matters для "хто сильніший")

**Таблиця:** distance_metrics_comparison.csv

#### 3.3 Експеримент 2: вплив data augmentation
- Що робили: порівняли orig (N матчів) vs aug (2N матчів)
- Результати: таблиця accuracy/F1/AUC для orig vs aug
- Аналіз: чи допомагає подвоєння? Для яких моделей більше?

**Таблиця:** orig_vs_aug_comparison.csv

#### 3.4 Експеримент 3: PCA sweep
- Що робили: тренували 11 моделей для PCA від 1 до max
- Результати: графік accuracy vs n_components
- Аналіз: які моделі потребують менше компонент, які — більше? Чи є "коліно"?

**Рисунок:** pca_accuracy_curves.png
**Таблиця:** pca_optimal_components.csv

#### 3.5 Експеримент 4: фінальне порівняння моделей
- Що робили: кожна модель з оптимальним PCA, повні метрики
- Результати: зведена таблиця (accuracy, precision, recall, F1, AUC)

**Таблиця:** final_model_comparison.csv

#### 3.6 Експеримент 5: Cross-Validation
- Що робили: 5-fold stratified CV для top-5 моделей
- Результати: mean ± std для accuracy і F1
- Аналіз: наскільки стабільні моделі? Чи немає overfitting?

**Рисунок:** cv_boxplot.png
**Таблиця:** cv_results.csv

#### 3.7 Аналіз помилок
- Confusion matrices для top-3 моделей
- Де модель частіше помиляється: коли прогнозує win team1 чи win team2?
- Precision vs recall tradeoff

**Рисунки:** confusion_matrix_*.png

#### 3.8 Інтерпретація: feature importance
- Для tree-based моделей: які distance-фічі найважливіші?
- Чи корелюють top-фічі між різними моделями?
- Інтерпретація: "winPercentage distance — найважливіша фіча, що логічно — різниця у відсотку перемог напряму пов'язана з силою команд"

**Рисунки:** feature_importance_*.png

#### 3.9 ROC-аналіз
- ROC-криві для всіх моделей
- Порівняння AUC
- Який класифікатор найкраще калібрований?

**Рисунок:** roc_curves.png

#### 3.10 Обговорення результатів
- Яка модель найкраща і чому?
- Порівняння з літературою (accuracy ~62–64% в існуючих роботах)
- Обмеження: розмір вибірки, відсутність live-features (in-game), стабільність ростерів
- Що можна покращити в майбутньому

---

### РОЗДІЛ 4: ПРОГРАМНА РЕАЛІЗАЦІЯ (~5–10 сторінок)

#### 4.1 Архітектура системи
- Діаграма: JSON → data_loader → preprocessing → features → training → evaluation
- Модульна структура, обґрунтування розбиття

#### 4.2 Опис модулів
- Коротко по кожному модулю: що робить, ключові функції
- UML або блок-схема pipeline

#### 4.3 Інструкція з використання
- Встановлення залежностей
- Запуск тренування
- Запуск оцінювання
- Структура результатів

#### 4.4 Репозиторій
- Посилання на GitHub
- Структура файлів

---

### ВИСНОВКИ (~2–3 сторінки)
- Підсумок по кожній задачі зі Вступу
- Конкретні цифри: "найкраща модель — X з accuracy Y%, F1 Z%"
- Порівняння з бейзлайнами з літератури
- Практична значимість
- Напрямки подальших досліджень:
  - Додати in-game фічі (раундова статистика)
  - Temporal features (форма команди за останні N днів)
  - Hyperparameter tuning (GridSearch / Optuna)
  - Ensemble stacking
  - Deep learning (LSTM для часових рядів матчів)

---

### СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ
- ~50 джерел з аналітичного розділу
- Додати джерела по конкретних бібліотеках (scikit-learn docs, XGBoost paper, etc.)
- Офіційна документація PandaScore API

---

## Порядок виконання

### Фаза 1: Код (evaluation + visualization)
| Крок | Що | Залежності |
|------|----|------------|
| 1 | Написати `src/evaluation.py` | numpy, sklearn.metrics |
| 2 | Написати `src/visualization.py` | matplotlib, seaborn |
| 3 | Написати `scripts/evaluate.py` | evaluation, visualization |
| 4 | Запустити pipeline для різних distance-метрик | train.py |
| 5 | Запустити pipeline для orig vs aug | train.py |
| 6 | Запустити evaluate.py, зібрати таблиці та рисунки | evaluate.py |

### Фаза 2: Документ
| Крок | Що | Потрібно з коду |
|------|----|-----------------|
| 7 | Написати Розділ 2 (дані + методологія) | — |
| 8 | Написати Розділ 3 (експерименти) | Таблиці + рисунки з Фази 1 |
| 9 | Написати Розділ 4 (реалізація) | GitHub repo, screenshots |
| 10 | Написати Вступ і Висновки | Конкретні цифри з Розділу 3 |
| 11 | Зібрати Список джерел | — |
| 12 | Фінальна вичитка | — |

---

## Чекліст "що має бути в дипломі для 5/5"

### Код
- [ ] Модульна структура на GitHub
- [ ] requirements.txt
- [ ] README з інструкціями
- [ ] Мінімум 5 різних експериментів
- [ ] Мінімум 8 рисунків (графіки, heatmaps, confusion matrices)
- [ ] Мінімум 5 таблиць з результатами
- [ ] Cross-validation для підтвердження стабільності

### Документ
- [ ] Аналітичний розділ з 40+ джерелами
- [ ] Чіткий опис даних: джерело, обсяг, фічі, preprocessing
- [ ] Формули для метрик відстані та метрик оцінювання
- [ ] Кожен експеримент: що зробили → що отримали → чому так
- [ ] Порівняння з існуючими роботами (accuracy ~62–64%)
- [ ] Feature importance з інтерпретацією
- [ ] Обговорення обмежень
- [ ] Напрямки подальших досліджень
