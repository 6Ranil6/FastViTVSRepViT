# FastViT и RepViT для классификации цветов

Сравнение FastViT-SA12 и RepViT-M0.9 на Oxford Flowers 102 с подготовкой моделей для Samsung Galaxy A34 5G.

## Архитектуры моделей

### FastViT-SA12

```mermaid
flowchart TD
    input["RGB · 3 × 224 × 224"] --> stem["Stem · 3 → 64 канала<br/>64 × 56 × 56"]
    stem --> s1["Стадия 1 · 2 блока RepMixer<br/>64 × 56 × 56"]
    s1 -->|Patch Embedding| s2["Стадия 2 · 2 блока RepMixer<br/>128 × 28 × 28"]
    s2 -->|Patch Embedding| s3["Стадия 3 · 6 блоков RepMixer<br/>256 × 14 × 14"]
    s3 -->|Patch Embedding| cpe["RepCPE 7 × 7 · один раз<br/>512 × 7 × 7"]
    cpe --> s4["Стадия 4 · 2 блока attention<br/>MHSA: 16 голов по 32 признака<br/>512 × 7 × 7"]
    s4 --> expand["MobileOne + SE + GELU<br/>512 → 1024 канала"]
    expand --> pool["Global Average Pooling<br/>1024 признака"]
    pool --> head["Linear · 1024 → 102"]
```

Первые три стадии смешивают пространственные признаки через RepMixer. На последней стадии attention работает с 49 токенами карты 7 × 7. [Детальная схема FastViT-SA12](docs/FastViT-SA12.drawio)

### RepViT-M0.9

```mermaid
flowchart TD
    input["RGB · 3 × 224 × 224"] --> stem["Stem · две свёртки 3 × 3<br/>3 → 24 → 48 каналов"]
    stem --> s1["Стадия 1 · 2 блока<br/>48 × 56 × 56"]
    s1 -->|Downsample| s2["Стадия 2 · 2 блока<br/>96 × 28 × 28"]
    s2 -->|Downsample| s3["Стадия 3 · 14 блоков<br/>192 × 14 × 14"]
    s3 -->|Downsample| s4["Стадия 4 · 2 блока<br/>384 × 7 × 7"]
    s4 --> pool["Global Average Pooling<br/>384 признака"]
    pool --> head["Linear · 384 → 102<br/>после объединения голов"]
```

В блоках RepViT пространственные признаки смешивает RepVGGDW, а каналы — FFN. Внутри каждой стадии чередуются блоки с SE и без SE. [Детальная схема RepViT-M0.9](docs/RepViT-M0.9.drawio)

## Что где лежит

| Где искать | Что там |
|---|---|
| [report.md](report.md) | Результаты по seed, график обучения и проверка экспорта |
| [Обучение и оценка](docs/TRAIN_EVAL.md) | Подготовка окружения, загрузка датасета, обучение, оценка и экспорт |
| [Мобильные замеры в отчёте](report.md#мобильные-замеры) | Таблица и диаграммы задержки и точности на Galaxy A34, пояснения к показателям и выводы |
| [Сводка мобильных замеров](mobile_comparison.csv) · [CSV отдельных прогонов](report_phone/) | Сводные показатели для 1, 2 и 4 потоков CPU и данные шести испытаний |
| [Тестирование на Galaxy A34](docs/PHONE.md) | Установка приложения, запуск замеров, выгрузка CSV и обновление отчёта |
| [Устройство проекта](docs/ARCHITECTURE.md) | Архитектуры моделей и схема работы |
| [FastViT-SA12](docs/FastViT-SA12.drawio) · [RepViT-M0.9](docs/RepViT-M0.9.drawio) | Редактируемые схемы моделей |
| [coursework/](coursework/) · [configs/](configs/) | Python-код и параметры экспериментов |
| [android/](android/) | Приложение Android |
| [requirements.txt](requirements.txt) | Версии Python-зависимостей для окружения экспериментов |
| `data/`, `runs/`, `exports/` | Датасет, локальные запуски и экспортированные модели |
