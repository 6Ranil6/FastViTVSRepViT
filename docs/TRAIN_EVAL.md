# Обучение и оценка моделей на Mac

Все команды выполняются из корня проекта.

## Подготовить окружение и данные

В новой копии проекта:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
python -m coursework.download
```

Если окружение и датасет уже подготовлены, достаточно активировать `.venv`. При первой загрузке датасета и предобученных весов нужен интернет.

### Что скачивает `coursework.download`

С официального сайта Oxford в `data/flowers-102/` загружаются:

- `102flowers.tgz` — архив фотографий; распаковывается в `data/flowers-102/jpg/`.
- `imagelabels.mat` — правильный класс для каждого изображения.
- `setid.mat` — официальное разделение на train / validation / test.

Модуль проверяет MD5 скачанных файлов, наличие 8189 фотографий и отсутствие пересечений между выборками: **1020 train, 1020 validation, 6149 test**. Источники и контрольные суммы MD5/SHA-256 сохраняются в `data/flowers-102/manifest.json`. Также создаётся `data/labels.json` с техническими названиями классов (`Flower class 001` и т. д.).

При повторном запуске файлы с правильной MD5 не скачиваются заново; распаковка и проверки выполняются повторно.

Чтение и преобразование изображений выполняет `coursework/core.py`. Предобученные веса загружает `timm` при создании модели для обучения.

## Обучить обе модели

```bash
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
seed=45
python -m coursework.train --model fastvit --seed "$seed" --device mps
python -m coursework.train --model repvit --seed "$seed" --device mps
```

Seed 45 в примере создаёт новый запуск. Для воспроизведения исходного сравнения в новой копии проекта используй 42, 43 и 44. Скрипт защищает существующие `best.pt` от перезаписи.

Обучение сохраняет конфигурацию, историю эпох и веса лучшей эпохи по точности на валидации в `runs/full/<модель>-seed<номер>/`.

## Оценить на test и обновить отчёт

```bash
python -m coursework.evaluate "runs/full/fastvit-seed$seed/best.pt" --device cpu
python -m coursework.evaluate "runs/full/repvit-seed$seed/best.pt" --device cpu
python -m coursework.report
```

В новом терминале активируй `.venv` и укажи нужный seed, например `seed=42` для базового запуска. Оценка проходит на всех 6149 изображениях официального test. Рядом с весами появляются `test_metrics.json`, прогнозы и матрица ошибок. Отчёт и график обучения находятся в корне проекта. [Отчёт](../report.md)

## Подготовить новые веса для телефона

```bash
python -m coursework.export "runs/full/fastvit-seed$seed/best.pt"
python -m coursework.export "runs/full/repvit-seed$seed/best.pt"
python -m coursework.bundle --seed "$seed"
```

Экспорт сохраняет `.pte` в `exports/full/` и проверяет совпадение результатов с исходной моделью. Последняя команда кладёт обе модели и контрольные фотографии в Android-проект. Затем пересобери приложение. [Установка и замеры на Galaxy A34](PHONE.md)


## Повторить эксперимент целиком

В новой копии проекта команда ниже скачивает данные, обучает и оценивает обе модели, экспортирует веса, собирает Android-пакет и обновляет отчёт.

```bash
bash scripts/run_experiment.sh 42
```

Для других запусков передай нужный seed. Сценарий собирает Android-пакет для указанного seed. [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) · [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)
