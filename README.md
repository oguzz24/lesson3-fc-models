# Lesson 3 — полносвязные сети

Решение [задания](HOMEWORK.md) на основе [примеров преподавателя](README_LESSON.md).
Источник: https://github.com/AKlimovUrfu/lesson3-fc-models, commit `8e72d62d2244cba448c58a0096931e051f65af47`.

## Установка и запуск

Python 3.12, CPU или CUDA:

```bash
python -m pip install -r requirements-homework.txt
python download_data.py
python -m unittest discover -s homework/tests -v
python run_experiments.py --epochs 8 --train-limit 6000 --test-limit 2000
python build_report.py
```

Для полного train/test набора уберите `--train-limit` и `--test-limit`.
Для синтетической проверки: `python run_experiments.py --smoke --epochs 1`.
Синтетические результаты находятся отдельно и не используются в выводах.

## Содержание

- `homework/homework_depth_experiments.py`: глубины 1/2/3/5/7, сравнение с Dropout + BatchNorm.
- `homework/homework_width_experiments.py`: четыре заданные ширины и grid search 3×3 для сужения, постоянной ширины и расширения.
- `homework/homework_regularization_experiments.py`: baseline, Dropout 0.1/0.3/0.5, BatchNorm, их сочетание, L2, адаптивный Dropout, momentum BatchNorm и комбинированный вариант.
- `homework/utils/`: общий цикл, визуализация и адаптация FullyConnectedModel из урока.
- `homework/tests/`: формы и градиенты, взвешенный loss, оценка без градиентов, воспроизводимость, адаптивный Dropout, обучение простой задачи.

## Результаты

[Русский отчёт](REPORT.md), [машиночитаемая таблица](homework/results/summary.csv).
Все измерения и графики сохраняются в `homework/results/`. Лучшие веса по validation сохраняются локально в `best.pt`; они и датасеты исключены из Git.
`metrics.json` содержит гиперпараметры, размеры выборок, историю, время, параметры модели и градиенты.
Официальный test отделён от train; validation выделяется только из официального train.
Тестовые кривые не используются для выбора архитектуры или эпохи.

Опубликованные эксперименты используют фиксированные подвыборки и один seed, поэтому не претендуют на максимальную точность или статистическую значимость.
Исходные файлы преподавателя сохранены в `fully_connected_basics/` без изменений.
