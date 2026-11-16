"""Generate Russian report and tables only from completed real experiment outputs."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import platform
import torchvision

ROOT = Path(__file__).resolve().parent


def main():
    runtime = {'python':platform.python_version(), 'torch':torch.__version__,
               'torchvision':torchvision.__version__, 'device':'cuda' if torch.cuda.is_available() else 'cpu',
               'platform':platform.platform()}
    (ROOT/'runtime.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    groups = [('depth_experiments','Глубина'), ('width_experiments','Ширина'), ('regularization_experiments','Регуляризация')]
    lines = ['# Отчёт: полносвязные сети', '', '## Задание и цель работы',
             'Исследовать влияние глубины, ширины и регуляризации на классификацию MNIST и CIFAR-10. Выполнены задания 1–3 из HOMEWORK.md.', '',
             '## Использованные методы',
             'Использован FullyConnectedModel преподавателя: Linear, ReLU, Dropout, BatchNorm; CrossEntropyLoss и Adam. Глубина включает выходной Linear. В задании о ширине три заданных размера интерпретируются как три скрытых слоя плюс классификатор.', '',
             '## Реализация',
             'Отдельные программы в homework/; общий цикл в utils/experiment_utils.py. Сохраняются история, параметры запуска, число параметров, время обучения и инференса, нормы градиентов и контрольные точки. Dropout изменяется от 0.1 до 0.5 с меньшим коэффициентом на ранних слоях. Momentum BatchNorm: 0.01, 0.1, 0.5.', '',
             '## Эксперименты',
             'Случайное разбиение официальной train-части: 90% train, 10% validation, seed=42. Для опубликованного CPU-эксперимента взяты 6000 train, 1200 validation, 2000 случайных test-примеров; 8 эпох, batch_size=64, Adam lr=0.001. Это ограниченный эксперимент, не обучение на полном датасете. Полный запуск доступен без --train-limit и --test-limit.',
             'Выбор эпохи и конфигурации производится по validation accuracy. Test не участвует в оптимизации или выборе. Train accuracy измеряется в eval-режиме. Время обучения включает backward и optimizer, но не оценку; инференс измеряется после прогрева. Результаты одного seed не являются оценкой статистической значимости.', '',
             '## Результаты']
    all_rows = []
    for group, title in groups:
        files = sorted((ROOT/'homework/results'/group/'real').glob('*/*/metrics.json'))
        results = [json.loads(p.read_text(encoding='utf-8')) for p in files]
        if not results:
            lines += ['', f'### {title}', 'Нет завершённых реальных запусков.']; continue
        lines += ['', f'### {title}', '', '| Данные | Модель | Параметры | Train, последняя | Val, лучшая | Test, выбранная | Секунды |', '|---|---|---:|---:|---:|---:|---:|']
        for r in results:
            lines.append(f"| {r['dataset']} | {r['name']} | {r['parameters']} | {r['history'][-1]['train']['accuracy']:.4f} | {r['best_val']:.4f} | {r['selected_test']['accuracy']:.4f} | {r['train_seconds']:.2f} |")
            all_rows.append(dict(group=group, dataset=r['dataset'], model=r['name'], parameters=r['parameters'], best_epoch=r['best_epoch'], val_accuracy=r['best_val'], test_accuracy=r['selected_test']['accuracy'], seconds=r['train_seconds'], inference_ms=1000*r['inference_seconds_per_image'], overfit_epoch=r['overfit_signal_epoch'], stability=r['validation_accuracy_std']))
        for dataset in sorted({r['dataset'] for r in results}):
            subset = [r for r in results if r['dataset']==dataset]
            best = max(subset,key=lambda r:r['best_val'])
            gap = best['history'][-1]['train']['accuracy'] - best['history'][-1]['test']['accuracy']
            lines += ['', f"Для {dataset} лучшая по validation конфигурация — **{best['name']}**: val={best['best_val']:.4f}, test={best['selected_test']['accuracy']:.4f}. Разрыв train−test последней эпохи: {gap:.4f}. Первый сигнал переобучения (train loss падает, val loss растёт): {best['overfit_signal_epoch']}. Стандартное отклонение val accuracy по эпохам: {best['validation_accuracy_std']:.4f}."]
            fig, axes = plt.subplots(1,2,figsize=(13,5))
            for r in subset:
                axes[0].plot(range(1,len(r['history'])+1),[h['val']['accuracy'] for h in r['history']],label=r['name'])
            axes[0].set(xlabel='Epoch',ylabel='Validation accuracy'); axes[0].legend(fontsize=6)
            axes[1].barh(range(len(subset)),[r['selected_test']['accuracy'] for r in subset])
            axes[1].set_yticks(range(len(subset)),[f"{r['name']} ({r['parameters']/1000:.1f}k)" for r in subset],fontsize=7)
            axes[1].set(xlabel='Test accuracy (validation-selected epoch)',xlim=(0,1))
            fig.tight_layout(); dest=ROOT/'homework/results'/group/'real'/f'{dataset}_comparison.png'; fig.savefig(dest); plt.close(fig)
            lines += ['',f'![{title}: {dataset}]({dest.relative_to(ROOT).as_posix()})']
            if group=='width_experiments':
                lines += ['', f'![Grid search {dataset}](homework/results/{group}/real/{dataset}/grid.png)']
            if group=='depth_experiments':
                plain = [r for r in subset if not r['name'].endswith('_dropout_bn')]
                plain_best = max(plain,key=lambda r:r['best_val'])
                lines += ['', f"Без регуляризации оптимальная исследованная глубина: {plain_best['name']}; val={plain_best['best_val']:.4f}. Сравнение пар:", '', '| Глубина | Δ val (Dropout+BN − baseline) | Разрыв train−test baseline | Разрыв train−test Dropout+BN |', '|---|---:|---:|---:|']
                for base in plain:
                    paired=next((r for r in subset if r['name']==base['name']+'_dropout_bn'),None)
                    if paired:
                        gap_a=base['history'][-1]['train']['accuracy']-base['history'][-1]['test']['accuracy']
                        gap_b=paired['history'][-1]['train']['accuracy']-paired['history'][-1]['test']['accuracy']
                        lines.append(f"| {base['name']} | {paired['best_val']-base['best_val']:+.4f} | {gap_a:.4f} | {gap_b:.4f} |")
            if group=='regularization_experiments':
                baseline=next(r for r in subset if r['name']=='baseline')
                lines += ['', '| Метод | Δ test к baseline | Std val accuracy по эпохам | Первый сигнал переобучения |', '|---|---:|---:|---|']
                for r in subset:
                    lines.append(f"| {r['name']} | {r['selected_test']['accuracy']-baseline['selected_test']['accuracy']:+.4f} | {r['validation_accuracy_std']:.4f} | {r['overfit_signal_epoch']} |")
        if group == 'regularization_experiments':
            lines += ['', '#### Влияние на слои', '', 'Таблица использует веса лучшей validation-эпохи; нормы градиентов усреднены по обучению. Гистограммы weights.png показывают последнюю эпоху.', '', '| Данные | Вариант | Слой | Std весов | Средняя норма градиента |', '|---|---|---|---:|---:|']
            for r, path in zip(results, files):
                if r['name'] not in ('baseline', 'adaptive', 'combined_adaptive'): continue
                state = torch.load(path.parent/'best.pt', map_location='cpu', weights_only=True)
                for name, weight in state.items():
                    if weight.ndim != 2: continue
                    grad = sum(h.get(name,0.) for h in r['gradient_history'])/len(r['gradient_history'])
                    lines.append(f"| {r['dataset']} | {r['name']} | {name} | {weight.std().item():.5f} | {grad:.5f} |")
    lines += ['', '## Графики и таблицы',
              'Сводные числовые данные: [summary.csv](homework/results/summary.csv). В каждом каталоге модели: curves.png (train/val/test loss и accuracy), weights.png (распределения весов по слоям), metrics.json (все исходные измерения).', '',
              '## Выводы',
              'Оптимальные конфигурации указаны отдельно для каждого датасета и группы выше, строго по validation. Большая глубина и ширина сами по себе не гарантируют улучшения; сравнение должно учитывать время, число параметров и разрыв train/test. Dropout и BatchNorm меняют динамику обучения, поэтому их эффект оценивается при одной базовой архитектуре [256,128,64].',
              'Один рост validation loss — лишь ранний сигнал, а не доказательство устойчивого переобучения. Для окончательного выбора нужны повторения по нескольким seed и более длительное обучение на полном train. Синтетические тесты не используются в таблицах результатов.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if all_rows:
        with (ROOT/'homework/results/summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)


if __name__=='__main__': main()
