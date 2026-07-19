# Generative ML Project Roadmap

**Обновлено:** 2026-07-19
**Статус:** generative ML работа временно приостановлена.
**Следующий эксперимент после возобновления:** transfer learning
`AFHQ Cats → AFHQ cat / dog / wild`.

Этот файл — единственный источник актуального ML-плана проекта. Подробные
результаты запусков остаются в `reports/` и `reports/experiment_ledger.jsonl`.
Пауза не отменяет backlog и не разрешает обучение или evaluation без отдельного
human-gated решения.

## 1. Правила ML-работы

1. Учиться через работающие эксперименты и разбирать теорию на их примере.
2. Сначала проверять минимальную рабочую версию, затем масштабировать.
3. Сравнивать модели только при одинаковых seeds, sampler, CFG, VAE,
   reference split, feature extractor и вычислительном бюджете.
4. Quick evaluation использовать для итераций; full evaluation — для
   отдельно утверждённых финалистов.
5. Не заявлять результаты обучения, CUDA, тестов или производительности, если
   соответствующие команды реально не запускались.
6. Материальные ML-операции фиксировать в experiment ledger с config,
   checkpoint/hash, командой, runtime, метриками и решением.
7. Перед изменением ML-направления читать этот roadmap; согласованные изменения
   вносить только сюда.
8. Long training и evaluation остаются ручными human gates.
9. RunPod и расширенный MLOps отложены до явного решения пользователя.

## 2. Завершённые ML-этапы

### 2.1 CIFAR-10 DDPM, U-Net, 32×32 — завершён

- Реализована class-conditioned DDPM на чистом PyTorch.
- Обучение завершено на 200k шагов на RTX 4090 с BF16.
- Проверены raw/EMA, checkpoint/resume, deterministic sampling и fixed
  previews.
- Добавлен DDIM-50 для быстрых previews; DDPM-1000 сохранён для основной
  оценки.
- Оптимизация дала примерно `+27.15%` throughput:
  `1787.57 → 2272.92 img/s`.

### 2.2 Tiny ImageNet U-Net, 64×64 — учебный этап закрыт

- Создана U-Net примерно на 48.4M параметров.
- Обучение остановлено примерно на 37%; checkpoint сохранён.
- Получен опыт перехода 32→64 px, batch/gradient accumulation и измерения
  производительности.
- Возобновлять этот run без отдельной исследовательской причины не требуется.

### 2.3 Latent SiT-S/2, Imagenette, 128×128 — завершён

- Использованы pretrained SD VAE и cached latents.
- Реализована SiT-S/2 с rectified-flow training и Heun sampling.
- Обучены baseline и отдельная REPA-версия с frozen DINOv2 teacher.
- REPA run остановлен и сохранён примерно на 365k.
- Проверены raw/EMA, checkpoints, deterministic sampling и скорость.

### 2.4 Training Evaluator — рабочая версия

- Зафиксированы quick и full протоколы.
- Метрики: FID, KID, precision, recall; для многоклассовых моделей также
  target accuracy.
- Диагностика: NaN/Inf, black/white/low-detail failures, feature duplicates,
  nearest neighbours и outliers.
- Реализовано сравнение checkpoints одной командой с JSON/CSV/Markdown и
  comparison grids.
- Технический долг: всегда выводить численную секцию `raw vs EMA`.

### 2.5 AFHQ Cats SiT-B/2 baseline, 128×128 — завершён

- Canonical checkpoint: raw 20k.
- Quick-200: FID `48.051`, KID `0.02052`, precision `0.340`,
  recall `0.754`.
- Raw лучше EMA; checkpoint заморожен как baseline.

### 2.6 AFHQ Cats SiT-B/2 + always-on REPA — завершён

- Отдельное обучение с нуля до 20k.
- REPA 10k→20k улучшила FID `62.305 → 52.384` и precision
  `0.210 → 0.310`, но recall снизился `0.862 → 0.722`.
- На одинаковом бюджете baseline raw 20k лучше REPA raw 20k по
  FID/KID/precision/recall.
- Always-on REPA исключена из дальнейшего отбора.

### 2.7 REPA early-stop: 10k REPA → 20k без REPA — завершён

- Продолжение выполнено из точного REPA checkpoint 10k без сброса SiT,
  optimizer, scheduler, EMA или RNG.
- После 10k teacher/projector отключены, `repa_weight = 0`.
- Выбран raw early-stop 20k:
  `outputs/afhq_cat_sit_b_128_repa_early_stop/checkpoints/best_raw_0020000.pt`.
- SHA-256:
  `300b5600b86d1a35ebf2c27307e480070cceee113735b23ffca8e46316e57bd0`.
- Quick-200: FID `45.787`, KID `0.01692`, precision `0.280`,
  recall `0.732`.
- Выбор сделан по FID/KID; baseline сохранил преимущество по precision/recall.
- Full-1000 сознательно не запускался; это ограничение freeze.

### 2.8 Portfolio packaging ML-результатов — завершён

- Подготовлены публичные README, case study, technical retrospective,
  воспроизводимый public verifier и компактные visual assets.
- Количественные ML claims связаны с reports/ledger; ограничения quick-200,
  precision/recall и незапущенного full-1000 изложены явно.
- Репозиторий опубликован как
  `Grigoriy-V/human-in-the-loop-generative-ml-lab` с MIT License и
  third-party attribution.

## 3. Текущий ML-статус

Generative ML работа поставлена на паузу. Checkpoints, outputs, configs,
метрики и experiment ledger сохраняются без пересмотра. Никакие training,
sampling, evaluation, dataset preparation или downloads сейчас не запланированы.

При возобновлении первое решение — отдельный human gate на эксперимент
`AFHQ Cats → все классы AFHQ`.

## 4. Ближайшие эксперименты после возобновления

### 4.1 Transfer learning: Cats → AFHQ cat / dog / wild

**Цель:** проверить пользу собственной pretrained модели на новом
распределении по сравнению с обучением с нуля.

1. Взять canonical AFHQ Cats early-stop checkpoint.
2. Расширить conditioning с одного класса до `cat / dog / wild`.
3. Перенести общие SiT weights; новые class embeddings инициализировать
   отдельно и задокументировать checkpoint surgery.
4. Запуск считать новым transfer/fine-tuning experiment, а не resume.
5. Перед full run выполнить dataset/split и VAE/cache validation, one-batch
   overfit, performance benchmark и короткий smoke.
6. Сравнить transfer и scratch при одинаковом бюджете.
7. Оценить каждый класс и общую выборку по frozen protocol.
8. Выполнить quick comparison; full evaluation запускать только для
   утверждённого финалиста.

### 4.2 Generative Training & Evaluation Playbook v1

После transfer-эксперимента оформить проверенный runbook:

- dataset/split и VAE/cache validation;
- one-batch overfit, benchmark и smoke;
- checkpoint/resume и fixed previews;
- quick/full evaluation и decision gate;
- report, model card и experiment-ledger evidence.

### 4.3 Img2img

1. Закодировать входное изображение существующим VAE.
2. Добавить шум по strength/стартовому времени.
3. Запустить обратный flow из этого состояния.
4. Сохранить conditioning, seed и metadata.
5. Построить strength grid и оценить сохранение композиции/изменение.

### 4.4 Hires fix и sampler ablation

- Первый проход 128×128, upscale до совместимого размера, затем слабый
  img2img pass.
- Первым безопасным кандидатом считать 192 px; размер проверять по VAE и
  patch-grid ограничениям.
- Сравнить Heun и Euler на одинаковых seeds и NFE.
- Выбрать отдельные preview и final-evaluation sampler.

## 5. Более поздний ML-backlog

### 5.1 Representation Autoencoder, 128×128

- Сначала измерить reconstruction ceiling и структуру latent space.
- Затем адаптировать SiT к новым latent channels/resolution.
- Сравнить с текущим VAE при одинаковом dataset/model budget.
- Не начинать до transfer, img2img и Playbook v1.

### 5.2 Native 256×256

- Переходить только после доказанного 128px-рецепта.
- Начать с capacity/data/VRAM benchmark.
- Выбирать SiT-S или SiT-B по фактической скорости и качеству.

### 5.3 Generated Image Inspector / Creative QA

- DINO/DINOv2 features, CLIP/SigLIP alignment, готовые IQA/aesthetic модели
  и простые statistical rules.
- Batch report с issue tags и confidence.
- Позже: style consistency, anatomy/artifacts, reference consistency,
  batch drift и regression testing.
- Собственный detector/разметку обсуждать только если готовые модели
  недостаточны.

### 5.4 CV navigation / robotics

1. 2D-машинка по данным датчиков.
2. Управление по изображению.
3. Indoor visual navigation.
4. Позже — реальные autonomous-driving datasets.

Фокус: depth, segmentation, obstacle detection, localization и path planning.

### 5.5 RunPod / MLOps

Отложено до явного решения пользователя и действительно тяжёлого облачного
обучения. Локальная RTX 4090 остаётся основной средой.

## 6. Порядок ближайших ML-действий

1. Сохранять generative ML в состоянии pause до отдельного решения.
2. После возобновления утвердить frozen protocol и бюджет transfer/scratch.
3. Выполнить `AFHQ Cats → cat / dog / wild`.
4. Зафиксировать Generative Training & Evaluation Playbook v1.
5. Реализовать img2img.
6. Проверить hires fix и sampler ablation.
7. Затем рассматривать RAE, native 256 и Generated Image Inspector.

## Как обновлять этот файл

После каждого ML-решения менять только:

1. статус этапа;
2. фактический результат и ссылку на evidence;
3. принятое решение;
4. следующий конкретный ML-шаг.

Сырые логи запусков сюда не копировать: они остаются в experiment ledger.
