# Federated Learning для CIFAR-100 и Google Landmarks v2

Проект демонстрирует федеративное обучение классификатора изображений на Flower.
Сервер запускает стратегию FedAvg с аудитом клиентских обновлений, клиенты обучают
локальные шарды данных, а после запуска сохраняются модель, метрики, CSV и графики.

## Что обучается

Основная модель находится в `for_fl/main_task.py`. Это компактная сверточная сеть:

- вход: RGB-изображение после `Resize`, `ToTensor`, ImageNet-normalization;
- блоки признаков: обычная стартовая свертка, depthwise separable convolutions,
  residual blocks, GroupNorm и SiLU;
- агрегация признаков: `AdaptiveAvgPool2d(1)`;
- embedding: `Linear -> LayerNorm -> SiLU`;
- выход: `Linear(embedding_dim, num_classes)`.

Модель решает задачу многоклассовой классификации изображений. В этом проекте она
используется как демонстрационный классификатор для:

- CIFAR-100: классы `0..99`;
- Google Landmarks v2: в смешанном режиме выбранные landmark-классы добавляются
  после CIFAR-100, то есть метки GLDv2 начинаются с `100`; в GLDv2-only режиме
  они начинаются с `0`;
- смешанного режима CIFAR-100 + GLDv2.

Для дипломной работы модель удобна не как state-of-the-art архитектура, а как
контролируемый экспериментальный стенд: можно сравнивать качество, loss, F1 и
устойчивость к gradient inversion при разных режимах защиты градиентов.

## Подготовка пустой Ubuntu/Debian машины

Ниже команды для полностью пустого сервера или клиента. Рекомендуемый вариант -
ставить Docker Engine и Docker Compose plugin из официального Docker apt-репозитория.

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg git lsb-release

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

Проверка:

```bash
docker --version
docker compose version
sudo docker run --rm hello-world
```

Чтобы запускать Docker без `sudo`:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker run --rm hello-world
```

Если apt-вариант Compose plugin недоступен, можно поставить standalone Compose
через `curl`. Тогда команды будут через `docker-compose`, а не `docker compose`.

```bash
COMPOSE_VERSION=$(curl -fsSL https://api.github.com/repos/docker/compose/releases/latest \
  | sed -n 's/.*"tag_name": "\(.*\)".*/\1/p' \
  | head -n 1)

sudo curl -SL \
  "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-$(uname -m)" \
  -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose
docker-compose version
```

Если используется GPU, дополнительно должен быть установлен NVIDIA driver и
NVIDIA Container Toolkit. Проверка:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## Установка локально без Docker

```bash
cd /home/hadder/Документы/Github/mag-main-hddr
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Проверка:

```bash
python -m compileall for_fl
flwr --help
```

## Датасеты

### CIFAR-100

CIFAR-100 скачивается автоматически через Hugging Face:

```python
load_dataset("uoft-cs/cifar100")
```

Отдельная ручная подготовка не нужна. Кэш будет находиться в Hugging Face cache.

### Google Landmarks v2

GLDv2 должен лежать локально. Ожидаемая структура:

```text
data/gld/
  train.csv
  train/
    a/
      b/
        c/
          abc....jpg
```

CSV должен иметь минимум поля:

```csv
id,landmark_id
abc123,12345
```

Путь к картинке строится как:

```text
gld-root / id[0] / id[1] / id[2] / id.jpg
```

Для дипломного эксперимента не стоит сразу брать все классы GLDv2. Практичный
вариант: `landmark-num-classes=1000` для основной серии и `5000` для расширенной.

## Быстрый локальный запуск CIFAR-100

```bash
flwr run . local-deployment --stream --run-config \
"num-server-rounds=3 fraction-train=0.5 fraction-evaluate=0.5 min-train-nodes=2 min-evaluate-nodes=2 min-available-nodes=2 batch-size=16 local-epochs=1 image-size=96 include-cifar100=true include-landmarks=false learning-rate=0.01 privacy-backend=manual_gradient_protection grad-noise-std=0.001 grad-clip-norm=1.0 save-client-updates=true"
```

## Рекомендуемый запуск CIFAR-100 для диплома

```bash
flwr run . local-deployment --stream --run-config \
"num-server-rounds=30 fraction-train=0.5 fraction-evaluate=0.5 min-train-nodes=10 min-evaluate-nodes=10 min-available-nodes=20 batch-size=32 local-epochs=1 image-size=96 include-cifar100=true include-landmarks=false learning-rate=0.01 privacy-backend=manual_gradient_protection grad-noise-std=0.001 grad-clip-norm=1.0 save-client-updates=true"
```

Базовые параметры:

```text
num-server-rounds=30
batch-size=16 или 32
local-epochs=1
image-size=96
learning-rate=0.01
fraction-train=0.5
fraction-evaluate=0.5
min-available-nodes=20
grad-clip-norm=1.0
grad-noise-std=0.001
```

Для `manual_gradient_protection` batch-size лучше держать умеренным: ручная
схема считает backward по каждому примеру внутри batch. Это медленнее обычного
обучения, но лучше демонстрирует архитектуру защиты градиентов.

## Рекомендуемый запуск GLDv2 для диплома

```bash
flwr run . local-deployment --stream --run-config \
"num-server-rounds=30 fraction-train=0.3 fraction-evaluate=0.2 min-train-nodes=10 min-evaluate-nodes=5 min-available-nodes=20 batch-size=16 local-epochs=1 image-size=128 include-cifar100=false include-landmarks=true landmark-num-classes=1000 gld-root=/home/hadder/Документы/Github/mag-main-hddr/data/gld/train gld-train-csv=/home/hadder/Документы/Github/mag-main-hddr/data/gld/train.csv gld-verify-files=true val-ratio=0.2 learning-rate=0.005 privacy-backend=manual_gradient_protection grad-noise-std=0.001 grad-clip-norm=1.0 save-client-updates=true"
```

Базовые параметры:

```text
num-server-rounds=30
batch-size=8 или 16
local-epochs=1
image-size=128
learning-rate=0.005
fraction-train=0.3
fraction-evaluate=0.2
landmark-num-classes=1000
grad-clip-norm=1.0
grad-noise-std=0.001
```

## Сравнение режимов защиты

Для наглядной дипломной таблицы запускай каждый датасет в трех режимах:

```text
privacy-backend=none
privacy-backend=manual_gradient_protection grad-clip-norm=1.0 grad-noise-std=0.001
privacy-backend=opacus opacus-noise-multiplier=1.0 opacus-delta=0.00001
```

Смысл сравнения:

- `none`: обычное обучение без дополнительной защиты;
- `manual_gradient_protection`: DP-SGD-подобная ручная схема. Для каждого
  примера в batch отдельно считается градиент, вклад каждого примера
  ограничивается через clipping, clipped gradients усредняются, после чего к
  итоговому градиенту добавляется гауссов шум перед `optimizer.step()`;
- `opacus`: DP-SGD через Opacus с per-sample gradients, clipping, noise и
  privacy accounting.

## Запуск на сервере через Docker Compose

На чистом сервере сначала установи Docker по разделу выше, затем клонируй проект:

```bash
git clone <URL_ТВОЕГО_РЕПОЗИТОРИЯ>
cd mag-main-hddr
```

CIFAR-100 скачивать вручную не нужно. Для GLDv2 заранее положи датасет на сервер
и на клиентские машины в одинаковую логическую структуру. Например:

```bash
mkdir -p /opt/datasets/gld
export GLD_HOST_DIR=/opt/datasets/gld
```

Внутри `/opt/datasets/gld` должны быть `train.csv` и каталог `train/`.

Собрать Docker-образ:

```bash
docker build -t for-fl:latest .
```

Запустить SuperLink на сервере:

```bash
export GLD_HOST_DIR=/opt/datasets/gld
docker compose -f docker-compose.server.yml up -d superlink
```

Открытые порты:

```text
9092 - подключение клиентских SuperNode
9093 - отправка Flower run
```

Логи:

```bash
docker compose -f docker-compose.server.yml logs -f superlink
```

После подключения клиентов запустить CIFAR-100:

```bash
NUM_SERVER_ROUNDS=30 \
FRACTION_TRAIN=0.5 \
FRACTION_EVALUATE=0.5 \
MIN_TRAIN_NODES=10 \
MIN_EVALUATE_NODES=10 \
MIN_AVAILABLE_NODES=20 \
BATCH_SIZE=32 \
LOCAL_EPOCHS=1 \
IMAGE_SIZE=96 \
INCLUDE_CIFAR100=true \
INCLUDE_LANDMARKS=false \
LEARNING_RATE=0.01 \
PRIVACY_BACKEND=manual_gradient_protection \
GRAD_NOISE_STD=0.001 \
GRAD_CLIP_NORM=1.0 \
docker compose -f docker-compose.server.yml --profile run up submit-run
```

Запустить GLDv2:

```bash
GLD_HOST_DIR=/absolute/path/to/data/gld \
NUM_SERVER_ROUNDS=30 \
FRACTION_TRAIN=0.3 \
FRACTION_EVALUATE=0.2 \
MIN_TRAIN_NODES=10 \
MIN_EVALUATE_NODES=5 \
MIN_AVAILABLE_NODES=20 \
BATCH_SIZE=16 \
LOCAL_EPOCHS=1 \
IMAGE_SIZE=128 \
INCLUDE_CIFAR100=false \
INCLUDE_LANDMARKS=true \
LANDMARK_NUM_CLASSES=1000 \
GLD_ROOT=/data/gld/train \
GLD_TRAIN_CSV=/data/gld/train.csv \
GLD_VERIFY_FILES=true \
LEARNING_RATE=0.005 \
PRIVACY_BACKEND=manual_gradient_protection \
GRAD_NOISE_STD=0.001 \
GRAD_CLIP_NORM=1.0 \
docker compose -f docker-compose.server.yml --profile run up submit-run
```

## Запуск клиентов

На каждой клиентской машине сначала установи Docker по разделу выше, затем:

```bash
git clone <URL_ТВОЕГО_РЕПОЗИТОРИЯ>
cd mag-main-hddr
docker build -t for-fl:latest .
```

Если используется GLDv2, положи датасет в такой же путь, как на сервере:

```bash
mkdir -p /opt/datasets/gld
export GLD_HOST_DIR=/opt/datasets/gld
```

На первой клиентской машине:

```bash
export SERVER_IP=<IP_СЕРВЕРА>
export VM_INDEX=0
export CLIENTS_PER_VM=10
export TOTAL_CLIENTS=20
export GLD_HOST_DIR=/opt/datasets/gld
docker compose -f docker-compose.clients.yml up -d
```

На второй клиентской машине:

```bash
export SERVER_IP=<IP_СЕРВЕРА>
export VM_INDEX=1
export CLIENTS_PER_VM=10
export TOTAL_CLIENTS=20
export GLD_HOST_DIR=/opt/datasets/gld
docker compose -f docker-compose.clients.yml up -d
```

Один `docker-compose.clients.yml` поднимает 10 SuperNode. Индекс клиента считается так:

```text
partition-id = VM_INDEX * CLIENTS_PER_VM + CLIENT_OFFSET
```

## Результаты

После запуска сохраняются:

```text
artifacts/models/final_model.pt
artifacts/metrics/<timestamp>-<privacy-backend>/metrics.json
artifacts/metrics/<timestamp>-<privacy-backend>/metrics.csv
artifacts/metrics/<timestamp>-<privacy-backend>/train_clientapp.csv
artifacts/metrics/<timestamp>-<privacy-backend>/evaluate_clientapp.csv
artifacts/metrics/<timestamp>-<privacy-backend>/evaluate_serverapp.csv
artifacts/metrics/<timestamp>-<privacy-backend>/plots/accuracy.png
artifacts/metrics/<timestamp>-<privacy-backend>/plots/loss.png
artifacts/metrics/<timestamp>-<privacy-backend>/plots/f1.png
artifacts/metrics/<timestamp>-<privacy-backend>/plots/overview.png
artifacts/updates/<privacy-backend>/round_001/
```

Если `matplotlib` не установлен, обучение не упадет: CSV/JSON сохранятся, а в
`metrics.json` появится `plot_warning`. При установке через `python -m pip install -e .`
или Docker графики будут доступны, потому что `matplotlib` указан в зависимостях.

Главные метрики для диплома:

```text
evaluate_serverapp.accuracy
evaluate_serverapp.loss
evaluate_serverapp.f1_macro
evaluate_serverapp.f1_weighted
evaluate_clientapp.eval_acc
evaluate_clientapp.eval_loss
evaluate_clientapp.eval_f1_macro
evaluate_clientapp.eval_f1_weighted
train_clientapp.train_loss
```

## Проверка gradient inversion

При `save-client-updates=true` сохраняются глобальные веса до раунда и веса
клиентов после локального обучения. Их можно использовать для демонстрации атаки:

```bash
python -m for_fl.sec_ops.attacks.gradient_inversion \
  --round-dir artifacts/updates/manual_gradient_protection/round_001 \
  --learning-rate 0.01 \
  --image-size 96 \
  --steps 300
```

Результат будет в:

```text
artifacts/attacks/<timestamp>-gradient-inversion/
```
