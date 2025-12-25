# Рекомендации по современным технологиям и сервисам для KYC-сервиса

## 1. Современные альтернативы и улучшения

### 1.1 Backend Framework - Детальное сравнение

#### FastAPI (Рекомендуется) ⭐
**Почему выбираем:**
- **Асинхронность из коробки**: Нативная поддержка async/await, что критично для I/O операций (загрузка файлов, вызовы Azure API)
- **Производительность**: Один из самых быстрых Python фреймворков (сопоставим с Go и Node.js)
- **Автодокументация**: OpenAPI/Swagger генерируется автоматически
- **Type hints**: Полная поддержка типизации через Pydantic
- **Простота**: Минимальный boilerplate код

**Современные практики:**
```python
# Пример современной структуры с dependency injection
from fastapi import Depends, UploadFile
from app.services.verification_service import VerificationService

@app.post("/kyc/verification")
async def create_verification(
    document: UploadFile,
    selfie: UploadFile,
    service: VerificationService = Depends(get_verification_service)
):
    return await service.create_verification(document, selfie)
```

#### Django REST Framework (Альтернатива)
**Когда использовать:**
- Команда уже знакома с Django
- Нужен встроенный admin panel
- Требуется много готовых пакетов (django-allauth, django-filter)

**Недостатки:**
- Меньшая производительность для async операций
- Больше boilerplate кода
- Сложнее настройка для чисто async приложений

### 1.2 База данных - PostgreSQL vs MySQL

#### PostgreSQL (Сильно рекомендуется) ⭐
**Современные возможности:**
- **JSONB**: Нативная поддержка JSON с индексацией (для метаданных)
- **UUID**: Встроенный тип UUID (для verification_id)
- **Full-text search**: Поиск по текстовым полям
- **Array types**: Хранение массивов без денормализации
- **Better concurrency**: MVCC (Multi-Version Concurrency Control)

**Пример схемы:**
```sql
CREATE TABLE verifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(20) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_verifications_status ON verifications(status);
CREATE INDEX idx_verifications_metadata ON verifications USING GIN(metadata);
```

#### MySQL 8.0+ (Альтернатива)
**Когда использовать:**
- Уже есть инфраструктура MySQL
- Команда более знакома с MySQL
- Нужна совместимость с legacy системами

**Ограничения:**
- Меньше возможностей для JSON (нет JSONB)
- UUID требует строкового типа
- Меньше возможностей для сложных запросов

### 1.3 Очередь задач - Современные подходы

#### Celery + Redis (Классический подход) ⭐
**Преимущества:**
- Проверенное решение
- Много возможностей (retry, scheduling, monitoring)
- Хорошая документация

**Современная конфигурация:**
```python
# celery_config.py
from celery import Celery
from kombu import Queue

app = Celery('kyc_service')
app.config_from_object('celeryconfig')

# Использование Redis как брокера и результата
app.conf.broker_url = 'redis://localhost:6379/0'
app.conf.result_backend = 'redis://localhost:6379/0'

# Настройка очередей
app.conf.task_queues = (
    Queue('verification', routing_key='verification.#'),
    Queue('media_processing', routing_key='media.#'),
)
```

#### RQ (Redis Queue) - Легковесная альтернатива
**Когда использовать:**
- Простые задачи без сложной логики
- Не нужен scheduling
- Хочется простоты

**Пример:**
```python
from rq import Queue
from redis import Redis

redis_conn = Redis()
q = Queue('verification', connection=redis_conn)

job = q.enqueue(process_verification, verification_id)
```

#### FastAPI BackgroundTasks - Для MVP
**Когда использовать:**
- Очень простой MVP
- Низкая нагрузка (< 100 запросов/час)
- Не нужен retry механизм

**Ограничения:**
- Нет персистентности (при перезапуске задачи теряются)
- Нет распределения нагрузки
- Нет мониторинга

### 1.4 Хранилище файлов - Современные подходы

#### Azure Blob Storage (Рекомендуется) ⭐
**Современные возможности:**
- **Lifecycle Management**: Автоматическое удаление через политики
- **Soft Delete**: Восстановление удалённых файлов
- **Immutability Policy**: Защита от изменения/удаления (для compliance)
- **Access Tiers**: Hot/Cool/Archive для оптимизации стоимости
- **Private Endpoints**: Изоляция трафика от интернета

**Настройка для GDPR:**
```python
# Создание контейнера с lifecycle policy
from azure.storage.blob import BlobServiceClient

# Lifecycle policy через Azure Portal или ARM template
# Автоматическое удаление файлов через 7 дней
lifecycle_policy = {
    "rules": [{
        "name": "DeleteOldFiles",
        "enabled": True,
        "type": "Lifecycle",
        "definition": {
            "filters": {
                "blobTypes": ["blockBlob"]
            },
            "actions": {
                "baseBlob": {
                    "delete": {"daysAfterModificationGreaterThan": 7}
                }
            }
        }
    }]
}
```

#### MinIO - Self-hosted альтернатива
**Когда использовать:**
- Нужен полный контроль над данными
- Требования к локализации данных
- S3-совместимое API

#### AWS S3 - Если не привязаны к Azure
- Более зрелая экосистема
- Лучшая интеграция с другими AWS сервисами
- Glacier для долгосрочного хранения

### 1.5 Обработка медиа - Современные библиотеки

#### OpenCV (cv2) ⭐
**Для извлечения кадров из видео:**
```python
import cv2

def extract_frames(video_path, max_frames=3):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Выбираем равномерно распределённые кадры
    indices = [int(frame_count * i / (max_frames + 1)) 
               for i in range(1, max_frames + 1)]
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames
```

#### Pillow (PIL) - Для работы с изображениями
**Предобработка для Azure Face API:**
```python
from PIL import Image
import io

def preprocess_image(image_bytes, max_size=(1920, 1920)):
    """Ресайз изображения для соответствия требованиям Azure"""
    img = Image.open(io.BytesIO(image_bytes))
    
    # Azure Face API требования:
    # - Минимум 36x36 пикселей
    # - Максимум 4096x4096 пикселей
    # - Размер файла до 6MB
    
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Конвертация в RGB если нужно
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=85, optimize=True)
    return output.getvalue()
```

#### FFmpeg - Альтернатива для видео
**Через python-ffmpeg:**
```python
import ffmpeg

def extract_frames_ffmpeg(video_path, output_dir, count=3):
    probe = ffmpeg.probe(video_path)
    duration = float(probe['streams'][0]['duration'])
    
    for i in range(count):
        timestamp = duration * (i + 1) / (count + 1)
        (
            ffmpeg
            .input(video_path, ss=timestamp)
            .output(f'{output_dir}/frame_{i}.jpg', vframes=1)
            .run(overwrite_output=True)
        )
```

### 1.6 Безопасность - Современные практики

#### JWT Authentication
**Использование python-jose:**
```python
from jose import JWTError, jwt
from datetime import datetime, timedelta

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
```

#### Azure Key Vault - Управление секретами
**Интеграция:**
```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://your-vault.vault.azure.net/", 
                      credential=credential)

# Получение секрета
azure_face_key = client.get_secret("azure-face-api-key").value
```

#### Rate Limiting
**Использование slowapi:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/kyc/verification")
@limiter.limit("10/minute")
async def create_verification(request: Request, ...):
    ...
```

### 1.7 Мониторинг и логирование

#### Sentry - Отслеживание ошибок
```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.celery import CeleryIntegration

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[
        FastApiIntegration(),
        CeleryIntegration(),
    ],
    traces_sample_rate=0.1,  # 10% транзакций для performance monitoring
    environment=os.getenv("ENVIRONMENT", "development"),
)
```

#### Prometheus + Grafana
**Экспорт метрик:**
```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

verification_requests = Counter(
    'verification_requests_total',
    'Total verification requests',
    ['status']
)

verification_duration = Histogram(
    'verification_duration_seconds',
    'Verification processing duration'
)

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

#### Structured Logging
**Использование structlog:**
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "verification_created",
    verification_id=verification_id,
    user_id=user_id,
    document_type="passport"
)
```

### 1.8 Тестирование - Современные практики

#### Pytest с async поддержкой
```python
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_create_verification():
    async with AsyncClient(app=app, base_url="http://test") as client:
        files = {
            "document": ("passport.jpg", open("test_passport.jpg", "rb")),
            "selfie": ("selfie.jpg", open("test_selfie.jpg", "rb"))
        }
        response = await client.post("/kyc/verification", files=files)
        assert response.status_code == 200
        assert "verification_id" in response.json()
```

#### Моки для Azure API
```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_verification_success():
    with patch('app.services.azure_face_service.AzureFaceService.detect_face') as mock_detect:
        mock_detect.return_value = {"faceId": "test-face-id"}
        
        result = await verification_service.verify(...)
        assert result["status"] == "verified"
```

### 1.9 DevOps и инфраструктура

#### Docker Multi-stage Build
```dockerfile
# Dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose для разработки
```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/kyc
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: kyc
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
  
  celery:
    build: .
    command: celery -A app.celery_app worker --loglevel=info
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
```

#### GitHub Actions CI/CD
```yaml
# .github/workflows/ci.yml
name: CI/CD
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      - uses: azure/docker-login@v1
        with:
          login-server: ${{ secrets.ACR_SERVER }}
          username: ${{ secrets.ACR_USERNAME }}
          password: ${{ secrets.ACR_PASSWORD }}
      - run: |
          docker build -t ${{ secrets.ACR_SERVER }}/kyc-service:${{ github.sha }} .
          docker push ${{ secrets.ACR_SERVER }}/kyc-service:${{ github.sha }}
```

## 2. Современные альтернативы Azure Face API

### 2.1 Сравнение сервисов распознавания лиц

#### Azure Face API ⭐ (Текущий выбор)
**Преимущества:**
- Интеграция с Azure экосистемой
- Хорошая документация
- Поддержка liveness detection (в некоторых регионах)

**Ограничения:**
- Free tier: 20 транзакций/минуту, 30K транзакций/месяц
- Размер изображения до 6MB
- Требования к качеству изображения

#### AWS Rekognition
**Преимущества:**
- Более щедрый free tier (5000 изображений/месяц)
- Лучшая производительность для больших объёмов
- Интеграция с другими AWS сервисами

**Недостатки:**
- Если уже используете Azure, сложнее интегрировать

#### Google Cloud Vision API
**Преимущества:**
- Очень точное распознавание
- Хорошая поддержка различных форматов
- Интеграция с ML моделями

**Недостатки:**
- Может быть дороже для больших объёмов

#### Face++ (Megvii)
**Преимущества:**
- Очень точное распознавание
- Хорошая поддержка азиатских лиц
- Конкурентные цены

**Недостатки:**
- Меньше документации на английском
- Меньше интеграций

### 2.2 Self-hosted решения (для будущего)

#### DeepFace
**Когда использовать:**
- Нужен полный контроль над данными
- Требования к локализации
- Большие объёмы (может быть дешевле)

**Пример:**
```python
from deepface import DeepFace

result = DeepFace.verify(
    img1_path="document.jpg",
    img2_path="selfie.jpg",
    model_name="VGG-Face"
)
```

#### InsightFace
- Open-source решение
- Очень высокая точность
- Требует больше ресурсов для развёртывания

## 3. Рекомендации по архитектуре для масштабирования

### 3.1 Микросервисная архитектура (будущее)

```
┌─────────────┐
│  API Gateway│
└──────┬──────┘
       │
   ┌───┴───┬──────────┬──────────┐
   │       │          │          │
┌──▼──┐ ┌──▼──┐  ┌───▼───┐  ┌───▼────┐
│ KYC │ │Media│  │Face   │  │Storage │
│ API │ │Proc │  │Verify │  │Service │
└─────┘ └─────┘  └───────┘  └────────┘
```

**Преимущества:**
- Независимое масштабирование
- Изоляция ошибок
- Разные технологии для разных сервисов

**Когда переходить:**
- Когда нагрузка > 1000 запросов/минуту
- Когда нужна независимая разработка команд
- Когда нужна разная SLA для разных компонентов

### 3.2 Event-Driven Architecture

**Использование Kafka или Azure Service Bus:**
```python
# Публикация события
from azure.servicebus import ServiceBusClient, ServiceBusMessage

client = ServiceBusClient.from_connection_string(conn_str)
sender = client.get_queue_sender(queue_name="verification-events")

message = ServiceBusMessage(json.dumps({
    "event_type": "verification.completed",
    "verification_id": verification_id,
    "status": "verified"
}))
sender.send_messages(message)
```

**Преимущества:**
- Слабая связанность сервисов
- Возможность replay событий
- Легче добавлять новые обработчики

## 4. Оптимизация производительности

### 4.1 Кэширование

**Redis для кэширования результатов:**
```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=1)

def get_verification_status(verification_id: str):
    cache_key = f"verification:{verification_id}"
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Загрузка из БД
    result = db.get_verification(verification_id)
    
    # Кэширование на 5 минут
    redis_client.setex(cache_key, 300, json.dumps(result))
    return result
```

### 4.2 Connection Pooling

**Для БД:**
```python
from sqlalchemy.pool import QueuePool

engine = create_async_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
```

**Для Azure API:**
```python
import aiohttp

async with aiohttp.ClientSession() as session:
    # Переиспользование сессии для всех запросов
    async with session.post(url, json=data) as response:
        return await response.json()
```

### 4.3 Batch Processing

**Обработка нескольких верификаций одновременно:**
```python
import asyncio

async def process_batch(verification_ids: list[str]):
    tasks = [process_verification(vid) for vid in verification_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

## 5. Соответствие требованиям и compliance

### 5.1 GDPR Compliance

**Автоматическое удаление данных:**
```python
from datetime import datetime, timedelta

async def cleanup_old_data():
    """Удаление данных старше 90 дней"""
    cutoff_date = datetime.utcnow() - timedelta(days=90)
    
    old_verifications = await db.query(Verification).filter(
        Verification.created_at < cutoff_date
    ).all()
    
    for verification in old_verifications:
        # Удаление файлов из Blob Storage
        await blob_storage.delete(verification.document_path)
        await blob_storage.delete(verification.selfie_path)
        
        # Удаление записи из БД
        await db.delete(verification)
```

**Право на удаление (Right to be Forgotten):**
```python
@app.delete("/kyc/verification/{verification_id}")
async def delete_verification(verification_id: str):
    """Удаление всех данных пользователя по запросу"""
    verification = await db.get_verification(verification_id)
    
    # Удаление файлов
    await blob_storage.delete(verification.document_path)
    await blob_storage.delete(verification.selfie_path)
    
    # Анонимизация или удаление записи
    await db.delete(verification)
    
    return {"status": "deleted"}
```

### 5.2 Audit Logging

**Логирование всех операций с PII:**
```python
from app.models.audit_log import AuditLog

async def log_audit_event(
    event_type: str,
    verification_id: str,
    user_id: str,
    details: dict
):
    await db.create(AuditLog(
        event_type=event_type,
        verification_id=verification_id,
        user_id=user_id,
        details=details,
        timestamp=datetime.utcnow(),
        ip_address=request.client.host
    ))
```

## 6. Заключение

Данные рекомендации основаны на:
- ✅ Современных best practices
- ✅ Опыте разработки подобных систем
- ✅ Требованиях к безопасности и compliance
- ✅ Готовности к масштабированию

Выбор конкретных технологий должен основываться на:
- Опыте команды
- Бюджете проекта
- Требованиях к локализации данных
- Планах масштабирования

