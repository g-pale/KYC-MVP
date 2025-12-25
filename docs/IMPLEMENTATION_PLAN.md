# Детальный план реализации KYC-сервиса (MVP)

## Общая оценка времени: 2-2.5 недели (команда из 2-3 разработчиков)

### Временные рамки по неделям:

**Неделя 1 (Дни 1-5):** Инфраструктура, загрузка файлов, обработка медиа
- Параллельная работа: Backend API + Azure настройка + Медиа обработка
- Результат: Работающий endpoint загрузки с предобработкой файлов

**Неделя 2 (Дни 6-14):** Интеграция Azure Face API, Celery, безопасность
- Параллельная работа: Azure интеграция + Celery + Безопасность + GDPR
- Результат: Полностью функциональный сервис с аутентификацией

**Неделя 3 (Дни 15-19):** Тестирование, документация, деплой
- Параллельная работа: Тесты + Документация + DevOps
- Результат: Готовый к production MVP

### Преимущества работы в команде:
- ✅ Параллельная разработка ускоряет процесс в 2-3 раза
- ✅ Code review повышает качество кода
- ✅ Разделение ответственности по экспертизе
- ✅ Возможность тестирования друг у друга

### Распределение ролей в команде

**Разработчик 1 (Backend Lead):**
- FastAPI приложение и API endpoints
- Интеграция с Azure Face API
- Celery и асинхронная обработка
- База данных и модели

**Разработчик 2 (Media & Security):**
- Обработка медиафайлов (изображения, видео)
- Безопасность и аутентификация
- Тестирование
- GDPR compliance

**Разработчик 3 (DevOps, опционально):**
- Docker и инфраструктура
- CI/CD pipeline
- Деплой на Azure
- Документация

### Координация команды

**Ежедневные стендапы (15 минут):**
- Что сделано вчера
- Что планируется сегодня
- Блокеры и зависимости

**Синхронизация работы:**
- Использование Git flow (feature branches)
- Code review для всех изменений
- Общие соглашения о коде (форматирование, naming)
- Общий канал для вопросов и обсуждений

**Критические точки синхронизации:**
- День 1: Согласование структуры проекта
- День 4: Интеграция медиа обработки с API
- День 6: Интеграция Azure Face API с обработкой медиа
- День 9: Полная интеграция всех компонентов
- День 14: Финальная интеграция безопасности и GDPR

---

## Неделя 1: Инфраструктура и базовая функциональность

### День 1: Настройка проекта и инфраструктуры (Параллельно)

#### Разработчик 1: Backend инфраструктура
1. **Инициализация проекта**
   - Создание структуры папок
   - Настройка виртуального окружения (poetry или venv)
   - Создание `requirements.txt` с базовыми зависимостями
   - Настройка `.env` файла и `.env.example`
   - Создание `.gitignore`

2. **Настройка FastAPI приложения**
   - Создание `app/main.py` с базовым FastAPI app
   - Настройка CORS
   - Настройка middleware для логирования
   - Базовая структура роутинга

3. **Настройка базы данных**
   - Установка SQLAlchemy и Alembic
   - Создание `app/database.py` с подключением к БД
   - Настройка Alembic для миграций
   - Создание первой миграции

4. **Создание модели Verification**
   ```python
   # app/models/verification.py
   from sqlalchemy import Column, String, DateTime, Enum
   from sqlalchemy.dialects.postgresql import UUID
   import uuid
   from app.database import Base
   
   class Verification(Base):
       __tablename__ = "verifications"
       
       id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
       status = Column(Enum('pending', 'processing', 'verified', 
                           'not_verified', 'manual_review', 'error'))
       document_path = Column(String)
       selfie_path = Column(String)
       metadata = Column(JSON)  # Для PostgreSQL
       created_at = Column(DateTime, default=datetime.utcnow)
       updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
   ```

#### Разработчик 2: Azure и медиа инфраструктура
5. **Настройка Azure**
   - Создание Azure Cognitive Services Face API ресурса
   - Получение ключей и endpoint
   - Создание Azure Blob Storage account
   - Создание контейнера с lifecycle policy

6. **Подготовка утилит для медиа**
   - Установка OpenCV, Pillow
   - Создание базовой структуры `app/utils/media_processing.py`
   - Изучение требований Azure Face API к изображениям

#### Разработчик 3 (если есть): DevOps подготовка
- Настройка Git репозитория
- Создание базового Dockerfile
- Подготовка docker-compose.yml структуры

**Результат:** Работающее FastAPI приложение с подключением к БД и Azure

---

### День 2-3: Загрузка файлов (Параллельно)

#### Разработчик 1: API и сервисы
1. **Создание Pydantic схем**
   ```python
   # app/schemas/verification.py
   from pydantic import BaseModel
   from uuid import UUID
   from datetime import datetime
   
   class VerificationCreate(BaseModel):
       pass  # Файлы приходят через multipart/form-data
   
   class VerificationResponse(BaseModel):
       verification_id: UUID
       status: str
       created_at: datetime
   ```

2. **Создание сервиса для работы с Blob Storage**
   ```python
   # app/services/blob_storage_service.py
   from azure.storage.blob import BlobServiceClient
   import os
   
   class BlobStorageService:
       def __init__(self):
           self.client = BlobServiceClient.from_connection_string(
               os.getenv("AZURE_STORAGE_CONNECTION_STRING")
           )
           self.container = "kyc-temp"
       
       async def upload_file(self, file_bytes: bytes, filename: str) -> str:
           blob_client = self.client.get_blob_client(
               container=self.container,
               blob=filename
           )
           blob_client.upload_blob(file_bytes, overwrite=True)
           return blob_client.url
       
       async def delete_file(self, blob_name: str):
           blob_client = self.client.get_blob_client(
               container=self.container,
               blob=blob_name
           )
           blob_client.delete_blob()
   ```

3. **Создание API endpoint для загрузки**
   ```python
   # app/api/verification.py
   from fastapi import APIRouter, UploadFile, File, Depends
   from app.services.verification_service import VerificationService
   
   router = APIRouter(prefix="/kyc/verification", tags=["KYC"])
   
   @router.post("", response_model=VerificationResponse)
   async def create_verification(
       document: UploadFile = File(...),
       selfie: UploadFile = File(...),
       service: VerificationService = Depends()
   ):
       # Валидация файлов
       # Сохранение в Blob Storage
       # Создание записи в БД
       # Запуск background task
       pass
   ```

#### Разработчик 2: Валидация и обработка медиа
4. **Валидация файлов**
   - Проверка типа файла (image/jpeg, image/png, video/mp4)
   - Проверка размера (максимум 10MB для изображений, 100MB для видео)
   - Базовая проверка формата
   - Создание валидаторов в `app/utils/validators.py`

5. **Начало работы над обработкой медиа**
   - Изучение требований Azure Face API
   - Подготовка структуры для обработки изображений

**Результат:** Работающий endpoint для загрузки файлов с сохранением в Azure Blob Storage

---

### День 4: Обработка медиа

#### Разработчик 2: Медиа обработка
1. **Создание утилит для обработки изображений**
   ```python
   # app/utils/media_processing.py
   from PIL import Image
   import io
   
   def preprocess_image(image_bytes: bytes) -> bytes:
       """Предобработка изображения для Azure Face API"""
       img = Image.open(io.BytesIO(image_bytes))
       
       # Ресайз если нужно
       max_size = (1920, 1920)
       img.thumbnail(max_size, Image.Resampling.LANCZOS)
       
       # Конвертация в RGB
       if img.mode != 'RGB':
           img = img.convert('RGB')
       
       output = io.BytesIO()
       img.save(output, format='JPEG', quality=85)
       return output.getvalue()
   ```

2. **Создание утилит для обработки видео**
   ```python
   import cv2
   
   def extract_frames(video_bytes: bytes, max_frames: int = 3) -> list[bytes]:
       """Извлечение кадров из видео"""
       # Сохранение во временный файл
       # Извлечение кадров через OpenCV
       # Возврат списка байтов изображений
       pass
   ```

#### Разработчик 1: Интеграция
3. **Интеграция в сервис верификации**
   - Создание `VerificationService`
   - Интеграция медиа обработки
   - Тестирование полного потока загрузки

**Результат:** Функционал предобработки медиафайлов

---

## Неделя 2: Интеграция с Azure Face API и асинхронная обработка

### День 5-6: Интеграция с Azure Face API (Параллельно)

#### Разработчик 1: Azure Face API интеграция
1. **Создание сервиса для работы с Azure Face API**
   ```python
   # app/services/azure_face_service.py
   import aiohttp
   import os
   
   class AzureFaceService:
       def __init__(self):
           self.endpoint = os.getenv("AZURE_FACE_ENDPOINT")
           self.key = os.getenv("AZURE_FACE_KEY")
           self.detect_url = f"{self.endpoint}/face/v1.0/detect"
           self.verify_url = f"{self.endpoint}/face/v1.0/verify"
       
       async def detect_face(self, image_bytes: bytes) -> dict:
           """Обнаружение лица на изображении"""
           headers = {
               "Ocp-Apim-Subscription-Key": self.key,
               "Content-Type": "application/octet-stream"
           }
           
           async with aiohttp.ClientSession() as session:
               async with session.post(
                   self.detect_url,
                   headers=headers,
                   data=image_bytes,
                   params={"returnFaceId": "true"}
               ) as response:
                   if response.status == 200:
                       faces = await response.json()
                       if len(faces) == 0:
                           raise ValueError("No face detected")
                       if len(faces) > 1:
                           raise ValueError("Multiple faces detected")
                       return faces[0]["faceId"]
                   else:
                       raise Exception(f"Azure API error: {response.status}")
       
       async def verify_faces(self, face_id1: str, face_id2: str) -> dict:
           """Сравнение двух лиц"""
           headers = {
               "Ocp-Apim-Subscription-Key": self.key,
               "Content-Type": "application/json"
           }
           
           data = {
               "faceId1": face_id1,
               "faceId2": face_id2
           }
           
           async with aiohttp.ClientSession() as session:
               async with session.post(
                   self.verify_url,
                   headers=headers,
                   json=data
               ) as response:
                   if response.status == 200:
                       return await response.json()
                   else:
                       raise Exception(f"Azure API error: {response.status}")
   ```

2. **Обработка ошибок Azure API**
   - Таймауты
   - Rate limiting
   - Недоступность сервиса
   - Некорректные ответы

3. **Логика верификации**
   - Detect для документа
   - Detect для селфи/видео
   - Verify для сравнения
   - Интерпретация результата (confidence threshold)

#### Разработчик 2: Тестирование интеграции
- Создание тестовых изображений
- Тестирование Azure Face API с реальными данными
- Документирование edge cases

**Результат:** Работающая интеграция с Azure Face API

---

### День 7-8: Настройка Celery и асинхронная обработка (Параллельно)

#### Разработчик 1: Celery настройка

#### Задачи:
1. **Настройка Celery**
   ```python
   # app/celery_app.py
   from celery import Celery
   import os
   
   celery_app = Celery(
       "kyc_service",
       broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
       backend=os.getenv("REDIS_URL", "redis://localhost:6379/0")
   )
   
   celery_app.conf.update(
       task_serializer='json',
       accept_content=['json'],
       result_serializer='json',
       timezone='UTC',
       enable_utc=True,
   )
   ```

2. **Создание Celery task**
   ```python
   # app/tasks/verification_tasks.py
   from app.celery_app import celery_app
   from app.services.verification_service import VerificationService
   from app.services.azure_face_service import AzureFaceService
   from app.services.blob_storage_service import BlobStorageService
   
   @celery_app.task(bind=True, max_retries=3)
   def process_verification(self, verification_id: str):
       """Асинхронная обработка верификации"""
       try:
           # Загрузка данных из БД
           # Загрузка файлов из Blob Storage
           # Обработка медиа
           # Вызов Azure Face API
           # Обновление статуса в БД
           # Очистка файлов
           pass
       except Exception as exc:
           # Retry с exponential backoff
           raise self.retry(exc=exc, countdown=2 ** self.request.retries)
   ```

3. **Интеграция с FastAPI**
   - Запуск task после создания verification
   - Обновление статуса в реальном времени

#### Разработчик 2: Redis и тестирование
4. **Настройка Redis**
   - Установка и настройка Redis
   - Тестирование очереди
   - Создание тестов для Celery tasks

#### Разработчик 3 (если есть): Docker для Celery
- Настройка Docker Compose для Celery worker
- Оптимизация конфигурации

**Результат:** Асинхронная обработка верификаций через Celery

---

### День 9: Endpoint для получения статуса и интеграция

#### Задачи:
1. **Создание endpoint GET /kyc/verification/{id}**
   ```python
   @router.get("/{verification_id}", response_model=VerificationResponse)
   async def get_verification_status(
       verification_id: UUID,
       service: VerificationService = Depends()
   ):
       verification = await service.get_verification(verification_id)
       if not verification:
           raise HTTPException(status_code=404, detail="Verification not found")
       return verification
   ```

2. **Оптимизация запросов**
   - Кэширование статусов в Redis
   - Индексы в БД

**Результат:** Работающий endpoint для проверки статуса

---

## Неделя 2-3: Безопасность, обработка ошибок и тестирование

### День 10-11: Безопасность и аутентификация (Параллельно)

#### Разработчик 2: Безопасность
1. **Реализация JWT аутентификации**
   ```python
   # app/utils/security.py
   from jose import JWTError, jwt
   from datetime import datetime, timedelta
   
   def create_access_token(data: dict):
       to_encode = data.copy()
       expire = datetime.utcnow() + timedelta(minutes=30)
       to_encode.update({"exp": expire})
       return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
   
   def verify_token(token: str):
       try:
           payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
           return payload
       except JWTError:
           return None
   ```

2. **Middleware для проверки токенов**
   ```python
   # app/middleware/auth.py
   from fastapi import HTTPException, Depends
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   async def get_current_user(token: str = Depends(security)):
       payload = verify_token(token.credentials)
       if payload is None:
           raise HTTPException(status_code=401, detail="Invalid token")
       return payload
   ```

3. **Rate limiting**
   - Установка slowapi
   - Настройка лимитов для endpoints

4. **Валидация входных данных**
   - Строгая валидация всех параметров
   - Защита от injection атак

#### Разработчик 1: Интеграция безопасности
- Добавление middleware в FastAPI
- Защита всех endpoints
- Тестирование аутентификации

**Результат:** Защищённое API с аутентификацией

---

### День 12-13: Обработка ошибок и статусы (Параллельно)

#### Разработчик 1: Обработка ошибок

#### Задачи:
1. **Расширенная обработка ошибок**
   ```python
   # app/exceptions.py
   class VerificationError(Exception):
       pass
   
   class FaceDetectionError(VerificationError):
       pass
   
   class AzureAPIError(VerificationError):
       pass
   
   # app/main.py
   @app.exception_handler(VerificationError)
   async def verification_error_handler(request, exc):
       return JSONResponse(
           status_code=400,
           content={"detail": str(exc)}
       )
   ```

2. **Логика определения статусов**
   - `pending`: Создана, ожидает обработки
   - `processing`: В процессе обработки
   - `verified`: Успешно верифицирован (confidence > 0.7)
   - `not_verified`: Не прошёл верификацию (confidence < 0.5)
   - `manual_review`: Требует ручной проверки (0.5 <= confidence <= 0.7 или ошибки)
   - `error`: Техническая ошибка

3. **Retry механизм**
   - Автоматические повторы для временных ошибок
   - Exponential backoff
   - Максимальное количество попыток

#### Разработчик 2: Логирование и мониторинг
4. **Логирование**
   - Structured logging
   - Логирование всех операций с PII
   - Интеграция с Sentry (опционально)
   - Настройка базового мониторинга

**Результат:** Надёжная обработка ошибок и корректные статусы

---

### День 14: GDPR compliance и очистка данных (Параллельно)

#### Разработчик 2: GDPR compliance

#### Задачи:
1. **Автоматическое удаление файлов**
   ```python
   # app/services/cleanup_service.py
   async def cleanup_after_verification(verification_id: str):
       """Удаление файлов после завершения верификации"""
       verification = await db.get_verification(verification_id)
       
       # Удаление из Blob Storage
       await blob_storage.delete(verification.document_path)
       await blob_storage.delete(verification.selfie_path)
       
       # Обновление записи (удаление путей к файлам)
       verification.document_path = None
       verification.selfie_path = None
       await db.commit()
   ```

2. **Endpoint для удаления данных**
   ```python
   @router.delete("/{verification_id}")
   async def delete_verification_data(
       verification_id: UUID,
       service: VerificationService = Depends()
   ):
       """Удаление всех данных верификации (GDPR)"""
       await service.delete_verification(verification_id)
       return {"status": "deleted"}
   ```

3. **Политика хранения метаданных**
   - Настройка TTL для метаданных в БД
   - Периодическая очистка старых данных

#### Разработчик 1: Интеграция cleanup
- Интеграция cleanup в Celery tasks
- Тестирование автоматического удаления

**Результат:** Соответствие требованиям GDPR

---

## Неделя 3: Тестирование, документация и деплой

### День 15-16: Тестирование (Параллельно)

#### Разработчик 1: Backend тесты
1. **Unit тесты**
   - Тесты сервисов
   - Тесты утилит
   - Моки для Azure API

2. **Integration тесты**
   - Тесты API endpoints
   - Тесты Celery tasks
   - Тесты с тестовой БД

#### Разработчик 2: E2E и нагрузочное тестирование
3. **E2E тесты**
   - Полный цикл верификации
   - Тестирование с реальными изображениями

4. **Нагрузочное тестирование**
   - Использование locust или k6
   - Проверка производительности

**Результат:** Покрытие тестами > 70%

---

### День 17: Документация (Параллельно)

#### Разработчик 1: API документация
1. **API документация**
   - Swagger UI (автоматически через FastAPI)
   - ReDoc
   - Примеры запросов
   - Описание ошибок

#### Разработчик 2: README и документация кода
2. **README**
   - Установка и настройка
   - Переменные окружения
   - Запуск локально
   - Запуск в Docker

3. **Документация кода**
   - Docstrings для всех функций и классов
   - Type hints везде

#### Разработчик 3 (если есть): Техническая документация
- Архитектурные диаграммы
- Описание деплоя

**Результат:** Полная документация проекта

---

### День 18-19: Деплой и CI/CD (Параллельно)

#### Разработчик 3 (если есть): DevOps
1. **Docker образ**
   - Создание Dockerfile
   - Оптимизация размера образа (multi-stage build)
   - Тестирование образа локально

2. **Docker Compose**
   - Настройка для локальной разработки
   - Включение всех сервисов (API, Celery, PostgreSQL, Redis)

3. **CI/CD pipeline**
   - GitHub Actions или Azure DevOps
   - Автоматические тесты
   - Автоматический деплой

4. **Деплой на Azure**
   - Настройка Azure App Service
   - Настройка Azure Database for PostgreSQL
   - Настройка Azure Cache for Redis
   - Настройка переменных окружения

#### Разработчик 1 и 2: Поддержка деплоя
- Тестирование на staging окружении
- Исправление проблем
- Финальная проверка функционала

**Результат:** Работающее приложение в production

---

## Чеклист готовности MVP

### Функциональность
- [ ] POST /kyc/verification - загрузка файлов
- [ ] GET /kyc/verification/{id} - получение статуса
- [ ] Обработка изображений и видео
- [ ] Интеграция с Azure Face API
- [ ] Асинхронная обработка через Celery
- [ ] Корректные статусы верификации

### Безопасность
- [ ] JWT аутентификация
- [ ] HTTPS только
- [ ] Rate limiting
- [ ] Валидация входных данных
- [ ] Секреты в переменных окружения / Key Vault

### Соответствие требованиям
- [ ] Автоматическое удаление файлов
- [ ] GDPR compliance
- [ ] Логирование операций
- [ ] Endpoint для удаления данных

### Качество
- [ ] Покрытие тестами > 70%
- [ ] Документация API
- [ ] README с инструкциями
- [ ] Обработка ошибок

### Инфраструктура
- [ ] Docker образ
- [ ] Docker Compose для разработки
- [ ] CI/CD pipeline
- [ ] Деплой на Azure

---

## Риски и митигация

### Технические риски

1. **Ограничения Azure Face API Free tier**
   - **Митигация**: Мониторинг использования, готовность к переходу на платный тариф

2. **Проблемы с качеством изображений**
   - **Митигация**: Предварительная валидация, улучшение качества через OpenCV

3. **Таймауты Azure API**
   - **Митигация**: Retry механизм, fallback на manual_review

### Временные риски

1. **Задержки в интеграции с Azure**
   - **Митигация**: Раннее начало работы с Azure, тестирование на ранних этапах

2. **Сложности с обработкой видео**
   - **Митигация**: Начать с простых случаев, постепенно усложнять

---

## Следующие шаги после MVP

1. **Мониторинг и метрики**
   - Настройка Application Insights
   - Дашборды в Grafana
   - Алерты на ошибки

2. **Оптимизация производительности**
   - Кэширование результатов
   - Оптимизация запросов к БД
   - Connection pooling

3. **Расширение функционала**
   - Liveness detection
   - Document OCR
   - Webhook notifications
   - Admin panel

4. **Масштабирование**
   - Горизонтальное масштабирование
   - Load balancing
   - Database read replicas

---

## Сводная таблица распределения задач

| День | Разработчик 1 (Backend) | Разработчик 2 (Media & Security) | Разработчик 3 (DevOps) |
|------|------------------------|----------------------------------|------------------------|
| 1 | FastAPI, БД, модели | Azure настройка, медиа утилиты | Git, Docker структура |
| 2-3 | API endpoints, Blob Storage | Валидация файлов, медиа обработка | - |
| 4 | Интеграция сервисов | Обработка изображений/видео | - |
| 5-6 | Azure Face API интеграция | Тестирование Azure API | - |
| 7-8 | Celery настройка, tasks | Redis, тесты Celery | Docker Compose |
| 9 | Endpoint статуса | - | - |
| 10-11 | Интеграция безопасности | JWT, Rate limiting | - |
| 12-13 | Обработка ошибок | Логирование, мониторинг | - |
| 14 | Cleanup интеграция | GDPR compliance | - |
| 15-16 | Unit/Integration тесты | E2E, нагрузочные тесты | - |
| 17 | API документация | README, docstrings | Техническая документация |
| 18-19 | Тестирование деплоя | Тестирование деплоя | CI/CD, Azure деплой |

### Заметки:
- **Разработчик 1** фокусируется на основной бизнес-логике и интеграциях
- **Разработчик 2** отвечает за медиа, безопасность и тестирование
- **Разработчик 3** (если есть) занимается инфраструктурой и деплоем
- Все разработчики участвуют в code review и тестировании

