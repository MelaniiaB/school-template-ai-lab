FROM python:3.8-slim

# Установка ffmpeg и зависимостей
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Установка зависимостей Python
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Запуск приложения (замените main.py на вашу точку входа)
CMD python bot.py
