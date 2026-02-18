FROM mambaorg/micromamba:1.5.8

ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app

# Сначала копируем только окружение, чтобы кэшировать установку зависимостей
COPY --chown=micromamba:micromamba environment.yml /tmp/environment.yml
RUN micromamba create -y -n moodbot -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Копируем исходный код
COPY --chown=micromamba:micromamba . .

# Запуск бота внутри подготовленного окружения
CMD ["micromamba", "run", "-n", "moodbot", "python", "bot.py"]
