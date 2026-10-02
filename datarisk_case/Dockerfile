# Utiliza imagem oficial Python slim para footprint otimizado
FROM python:3.12-slim

# Evita geração de arquivos .pyc e força unbuffered output para logs em tempo real
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Define diretório de trabalho no contêiner
WORKDIR /app

# Instala dependências do sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python primeiro (aproveitamento de cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia código da aplicação e diretório de artefatos
COPY . /app

# Executa script para gerar os artefatos caso não existam
RUN python kmeans_pipeline.py

# Expõe a porta da API
EXPOSE 8000

# Healthcheck interno do Docker
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicialização do servidor Uvicorn em produção
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
