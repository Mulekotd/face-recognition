FROM python:3.13-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório da aplicação
WORKDIR /app

# Copiar arquivos da aplicação
COPY . .

# Instalar dependências do Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Comando para rodar a aplicação (ajuste conforme necessário)
CMD ["python", "main.py"]

