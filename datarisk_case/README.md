# Datarisk Credit Risk Case - Refatoração & Arquitetura de Produção

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.8522-success.svg)](#métricas-de-performance)

Este repositório contém a arquitetura refatorada e pronta para produção de um modelo de Risco de Crédito. O projeto prevê a probabilidade de inadimplência de clientes a partir do histórico de pagamentos e dados cadastrais, disponibilizando o modelo preditivo através de uma API REST em contêiner Docker.

## Arquitetura da Solução

```mermaid
flowchart TD
    subgraph Cloud["1. Ingestão AWS S3"]
        S3[("AWS S3 Bucket\n(base_pagamentos, base_info, base_cadastral)")]
    end

    subgraph Pipeline["2. Pipeline ML (Sem Data Leakage)"]
        Ingest["Leitura e Agregação"]
        Split["Divisão Treino / Teste (Stratified)"]
        Scaler["StandardScaler (Ajustado no Treino)"]
        KMeans["K-Means Clustering (n_clusters=4)"]
        GB["Gradient Boosting Classifier"]
        
        Ingest --> Split
        Split --> Scaler
        Scaler --> KMeans
        KMeans -->|Feature CLUSTER_CLIENTE| GB
    end

    subgraph Artifacts["3. Artefatos de Produção"]
        Pkls["artifacts/\n├── encoder.pkl\n├── scaler.pkl\n├── kmeans.pkl\n└── modelo_gb.pkl"]
    end

    subgraph Deploy["4. Produção (Docker e FastAPI)"]
        API["FastAPI (main.py)"]
        Docker["Docker Container (Porta 8000)"]
    end

    S3 --> Ingest
    GB --> Pkls
    Pkls --> API
    API --> Docker
