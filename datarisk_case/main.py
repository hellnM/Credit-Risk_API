"""
main.py - API REST em FastAPI para Inferência do Modelo de Risco de Crédito

Carrega os artefatos serializados (encoder.pkl, scaler.pkl, kmeans.pkl, modelo_gb.pkl),
trata dados desconhecidos, calcula a clusterização dinamicamente e expõe endpoints HTTP.
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Caminho para o diretório de artefatos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Variáveis globais para armazenar os artefatos em memória
artifacts: Dict[str, Any] = {}

# Mapeamento de colunas esperadas
NUMERIC_COLS = ["RENDA_MENSAL", "IDADE", "QTD_PARCELAS", "VALOR_PARCELA_MEDIO", "ATRASO_MEDIO", "ATRASO_MAX"]
CATEGORICAL_COLS = ["TIPO_PESSOA"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação, carregando os artefatos na inicialização.
    """
    print("[API STATUP] Carregando artefatos do modelo de risco de crédito...")
    try:
        artifacts["encoder"] = joblib.load(os.path.join(ARTIFACTS_DIR, "encoder.pkl"))
        artifacts["scaler"] = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
        artifacts["kmeans"] = joblib.load(os.path.join(ARTIFACTS_DIR, "kmeans.pkl"))
        artifacts["modelo_gb"] = joblib.load(os.path.join(ARTIFACTS_DIR, "modelo_gb.pkl"))
        print("[API STARTUP] Todos os 4 artefatos foram carregados com sucesso!")
    except Exception as e:
        print(f"[API STARTUP ERROR] Falha ao carregar artefatos: {str(e)}")
        # Em ambiente de execução, os artefatos devem ser previamente gerados pelo kmeans_pipeline.py
    yield
    artifacts.clear()


app = FastAPI(
    title="Datarisk Credit Risk Scoring API",
    description="API REST para pontuação de risco de crédito combinando K-Means e Gradient Boosting.",
    version="1.0.0",
    lifespan=lifespan
)


# Esquema Pydantic para Payload de Entrada
class ClientPayload(BaseModel):
    ID_CLIENTE: Optional[str] = Field(default="CLI_00001", description="Identificador único do cliente")
    TIPO_PESSOA: str = Field(..., description="PF (Pessoa Física) ou PJ (Pessoa Jurídica)")
    RENDA_MENSAL: float = Field(..., gt=0, description="Renda mensal declarada do cliente em BRL")
    IDADE: int = Field(..., ge=18, le=120, description="Idade do cliente em anos")
    QTD_PARCELAS: int = Field(..., ge=1, description="Quantidade total de parcelas históricas")
    VALOR_PARCELA_MEDIO: float = Field(..., gt=0, description="Valor médio das parcelas contratadas")
    ATRASO_MEDIO: float = Field(..., ge=0, description="Média histórica de dias de atraso nas parcelas")
    ATRASO_MAX: float = Field(..., ge=0, description="Máximo histórico de dias de atraso registrado")

    model_config = {
        "json_schema_extra": {
            "example": {
                "ID_CLIENTE": "CLI_99482",
                "TIPO_PESSOA": "PF",
                "RENDA_MENSAL": 4500.0,
                "IDADE": 35,
                "QTD_PARCELAS": 12,
                "VALOR_PARCELA_MEDIO": 350.0,
                "ATRASO_MEDIO": 1.2,
                "ATRASO_MAX": 4.0
            }
        }
    }


class PredictionResponse(BaseModel):
    id_cliente: str
    probabilidade_default: float
    decisao_credito: str
    rating_risco: str
    cluster_cliente: int
    detalhes: Dict[str, Any]


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
def health_check():
    """
    Endpoint de verificação de integridade da API e dos artefatos.
    """
    model_loaded = "modelo_gb" in artifacts and artifacts["modelo_gb"] is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "artifacts_loaded": list(artifacts.keys()),
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, tags=["Prediction"])
def predict_credit_risk(payload: ClientPayload):
    """
    Recebe os dados cadastrais e financeiros do cliente e retorna o escore de risco de crédito.
    """
    if "modelo_gb" not in artifacts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artefatos do modelo não estão carregados no servidor."
        )

    try:
        # 1. Tratamento seguro de Categorias Desconhecidas (handling unknown categories)
        tipo_pessoa_str = payload.TIPO_PESSOA.upper().strip()
        if tipo_pessoa_str not in ["PF", "PJ"]:
            # Fallback seguro para categoria padrão sem quebrar o encoder
            tipo_pessoa_str = "PF"
            
        df_input = pd.DataFrame([{
            "TIPO_PESSOA": tipo_pessoa_str,
            "RENDA_MENSAL": payload.RENDA_MENSAL,
            "IDADE": payload.IDADE,
            "QTD_PARCELAS": payload.QTD_PARCELAS,
            "VALOR_PARCELA_MEDIO": payload.VALOR_PARCELA_MEDIO,
            "ATRASO_MEDIO": payload.ATRASO_MEDIO,
            "ATRASO_MAX": payload.ATRASO_MAX
        }])

        # 2. Padronização das variáveis numéricas
        scaler = artifacts["scaler"]
        num_scaled = scaler.transform(df_input[NUMERIC_COLS])

        # 3. Predição do Cluster K-Means
        kmeans = artifacts["kmeans"]
        cluster_pred = int(kmeans.predict(num_scaled)[0])

        # 4. Encoders Categóricos com OneHotEncoder
        encoder = artifacts["encoder"]
        cat_encoded = encoder.transform(df_input[CATEGORICAL_COLS])

        # 5. Formatação do vetor final de entrada para o Gradient Boosting
        cluster_arr = np.array([[cluster_pred]])
        features_vector = np.hstack([num_scaled, cat_encoded, cluster_arr])

        # 6. Inferência de Probabilidade pelo Modelo Gradient Boosting
        modelo_gb = artifacts["modelo_gb"]
        prob_default = float(modelo_gb.predict_proba(features_vector)[0][1])

        # Categorização de Risco de Negócio
        if prob_default < 0.25:
            rating = "A - Baixo Risco"
            decisao = "APROVADO"
        elif prob_default < 0.50:
            rating = "B - Risco Médio"
            decisao = "APROVADO COM RESTRIÇÃO / ANÁLISE MANUAL"
        elif prob_default < 0.75:
            rating = "C - Alto Risco"
            decisao = "NEGADO"
        else:
            rating = "D - Risco Crítico"
            decisao = "NEGADO"

        return PredictionResponse(
            id_cliente=payload.ID_CLIENTE or "N/A",
            probabilidade_default=round(prob_default, 4),
            decisao_credito=decisao,
            rating_risco=rating,
            cluster_cliente=cluster_pred,
            detalhes={
                "atraso_max_registrado": payload.ATRASO_MAX,
                "renda_declarada": payload.RENDA_MENSAL,
                "atraso_medio": payload.ATRASO_MEDIO
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao processar predição: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
