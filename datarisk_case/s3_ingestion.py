"""
s3_ingestion.py - Módulo de Ingestão de Dados via AWS S3

Adapta o carregamento das bases do Case Datarisk para ler diretamente de um bucket S3
utilizando s3fs / boto3 e pandas.
"""

import os
from typing import Dict, Optional
import pandas as pd
import numpy as np

def carregar_base_s3(
    s3_path: str,
    storage_options: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Carrega um arquivo CSV diretamente da AWS S3 ou caminho remoto/local.

    Parameters:
    -----------
    s3_path : str
        Caminho S3 completo (ex: 's3://meu-bucket-datarisk/dados/base_info.csv')
    storage_options : dict, optional
        Dicionário com credenciais AWS (key, secret, token, region)

    Returns:
    --------
    pd.DataFrame: DataFrame carregado
    """
    print(f"[AWS S3] Carregando arquivo: {s3_path}")
    
    # Se storage_options não for informado, busca de variáveis de ambiente padrão AWS
    if storage_options is None and s3_path.startswith("s3://"):
        storage_options = {
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "token": os.getenv("AWS_SESSION_TOKEN")
        }
        # Limpa chaves None se variáveis de ambiente não estiverem definidas
        storage_options = {k: v for k, v in storage_options.items() if v is not None}
        if not storage_options:
            storage_options = None

    try:
        if storage_options:
            df = pd.read_csv(s3_path, storage_options=storage_options)
        else:
            df = pd.read_csv(s3_path)
        print(f"[AWS S3] Sucesso! Dimensões: {df.shape}")
        return df
    except Exception as e:
        print(f"[AWS S3] Erro ao carregar {s3_path}: {str(e)}")
        raise e


def carregar_todas_as_bases(
    bucket_uri: str = "s3://datarisk-credit-risk-case/",
    storage_options: Optional[Dict[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Carrega as 4 bases principais do Case Datarisk a partir de um Bucket S3.

    Bases:
    1. base_pagamentos_desenvolvimento.csv
    2. base_pagamentos_teste.csv
    3. base_info.csv
    4. base_cadastral.csv
    """
    bucket_uri = bucket_uri.rstrip("/") + "/"
    
    arquivos = {
        "pagamentos_dev": f"{bucket_uri}base_pagamentos_desenvolvimento.csv",
        "pagamentos_teste": f"{bucket_uri}base_pagamentos_teste.csv",
        "info": f"{bucket_uri}base_info.csv",
        "cadastral": f"{bucket_uri}base_cadastral.csv",
    }
    
    bases = {}
    for nome, url in arquivos.items():
        bases[nome] = carregar_base_s3(url, storage_options=storage_options)
        
    return bases


def engenharia_de_features_pagamentos(df_pagamentos: pd.DataFrame, is_dev: bool = True) -> pd.DataFrame:
    """
    Processa o histórico de pagamentos sem vazamento de dados (Data Leakage).
    Define o Target de risco de crédito (Atraso >= 5 dias).
    """
    df = df_pagamentos.copy()
    
    # Target: Atraso >= 5 dias em qualquer parcela histórica (apenas para base dev)
    if 'ATRASO' in df.columns and is_dev:
        df['TARGET_DEF'] = (df['ATRASO'] >= 5).astype(int)
    
    # Agregações por cliente (ID_CLIENTE)
    agregacoes = df.groupby('ID_CLIENTE').agg(
        QTD_PARCELAS=('ID_CLIENTE', 'count'),
        VALOR_PARCELA_MEDIO=('VALOR_PARCELA', 'mean') if 'VALOR_PARCELA' in df.columns else ('ID_CLIENTE', 'count'),
        VALOR_PARCELA_TOTAL=('VALOR_PARCELA', 'sum') if 'VALOR_PARCELA' in df.columns else ('ID_CLIENTE', 'count'),
        ATRASO_MAX=('ATRASO', 'max') if 'ATRASO' in df.columns else ('ID_CLIENTE', 'count'),
        ATRASO_MEDIO=('ATRASO', 'mean') if 'ATRASO' in df.columns else ('ID_CLIENTE', 'count'),
        DIAS_DESDE_PRIMEIRA_PARCELA=('DIAS_PAGO', 'max') if 'DIAS_PAGO' in df.columns else ('ID_CLIENTE', 'count'),
    ).reset_index()
    
    if is_dev and 'TARGET_DEF' in df.columns:
        target_cliente = df.groupby('ID_CLIENTE')['TARGET_DEF'].max().reset_index()
        agregacoes = agregacoes.merge(target_cliente, on='ID_CLIENTE', how='left')
        agregacoes.rename(columns={'TARGET_DEF': 'TARGET'}, inplace=True)
        
    return agregacoes


if __name__ == "__main__":
    print("Módulo s3_ingestion pronto para uso.")
