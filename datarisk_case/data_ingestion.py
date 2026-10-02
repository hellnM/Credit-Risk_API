"""
data_ingestion.py - Módulo profissional de ingestão de dados via AWS S3

Fornece funções para carregar os datasets do Case Datarisk diretamente de um bucket S3
usando pandas + s3fs / boto3. Mantém a mesma funcionalidade do antigo s3_ingestion.py,
mas com nome e documentação mais claros.
"""

import os
from typing import Dict, Optional
import pandas as pd

def load_s3_csv(
    s3_path: str,
    storage_options: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Carrega um arquivo CSV a partir de um caminho S3.

    Parameters
    ----------
    s3_path : str
        URL completa do S3, por exemplo ``s3://bucket/dataset.csv``.
    storage_options : dict, optional
        Dicionário com credenciais AWS (key, secret, token). Se não fornecido,
        o método tenta obter as credenciais das variáveis de ambiente padrão.

    Returns
    -------
    pd.DataFrame
        DataFrame contendo os dados do CSV.
    """
    print(f"[AWS S3] Carregando arquivo: {s3_path}")

    if storage_options is None and s3_path.startswith("s3://"):
        storage_options = {
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "token": os.getenv("AWS_SESSION_TOKEN")
        }
        # Remove chaves com valores None
        storage_options = {k: v for k, v in storage_options.items() if v}
        if not storage_options:
            storage_options = None

    try:
        df = pd.read_csv(s3_path, storage_options=storage_options) if storage_options else pd.read_csv(s3_path)
        print(f"[AWS S3] Sucesso – shape={df.shape}")
        return df
    except Exception as exc:
        print(f"[AWS S3] Erro ao ler {s3_path}: {exc}")
        raise


def load_all_datasets(
    bucket_uri: str = "s3://datarisk-credit-risk-case/",
    storage_options: Optional[Dict[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """Carrega as quatro bases principais do case a partir do bucket S3.

    Returns
    -------
    dict
        Mapping com chaves ``pagamentos_dev``, ``pagamentos_test``, ``info`` e ``cadastral``.
    """
    bucket_uri = bucket_uri.rstrip('/') + '/'
    files = {
        "pagamentos_dev": f"{bucket_uri}base_pagamentos_desenvolvimento.csv",
        "pagamentos_test": f"{bucket_uri}base_pagamentos_teste.csv",
        "info": f"{bucket_uri}base_info.csv",
        "cadastral": f"{bucket_uri}base_cadastral.csv",
    }
    datasets = {}
    for name, path in files.items():
        datasets[name] = load_s3_csv(path, storage_options=storage_options)
    return datasets

if __name__ == "__main__":
    print("Carregando datasets do bucket S3...")
    ds = load_all_datasets()
    for k, v in ds.items():
        print(f"{k}: {v.shape}")
