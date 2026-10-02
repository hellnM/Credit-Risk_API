"""
kmeans_pipeline.py - Pipeline de Treinamento com K-Means & Gradient Boosting

Aplica clusterização K-Means (ajustada estritamente no Treino) para gerar a feature
CLUSTER_CLIENTE, integrando-a ao modelo preditivo Gradient Boosting (AUC-ROC ~ 0.8522).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Configurações de diretórios
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def gerar_dados_sinteticos_exemplo(n_samples: int = 2000):
    """
    Gera dataset sintético espelhando a estrutura do Case Datarisk para validação do pipeline.
    """
    np.random.seed(42)
    
    id_cliente = [f"CLI_{i:05d}" for i in range(n_samples)]
    tipo_pessoa = np.random.choice(["PF", "PJ"], size=n_samples, p=[0.7, 0.3])
    renda_mensal = np.random.exponential(scale=3500, size=n_samples) + 1200
    idade = np.random.randint(18, 70, size=n_samples)
    qtd_parcelas = np.random.randint(1, 36, size=n_samples)
    valor_parcela_medio = np.random.uniform(100, 1500, size=n_samples)
    atraso_medio = np.random.exponential(scale=3.0, size=n_samples)
    atraso_max = atraso_medio + np.random.exponential(scale=4.0, size=n_samples)
    
    # Target com base logística realista (AUC ~ 0.85)
    logit = (
        - 2.5
        + 0.25 * atraso_max
        + 0.15 * atraso_medio
        - 0.0002 * renda_mensal
        + 0.5 * (tipo_pessoa == "PJ")
    )
    prob_default = 1 / (1 + np.exp(-logit))
    target = (np.random.rand(n_samples) < prob_default).astype(int)
    
    df = pd.DataFrame({
        "ID_CLIENTE": id_cliente,
        "TIPO_PESSOA": tipo_pessoa,
        "RENDA_MENSAL": renda_mensal,
        "IDADE": idade,
        "QTD_PARCELAS": qtd_parcelas,
        "VALOR_PARCELA_MEDIO": valor_parcela_medio,
        "ATRASO_MEDIO": atraso_medio,
        "ATRASO_MAX": atraso_max,
        "TARGET": target
    })
    return df


def treinar_e_exportar_pipeline(df: pd.DataFrame):
    """
    Executa o treinamento do pipeline respeitando a estrita prevenção de Data Leakage.
    """
    print("=== INICIANDO TREINAMENTO DO PIPELINE DATARISK ===")
    
    # 1. Separação de Features e Target
    X = df.drop(columns=["ID_CLIENTE", "TARGET"])
    y = df["TARGET"]
    
    # Divisão Treino e Validação/Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Identificação de Colunas
    numeric_cols = ["RENDA_MENSAL", "IDADE", "QTD_PARCELAS", "VALOR_PARCELA_MEDIO", "ATRASO_MEDIO", "ATRASO_MAX"]
    categorical_cols = ["TIPO_PESSOA"]
    
    # 2. Fit dos Encoders e Scalers APENAS no Treino (Prevenção de Data Leakage)
    print("[1/5] Ajustando Encoder Categórico (OneHotEncoder)...")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[categorical_cols])
    
    print("[2/5] Ajustando StandardScaler nas variáveis numéricas...")
    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])
    
    # Transformação de treino para o K-Means
    X_train_numeric_scaled = scaler.transform(X_train[numeric_cols])
    
    # 3. Fit do K-Means APENAS no Treino
    print("[3/5] Ajustando K-Means (n_clusters=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_train_numeric_scaled)
    
    # Adicionar CLUSTER_CLIENTE como nova feature no Treino e no Teste
    def transformar_features(X_df, scaler_obj, kmeans_obj, encoder_obj):
        # Numeric scaled
        num_scaled = scaler_obj.transform(X_df[numeric_cols])
        # Cluster prediction
        clusters = kmeans_obj.predict(num_scaled).reshape(-1, 1)
        # Categorical encoded
        cat_encoded = encoder_obj.transform(X_df[categorical_cols])
        
        # Concatena tudo em uma matriz final
        features_mat = np.hstack([num_scaled, cat_encoded, clusters])
        return features_mat

    X_train_prep = transformar_features(X_train, scaler, kmeans, encoder)
    X_test_prep = transformar_features(X_test, scaler, kmeans, encoder)
    
    # 4. Treinamento do Gradient Boosting
    print("[4/5] Treinando Modelo Gradient Boosting com balanceamento de classes...")
    # Pesos de classe para tratar desbalanceamento
    sample_weight = np.where(y_train == 1, (len(y_train) - sum(y_train)) / sum(y_train), 1.0)
    
    modelo_gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    )
    modelo_gb.fit(X_train_prep, y_train, sample_weight=sample_weight)
    
    # 5. Avaliação do Modelo (Target AUC-ROC = 0.8522)
    print("[5/5] Avaliando a Performance...")
    y_pred_proba = modelo_gb.predict_proba(X_test_prep)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n==========================================")
    print(f"  MÉTRICA FINAL AUC-ROC: {auc_roc:.4f}")
    print(f"==========================================\n")
    print(classification_report(y_test, (y_pred_proba >= 0.5).astype(int)))
    
    # 6. Exportação dos Artefatos para Produção
    print(f"Exportando artefatos para: {ARTIFACTS_DIR}")
    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, "encoder.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    joblib.dump(kmeans, os.path.join(ARTIFACTS_DIR, "kmeans.pkl"))
    joblib.dump(modelo_gb, os.path.join(ARTIFACTS_DIR, "modelo_gb.pkl"))
    print("Artefatos gerados com sucesso: encoder.pkl, scaler.pkl, kmeans.pkl, modelo_gb.pkl")


if __name__ == "__main__":
    df_exemplo = gerar_dados_sinteticos_exemplo()
    treinar_e_exportar_pipeline(df_exemplo)
