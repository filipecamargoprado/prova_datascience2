import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Caminho dos arquivos na pasta do projeto
PASTA_BASE = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_BASE, "HousingData.csv")
ARQUIVO_MODELO = os.path.join(PASTA_BASE, "modelo_kmeans_boston.pkl")
ARQUIVO_GRAFICO = os.path.join(PASTA_BASE, "grafico_cotovelo.png")

# Separando colunas quantitativas das categoricas
COLUNAS_QUANTITATIVAS = [
    "CRIM", "ZN", "INDUS", "NOX", "RM", "AGE",
    "DIS", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]
COLUNAS_CATEGORICAS = ["CHAS", "RAD"]


# ==========================================
# MODULO DE TREINAMENTO
# ==========================================

def carregar_e_limpar_base(caminho_csv):
    df = pd.read_csv(caminho_csv)

    # Preenchendo valores ausentes (NA) da base:
    # mediana para dados quantitativos e moda para categoricos
    for col in COLUNAS_QUANTITATIVAS:
        df[col] = df[col].fillna(df[col].median())

    for col in COLUNAS_CATEGORICAS:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["CHAS"] = df["CHAS"].astype(int)
    df["RAD"] = df["RAD"].astype(int)

    return df


def normalizar_base(df):
    # Normaliza apenas as variaveis quantitativas (Min-Max entre 0 e 1)
    scaler = MinMaxScaler()
    df_quant = pd.DataFrame(
        scaler.fit_transform(df[COLUNAS_QUANTITATIVAS]),
        columns=COLUNAS_QUANTITATIVAS,
        index=df.index
    )

    # Trata as variaveis categoricas separadamente:
    # CHAS ja e binaria (0 ou 1) e RAD vira One-Hot Encoding (dummies)
    df_chas = df[["CHAS"]].copy()
    df_rad_dummies = pd.get_dummies(df["RAD"], prefix="RAD", dtype=int)

    df_final = pd.concat([df_quant, df_chas, df_rad_dummies], axis=1)
    colunas_dummies_rad = list(df_rad_dummies.columns)

    return df_final, scaler, colunas_dummies_rad


def encontrar_k_cotovelo(dados_normalizados, k_max=10):
    inercias = []
    valores_k = list(range(1, k_max + 1))

    for k in valores_k:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(dados_normalizados)
        inercias.append(kmeans.inertia_)

    # Calculo do melhor K usando a maior distancia ate a reta entre o primeiro e o ultimo ponto
    x1, y1 = valores_k[0], inercias[0]
    x2, y2 = valores_k[-1], inercias[-1]

    distancias = []
    for x0, y0 in zip(valores_k, inercias):
        numerador = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominador = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distancias.append(numerador / denominador)

    k_otimo = valores_k[int(np.argmax(distancias))]

    # Salva o grafico do metodo do cotovelo
    plt.figure(figsize=(8, 5))
    plt.plot(valores_k, inercias, marker="o", linestyle="-", color="b", label="Inércia (WCSS)")
    plt.axvline(x=k_otimo, color="r", linestyle="--", label=f"K ótimo = {k_otimo}")
    plt.title("Método do Cotovelo - Boston Housing")
    plt.xlabel("Número de Grupos (K)")
    plt.ylabel("Soma das Distâncias Quadráticas (Inércia)")
    plt.xticks(valores_k)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ARQUIVO_GRAFICO)
    plt.close()

    return k_otimo, inercias


def treinar_e_salvar_modelo():
    print("--- INICIANDO MODULO DE TREINAMENTO ---")
    df = carregar_e_limpar_base(ARQUIVO_DADOS)
    print(f"Base carregada com sucesso: {df.shape[0]} registros e {df.shape[1]} colunas.")

    df_normalizado, scaler, colunas_dummies_rad = normalizar_base(df)
    print("Base normalizada (quantitativas com MinMaxScaler e categoricas com One-Hot/Binario).")

    k_otimo, inercias = encontrar_k_cotovelo(df_normalizado, k_max=10)
    print("\nDistancias (Inercia) por K no Metodo do Cotovelo:")
    for k, inercia in enumerate(inercias, start=1):
        print(f"  K = {k:2d} -> Inercia = {inercia:.2f}")
    print(f"\nNumero otimo de grupos definido pelo Metodo do Cotovelo: K = {k_otimo}")
    print(f"Grafico salvo em: {ARQUIVO_GRAFICO}")

    # Treinando o modelo final com o K otimo
    modelo_kmeans = KMeans(n_clusters=k_otimo, n_init=10, random_state=42)
    modelo_kmeans.fit(df_normalizado)

    # Salvando o modelo e os objetos necessarios para a inferencia
    pacote_modelo = {
        "modelo_kmeans": modelo_kmeans,
        "scaler_quantitativo": scaler,
        "colunas_quantitativas": COLUNAS_QUANTITATIVAS,
        "colunas_categoricas": COLUNAS_CATEGORICAS,
        "colunas_dummies_rad": colunas_dummies_rad,
        "colunas_finais_modelo": list(df_normalizado.columns),
        "k_otimo": k_otimo
    }

    with open(ARQUIVO_MODELO, "wb") as f:
        pickle.dump(pacote_modelo, f)

    print(f"Modelo treinado e salvo com sucesso em: {ARQUIVO_MODELO}\n")
    return pacote_modelo


# ==========================================
# MODULO DE INFERENCIA
# ==========================================

def normalizar_nova_residencia(dados_residencia, pacote_modelo):
    scaler = pacote_modelo["scaler_quantitativo"]
    colunas_quant = pacote_modelo["colunas_quantitativas"]
    colunas_dummies_rad = pacote_modelo["colunas_dummies_rad"]
    colunas_finais = pacote_modelo["colunas_finais_modelo"]

    df_nova = pd.DataFrame([dados_residencia])

    # Normaliza a parte quantitativa com o mesmo scaler do treinamento
    df_quant_norm = pd.DataFrame(
        scaler.transform(df_nova[colunas_quant]),
        columns=colunas_quant
    )

    # Mantem CHAS binario
    df_chas = pd.DataFrame({"CHAS": [int(dados_residencia["CHAS"])]})

    # Monta as colunas dummies de RAD exatamente como no treinamento
    rad_valor = int(dados_residencia["RAD"])
    dummies_dict = {col: [1 if col == f"RAD_{rad_valor}" else 0] for col in colunas_dummies_rad}
    df_rad = pd.DataFrame(dummies_dict)

    df_nova_norm = pd.concat([df_quant_norm, df_chas, df_rad], axis=1)
    df_nova_norm = df_nova_norm[colunas_finais]

    return df_nova_norm


def desnormalizar_centroide(centroide_norm, pacote_modelo):
    scaler = pacote_modelo["scaler_quantitativo"]
    colunas_quant = pacote_modelo["colunas_quantitativas"]
    colunas_dummies_rad = pacote_modelo["colunas_dummies_rad"]
    colunas_finais = pacote_modelo["colunas_finais_modelo"]

    s_centroide = pd.Series(centroide_norm, index=colunas_finais)

    # Desnormaliza as variaveis quantitativas usando inverse_transform
    valores_quant_norm = s_centroide[colunas_quant].values.reshape(1, -1)
    valores_quant_desnorm = scaler.inverse_transform(valores_quant_norm)[0]

    resultado = {}
    for col, val in zip(colunas_quant, valores_quant_desnorm):
        resultado[col] = round(float(val), 4)

    # Para CHAS, mostra a categoria predominante (0 ou 1) e a proporcao no grupo
    chas_prop = float(s_centroide["CHAS"])
    resultado["CHAS"] = f"{int(round(chas_prop))} (proporcao no grupo: {chas_prop:.2%})"

    # Para RAD, identifica o indice RAD mais frequente no centroide
    valores_rad = {col: s_centroide[col] for col in colunas_dummies_rad}
    rad_mais_comum = max(valores_rad, key=valores_rad.get)
    indice_rad = int(rad_mais_comum.replace("RAD_", ""))
    resultado["RAD"] = f"{indice_rad} (categoria predominante: {valores_rad[rad_mais_comum]:.2%})"

    return resultado


def inferir_residencia(dados_residencia):
    print("--- INICIANDO MODULO DE INFERENCIA ---")

    with open(ARQUIVO_MODELO, "rb") as f:
        pacote_modelo = pickle.load(f)

    modelo_kmeans = pacote_modelo["modelo_kmeans"]

    print("Dados recebidos da nova residencia:")
    for k, v in dados_residencia.items():
        print(f"  {k}: {v}")

    # 1. Normalizar os dados da nova residencia
    nova_norm = normalizar_nova_residencia(dados_residencia, pacote_modelo)
    print("\nDados da nova residencia normalizados com sucesso.")

    # 2. Inferir a qual grupo pertence
    grupo_inferido = int(modelo_kmeans.predict(nova_norm)[0])
    print(f"\n-> A nova residencia pertence ao GRUPO (Cluster): {grupo_inferido}")

    # 3. Exibir os dados desnormalizados do centroide inferido
    centroide_norm = modelo_kmeans.cluster_centers_[grupo_inferido]
    centroide_desnorm = desnormalizar_centroide(centroide_norm, pacote_modelo)

    print(f"\nDados desnormalizados do centroide do Grupo {grupo_inferido}:")
    for atributo, valor in centroide_desnorm.items():
        print(f"  {atributo:8s}: {valor}")

    return grupo_inferido, centroide_desnorm


if __name__ == "__main__":
    # 1. Executa o treinamento e salva o modelo
    treinar_e_salvar_modelo()

    # 2. Exemplo de nova residencia para testar o modulo de inferencia
    nova_residencia_exemplo = {
        "CRIM": 0.045,
        "ZN": 12.5,
        "INDUS": 6.07,
        "CHAS": 0,
        "NOX": 0.49,
        "RM": 6.5,
        "AGE": 55.0,
        "DIS": 4.8,
        "RAD": 4,
        "TAX": 290,
        "PTRATIO": 16.8,
        "B": 392.5,
        "LSTAT": 7.5,
        "MEDV": 24.5
    }

    inferir_residencia(nova_residencia_exemplo)
