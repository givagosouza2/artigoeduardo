import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import shapiro, rankdata
from sklearn.metrics import mean_absolute_error

# ------------------------------
# Função para calcular ICC simples
def calcular_icc(data):
    n = data.shape[0]
    mean_per_person = np.mean(data, axis=1)
    grand_mean = np.mean(data)
    ms_between = np.var(mean_per_person, ddof=1) * n
    ms_within = np.mean(np.var(data, axis=1, ddof=1))
    denominator = ms_between + (n - 1) * ms_within
    icc = (ms_between - ms_within) / denominator if denominator != 0 else np.nan
    return icc

# ------------------------------
# Função para tentar diferentes separadores
def tentar_carregar_csv(uploaded_file):
    separadores = {',': 'Vírgula', ';': 'Ponto e vírgula', '\t': 'Tabulação'}
    for sep, nome in separadores.items():
        try:
            df = pd.read_csv(uploaded_file, sep=sep)
            if df.shape[1] >= 2:
                return df, nome
        except Exception:
            continue
    return None, None

# ------------------------------
# Interface Streamlit
st.title("Análise de Confiabilidade Inter-Dias")

uploaded_file = st.file_uploader("Carregue um arquivo CSV com duas colunas: Dia 1 e Dia 2", type=["csv"])

if uploaded_file is not None:
    df, separador_usado = tentar_carregar_csv(uploaded_file)

    if df is None:
        st.error("Não foi possível ler o arquivo. Verifique o formato e o separador.")
    else:
        st.success(f"Arquivo lido com sucesso usando o separador: {separador_usado}")
        st.write("Visualização dos dados carregados:")
        st.dataframe(df.head())

        # Remover linhas com NaN
        df = df.dropna()

        if df.shape[1] != 2:
            st.error("O arquivo precisa ter exatamente duas colunas (Dia 1 e Dia 2).")
        else:
            dia1 = df.iloc[:, 0]
            dia2 = df.iloc[:, 1]

            # Teste de normalidade
            shapiro_dia1 = shapiro(dia1)
            shapiro_dia2 = shapiro(dia2)

            st.subheader("Teste de Normalidade (Shapiro-Wilk)")
            normalidade_df = pd.DataFrame({
                "Dia": ["Dia 1", "Dia 2"],
                "W": [shapiro_dia1.statistic, shapiro_dia2.statistic],
                "p-valor": [shapiro_dia1.pvalue, shapiro_dia2.pvalue]
            })
            st.dataframe(normalidade_df)

            normal = (shapiro_dia1.pvalue > 0.05) and (shapiro_dia2.pvalue > 0.05)

            if normal:
                st.success("Dados com distribuição normal. Executando análise paramétrica...")

                media_dia1 = np.mean(dia1)
                media_dia2 = np.mean(dia2)
                dp_dia1 = np.std(dia1, ddof=1)
                dp_dia2 = np.std(dia2, ddof=1)

                sem = np.sqrt(((dp_dia1**2 + dp_dia2**2) / 2))
                cv_dia1 = (dp_dia1 / media_dia1) * 100
                cv_dia2 = (dp_dia2 / media_dia2) * 100
                mdc = sem * 1.96 * np.sqrt(2)
                ape = np.mean(np.abs((dia2 - dia1) / dia1)) * 100
                mean_global = np.mean(np.concatenate([dia1, dia2]))
                mae = mean_absolute_error(dia1, dia2)
                accuracy = (1 - (mae / mean_global)) * 100
                icc_value = calcular_icc(df.values)

                resultados_df = pd.DataFrame({
                    "Média Dia 1": [media_dia1],
                    "Média Dia 2": [media_dia2],
                    "DP Dia 1": [dp_dia1],
                    "DP Dia 2": [dp_dia2],
                    "SEM": [sem],
                    "CV% Dia 1": [cv_dia1],
                    "CV% Dia 2": [cv_dia2],
                    "MDC": [mdc],
                    "APE (%)": [ape],
                    "Accuracy (%)": [accuracy],
                    "ICC": [icc_value]
                })

                st.subheader("Resultados - Análise Paramétrica")
                st.dataframe(resultados_df)

            else:
                st.warning("Pelo menos um dos dias não apresentou normalidade. Executando análise não paramétrica...")

                mediana_dia1 = np.median(dia1)
                mediana_dia2 = np.median(dia2)
                iqr_dia1 = np.percentile(dia1, 75) - np.percentile(dia1, 25)
                iqr_dia2 = np.percentile(dia2, 75) - np.percentile(dia2, 25)

                # Mediana do erro absoluto
                md_ae = np.median(np.abs(dia2 - dia1))

                # ICC com ranks (Rank ICC simplificado)
                dia1_rank = rankdata(dia1)
                dia2_rank = rankdata(dia2)
                icc_rank = calcular_icc(np.column_stack([dia1_rank, dia2_rank]))

                resultados_np_df = pd.DataFrame({
                    "Mediana Dia 1": [mediana_dia1],
                    "Mediana Dia 2": [mediana_dia2],
                    "IQR Dia 1": [iqr_dia1],
                    "IQR Dia 2": [iqr_dia2],
                    "Mediana Erro Absoluto": [md_ae],
                    "ICC (Ranks)": [icc_rank]
                })

                st.subheader("Resultados - Análise Não Paramétrica")
                st.dataframe(resultados_np_df)
