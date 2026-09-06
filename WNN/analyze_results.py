import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuração ---
# Caminho para o arquivo JSON gerado por main.py
RESULTS_JSON_PATH = os.path.join(os.path.dirname(__file__), "results", "experiments.json")
# Diretório para salvar os gráficos de análise
PLOTS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "analysis_plots")


def load_and_flatten_data(json_path: str) -> pd.DataFrame:
    """
    Carrega os resultados dos experimentos de um arquivo JSON e achata a estrutura aninhada
    em um DataFrame do pandas, adequado para análise.

    Args:
        json_path: O caminho para o arquivo JSON de entrada.

    Returns:
        Um DataFrame do pandas onde cada linha representa o desempenho de um classificador
        em um dataset específico.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Arquivo de resultados não encontrado em: {json_path}. Por favor, execute main.py primeiro.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat_data = []
    for experiment in data:
        dataset_info = experiment.get("dataset", {})
        for clf_name, metrics in experiment.get("classifiers", {}).items():
            row = {
                "run_id": experiment.get("run_id"),
                "dataset_name": dataset_info.get("name"),
                "n_samples": dataset_info.get("n_samples"),
                "n_features": dataset_info.get("n_features"),
                "n_classes": dataset_info.get("n_classes"),
                "kernel_strategy": experiment.get("representation", {}).get("kernel_strategy", "N/A"),
                "classifier": clf_name,
                "accuracy_mean": metrics.get("accuracy_mean"),
                "accuracy_std": metrics.get("accuracy_std"),
                "f1_mean": metrics.get("f1_mean"),
            }
            flat_data.append(row)

    return pd.DataFrame(flat_data)


def plot_overall_performance(df: pd.DataFrame, output_dir: str):
    """
    Gera e salva um gráfico de barras comparando a acurácia média geral de cada classificador.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    # Calcula a acurácia média em todos os datasets para cada classificador
    overall_perf = df.groupby("classifier")["accuracy_mean"].mean().sort_values(ascending=False)

    ax = sns.barplot(x=overall_perf.index, y=overall_perf.values, palette="viridis")

    ax.set_title("Desempenho Geral dos Classificadores (Acurácia Média)", fontsize=16, pad=20)
    ax.set_xlabel("Classificador", fontsize=12)
    ax.set_ylabel("Acurácia Média", fontsize=12)
    ax.set_ylim(0, max(1.0, overall_perf.max() * 1.1))

    # Adiciona rótulos de valor no topo das barras
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points',
                    fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "overall_classifier_performance.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Gráfico de desempenho geral salvo em: {save_path}")


def plot_performance_per_dataset(df: pd.DataFrame, output_dir: str):
    """
    Gera e salva um gráfico de barras para cada dataset, comparando a acurácia dos classificadores.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Cria um subdiretório para esses plots para não poluir a pasta principal
    per_dataset_dir = os.path.join(output_dir, "per_dataset")
    os.makedirs(per_dataset_dir, exist_ok=True)

    datasets = df['dataset_name'].unique()
    
    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        
        # Filtra os dados para o dataset atual e ordena para melhor visualização
        df_dataset = df[df['dataset_name'] == dataset].sort_values("accuracy_mean", ascending=False)
        
        if df_dataset.empty:
            continue

        ax = sns.barplot(
            data=df_dataset,
            x="classifier",
            y="accuracy_mean",
            palette="plasma"
        )

        ax.set_title(f"Desempenho dos Classificadores no Dataset: {dataset}", fontsize=16, pad=20)
        ax.set_xlabel("Classificador", fontsize=12)
        ax.set_ylabel("Acurácia Média", fontsize=12)
        ax.set_ylim(0, max(1.0, df_dataset["accuracy_mean"].max() * 1.15))
        plt.xticks(rotation=45, ha='right')

        # Adiciona rótulos de valor no topo das barras
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        fontsize=10)

        plt.tight_layout()
        
        # Nome do arquivo específico para o dataset
        save_path = os.path.join(per_dataset_dir, f"performance_{dataset}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Gráfico de desempenho para '{dataset}' salvo em: {save_path}")


def plot_accuracy_vs_features(df: pd.DataFrame, output_dir: str):
    """
    Gera um gráfico de dispersão da acurácia vs. número de features binárias,
    colorido por classificador.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))

    ax = sns.scatterplot(
        data=df,
        x="n_features",
        y="accuracy_mean",
        hue="classifier",
        size="n_samples",
        sizes=(50, 500),
        palette="deep",
        alpha=0.8
    )

    ax.set_title("Acurácia vs. Tamanho do Vetor Binário", fontsize=16, pad=20)
    ax.set_xlabel("Número de Features (Tamanho do Vetor Binário)", fontsize=12)
    ax.set_ylabel("Acurácia Média", fontsize=12)
    ax.legend(title="Classificador", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_path = os.path.join(output_dir, "accuracy_vs_features.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Gráfico de acurácia vs. features salvo em: {save_path}")


def main():
    """
    Função principal para executar o pipeline de análise.
    """
    print("--- Iniciando Análise de Resultados ---")
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

    try:
        df = load_and_flatten_data(RESULTS_JSON_PATH)
        print("Dados carregados e processados com sucesso.")
        print(f"Encontrados {len(df)} registros de resultados.")

        print("\n--- Estatísticas Descritivas (Acurácia Média por Classificador) ---")
        desc_stats = df.groupby("classifier")["accuracy_mean"].describe()
        print(desc_stats.to_string())

        print("\n--- Gerando Visualizações ---")
        plot_overall_performance(df, PLOTS_OUTPUT_DIR)
        plot_performance_per_dataset(df, PLOTS_OUTPUT_DIR)
        plot_accuracy_vs_features(df, PLOTS_OUTPUT_DIR)

        print("\n--- Análise Concluída ---")
        print(f"Plots salvos em: {os.path.abspath(PLOTS_OUTPUT_DIR)}")

    except Exception as e:
        print(f"\n[ERRO] Ocorreu um erro durante a análise: {e}")


if __name__ == "__main__":
    main()