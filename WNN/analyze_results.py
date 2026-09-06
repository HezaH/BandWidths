import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
        representation_info = experiment.get("representation", {})
        diagnostics_info = representation_info.get("diagnostics", {})

        # Extrai os parâmetros da representação para análise posterior
        representation_params = {
            "n_kernels": representation_info.get("n_kernels"),
            "bits_per_kernel": representation_info.get("bits_per_kernel"),
            "k_activate": representation_info.get("k_activate"),
        }

        for clf_name, metrics in experiment.get("classifiers", {}).items():
            row = {
                "run_id": experiment.get("run_id"),
                "dataset_name": dataset_info.get("name"),
                "n_samples": dataset_info.get("n_samples"),
                "n_features_out": dataset_info.get("n_features"),
                "n_classes": dataset_info.get("n_classes"),
                "kernel_strategy": experiment.get("representation", {}).get("kernel_strategy", "N/A"),
                "classification_mode": experiment.get("classification_mode", "N/A"),
                "classifier": clf_name,
                "accuracy_mean": metrics.get("accuracy_mean"),
                "accuracy_std": metrics.get("accuracy_std"),
                "f1_mean": metrics.get("f1_mean"),
                "all_scores": metrics.get("all_scores"),
                "confusion_matrix": metrics.get("confusion_matrix"),
                "class_distribution": dataset_info.get("class_distribution"),
                **{f"param_{k}": v for k, v in representation_params.items()}
            }
            # Adiciona as métricas de diagnóstico (correlações)
            for diag_key, diag_val in diagnostics_info.items():
                row[f"diag_{diag_key}"] = diag_val
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
        x="n_features_out",
        y="accuracy_mean",
        hue="classifier",
        size="n_samples",
        sizes=(50, 500),
        palette="deep",
        alpha=0.8
    )

    ax.set_title("Acurácia vs. Tamanho do Vetor Binário", fontsize=16, pad=20)
    ax.set_xlabel("Tamanho do Vetor Binário (n_features_out)", fontsize=12)
    ax.set_ylabel("Acurácia Média", fontsize=12)
    ax.legend(title="Classificador", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_path = os.path.join(output_dir, "accuracy_vs_features.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Gráfico de acurácia vs. features salvo em: {save_path}")


def plot_performance_by_kernel_strategy(df: pd.DataFrame, output_dir: str):
    """
    Compara o desempenho (acurácia) das estratégias de kernel.
    """
    if 'kernel_strategy' not in df.columns:
        print("Coluna 'kernel_strategy' não encontrada. Pulando gráfico de estratégia de kernel.")
        return

    df_plot = df.copy()
    df_plot['kernel_strategy'] = df_plot['kernel_strategy'].fillna('kmeans')

    if df_plot['kernel_strategy'].nunique() < 2:
        print("Apenas uma estratégia de kernel encontrada. Pulando gráfico de comparação de estratégias.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))

    ax = sns.barplot(
        data=df_plot,
        x="classifier",
        y="accuracy_mean",
        hue="kernel_strategy",
        palette="coolwarm",
        estimator=np.mean,
        errorbar="sd"
    )

    ax.set_title("Desempenho por Estratégia de Kernel (Acurácia Média)", fontsize=16, pad=20)
    ax.set_xlabel("Classificador", fontsize=12)
    ax.set_ylabel("Acurácia Média", fontsize=12)
    ax.set_ylim(0, max(1.0, df_plot["accuracy_mean"].max() * 1.1))
    plt.xticks(rotation=45, ha='right')
    ax.legend(title="Estratégia de Kernel")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "performance_by_kernel_strategy.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Gráfico de desempenho por estratégia de kernel salvo em: {save_path}")


def plot_accuracy_rank_over_datasets(df: pd.DataFrame, output_dir: str):
    """
    Cria um heatmap mostrando o ranking de acurácia de cada classificador em cada dataset.
    """
    if df.empty or 'dataset_name' not in df.columns or 'classifier' not in df.columns:
        return

    # Garante que cada combinação (dataset, classifier) seja única, pegando a melhor run_id se houver duplicatas
    df_unique = df.loc[df.groupby(['dataset_name', 'classifier'])['accuracy_mean'].idxmax()]

    df_unique['rank'] = df_unique.groupby('dataset_name')['accuracy_mean'].rank(method='dense', ascending=False)
    rank_pivot = df_unique.pivot_table(index='classifier', columns='dataset_name', values='rank')

    if rank_pivot.empty:
        print("Não foi possível criar a tabela de ranking. Pulando o heatmap de ranking.")
        return

    plt.style.use('default')
    plt.figure(figsize=(max(10, rank_pivot.shape[1] * 1.2), max(6, rank_pivot.shape[0] * 0.6)))

    ax = sns.heatmap(
        rank_pivot,
        annot=True,
        fmt=".0f",
        cmap="viridis_r",
        linewidths=.5,
        cbar_kws={'label': 'Ranking de Acurácia (1 = Melhor)'}
    )

    ax.set_title("Ranking de Classificadores por Dataset", fontsize=16, pad=20)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Classificador", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "classifier_rank_heatmap.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Heatmap de ranking salvo em: {save_path}")


def plot_confusion_matrices(df: pd.DataFrame, output_dir: str):
    """
    Gera e salva as matrizes de confusão agregadas para cada classificador em cada dataset.
    """
    if 'confusion_matrix' not in df.columns or df['confusion_matrix'].isnull().all():
        print("Nenhuma matriz de confusão encontrada. Pulando a geração de plots.")
        return

    cm_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    # Garante que cada combinação (dataset, classifier) seja única, pegando a melhor run_id se houver duplicatas
    df_unique = df.loc[df.groupby(['dataset_name', 'classifier'])['accuracy_mean'].idxmax()]

    for _, row in df_unique.iterrows():
        cm = np.array(row['confusion_matrix'])
        if cm.size == 0: continue

        dataset_name, clf_name = row['dataset_name'], row['classifier']
        class_labels = sorted(row['class_distribution'].keys()) if isinstance(row['class_distribution'], dict) else [i for i in range(cm.shape[0])]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_labels, yticklabels=class_labels)
        ax.set_title(f"Matriz de Confusão: {dataset_name} - {clf_name}", fontsize=14, pad=20)
        ax.set_xlabel("Rótulo Previsto", fontsize=12)
        ax.set_ylabel("Rótulo Verdadeiro", fontsize=12)

        plt.tight_layout()
        save_path = os.path.join(cm_dir, f"cm_{dataset_name}_{clf_name}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    print(f"Matrizes de confusão salvas em: {os.path.abspath(cm_dir)}")


def plot_accuracy_distribution(df: pd.DataFrame, output_dir: str):
    """
    Gera um boxplot mostrando a distribuição da acurácia entre os folds
    para cada classificador e dataset.
    """
    if 'all_scores' not in df.columns or df['all_scores'].isnull().all():
        print("Coluna 'all_scores' não encontrada ou vazia. Pulando gráfico de distribuição de acurácia.")
        return

    df_folds = df.explode('all_scores').dropna(subset=['all_scores'])
    df_folds['all_scores'] = df_folds['all_scores'].astype(float)

    if df_folds.empty:
        print("Nenhum dado de score por fold para plotar.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    g = sns.catplot(
        data=df_folds, x="classifier", y="all_scores", col="dataset_name",
        kind="box", col_wrap=3, palette="muted", height=4, aspect=1.2, sharey=False
    )
    g.fig.suptitle("Distribuição da Acurácia por Fold", y=1.03, fontsize=16)
    g.set_axis_labels("Classificador", "Acurácia (por Fold)")
    g.set_titles("Dataset: {col_name}")
    g.set_xticklabels(rotation=45, ha='right')

    for ax in g.axes.flat:
        try:
            n_classes = df[df['dataset_name'] == ax.get_title().replace('Dataset: ', '')]['n_classes'].iloc[0]
            random_acc = 1 / n_classes if n_classes > 0 else 0.5
            ax.axhline(y=random_acc, color='r', linestyle='--', linewidth=1, alpha=0.7, label=f'Aleatório ({random_acc:.1%})')
            ax.legend(loc='lower right', fontsize='small')
        except (IndexError, ZeroDivisionError):
            pass # Ignora se não conseguir encontrar n_classes para o dataset
        ax.set_ylim(bottom=min(0, df_folds['all_scores'].min() * 0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(output_dir, "accuracy_distribution_by_fold.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Gráfico de distribuição de acurácia salvo em: {save_path}")


def plot_performance_by_mode(df: pd.DataFrame, output_dir: str):
    """
    Compara o desempenho (acurácia) dos diferentes modos de classificação.
    """
    if 'classification_mode' not in df.columns or df['classification_mode'].nunique() < 2:
        print("Apenas um modo de classificação encontrado. Pulando gráfico de comparação de modos.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        data=df, x="dataset_name", y="accuracy_mean", hue="classification_mode",
        palette="Set2", estimator=np.mean, errorbar="sd"
    )
    ax.set_title("Desempenho por Modo de Classificação", fontsize=16, pad=20)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Acurácia Média", fontsize=12)
    ax.set_ylim(0, max(1.0, df["accuracy_mean"].max() * 1.1))
    plt.xticks(rotation=45, ha='right')
    ax.legend(title="Modo de Classificação")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "performance_by_classification_mode.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Gráfico de desempenho por modo de classificação salvo em: {save_path}")


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
        plot_performance_by_mode(df, PLOTS_OUTPUT_DIR)
        plot_accuracy_vs_features(df, PLOTS_OUTPUT_DIR)
        plot_accuracy_distribution(df, PLOTS_OUTPUT_DIR)
        plot_accuracy_rank_over_datasets(df, PLOTS_OUTPUT_DIR)
        plot_confusion_matrices(df, PLOTS_OUTPUT_DIR)

        print("\n--- Análise Concluída ---")
        print(f"Plots salvos em: {os.path.abspath(PLOTS_OUTPUT_DIR)}")

    except Exception as e:
        print(f"\n[ERRO] Ocorreu um erro durante a análise: {e}")


if __name__ == "__main__":
    main()