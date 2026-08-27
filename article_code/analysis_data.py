# lendo todas as instancias de uma classe

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from modules.utils.statistical_tests import StatisticalDecisionSupport, WilcoxonHypothesisValidator

base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "data", "newdata", "global_analysis_inputs.json")

# 1. Ler o arquivo JSON 
with open(json_path, "r", encoding="utf-8") as f: 
    data = json.load(f) 

# 2. Converter para DataFrame 
df = pd.DataFrame(data)

list_of_columns = ['Initial Bandwidth','bandwidth', 'centrality', 'Instance', 'Edges', 'Nodes', 'Diameter', 'Node Connectivity', 'Edge Connectivity', 'Algebraic Connectivity', 'Graph Density', 'Average Shortest Path Length'] #df.columns.tolist()
list_of_instances = df['Instance'].drop_duplicates().to_list()

legend_labels = { 'DEG': 'DEG: Degree', 'CLO': 'CLO: Closeness', 'BTW': 'BTW: Betweenness', 'EIG': 'EIG: Eigenvector', 'KAT': 'KAT: Katz Centrality', 'PRK': 'PRK: PageRank', 'HAR': 'HAR: Harmonic Centrality' }
order = ['Degree', 'Closeness', 'Betweenness', 'Eigenvector', 'Katz Centrality', 'PageRank', 'Harmonic Centrality']
abbr = {'Degree': 'DEG', 'Closeness': 'CLO', 'Betweenness': 'BTW', 'Eigenvector': 'EIG', 'Katz Centrality': 'KAT', 'PageRank': 'PRK', 'Harmonic Centrality': 'HAR'}

# Definir cores para cada sigla
color_map = {
    'DEG': 'blue',
    'CLO': 'green',
    'BTW': 'red',
    'EIG': 'purple',
    'KAT': 'orange',
    'PRK': 'brown',
    'HAR': 'pink'
}

best_solutions = pd.DataFrame()

frequences = pd.DataFrame()

for instance in list_of_instances:
    plt_path = os.path.join(base_dir, "data", "newdata", "analysis_results")
    image_path = os.path.join(plt_path, f"plot_frequency_{instance}.png")
    html_path = os.path.join(plt_path, f"grafico_bandwidth_{instance}.html")
    # Filtering by instance
    df_instance = df[df['Instance'] == instance][list_of_columns]


    # Filtrar pelo instance
    df_instance = df[df['Instance'] == instance][list_of_columns]

    # Garantir ordem categórica
    df_instance['centrality'] = pd.Categorical(
        df_instance['centrality'],
        categories=order,
        ordered=True
    )

    # Criar coluna com sigla
    df_instance['centrality_abbr'] = df_instance['centrality'].map(abbr)

    if not os.path.exists(html_path):
        # Criar violin plot
        fig = px.violin(
            df_instance,
            x="centrality_abbr",
            y="bandwidth",
            color="centrality_abbr",
            box=True,
            points="all",
            category_orders={"centrality_abbr": [abbr[c] for c in order]},
            color_discrete_map=color_map,
            labels={"centrality_abbr": "Centralidade (sigla)", "bandwidth": "Bandwidth"},
            title=f"Distribuição de Bandwidth por Centralidade - Instância {instance}"
        )

        # Atualizar legenda para mostrar Sigla: Nome completo
        fig.for_each_trace(
            lambda t: t.update(name=legend_labels[t.name])
        )

        fig.update_layout(legend_title_text="Centralidade")
        fig.write_html(html_path, include_plotlyjs="cdn")
        # fig.write_image(html_path.replace('.html', '.jpeg'), format="jpeg")

    # Definig frequency of centrality usage
    freq_df = df_instance["centrality"].value_counts().reset_index()
    freq_df.columns = ["centrality", "frequency"]

    # transformar a coluna em categórica com ordem definida
    freq_df["centrality"] = pd.Categorical(freq_df["centrality"], categories=order, ordered=True)
    # ordenar pelo nível categórico
    freq_df = freq_df.sort_values("centrality").reset_index(drop=True)
    for c in list_of_columns[3::]:
        freq_df[c] = df_instance[c].iloc[0]

    frequences = pd.concat([frequences, freq_df], ignore_index=True)
    if not os.path.exists(image_path):
        plt.figure(figsize=(10, 6))
        plt.bar(freq_df["centrality"], freq_df["frequency"], color='skyblue')
        plt.xlabel("Centrality Measures")
        plt.ylabel("Frequency")
        plt.title(f"Frequency of Centrality Measures for Instance: {instance}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        os.makedirs(plt_path, exist_ok=True)
        plt.savefig(image_path)

    # Setting better solutions of each instance
    df_instance = df_instance.drop_duplicates().reset_index(drop=True)
    better_bandwidth = df_instance[df_instance['bandwidth'] == df_instance['bandwidth'].min()]

    best_solutions = pd.concat([best_solutions, better_bandwidth], ignore_index=True)

best_solutions = best_solutions.reset_index(drop=True)
best_solutions["ReasonEdgeNodes"] = best_solutions["Edges"] / best_solutions["Nodes"]
best_adap = best_solutions.drop_duplicates(subset=["Instance"], keep="first").reset_index(drop=True)
sol = best_adap[list_of_columns[:6]]
sol['Decay %'] = round(100* ((best_adap["Initial Bandwidth"] - best_adap["bandwidth"])/best_adap["Initial Bandwidth"]), 2)
df_repeated = best_solutions[best_solutions.duplicated(subset=["Instance"], keep=False)]

x = 'Instance'
for y in list_of_columns[4::]:

    fig_path = os.path.join(plt_path, f"instace_{y}.jpeg")
    
    if not os.path.exists(fig_path):
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=x, y=y, data=best_adap)

        # Calcula a média da coluna Graph Density
        avr = best_adap[y].mean()

        # Adiciona a linha horizontal em vermelho
        plt.axhline(y=avr, color='red', linestyle='--', label=f'Average = {avr:.2f}')

        plt.xticks(rotation=45)
        plt.title(f'{y} by {x}')
        plt.xlabel(f'{x}')
        plt.ylabel(y)
        plt.legend()  # mostra a legenda com a média
        plt.savefig(fig_path)

    bins = pd.qcut(best_adap[y], q=5, duplicates='drop')
    best_adap[f'{y}_bins'] = bins

# Colunas prioritárias
first_cols = ['bandwidth', 'centrality', 'Instance']

# Demais colunas em ordem alfabética
other_cols = sorted([c for c in best_adap.columns if c not in first_cols])

# Nova ordem de colunas
new_order = first_cols + other_cols

# Reorganizar DataFrame
best_adap = best_adap[new_order]

list_of_centralities = list(set(best_adap['centrality'].to_list()))

for centrality in list_of_centralities:
    subset = best_adap[best_adap['centrality'] == centrality].reset_index(drop=True)
    for y in list_of_columns[4:]:
        fig_beans = os.path.join(plt_path, f"beans_{centrality}_{y}.html")
        # todas as categorias possíveis do qcut (mantém ordem crescente)
        all_bins = best_adap[f'{y}_bins'].cat.categories

        # contar frequência no subset
        freq = subset[f'{y}_bins'].value_counts(sort=False)

        # reindexar para incluir bins vazios e manter ordem
        freq = freq.reindex(all_bins, fill_value=0).reset_index()
        freq.columns = ['Bin', 'Count']

        # converter os bins para string só na hora de plotar
        freq['Bin'] = freq['Bin'].astype(str)

        if not os.path.exists(fig_beans):
            fig = px.bar(
                freq,
                x="Bin",
                y="Count",
                text="Count",
                labels={"Bin": "Intervalo (qcut)", "Count": "Frequência"},
                title=f"Distribuição de {y} em {len(all_bins)} grupos, pela centralidade {centrality}"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
            fig.write_html(fig_beans, include_plotlyjs="cdn")
            # fig.write_image(fig_beans.replace('.html', '.jpeg'), format="jpeg", scale=2)


def run_nonparametric_statistical_analysis(dataframe: pd.DataFrame, output_path: str) -> None:
    """Apply descriptive, Wilcoxon, Friedman and Holm analyses with logs."""
    print("[STAT] Starting non-parametric statistical analysis...")

    required_cols = ["Instance", "centrality", "bandwidth"]
    missing = [c for c in required_cols if c not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns for statistics: {missing}")

    os.makedirs(output_path, exist_ok=True)
    alpha: float = 0.05
    aggregation: str = "mean"
    lower_is_better: bool = True

    decision_engine = StatisticalDecisionSupport(alpha=alpha)
    wilcoxon_validator = WilcoxonHypothesisValidator(alpha=alpha)

    aggregated_df = decision_engine.aggregate_runs(
        dataframe,
        instance_col="Instance",
        method_col="centrality",
        value_col="bandwidth",
        aggregation=aggregation,
    )

    mean_by_method = (
        aggregated_df.groupby("centrality", as_index=False)["aggregated_value"]
        .mean()
        .sort_values("aggregated_value", ascending=lower_is_better)
        .reset_index(drop=True)
    )
    reference_method: str = str(mean_by_method.loc[0, "centrality"])
    methods = sorted(aggregated_df["centrality"].unique().tolist())
    competitors = [m for m in methods if m != reference_method]

    print(f"[STAT] Reference method selected by aggregated mean: {reference_method}")
    print(f"[STAT] Methods considered: {methods}")

    descriptive_df = decision_engine.descriptive_statistics(
        dataframe,
        method_col="centrality",
        value_col="bandwidth",
    )
    descriptive_df.to_csv(os.path.join(output_path, "descriptive_statistics.csv"), index=False)

    wilcoxon_df = wilcoxon_validator.compare_methods(
        dataframe,
        reference_method=reference_method,
        competitor_methods=competitors,
        instance_col="Instance",
        method_col="centrality",
        value_col="bandwidth",
        aggregation=aggregation,
        lower_is_better=lower_is_better,
    )
    wilcoxon_df.to_csv(os.path.join(output_path, "wilcoxon_vs_reference.csv"), index=False)

    friedman_outputs = decision_engine.friedman_holm_against_reference(
        dataframe,
        reference_method=reference_method,
        methods=methods,
        instance_col="Instance",
        method_col="centrality",
        value_col="bandwidth",
        aggregation=aggregation,
        lower_is_better=lower_is_better,
    )

    friedman_result = friedman_outputs["friedman_result"]
    posthoc_df = friedman_outputs["posthoc"]
    ranks_df = friedman_outputs["average_ranks"]

    posthoc_df.to_csv(os.path.join(output_path, "friedman_holm_posthoc.csv"), index=False)
    ranks_df.to_csv(os.path.join(output_path, "friedman_average_ranks.csv"), index=False)

    with open(os.path.join(output_path, "friedman_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("Friedman test summary\n")
        fh.write(f"statistic={friedman_result.statistic:.6f}\n")
        fh.write(f"p_value={friedman_result.p_value:.6f}\n")
        fh.write(f"alpha={friedman_result.alpha}\n")
        fh.write(f"reject_h0={friedman_result.reject_h0}\n")
        fh.write(f"n_instances={friedman_result.n_instances}\n")
        fh.write(f"n_methods={friedman_result.n_methods}\n")

    pairwise_matrix = pd.DataFrame(np.nan, index=methods, columns=methods)
    pivot = friedman_outputs["paired_matrix"]
    for i, method_i in enumerate(methods):
        pairwise_matrix.loc[method_i, method_i] = 1.0
        x = pivot[method_i].to_numpy(dtype=float)
        for j in range(i + 1, len(methods)):
            method_j = methods[j]
            y = pivot[method_j].to_numpy(dtype=float)
            pair_result = wilcoxon_validator.test_paired_samples(x, y)
            pairwise_matrix.loc[method_i, method_j] = pair_result.p_value
            pairwise_matrix.loc[method_j, method_i] = pair_result.p_value
    pairwise_matrix.to_csv(os.path.join(output_path, "pairwise_wilcoxon_pvalues.csv"))

    plt.figure(figsize=(10, 5))
    ranks_sorted = ranks_df.sort_values("average_rank", ascending=True)
    plt.bar(ranks_sorted["method"], ranks_sorted["average_rank"], color="darkorange")
    plt.title("Average Ranks by Method (Friedman)")
    plt.ylabel("Average Rank (lower is better)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "friedman_average_ranks.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(pairwise_matrix, annot=True, fmt=".3f", cmap="viridis_r", vmin=0.0, vmax=1.0)
    plt.title("Pairwise Wilcoxon p-values")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "pairwise_wilcoxon_pvalues.png"), dpi=150)
    plt.close()

    print(
        f"[STAT] Friedman: statistic={friedman_result.statistic:.6f}, "
        f"p-value={friedman_result.p_value:.6f}, reject_h0={friedman_result.reject_h0}"
    )
    print("[STAT] Top methods by average rank:")
    print(ranks_df.sort_values("average_rank", ascending=True).head(5).to_string(index=False))
    print(f"[STAT] Statistical outputs saved to: {output_path}")


stat_output_dir = os.path.join(base_dir, "data", "newdata", "analysis_results", "statistics")
run_nonparametric_statistical_analysis(df, stat_output_dir)