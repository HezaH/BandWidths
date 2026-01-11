# lendo todas as instancias de uma classe

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import plotly.express as px

base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "data", "newdata", "global_analysis_inputs.json")

# 1. Ler o arquivo JSON 
with open(json_path, "r", encoding="utf-8") as f: 
    data = json.load(f) 

# 2. Converter para DataFrame 
df = pd.DataFrame(data)

list_of_columns = ['bandwidth', 'centrality', 'Instance', 'Edges', 'Nodes', 'LargestCompSize', 'Diameter', 'Node Connectivity', 'Edge Connectivity', 'Algebraic Connectivity', 'Graph Density', 'Average Shortest Path Length'] #df.columns.tolist()
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

    # Definig frequency of centrality usage
    freq_df = df_instance["centrality"].value_counts().reset_index()
    freq_df.columns = ["centrality", "frequency"]

    # transformar a coluna em categórica com ordem definida
    freq_df["centrality"] = pd.Categorical(freq_df["centrality"], categories=order, ordered=True)
    # ordenar pelo nível categórico
    freq_df = freq_df.sort_values("centrality").reset_index(drop=True)
    for c in list_of_columns[2::]:
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
    # if len((better_bandwidth)) > 1:
    #     better_bandwidth = better_bandwidth.iloc[[0]]
    best_solutions = pd.concat([best_solutions, better_bandwidth], ignore_index=True)

best_solutions = best_solutions.reset_index(drop=True)
best_adap = best_solutions.drop_duplicates(subset=["Instance"], keep="first").reset_index(drop=True)
df_repeated = best_solutions[best_solutions.duplicated(subset=["Instance"], keep=False)]


