import os
from pipeline import build_binary_representations
from reading_graphs import process_dataset
from graph_classifier import (
    GraphClassifier,
    benchmark_classifiers
)

# --------------------------------------------------------------------------------
# 0) Execução: PARTE 1 (extração) seguida da PARTE 2 (KernelCanvas++)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 1) CONFIGURAÇÃO
# --------------------------------------------------------------------------------
BASE_DIR = os.path.join("WNN", "data_sets")
N_WORKERS = None  # None = usa todos os núcleos disponíveis


if __name__ == "__main__":
    # Ajuste aqui os nomes das pastas de dataset que você tem disponíveis.
    # Cada nome deve corresponder a uma pasta dentro de BASE_DIR contendo os
    # arquivos <NOME>_A.txt, <NOME>_graph_indicator.txt, etc.
    datasets_to_process = os.listdir(BASE_DIR)

    # opcional: centralidades extras já filtradas por benchmark_centralities()
    extra_centralities = None  # ex.: {"Eigenvector": nx.eigenvector_centrality}

    for dataset_name in datasets_to_process:
        print(f"\n=== Processando dataset {dataset_name} ===")
        try:
            # PARTE 1 -- extração paralela das métricas (M)
            csv_nodes, csv_edges = process_dataset(dataset_name, extra_centralities=extra_centralities)

            # PARTE 2 -- KernelCanvas++ com ECDF + K-means (Q)
            X, y, models, diagnostics = build_binary_representations(
                csv_nodes, edge_metrics_csv_path=csv_edges,
                n_kernels=8, bits_per_kernel=4, k_activate=2,
            )

            # PARTE 3 -- Classificação dos vetores binários (X) com SVM, RF, KNN e MLP
            if False:
                clf = GraphClassifier(
                    classifier_name="rf"
                )

                results = clf.evaluate(
                    X,
                    y
                )

            else:
                results = benchmark_classifiers(X, y)
 
            print(results)

            print("\nExemplo -- vetor binário do grafo 0:")
            print(X[0])
            print("Rótulo correspondente:", y[0])

        except FileNotFoundError as e:
            print(f"[ERRO] {dataset_name}: {e}")