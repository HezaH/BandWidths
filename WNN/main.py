import os
import json
import math
from datetime import datetime
from pipeline import build_binary_representations
from reading_graphs import process_dataset
from graph_classifier import (
    GraphClassifier,
    benchmark_classifiers
)
import numpy as np

# --------------------------------------------------------------------------------
# 0) Execução: PARTE 1 (extração) seguida da PARTE 2 (KernelCanvas++)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 1) CONFIGURAÇÃO
# --------------------------------------------------------------------------------
BASE_DIR = os.path.join(os.getcwd(),"WNN", "data_sets")
N_WORKERS = None  # None = usa todos os núcleos disponíveis
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "experiments.json")


def _jsonable(value):
    """Converte recursivamente tipos NumPy e valores não finitos para JSON."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_results(results):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(_jsonable(results), file, indent=2, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    # Ajuste aqui os nomes das pastas de dataset que você tem disponíveis.
    # Cada nome deve corresponder a uma pasta dentro de BASE_DIR contendo os
    # arquivos <NOME>_A.txt, <NOME>_graph_indicator.txt, etc.
    datasets_to_process = os.listdir(BASE_DIR)

    # opcional: centralidades extras já filtradas por benchmark_centralities()
    extra_centralities = None  # ex.: {"Eigenvector": nx.eigenvector_centrality}
    kernel_strategy = np.random.choice(["fps", None])
    print(f"\nEstratégia de kernel escolhida: {kernel_strategy}")
    results = []

    # Opções: "single_rf", "wisard", "benchmark".
    classification_mode = np.random.choice(["single_rf", "wisard", "benchmark"])
    print(f"\nClassificação dos vetores binários (X) usando o modo: {classification_mode}")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset_name in datasets_to_process:
        print(f"\n=== Processando dataset {dataset_name} ===")
        try:
            # PARTE 1 -- extração paralela das métricas (M)
            csv_nodes, csv_edges = process_dataset(dataset_name, extra_centralities=extra_centralities)

            # PARTE 2 -- KernelCanvas++ com ECDF + K-means (Q)
            X, y, models, diagnostics = build_binary_representations(
                csv_nodes, edge_metrics_csv_path=csv_edges,
                n_kernels=8, bits_per_kernel=4, k_activate=2,kernel_strategy=kernel_strategy,
            )

            sample_ids = np.arange(len(y))

            # PARTE 3 -- Classificação dos vetores binários (X)
            if classification_mode == "single_rf":
                clf = GraphClassifier(
                    classifier_name="rf"
                )

                classifier_results = {
                    "rf": clf.evaluate(X, y, sample_ids=sample_ids)
                }

            elif classification_mode == "wisard":
                clf = GraphClassifier(
                    classifier_name="wisard"
                )

                classifier_results = {
                    "wisard": clf.evaluate(X, y, sample_ids=sample_ids)
                }

            else:
                classifier_results = benchmark_classifiers(
                    X,
                    y,
                    sample_ids=sample_ids
                )

            dataset_result = {
                "run_id": run_id,
                "dataset": {
                    "name": dataset_name,
                    "n_samples": int(len(y)),
                    "n_features": int(X.shape[1]),
                    "n_classes": int(len(np.unique(y))),
                    "class_distribution": {
                        str(label): int(count)
                        for label, count in zip(*np.unique(y, return_counts=True))
                    },
                },
                "representation": {
                    "method": "KernelCanvasPP",
                    "kernel_strategy": kernel_strategy,
                    "n_kernels": 8,
                    "bits_per_kernel": 4,
                    "k_activate": 2,
                    "diagnostics": diagnostics,
                },
                "classification_mode": classification_mode,
                "classifiers": classifier_results,
            }
            results.append(dataset_result)
            _save_results(results)

            print(f"Resultados salvos em: {RESULTS_PATH}")

            print("\nExemplo -- vetor binário do grafo 0:")
            print(X[0])
            print("Rótulo correspondente:", y[0])

        except FileNotFoundError as e:
            print(f"[ERRO] {dataset_name}: {e}")