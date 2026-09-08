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

# Tenta importar o script de análise para execução automática no final
try:
    from analyze_results import main as run_analysis
except ImportError:
    run_analysis = None
    print("[AVISO] Script de análise 'analyze_results.py' não encontrado. A análise final não será executada.")

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
    """Carrega resultados existentes, anexa os novos e salva o arquivo JSON."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    
    all_results = []
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                # Evita erro se o arquivo estiver vazio
                content = f.read()
                if content:
                    all_results = json.loads(content)
        except json.JSONDecodeError:
            print(f"[AVISO] O arquivo {RESULTS_PATH} está corrompido ou vazio. Um novo arquivo será criado.")
            all_results = []

    all_results.extend(results)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(_jsonable(all_results), f, indent=2, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    datasets_to_process = os.listdir(BASE_DIR)
    extra_centralities = None  # ex.: {"Eigenvector": nx.eigenvector_centrality}

    # --- Definição das configurações de experimento para avaliação ---
    # Cada dicionário define uma combinação de parâmetros a ser testada.
    experiment_configs = [
        # Configuração 1: K-means (padrão) com parâmetros base e benchmark de classificadores
        {"kernel_strategy": None, "n_kernels": 8, "bits_per_kernel": 4, "k_activate": 2, "classification_mode": "benchmark"},
        # Configuração 2: K-means com mais kernels
        {"kernel_strategy": None, "n_kernels": 16, "bits_per_kernel": 4, "k_activate": 3, "classification_mode": "benchmark"},
        # Configuração 3: FPS (Farthest Point Sampling) com parâmetros base e benchmark
        {"kernel_strategy": "fps", "n_kernels": 8, "bits_per_kernel": 4, "k_activate": 2, "classification_mode": "benchmark"},
        # Configuração 4: FPS com mais kernels e apenas WiSARD
        {"kernel_strategy": "fps", "n_kernels": 16, "bits_per_kernel": 8, "k_activate": 4, "classification_mode": "wisard"},
    ]

    for i, config in enumerate(experiment_configs):
        print("\n" + "="*80)
        print(f"--- INICIANDO EXPERIMENTO {i+1}/{len(experiment_configs)} ---")
        print(f"Configuração: {config}")
        print("="*80 + "\n")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_for_config = []

        for dataset_name in datasets_to_process:
            print(f"\n--- Processando dataset {dataset_name} para a configuração atual ---")
            try:
                # PARTE 1 -- extração paralela das métricas (M)
                csv_nodes, csv_edges = process_dataset(dataset_name, extra_centralities=extra_centralities)

                # PARTE 2 -- KernelCanvas++ com ECDF + K-means (Q)
                X, y, models, diagnostics = build_binary_representations(
                    csv_nodes, edge_metrics_csv_path=csv_edges,
                    n_kernels=config["n_kernels"],
                    bits_per_kernel=config["bits_per_kernel"],
                    k_activate=config["k_activate"],
                    kernel_strategy=config["kernel_strategy"],
                )

                sample_ids = np.arange(len(y))

                # PARTE 3 -- Classificação dos vetores binários (X)
                classification_mode = config["classification_mode"]
                if classification_mode == "single_rf":
                    clf = GraphClassifier(classifier_name="rf")
                    classifier_results = {"rf": clf.evaluate(X, y, sample_ids=sample_ids)}
                elif classification_mode == "wisard":
                    clf = GraphClassifier(classifier_name="wisard")
                    classifier_results = {"wisard": clf.evaluate(X, y, sample_ids=sample_ids)}
                else: # "benchmark"
                    classifier_results = benchmark_classifiers(X, y, sample_ids=sample_ids)

                dataset_result = {
                    "run_id": run_id,
                    "dataset": {
                        "name": dataset_name,
                        "n_samples": int(len(y)),
                        "n_features_in": len(diagnostics), # Número de métricas de entrada
                        "n_classes": int(len(np.unique(y))),
                        "class_distribution": {str(label): int(count) for label, count in zip(*np.unique(y, return_counts=True))},
                    },
                    "representation": {
                        "method": "KernelCanvasPP",
                        "kernel_strategy": config["kernel_strategy"],
                        "n_kernels": config["n_kernels"],
                        "bits_per_kernel": config["bits_per_kernel"],
                        "k_activate": config["k_activate"],
                        "n_features_out": int(X.shape[1]),
                        "diagnostics": diagnostics,
                    },
                    "classification_mode": classification_mode,
                    "classifiers": classifier_results,
                }
                results_for_config.append(dataset_result)

                print(f"\nResultados para {dataset_name} com config {i+1} processados.")
                print(f"Vetor binário resultante: {X.shape[0]} amostras x {X.shape[1]} features")

            except FileNotFoundError as e:
                print(f"[ERRO] {dataset_name}: {e}")
            except Exception as e:
                print(f"[ERRO INESPERADO] ao processar {dataset_name}: {e}")

        # Salva os resultados desta configuração no final do loop de datasets
        if results_for_config:
            _save_results(results_for_config)
            print(f"\nResultados da configuração {i+1} salvos em: {RESULTS_PATH}")

    # ---- NOVO: Executa a análise de resultados ao final de todos os datasets ----
    print("\n\n" + "="*50)
    print("=== Execução do pipeline principal concluída.      ===")
    print("=== Iniciando a análise e geração de gráficos...   ===")
    print("="*50)
    if run_analysis:
        try:
            run_analysis()
        except Exception as e:
            print(f"\n[ERRO] Falha ao executar a análise de resultados: {e}")