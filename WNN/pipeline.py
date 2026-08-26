"""
==================================================================================
Pipeline de representação binária via KernelCanvas++ (Q), usando ECDF e
K-means, pronto para alimentar o classificador WiSARD (C).

A extração paralela de métricas de centralidade (M) foi movida para
WNN/reading_graphs.py para evitar duplicação de código.
==================================================================================
"""

from itertools import combinations
import numpy as np
import pandas as pd

try:
    from .reading_graphs import process_dataset
    from .kernel_canvas_pp import KernelCanvasPP
except ImportError:
    from reading_graphs import process_dataset
    from kernel_canvas_pp import KernelCanvasPP

# --------------------------------------------------------------------------------
# 6) NOVO — Histograma + Termômetro (para atributos CATEGÓRICOS, Seção 3.1)
# --------------------------------------------------------------------------------
def thermometer_encode(value, vmin, vmax, bits):
    """Termômetro linear clássico (equal-width)."""
    if vmax <= vmin:
        level = bits if value >= vmax else 0
    elif value <= vmin:
        level = 0
    elif value >= vmax:
        level = bits
    else:
        level = int((value - vmin) / (vmax - vmin) * bits)
    return np.array([1 if i < level else 0 for i in range(bits)], dtype=int)


def distributive_thermometer_encode(value, quantiles, bits):
    """Termômetro distributivo: limiares vêm dos quantis observados no treino."""
    level = int(np.searchsorted(quantiles, value, side="right"))
    level = min(level, bits)
    return np.array([1 if i < level else 0 for i in range(bits)], dtype=int)


def categorical_histogram_binarize(values, all_possible_values, bits_per_bin,
                                    use_distributive=False, quantiles_by_value=None):
    """
    values              : rótulos observados num grafo (lista de nós ou arestas)
    all_possible_values : vocabulário GLOBAL de rótulos (definido no treino)
    bits_per_bin        : bits do termômetro alocados a cada rótulo do vocabulário
    """
    counts = {v: 0 for v in all_possible_values}
    for v in values:
        if v in counts:
            counts[v] += 1
    total = sum(counts.values()) or 1

    bits = []
    for v in all_possible_values:
        if use_distributive:
            pct = counts[v] / total
            q = quantiles_by_value[v]
            bits.append(distributive_thermometer_encode(pct, q, bits_per_bin))
        else:
            bits.append(thermometer_encode(counts[v], 0, total, bits_per_bin))
    return np.concatenate(bits) if bits else np.array([], dtype=int)


def _compute_categorical_quantiles(df, group_col, cat_col, all_values, bits_per_bin):
    """
    Para cada valor categórico possível, calcula a distribuição da PORCENTAGEM
    de ocorrências por grafo (agrupando por `group_col`), e extrai os quantis
    dessa distribuição -- usados pelo termômetro distributivo (Seção 3.1).
    """
    pct_by_value = {v: [] for v in all_values}
    for _, sub in df.groupby(group_col):
        vals = sub[cat_col].dropna().tolist()
        total = len(vals) or 1
        for v in all_values:
            pct_by_value[v].append(vals.count(v) / total)

    quantiles_by_value = {}
    for v in all_values:
        qs = np.quantile(pct_by_value[v], np.linspace(0, 1, bits_per_bin, endpoint=False)[1:])
        quantiles_by_value[v] = qs
    return quantiles_by_value


def _diagnostico_correlacao(coluna, modelo, max_valores, random_state):
    """Correlação distância-valor x distância-Hamming, com amostragem para atributos
    contínuos com muitos valores únicos (evita explosão O(n²) de combinations)."""
    valores_unicos = np.sort(coluna.dropna().unique())
    if len(valores_unicos) > max_valores:
        rng = np.random.default_rng(random_state)
        valores_unicos = np.sort(rng.choice(valores_unicos, size=max_valores, replace=False))

    vetores = {v: modelo.transform_sequence(np.array([v])) for v in valores_unicos}
    dif_valor, dif_hamming = [], []
    for v1, v2 in combinations(valores_unicos, 2):
        dif_valor.append(abs(v1 - v2))
        dif_hamming.append(np.sum(vetores[v1] != vetores[v2]))

    if len(dif_valor) >= 2 and np.std(dif_hamming) > 0:
        return np.corrcoef(dif_valor, dif_hamming)[0, 1]
    return float("nan")

def _allocate_bit_budget(total_budget, attr_names, min_info=64, random_state=None):
    """
    Divide um orçamento total de bits entre vários atributos, seguindo o
    procedimento guloso descrito na Seção 4.2 do artigo:
    "an attribute i is first uniformly sampled, then its βi is uniformly
    sampled from the remaining bits ..., respecting min_info".

    total_budget : Bs ou Bns (o orçamento do bloco estrutural/não-estrutural)
    attr_names   : lista de nomes de atributos que vão dividir esse orçamento
    min_info     : mínimo de bits garantido para cada atributo (padrão do
                   artigo: 64)

    Retorna: dict {atributo: βᵢ}
    """
    if not attr_names:
        return {}

    rng = np.random.default_rng(random_state)
    ordem = list(attr_names)
    rng.shuffle(ordem)

    # segurança: se o orçamento não alcança min_info para todos, reduz o
    # mínimo proporcionalmente (evita orçamento negativo) e avisa
    if total_budget < min_info * len(ordem):
        min_info_ajustado = max(1, total_budget // len(ordem))
        print(f"  [aviso] orçamento de {total_budget} bits é pequeno demais para "
              f"{len(ordem)} atributos com min_info={min_info}; usando min_info={min_info_ajustado}")
        min_info = min_info_ajustado

    alocacao = {}
    orcamento_restante = total_budget
    n_restantes = len(ordem)
    for attr in ordem:
        n_restantes_depois = n_restantes - 1
        teto = orcamento_restante - min_info * n_restantes_depois
        if n_restantes_depois == 0:
            beta_i = orcamento_restante  # o último atributo fica com o que sobrar
        else:
            teto = max(min_info, teto)
            beta_i = int(rng.integers(min_info, teto + 1))
        alocacao[attr] = beta_i
        orcamento_restante -= beta_i
        n_restantes -= 1
    return alocacao


def _random_factor_pair(beta_i, max_n_kernels=None, random_state=None):
    """
    Fatora βᵢ em (n_kernels, bits_per_kernel), tal que idealmente
    n_kernels x bits_per_kernel == βᵢ (Algoritmo 1 / Seção 4.2).

    max_n_kernels : teto opcional para n_kernels (ex.: o nº de valores
        únicos do atributo nos dados de treino).

    Quando o teto É restritivo (max_n_kernels < βᵢ), sorteia n_kernels
    dentro do teto e usa bits_per_kernel = βᵢ // n_kernels -- pode sobrar
    um resto pequeno (< n_kernels bits) não aproveitado, mas evita
    fatorações degeneradas: sem essa checagem, βᵢ com poucos divisores
    (ex.: números primos) forçaria n_kernels=1 com todo o orçamento
    concentrado num único kernel -- um bloco de bits CONSTANTE (sempre 1),
    sem nenhum poder de discriminação, o que é pior do que perder alguns
    bits de arredondamento.
    """
    rng = np.random.default_rng(random_state)
    beta_i = max(1, int(beta_i))

    if max_n_kernels is not None and 0 < max_n_kernels < beta_i:
        n_kernels = int(rng.integers(1, max_n_kernels + 1))
        bits_per_kernel = max(1, beta_i // n_kernels)
        return n_kernels, bits_per_kernel

    # sem teto restritivo: fatoração exata por divisores (usa βᵢ por completo)
    divisores = [d for d in range(1, beta_i + 1) if beta_i % d == 0]
    bits_per_kernel = int(rng.choice(divisores))
    n_kernels = beta_i // bits_per_kernel
    return n_kernels, bits_per_kernel


def _plan_budget_allocation(attribute_cols, categorical_cols, edge_attribute_cols,
                             edge_categorical_cols, df, edge_df, B, alpha, min_info,
                             random_state, verbose):
    """
    Implementa as Equações 1-3 do artigo de ponta a ponta:
      1) B = α·Bs + (1-α)·Bns
      2) βᵢ de cada atributo, sorteado dentro do orçamento do seu bloco
      3) para contínuos: βᵢ = n_kernels_i x bits_per_kernel_i (fatoração aleatória)
         para categóricos: bits_per_bin_i = βᵢ / nº de valores distintos (Seção 4.2)

    "Estrutural" = as centralidades vindas da etapa M (degree, pagerank,
    onion_layer, k_core, extras) -- tudo que NÃO é node_attr_*/node_label/
    aresta é considerado não-estrutural (Bns), incluindo TODOS os atributos
    de aresta.

    Retorna dicts prontos para substituir n_kernels/bits_per_kernel/
    categorical_bits_per_bin em build_binary_representations.
    """
    rng_seed = random_state
    Bs = round(alpha * B)
    Bns = B - Bs

    estruturais = [c for c in attribute_cols if not c.startswith("node_attr_")]
    nao_estruturais_continuos_no = [c for c in attribute_cols if c.startswith("node_attr_")]

    # todo atributo não-estrutural (nó categórico + nó contínuo não-estrutural +
    # QUALQUER atributo de aresta) disputa o mesmo orçamento Bns
    nao_estruturais_continuos = nao_estruturais_continuos_no + list(edge_attribute_cols)
    nao_estruturais_categoricos = list(categorical_cols) + list(edge_categorical_cols)
    todos_nao_estruturais = nao_estruturais_continuos + nao_estruturais_categoricos

    beta_estrutural = _allocate_bit_budget(Bs, estruturais, min_info, rng_seed)
    beta_nao_estrutural = _allocate_bit_budget(Bns, todos_nao_estruturais, min_info, rng_seed)
    beta_todos = {**beta_estrutural, **beta_nao_estrutural}

    n_kernels_plan, bits_per_kernel_plan, bits_per_bin_plan = {}, {}, {}

    for attr in estruturais + nao_estruturais_continuos:
        fonte = edge_df if attr in edge_attribute_cols else df
        # teto = nº de valores únicos REALMENTE observados -- evita fatorações
        # extremas (ex.: 587 kernels de 1 bit) que depois seriam cortadas de
        # volta pelo clamp de "kernels efetivos", desperdiçando o orçamento βᵢ
        max_n_kernels = max(1, fonte[attr].dropna().nunique())
        n_k, b_k = _random_factor_pair(beta_todos[attr], max_n_kernels, rng_seed)
        n_kernels_plan[attr] = n_k
        bits_per_kernel_plan[attr] = b_k

    for col in nao_estruturais_categoricos:
        fonte = edge_df if col in edge_categorical_cols else df
        n_distintos = max(1, fonte[col].dropna().nunique())
        bits_per_bin_plan[col] = max(1, beta_todos[col] // n_distintos)

    if verbose:
        print(f"\n  [alocação automática] B={B}, α={alpha}  ->  Bs={Bs}, Bns={Bns}")
        for attr in estruturais:
            print(f"    [estrutural]     {attr:15s}: βᵢ={beta_todos[attr]:4d}  "
                  f"-> n_kernels={n_kernels_plan[attr]}, bits_per_kernel={bits_per_kernel_plan[attr]}")
        for attr in nao_estruturais_continuos:
            print(f"    [não-estrutural] {attr:15s}: βᵢ={beta_todos[attr]:4d}  "
                  f"-> n_kernels={n_kernels_plan[attr]}, bits_per_kernel={bits_per_kernel_plan[attr]}")
        for col in nao_estruturais_categoricos:
            print(f"    [não-estrutural] {col:15s}: βᵢ={beta_todos[col]:4d}  "
                  f"-> bits_per_bin={bits_per_bin_plan[col]}")

    return n_kernels_plan, bits_per_kernel_plan, bits_per_bin_plan


# --------------------------------------------------------------------------------
# 7) PARTE 2: lê os CSVs (nó + aresta) e monta os vetores binários finais
# --------------------------------------------------------------------------------
def build_binary_representations(
    metrics_csv_path,
    edge_metrics_csv_path=None,
    attribute_cols=None,
    categorical_cols=None,
    kernel_strategy=None, #"fps"
    n_kernels=16, bits_per_kernel=4, k_activate=3,
    activation_rate=None,
    categorical_bits_per_bin=8, categorical_use_distributive=True,
    B=None, alpha=None, min_info=64,
    random_state=0,
    check_correlation=True, max_values_for_correlation=200,
    verbose=True,
):
    """
    Lê o(s) CSV(s) gerado(s) por process_dataset() e constrói, para cada
    grafo, o vetor binário final B (etapa Q completa):
      - atributos CONTÍNUOS (estruturais + node_attr_* + edge_attr_*) -> KernelCanvasPP
      - atributos CATEGÓRICOS (node_label + edge_label) -> histograma + termômetro

    attribute_cols / categorical_cols : None = detecta automaticamente.
        Colunas terminadas em "_label" são tratadas como categóricas por
        convenção (node_label, edge_label); todas as outras colunas
        numéricas (exceto graph_id/graph_label/node_id) são contínuas.

    n_kernels, bits_per_kernel, k_activate, categorical_bits_per_bin :
        aceitam valor único (aplicado a todos os atributos) ou dict
        {"atributo": valor}. Quando o dict não cobre um atributo, usa-se o
        valor padrão do PRÓPRIO parâmetro, nunca um número fixo genérico.
        IGNORADOS se B e alpha forem definidos (ver abaixo).

    activation_rate : se definido (ex.: 0.07, o valor fixo do artigo -- Eq. 4),
        SUBSTITUI k_activate: cada atributo passa a usar
        k_activate_i = round(activation_rate * n_kernels_efetivo_i).
        Se None e B/alpha estiverem definidos, assume 0.07 automaticamente
        (fiel à metodologia do artigo). Se None e B/alpha não definidos,
        mantém o comportamento manual de k_activate.

    B, alpha, min_info : quando AMBOS B e alpha são definidos, ativa a
        ALOCAÇÃO AUTOMÁTICA de bits (Eqs. 1-3 do artigo): o orçamento total
        B é dividido entre Bs (estrutural, fração α) e Bns (não-estrutural,
        fração 1-α); cada atributo recebe uma fatia βᵢ sorteada dentro do
        seu bloco (respeitando min_info bits mínimos), e essa fatia é
        fatorada em (n_kernels, bits_per_kernel) para contínuos, ou vira
        bits_per_bin = βᵢ / nº de valores distintos para categóricos.
        Nesse modo, n_kernels/bits_per_kernel/categorical_bits_per_bin
        passados manualmente são ignorados.

    Retorna:
      X            : matriz numpy (n_grafos x B)
      graph_labels : array com o rótulo de cada grafo
      models       : dict com "continuous" (KernelCanvasPP por atributo) e
                     "categorical" (vocabulário/quantis por atributo)
      diagnostics  : dict {atributo contínuo: correlação distância-valor x distância-Hamming}
    """
    df = pd.read_csv(metrics_csv_path)
    edge_df = pd.read_csv(edge_metrics_csv_path) if edge_metrics_csv_path else None

    # ---------- Leitura e validação básica ----------
    # `df` contém métricas por nó (cada linha é um nó com seu graph_id)
    # `edge_df`, se fornecido, contém métricas por aresta. Ambos são
    # usados para construir representações contínuas (KernelCanvasPP)
    # e categóricas (histograma + termômetro) por grafo.

    id_cols = {"graph_id", "graph_label", "node_id"}
    if categorical_cols is None:
        categorical_cols = [c for c in df.columns if c not in id_cols and c.endswith("_label")]
    if attribute_cols is None:
        attribute_cols = [c for c in df.columns if c not in id_cols and c not in categorical_cols]

    edge_categorical_cols, edge_attribute_cols = [], []
    if edge_df is not None:
        edge_id_cols = {"graph_id"}
        edge_categorical_cols = [c for c in edge_df.columns if c not in edge_id_cols and c.endswith("_label")]
        edge_attribute_cols = [c for c in edge_df.columns if c not in edge_id_cols and c not in edge_categorical_cols]

    # ---------- Alocação automática de bits (opcional) ----------
    # Quando B e alpha são fornecidos, acionamos o planejamento automático
    # de orçamento de bits (Eqs. 1-3). Isso sobrescreve os valores manuais
    # de n_kernels/bits_per_kernel/categorical_bits_per_bin.

    # ---- NOVO: alocação automática de bits (Eqs. 1-3), se B e alpha forem definidos ----
    if B is not None and alpha is not None:
        if verbose:
            print(f"\n=== [Q] Alocação automática de bits ativada (B={B}, α={alpha}, min_info={min_info}) ===")
        n_kernels, bits_per_kernel, categorical_bits_per_bin = _plan_budget_allocation(
            attribute_cols, categorical_cols, edge_attribute_cols, edge_categorical_cols,
            df, edge_df, B, alpha, min_info, random_state, verbose,
        )
        if activation_rate is None:
            activation_rate = 0.07  # Seção 3.2/4.2: taxa fixa do artigo

    if verbose:
        print(f"\n=== [Q] Construindo representação binária a partir de: {metrics_csv_path} ===")
        print(f"Atributos CONTÍNUOS de nó (KernelCanvas++): {attribute_cols}")
        print(f"Atributos CATEGÓRICOS de nó (histograma): {categorical_cols}")
        if edge_df is not None:
            print(f"Atributos CONTÍNUOS de aresta (KernelCanvas++): {edge_attribute_cols}")
            print(f"Atributos CATEGÓRICOS de aresta (histograma): {edge_categorical_cols}")

    def _param_for(param, attr, default):
        # CORREÇÃO: antes usava um fallback fixo (16) para qualquer parâmetro
        # quando o dict não cobria o atributo -- agora usa o default REAL
        # daquele parâmetro específico (16 para n_kernels, 4 para
        # bits_per_kernel, 3 para k_activate), evitando o bug de "vazamento"
        # silencioso que vimos com pagerank (virava 16 em vez do valor pedido).
        return param.get(attr, default) if isinstance(param, dict) else param

    # ---- ajusta KernelCanvasPP para cada atributo contínuo (nó) ----
    kcpp_models = {}
    diagnostics = {}
    for attr in attribute_cols:
        valores_treino = df[attr].dropna().values.reshape(-1, 1)
        if len(valores_treino) == 0:
            continue
        modelo = KernelCanvasPP(
            n_kernels=_param_for(n_kernels, attr, default=16),
            bits_per_kernel=_param_for(bits_per_kernel, attr, default=4),
            k_activate=_param_for(k_activate, attr, default=3),
            activation_rate=_param_for(activation_rate, attr, default=None),
            random_state=random_state,
        )

        if kernel_strategy == "fps":

            labels_treino = (
                df.loc[
                    df[attr].notna(),
                    "graph_label"
                ]
                .values
            )

            modelo.fit_fps_pipeline(
                valores_treino,
                labels_treino
            )

        else:

            modelo.fit(
                valores_treino
            )

        kcpp_models[attr] = modelo

        # Imprime as estatísticas de uso dos kernels
        report = kcpp_models[attr].kernel_report(
            df[attr].values.reshape(-1,1)
        )
        print(report)

        if verbose:
            print(f"  [nó-contínuo] '{attr}' -> {kcpp_models[attr].n_kernels} kernels efetivos")

        if check_correlation:
            diagnostics[attr] = _diagnostico_correlacao(
                df[attr], kcpp_models[attr], max_values_for_correlation, random_state
            )
            if verbose:
                print(f"    correlação distância-valor x distância-Hamming: {diagnostics[attr]:.3f}")

    # Para cada atributo contínuo:
    # - obtém valores de treino (linhas do CSV) e cria um `KernelCanvasPP`
    # - se `kernel_strategy=='fps'`, usa a pipeline FPS+MI para selecionar
    #   kernels informativos; caso contrário, usa KMeans simples via `fit()`
    # - armazena o modelo no dicionário `kcpp_models` para uso posterior
    # - opcionalmente calcula uma métrica diagnóstica de correlação

    # ---- ajusta vocabulário/quantis para cada atributo categórico (nó) ----
    categorical_models = {}
    for col in categorical_cols:
        valores = df[col].dropna()
        if valores.empty:
            continue
        all_values = sorted(valores.unique().tolist())
        bits_per_bin_col = _param_for(categorical_bits_per_bin, col, default=8)
        quantiles_by_value = None
        if categorical_use_distributive:
            quantiles_by_value = _compute_categorical_quantiles(
                df, "graph_id", col, all_values, bits_per_bin_col
            )
        categorical_models[col] = {
            "all_values": all_values,
            "bits_per_bin": bits_per_bin_col,
            "use_distributive": categorical_use_distributive,
            "quantiles_by_value": quantiles_by_value,
        }
        if verbose:
            print(f"  [nó-categórico] '{col}' -> vocabulário: {all_values}")

    # Para atributos categóricos:
    # - constrói o vocabulário global `all_values` observado no treino
    # - decide `bits_per_bin` (pode ser uniforme ou derivado via alocação)
    # - se `use_distributive` ativado, pré-computa quantis por valor para o
    #   termômetro distributivo, que mapeia frequências relativas para bits

    # ---- o mesmo, agora para atributos de ARESTA ----
    edge_kcpp_models, edge_categorical_models = {}, {}
    if edge_df is not None:
        for attr in edge_attribute_cols:
            valores_treino = edge_df[attr].dropna().values.reshape(-1, 1)
            if len(valores_treino) == 0:
                continue
            edge_kcpp_models[attr] = KernelCanvasPP(
                n_kernels=_param_for(n_kernels, attr, default=16),
                bits_per_kernel=_param_for(bits_per_kernel, attr, default=4),
                k_activate=_param_for(k_activate, attr, default=3),
                activation_rate=_param_for(activation_rate, attr, default=None),
                random_state=random_state,
            ).fit(valores_treino)
            if verbose:
                print(f"  [aresta-contínuo] '{attr}' -> {edge_kcpp_models[attr].n_kernels} kernels efetivos")

        for col in edge_categorical_cols:
            valores = edge_df[col].dropna()
            if valores.empty:
                continue
            all_values = sorted(valores.unique().tolist())
            bits_per_bin_col = _param_for(categorical_bits_per_bin, col, default=8)
            quantiles_by_value = None
            if categorical_use_distributive:
                quantiles_by_value = _compute_categorical_quantiles(
                    edge_df, "graph_id", col, all_values, bits_per_bin_col
                )
            edge_categorical_models[col] = {
                "all_values": all_values,
                "bits_per_bin": bits_per_bin_col,
                "use_distributive": categorical_use_distributive,
                "quantiles_by_value": quantiles_by_value,
            }
            if verbose:
                print(f"  [aresta-categórico] '{col}' -> vocabulário: {all_values}")

    # ---- monta o vetor binário final de cada grafo (Bs ∪ Bns, Eq. 5) ----
    graph_ids = sorted(df["graph_id"].unique())
    X, graph_labels = [], []
    for g_id in graph_ids:
        sub = df[df["graph_id"] == g_id]
        graph_labels.append(sub["graph_label"].iloc[0])

        pedacos = []
        for attr, modelo in kcpp_models.items():
            valores_do_grafo = sub[attr].dropna().values.reshape(-1, 1)
            pedacos.append(modelo.transform_sequence(valores_do_grafo))

        for col, cfg in categorical_models.items():
            valores_do_grafo = sub[col].dropna().tolist()
            pedacos.append(categorical_histogram_binarize(
                valores_do_grafo, cfg["all_values"], cfg["bits_per_bin"],
                cfg["use_distributive"], cfg["quantiles_by_value"],
            ))

        if edge_df is not None:
            esub = edge_df[edge_df["graph_id"] == g_id]
            for attr, modelo in edge_kcpp_models.items():
                valores_do_grafo = esub[attr].dropna().values.reshape(-1, 1)
                pedacos.append(modelo.transform_sequence(valores_do_grafo))
            for col, cfg in edge_categorical_models.items():
                valores_do_grafo = esub[col].dropna().tolist()
                pedacos.append(categorical_histogram_binarize(
                    valores_do_grafo, cfg["all_values"], cfg["bits_per_bin"],
                    cfg["use_distributive"], cfg["quantiles_by_value"],
                ))

        # Concatena os pedaços (contínuos + categóricos de nós e arestas) na
        # ordem fixa definida pelo processamento acima. Cada `pedacos` é um
        # vetor binário (resultante de transform_sequence ou histogramas).
        vetor_final = np.concatenate(pedacos) if pedacos else np.array([], dtype=int)
        X.append(vetor_final)

    X = np.array(X)
    graph_labels = np.array(graph_labels)

    models = {
        "node_continuous": kcpp_models,
        "node_categorical": categorical_models,
        "edge_continuous": edge_kcpp_models,
        "edge_categorical": edge_categorical_models,
    }

    if verbose:
        print(f"Vetores binários construídos: {X.shape[0]} grafos x {X.shape[1]} bits cada")

    return X, graph_labels, models, diagnostics
