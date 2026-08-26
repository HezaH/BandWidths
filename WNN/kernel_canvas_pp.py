import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import cdist

"""
Módulo KernelCanvas++ (extensão do KernelCanvas original).

Fornece uma implementação que combina:
- normalização por ECDF (Probability Integral Transform) para mapear
    cada dimensão para [0,1];
- seleção/posicionamento de kernels via K-means ou via pipeline FPS+MI;
- transformação de uma sequência de pontos em um vetor binário por ativação
    dos kernels (cada kernel corresponde a um grupo de bits).

Os comentários neste arquivo explicam os passos principais de cada método
para facilitar a compreensão e manutenção.
"""
# --------------------------------------------------------------------------------
# 1) KernelCanvas++ — ECDF + K-means
# --------------------------------------------------------------------------------
class KernelCanvasPP:
    """
    Extensão do KernelCanvas original [28] (Seção 3.2 do artigo):
      - normalização por ECDF (Probability Integral Transform) -> kernels em [0,1]^d
      - kernels posicionados pelos centróides de um K-means
    
        Parâmetros principais:
        - n_kernels: número desejado de kernels/centroides finais.
        - bits_per_kernel: quantos bits no vetor de saída são atribuídos a cada kernel.
        - k_activate: quantos kernels mais próximos são ativados por ponto (pode ser
            recalculado a partir de `activation_rate` durante o fit).
        - activation_rate: alternativa para definir `k_activate` como uma fração
            do número de kernels efetivos (Eq.4 do artigo).
        - random_state: semente usada para operações aleatórias (KMeans, FPS inicial).

        Fluxo geral:
        1) `fit()` normaliza os pontos pelo ECDF e posiciona kernels (KMeans).
        2) `transform_sequence()` transforma novos pontos em um vetor binário
             ativando os kernels mais próximos.
        3) Métodos auxiliares (`fit_fps_pipeline`) permitem construir kernels através
             de uma pipeline que usa Farthest Point Sampling ponderado por densidade,
             filtra por uso e por informação mútua e remove redundâncias.
    """

    def __init__(self, n_kernels=16, bits_per_kernel=4, k_activate=3,
                 activation_rate=None, random_state=None):
        self.n_kernels = n_kernels
        self.bits_per_kernel = bits_per_kernel
        self.k_activate = min(k_activate, n_kernels)
        # NOVO: se activation_rate for passado, ele MANDA sobre k_activate --
        # replica a Eq. 4 do artigo: k_activate = round(activationRate x n_kernels),
        # com activationRate fixo em 0.07 na metodologia original. Calculado
        # dentro do fit(), usando o n_kernels EFETIVO (pode ser menor que o
        # pedido, se houver poucos valores únicos -- ver conversa sobre
        # "kernels efetivos").
        self.activation_rate = activation_rate
        self.random_state = random_state
        self.kernels_ = None
        self.train_sorted_ = None

    def fit(self, training_points):
        training_points = np.asarray(training_points, dtype=float)
        if training_points.ndim == 1:
            training_points = training_points.reshape(-1, 1)

        n_dims = training_points.shape[1]
        self.train_sorted_ = [np.sort(training_points[:, j]) for j in range(n_dims)]
        # Normaliza cada dimensão usando ECDF para obter valores em [0,1].
        # Isso garante que kernels e distâncias sejam comparáveis entre
        # diferentes escalas de dimensão.
        norm_pts = self._ecdf_transform_matrix(training_points)

        # limita tanto pelo nº de linhas quanto pelo nº de valores DISTINTOS
        # (evita kernels duplicados/desperdiçados -- ver conversa sobre "kernels efetivos")
        n_valores_unicos = len(np.unique(norm_pts, axis=0))
        n_clusters = min(self.n_kernels, len(norm_pts), n_valores_unicos)

        # Aplica K-means no espaço normalizado para posicionar os centróides
        # que serão usados como kernels. O KMeans opera em `norm_pts`.
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_state)
        km.fit(norm_pts)
        self.kernels_ = km.cluster_centers_
        self.n_kernels = n_clusters

        if self.activation_rate is not None:
            # Eq. 4 do artigo -- k_activate NASCE da taxa fixa, não é escolhido direto
            self.k_activate = max(1, round(self.activation_rate * n_clusters))
        else:
            # comportamento antigo: k_activate fixo, só "clampado" por segurança
            self.k_activate = min(self.k_activate, n_clusters)
        return self

    def _ecdf_transform_matrix(self, points):
        return np.column_stack([
            self._ecdf_transform(points[:, j], self.train_sorted_[j])
            for j in range(points.shape[1])
        ])

    @staticmethod
    def _ecdf_transform(arr, sorted_train_arr):
        # `ranks` é o número de elementos do treino menores ou iguais a cada
        # elemento de `arr` — ao dividir pelo tamanho do vetor de treino
        # obtemos uma estimativa empírica da CDF no ponto.
        ranks = np.searchsorted(sorted_train_arr, arr, side="right")
        return ranks / max(len(sorted_train_arr), 1)

    def transform_sequence(self, sequence_points):
        B = self.n_kernels * self.bits_per_kernel
        out = np.zeros(B, dtype=int)
        if sequence_points is None or len(sequence_points) == 0:
            return out

        sequence_points = np.asarray(sequence_points, dtype=float)
        if sequence_points.ndim == 1:
            sequence_points = sequence_points.reshape(-1, 1)

        norm_seq = self._ecdf_transform_matrix(sequence_points)

        for p in norm_seq:
            dists = np.linalg.norm(self.kernels_ - p, axis=1)
            idxs = np.argsort(dists)[: self.k_activate]
            # Para cada kernel ativado, definimos o bloco de `bits_per_kernel`
            # correspondentes como 1 no vetor de saída.
            for k_idx in idxs:
                start = k_idx * self.bits_per_kernel
                out[start: start + self.bits_per_kernel] = 1
        return out

    def plot_embedding_2d(
        self,
        sequence_points,
        show_connections=False,
        annotate_kernels=True,
        figsize=(12, 10)
    ):

        if self.kernels_ is None:
            raise ValueError("Execute fit() primeiro")

        pts = np.asarray(sequence_points, dtype=float)

        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)

        pts_norm = self._ecdf_transform_matrix(pts)

        combined = np.vstack([
            pts_norm,
            self.kernels_
        ])

        d = combined.shape[1]

        if d > 2:

            pca = PCA(n_components=2)

            embedding = pca.fit_transform(combined)

            explained = pca.explained_variance_ratio_.sum()

            title = f"PCA → R² ({explained:.2%})"

        elif d == 1:

            embedding = np.column_stack([
                combined[:, 0],
                np.zeros(len(combined))
            ])

            title = "R¹ → R²"

        else:

            embedding = combined

            title = "Espaço Original R²"

        n_pts = len(pts_norm)

        pts_2d = embedding[:n_pts]
        kernels_2d = embedding[n_pts:]

        activated = set()
        point_to_kernel = []

        for p in pts_norm:

            dists = np.linalg.norm(
                self.kernels_ - p,
                axis=1
            )

            idxs = np.argsort(dists)[: self.k_activate]

            activated.update(idxs)

            point_to_kernel.append(idxs)

        activated = list(activated)

        fig, ax = plt.subplots(figsize=figsize)

        # -------------------------
        # Voronoi
        # -------------------------

        if len(kernels_2d) >= 4:

            vor = Voronoi(kernels_2d)

            voronoi_plot_2d(
                vor,
                ax=ax,
                show_vertices=False,
                line_width=1,
                line_alpha=0.5,
                point_size=0
            )

        # -------------------------
        # Pontos
        # -------------------------

        ax.scatter(
            pts_2d[:, 0],
            pts_2d[:, 1],
            c="royalblue",
            alpha=0.5,
            s=40,
            label="Pontos"
        )

        # -------------------------
        # Kernels
        # -------------------------

        ax.scatter(
            kernels_2d[:, 0],
            kernels_2d[:, 1],
            c="red",
            marker="X",
            s=180,
            label="Kernels"
        )

        # -------------------------
        # Kernels ativados
        # -------------------------

        if activated:

            ax.scatter(
                kernels_2d[activated, 0],
                kernels_2d[activated, 1],
                c="lime",
                marker="*",
                s=350,
                edgecolors="black",
                label="Ativados"
            )

        # -------------------------
        # IDs
        # -------------------------

        if annotate_kernels:

            for idx, k in enumerate(kernels_2d):

                ax.text(
                    k[0],
                    k[1],
                    f"K{idx}",
                    fontsize=9
                )

        # -------------------------
        # Conexões
        # -------------------------

        if show_connections:

            for p_idx, kernel_idxs in enumerate(point_to_kernel):

                p = pts_2d[p_idx]

                for k_idx in kernel_idxs:

                    k = kernels_2d[k_idx]

                    ax.plot(
                        [p[0], k[0]],
                        [p[1], k[1]],
                        color="gray",
                        alpha=0.15
                    )

        ax.set_title(title)

        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_embedding_3d(
        self,
        sequence_points,
        show_connections=False,
        annotate_kernels=True,
        figsize=(12, 10)
    ):

        if self.kernels_ is None:
            raise ValueError("Execute fit() primeiro")

        pts = np.asarray(sequence_points, dtype=float)

        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)

        pts_norm = self._ecdf_transform_matrix(pts)

        combined = np.vstack([
            pts_norm,
            self.kernels_
        ])

        d = combined.shape[1]

        if d > 3:

            pca = PCA(n_components=3)

            embedding = pca.fit_transform(combined)

            explained = pca.explained_variance_ratio_.sum()

            title = f"PCA → R³ ({explained:.2%})"

        elif d == 2:

            embedding = np.column_stack([
                combined,
                np.zeros(len(combined))
            ])

            title = "R² → R³"

        elif d == 1:

            embedding = np.column_stack([
                combined[:, 0],
                np.zeros(len(combined)),
                np.zeros(len(combined))
            ])

            title = "R¹ → R³"

        else:

            embedding = combined

            title = "Espaço Original R³"

        n_pts = len(pts_norm)

        pts_3d = embedding[:n_pts]
        kernels_3d = embedding[n_pts:]

        activated = set()

        point_to_kernel = []

        for p in pts_norm:

            dists = np.linalg.norm(
                self.kernels_ - p,
                axis=1
            )

            idxs = np.argsort(dists)[: self.k_activate]

            activated.update(idxs)

            point_to_kernel.append(idxs)

        activated = list(activated)

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        ax.scatter(
            pts_3d[:, 0],
            pts_3d[:, 1],
            pts_3d[:, 2],
            c="royalblue",
            alpha=0.5,
            label="Pontos"
        )

        ax.scatter(
            kernels_3d[:, 0],
            kernels_3d[:, 1],
            kernels_3d[:, 2],
            c="red",
            marker="X",
            s=220,
            label="Kernels"
        )

        if activated:

            ax.scatter(
                kernels_3d[activated, 0],
                kernels_3d[activated, 1],
                kernels_3d[activated, 2],
                c="lime",
                marker="*",
                s=350,
                edgecolors="black",
                label="Ativados"
            )

        if annotate_kernels:

            for idx, k in enumerate(kernels_3d):

                ax.text(
                    k[0],
                    k[1],
                    k[2],
                    f"K{idx}"
                )

        if show_connections:

            for p_idx, kernel_idxs in enumerate(point_to_kernel):

                p = pts_3d[p_idx]

                for k_idx in kernel_idxs:

                    k = kernels_3d[k_idx]

                    ax.plot(
                        [p[0], k[0]],
                        [p[1], k[1]],
                        [p[2], k[2]],
                        color="gray",
                        alpha=0.08
                    )

        ax.set_title(title)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.legend()

        plt.tight_layout()
        plt.show()

    ## FPS ponderado por densidade
    def _fps_density_sampling(
        self,
        points,
        n_candidates,
        density_neighbors=10
    ):
        """
        Farthest Point Sampling ponderado por densidade.

        Retorna:
            kernels candidatos
        """

        points = np.asarray(points)

        n = len(points)

        if n <= n_candidates:
            return points.copy()

        # -----------------------------------
        # densidade local
        # -----------------------------------
        # Calcula uma estimativa de densidade local baseada na média das
        # distâncias aos k vizinhos mais próximos. Pontos em regiões mais
        # densas terão valor de `density` maior (inverso da distância média).

        knn = NearestNeighbors(
            n_neighbors=min(
                density_neighbors + 1,
                len(points)
            )
        )

        knn.fit(points)

        dists, _ = knn.kneighbors(points)

        density = 1.0 / (
            np.mean(dists[:, 1:], axis=1)
            + 1e-9
        )

        rng = np.random.default_rng(
            self.random_state
        )

        selected = [rng.integers(n)]

        remaining = set(range(n))

        remaining.remove(selected[0])

        while len(selected) < n_candidates:

            best_idx = None
            best_score = -np.inf

            for idx in remaining:

                p = points[idx]

                min_dist = np.min(
                    np.linalg.norm(
                        points[selected] - p,
                        axis=1
                    )
                )

                # Score combina distância mínima ao conjunto já selecionado
                # (preferimos pontos mais distantes) com um termo que aumenta
                # a preferência por regiões mais densas via log1p(density).
                # Formula: score = min_dist * log1p(density)
                score = min_dist * np.log1p(
                    density[idx]
                )

                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        return points[selected]
    
    # Construir kernels candidatos
    def build_candidate_kernels(
        self,
        training_points,
        multiplier=4
    ):
        """
        Cria um conjunto rico de kernels.

        Ex:

        32 kernels finais
        →

        128 candidatos
        """

        # Número de candidatos gerado multiplicando a quantidade desejada
        # de kernels; isso cria um conjunto rico do qual filtrar depois.
        n_candidates = max(
            self.n_kernels * multiplier,
            self.n_kernels
        )

        return self._fps_density_sampling(
            training_points,
            n_candidates
        )
    # Uso dos kernels pelos pontos (quantos pontos ativam cada kernel)
    def kernel_usage(
        self,
        points,
        kernels
    ):

        # Conta quantos pontos ativam cada kernel (uso bruto). Útil para
        # eliminar kernels que raramente são acionados ("kernels mortos").
        usage = np.zeros(
            len(kernels),
            dtype=int
        )

        for p in points:

            dists = np.linalg.norm(
                kernels - p,
                axis=1
            )

            idxs = np.argsort(
                dists
            )[:self.k_activate]

            usage[idxs] += 1

        return usage
    
    # Uso dos kernels pelos pontos (quantos pontos ativam cada kernel) ponderado por densidade
    def kernel_mutual_information(
        self,
        points,
        labels,
        kernels
    ):

        n_kernels = len(kernels)

        activation_matrix = np.zeros(
            (
                len(points),
                n_kernels
            ),
            dtype=int
        )

        for i, p in enumerate(points):

            dists = np.linalg.norm(
                kernels - p,
                axis=1
            )

            idxs = np.argsort(
                dists
            )[:self.k_activate]

            activation_matrix[i, idxs] = 1

        # Calcula a informação mútua entre os rótulos e cada vetor de
        # ativação binária do kernel. Valores mais altos indicam que o
        # kernel é informativo para distinguir os rótulos fornecidos.
        mi_scores = []

        for k in range(n_kernels):

            mi = mutual_info_score(
                labels,
                activation_matrix[:, k]
            )

            mi_scores.append(mi)

        return np.array(mi_scores)
    
    # Remover kernels mortos (usados por menos de min_usage pontos)
    def remove_dead_kernels(
        self,
        kernels,
        usage,
        min_usage=5
    ):

        # Remove kernels que são ativados por menos que `min_usage` pontos.
        # Isso reduz ruído e evita manter centroids irrelevantes.
        mask = usage >= min_usage

        return kernels[mask]
    
    # Remover kernels de baixa informação mútua (mantendo apenas a fração superior)
    def remove_low_mi_kernels(
        self,
        kernels,
        mi_scores,
        top_fraction=0.5
    ):

        # Mantém apenas a fração superior de kernels ordenados por MI.
        n_keep = max(
            1,
            int(
                len(kernels)
                * top_fraction
            )
        )

        idxs = np.argsort(
            mi_scores
        )[::-1][:n_keep]

        return kernels[idxs]
    
    # Remover kernels redundantes (mantendo apenas os mais distantes)
    def fit_fps_pipeline(
        self,
        training_points,
        labels
    ):

        training_points = np.asarray(
            training_points,
            dtype=float
        )

        n_dims = training_points.shape[1]

        self.train_sorted_ = [
            np.sort(training_points[:, j])
            for j in range(n_dims)
        ]

        # Normaliza via ECDF para trabalhar no espaço [0,1]^d
        norm_pts = self._ecdf_transform_matrix(
            training_points
        )

        # -------------------
        # FPS
        # -------------------

        kernels = self.build_candidate_kernels(
            norm_pts,
            multiplier=4
        )

        # -------------------
        # Usage
        # -------------------

        usage = self.kernel_usage(
            norm_pts,
            kernels
        )

        kernels = self.remove_dead_kernels(
            kernels,
            usage
        )

        # -------------------
        # Mutual Information
        # -------------------

        mi = self.kernel_mutual_information(
            norm_pts,
            labels,
            kernels
        )

        kernels = self.remove_low_mi_kernels(
            kernels,
            mi,
            top_fraction=0.5
        )

        # -------------------
        # Redundância
        # -------------------

        kernels = self.remove_redundant_kernels(
            kernels
        )

        self.kernels_ = kernels

        self.n_kernels = len(
            self.kernels_
        )

        if self.activation_rate:

            self.k_activate = max(
                1,
                round(
                    self.activation_rate
                    * self.n_kernels
                )
            )

        else:

            self.k_activate = min(
                self.k_activate,
                self.n_kernels
            )

        return self

    def kernel_usage_stats(self, points):

        points = np.asarray(points)

        if points.ndim == 1:
            points = points.reshape(-1, 1)

        norm_pts = self._ecdf_transform_matrix(points)

        usage = np.zeros(self.n_kernels, dtype=int)

        for p in norm_pts:

            dists = np.linalg.norm(
                self.kernels_ - p,
                axis=1
            )

            idxs = np.argsort(dists)[
                :self.k_activate
            ]

            usage[idxs] += 1

        return usage
    
    def kernel_report(self, points):

        usage = self.kernel_usage_stats(points)

        used = np.sum(usage > 0)

        return {
            "kernels_total": self.n_kernels,
            "kernels_utilizados": used,
            "kernels_mortos": self.n_kernels - used,
            "ocupacao": used / self.n_kernels,
            "usage": usage
        }
