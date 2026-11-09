# Gera um relatorio HTML (Plotly) comparando os resultados das execucoes em
# data/newdata/plots e data/survey/plots com o que e reportado em:
#   - Artigo SBPO 2024 (Q-MCH)          -> Tabelas 3, 4 e 5
#   - Gonzaga de Oliveira (2025) survey -> Tabela 1 e recomendacoes experimentais
#
# Pre-requisito: comparison/baselines.csv (gerado por compute_baselines.py)
# Saida: comparison/report_comparacao.html e comparison/tabelas/*.csv

import os
import json
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HERE)
PLOTS = {
    "newdata": os.path.join(BASE_DIR, "data", "newdata", "plots"),
    "survey": os.path.join(BASE_DIR, "data", "survey", "plots"),
}
RUN_TAG = "centrality"  # resultados de main_survey.py: analysis_inputs_<inst>_<tag>.json (Q-MCH original)
OUT_HTML = os.path.join(HERE, "report_comparacao.html")
OUT_TABLES = os.path.join(HERE, "tabelas")

# ---------------------------------------------------------------------------
# Dados transcritos dos PDFs
# ---------------------------------------------------------------------------
# SBPO 2024, Tabela 3: instancia, classe, Original, Cuthill best, Cuthill tempo, Q-MCH best, Q-MCH tempo
SBPO_TABLE3 = [
    ("bp_0", "optimization", 1549, 1031, 0.021288, 585, 0.356708),
    ("bp_1000", "optimization", 1643, 1071, 0.025355, 640, 0.452862),
    ("bp_1200", "optimization", 1643, 1099, 0.019303, 647, 0.446598),
    ("bp_1400", "optimization", 1643, 1095, 0.031912, 642, 0.457749),
    ("bp_1600", "optimization", 1643, 1079, 0.033441, 650, 0.465606),
    ("bp_200", "optimization", 1633, 1019, 0.021413, 596, 0.390006),
    ("bp_400", "optimization", 1627, 999, 0.025189, 621, 0.407727),
    ("bp_600", "optimization", 1627, 1075, 0.022407, 631, 0.408325),
    ("bp_800", "optimization", 1643, 1085, 0.09957, 639, 0.450543),
    ("shl_0", "optimization", 1285, 849, 0.018475, 531, 0.215352),
    ("bcsstk01", "structural", 71, 55, 0.00112, 28, 0.026956),
    ("bcsstk02", "structural", 131, 131, 0.003741, 65, 0.146682),
    ("bcsstk04", "structural", 109, 109, 0.004125, 58, 0.12903),
    ("bcsstk05", "structural", 55, 49, 0.004122, 31, 0.095921),
    ("bcsstk06", "structural", 147, 101, 0.010224, 78, 0.289728),
    ("bcsstk07", "structural", 147, 101, 0.010306, 76, 0.287401),
    ("lshp_265", "thermal", 511, 37, 0.004225, 25, 0.089789),
    ("lshp_406", "thermal", 783, 45, 0.006336, 32, 0.138994),
    ("lshp1009", "thermal", 1953, 69, 0.028204, 54, 0.343484),
    ("lshp1270", "thermal", 2457, 77, 0.024355, 61, 0.422652),
    ("lshp1561", "thermal", 3019, 85, 0.027688, 66, 0.516754),
    ("lshp1882", "thermal", 3641, 93, 0.037781, 73, 0.62875),
    ("lshp2233", "thermal", 4333, 101, 0.056463, 82, 0.743319),
    ("lshp2614", "thermal", 5085, 109, 0.050563, 87, 0.86284),
    ("lshp3025", "thermal", 5897, 117, 0.067248, 94, 1.006613),
    ("lshp3466", "thermal", 6769, 125, 0.085357, 104, 1.182524),
]
# SBPO 2024, Tabela 4 (numero de instancias em que o Q-MCH melhorou)
SBPO_TABLE4 = {"optimization": {"Original": 10, "Cuthill": 10, "MCH": 5},
               "structural": {"Original": 10, "Cuthill": 10, "MCH": 7},
               "thermal": {"Original": 6, "Cuthill": 6, "MCH": 5}}
# SBPO 2024, Tabela 5 (tempo medio em segundos)
SBPO_TABLE5 = {"Cuthill-McKee": 0.03, "VNS Constructive": 0.16, "Q-MCH": 0.42,
               "GPS": 31.80, "FNCHC": 146.20, "VNS-band": 157.33}
SBPO_CLAIMED_REDUCTION = {"optimization": 40.5, "structural": 38.5}
SBPO_WILCOXON_P = 2.98e-8

# Survey (Computer Science Review 56, 2025), Tabela 1: matriz, grupo, |V|, |E| (nao nulos fora da diagonal)
SURVEY_TABLE1 = [
    ("MISKnowledgeMap", "Moqri", 2427, 57022),
    ("har_10NN", "ML_Graph", 10299, 151736),
    ("kmnist_norm_10NN", "ML_Graph", 10000, 156932),
    ("Fashion_MNIST_norm_10NN", "ML_Graph", 10000, 158304),
    ("worms20_10NN", "ML_Graph", 20055, 240826),
    ("k49_norm_10NN", "ML_Graph", 38547, 618158),
    ("power", "Newman", 4941, 13188),
    ("smallworld", "DIMACS10", 100000, 999996),
    ("vsp_befref_fxm_2_4_air02", "DIMACS10", 14109, 196448),
    ("vsp_barth5_1Ksep_50in_5Kout", "DIMACS10", 32212, 203610),
    ("vsp_sctap1-2b_and_seymourl", "DIMACS10", 40174, 281662),
    ("vsp_south31_slptsk", "DIMACS10", 39668, 379828),
    ("vsp_model1_crew1_cr42_south31", "DIMACS10", 45101, 379952),
    ("vsp_bump2_e18_aa01_model1_crew1", "DIMACS10", 56438, 601602),
    ("vsp_vibrobox_scagr7-2c_rlfddd", "DIMACS10", 77328, 871172),
    ("vsp_finan512_scagr7-2c_rlfddd", "DIMACS10", 139752, 1104040),
    ("vsp_msc10848_300sep_100in_1Kout", "DIMACS10", 21996, 2442056),
    ("vsp_c-30_data_data", "DIMACS10", 11023, 124368),
    ("vsp_data_and_seymourl", "DIMACS10", 9167, 111732),
    ("vsp_p0291_seymourl_iiasa", "DIMACS10", 10498, 107736),
    ("preferentialAttachment", "DIMACS10", 100000, 999970),
    ("boyd1", "GHS_indef", 93279, 1211231),
    ("lp1", "Andrianov", 534388, 1643420),
]
SURVEY_TIMEOUT_S = 1500   # timeout sugerido pelo survey para matrizes grandes
VNS_BAND_TIMEOUT_S = 500  # timeout do VNS-band (Mladenovic et al. 2010)

FOLDER_CLASS = {
    "optimization": "optimization", "structural": "structural", "thermal": "thermal",
    "computational_fluid_dynamics": "cfd", "computational_fd_test": "cfd (lns)",
    "electromagnetics": "electromagnetics", "trash": "outras (HB/Mladenovic)",
}

# ---------------------------------------------------------------------------
# Estilo (paleta de referencia validada; ordem categorica fixa)
# ---------------------------------------------------------------------------
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD, BAD, NEUTRAL = "#2a78d6", "#e34948", "#c3c2b7"   # polos do par divergente + meio cinza

METHOD_COLOR = {
    "Q-MCH (seus resultados)": SLOTS[0],
    "Q-MCH (artigo SBPO)": SLOTS[1],
    "RCM (scipy)": SLOTS[2],
    "Cuthill (artigo SBPO)": SLOTS[3],
    "Ordem aleatória (VNS-band constr.)": SLOTS[4],
    "Original": MUTED,
}
CENTRALITY_ORDER = ["Degree", "Closeness", "Betweenness", "Eigenvector", "PageRank",
                    "Katz Centrality", "Harmonic Centrality"]
CENTRALITY_COLOR = dict(zip(CENTRALITY_ORDER, SLOTS))
DATASET_COLOR = {"newdata": SLOTS[0], "survey": SLOTS[1]}
DATASET_LABEL = {"newdata": "newdata (SBPO + HB)", "survey": "survey (Tabela 1)"}


def style(fig, title, height=460, **kw):
    kw.setdefault("margin", dict(l=60, r=30, t=110, b=60))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.985, yref="container", yanchor="top",
                   font=dict(size=16, color=INK)),
        template="plotly_white", height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK2),
        barcornerradius=4, bargap=0.25, bargroupgap=0.08,
        legend=dict(orientation="h", yanchor="bottom", y=1.07, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="white", font=dict(color=INK)), **kw)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_traces(selector=dict(type="bar"), marker_line_color=SURFACE, marker_line_width=1.5)
    return fig


# ---------------------------------------------------------------------------
# Carga dos resultados
# ---------------------------------------------------------------------------
def load_runs():
    summaries, iterations = [], []
    for dataset, root in PLOTS.items():
        for inst in sorted(os.listdir(root)):
            run_file = os.path.join(root, inst, f"analysis_inputs_{inst}_{RUN_TAG}.json")
            if not os.path.exists(run_file):
                if os.path.exists(os.path.join(root, inst, f"analysis_inputs_{inst}.json")):
                    print(f"[WARN] {dataset}/{inst}: so ha resultado antigo (sem sufixo _{RUN_TAG}); ignorado")
                continue
            with open(run_file, encoding="utf-8") as f:
                df = pd.DataFrame(json.load(f))
            if "Run" in df.columns:  # varias execucoes independentes: as analises por construcao usam a primeira
                df = df[df["Run"] == df["Run"].min()].reset_index(drop=True)
            df["dataset"] = dataset
            df["instance"] = inst
            df["step"] = np.arange(len(df))
            if "Episode" in df.columns:
                df["episode"] = df["Episode"] - 1
            else:
                df["episode"] = df["step"] // 15  # 5 chamadas x 3 construcoes por episodio
            iterations.append(df)
            best_file = os.path.join(root, inst, f"best_bandwidth_{inst}_{RUN_TAG}.json")
            hist_best = None
            if os.path.exists(best_file):
                with open(best_file, encoding="utf-8") as f:
                    hist_best = float(json.load(f))
            bw = df["bandwidth"]
            summaries.append({
                "dataset": dataset, "instance": inst,
                "run_best": bw.min(), "run_mean": bw.mean(), "run_std": bw.std(),
                "run_median": bw.median(), "run_worst": bw.max(),
                "hist_best": hist_best,
                "global_time_s": df["Global Time (s)"].iloc[0],
                "construction_time_mean_s": df["local_time"].mean(),
                "n_constructions": len(df),
                "best_centrality": df.loc[bw.idxmin(), "centrality"],
            })
    return pd.DataFrame(summaries), pd.concat(iterations, ignore_index=True)


def build_tables():
    runs, iters = load_runs()
    base = pd.read_csv(os.path.join(HERE, "baselines.csv"))
    df = runs.merge(base, on=["dataset", "instance"], how="left")
    df["classe"] = np.where(df["dataset"] == "survey", "survey",
                            df["folder"].map(FOLDER_CLASS).fillna(df["folder"]))
    df["vs_rcm_pct"] = 100 * (df["run_best"] - df["rcm_bw"]) / df["rcm_bw"]
    df["vs_random_pct"] = 100 * (df["run_best"] - df["random_best"]) / df["random_best"]
    df["gap_lb_pct"] = 100 * (df["run_best"] - df["lower_bound"]) / df["lower_bound"]
    df["reduction_vs_orig_pct"] = 100 * (df["original_bw"] - df["run_best"]) / df["original_bw"]
    df["cv_pct"] = 100 * df["run_std"] / df["run_mean"]

    sbpo = pd.DataFrame(SBPO_TABLE3, columns=["instance", "classe_sbpo", "art_original", "art_cuthill",
                                              "art_cuthill_time", "art_qmch", "art_qmch_time"])
    sbpo = sbpo.merge(df[df["dataset"] == "newdata"], on="instance", how="left")
    sbpo["ours_best"] = sbpo[["run_best", "hist_best"]].min(axis=1)
    sbpo["full_rcm"] = 2 * sbpo["rcm_bw"] + 1     # banda completa 2*beta+1
    sbpo["full_orig"] = 2 * sbpo["original_bw"] + 1
    return df, sbpo, iters


# ---------------------------------------------------------------------------
# Figuras - Artigo SBPO
# ---------------------------------------------------------------------------
def fig_sbpo_table3(sbpo):
    classes = ["optimization", "structural", "thermal"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=classes, horizontal_spacing=0.06)
    series = [("Q-MCH (artigo SBPO)", "art_qmch"), ("Q-MCH (seus resultados)", "run_best"),
              ("RCM (scipy)", "rcm_bw"), ("Ordem aleatória (VNS-band constr.)", "random_best")]
    for c, cls in enumerate(classes, start=1):
        sub = sbpo[sbpo["classe_sbpo"] == cls]
        for name, col in series:
            fig.add_bar(x=sub["instance"], y=sub[col], name=name, marker_color=METHOD_COLOR[name],
                        legendgroup=name, showlegend=(c == 1),
                        hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.0f}}<extra></extra>",
                        row=1, col=c)
        fig.add_scatter(x=sub["instance"], y=sub["lower_bound"], mode="markers", name="Limite inferior",
                        marker=dict(symbol="line-ew-open", size=18, color=INK, line_width=2),
                        legendgroup="lb", showlegend=(c == 1),
                        hovertemplate="<b>%{x}</b><br>Limite inferior: %{y:.0f}<extra></extra>", row=1, col=c)
    fig.update_yaxes(title_text="largura de banda β (menor = melhor)", row=1, col=1)
    return style(fig, "Q1 · Tabela 3 do SBPO: Q-MCH do artigo vs. suas execuções (mesmas 26 instâncias)",
                 height=500, barmode="group")


def fig_sbpo_diff(sbpo):
    d = sbpo.assign(diff=100 * (sbpo["run_best"] - sbpo["art_qmch"]) / sbpo["art_qmch"])
    d = d.sort_values(["classe_sbpo", "diff"])
    colors = np.where(d["diff"] < 0, GOOD, np.where(d["diff"] > 0, BAD, NEUTRAL))
    fig = go.Figure(go.Bar(
        x=d["diff"], y=d["instance"] + "  ·  " + d["classe_sbpo"], orientation="h", marker_color=colors,
        customdata=np.c_[d["art_qmch"], d["run_best"], d["hist_best"].fillna(np.nan)],
        hovertemplate="<b>%{y}</b><br>artigo: %{customdata[0]:.0f}<br>sua execução: %{customdata[1]:.0f}"
                      "<br>melhor histórico: %{customdata[2]:.0f}<br>diferença: %{x:+.1f}%<extra></extra>"))
    fig.add_vline(x=0, line_color=AXIS)
    fig.update_xaxes(title_text="(seu β − β do artigo) / β do artigo  [%]   ← melhor que o artigo | pior →",
                     ticksuffix="%")
    return style(fig, "Q1b · Diferença relativa por instância (azul = você melhorou, vermelho = piorou)",
                 height=720, showlegend=False)


def fig_sbpo_cuthill_metric(sbpo):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
        "Coluna 'Cuthill' do artigo vs. RCM recalculado", "Coluna 'Original' do artigo vs. banda do arquivo"))
    for col, (ycol, beta, full, lab) in enumerate([
            ("art_cuthill", "rcm_bw", "full_rcm", "RCM"), ("art_original", "original_bw", "full_orig", "original")],
            start=1):
        for xcol, name, color, sym in [(beta, f"β {lab} (semi-banda)", SLOTS[2], "circle"),
                                       (full, f"2β+1 {lab} (banda completa)", SLOTS[0], "diamond")]:
            fig.add_scatter(x=sbpo[xcol], y=sbpo[ycol], mode="markers", name=name,
                            marker=dict(size=10, color=color, symbol=sym, line=dict(color=SURFACE, width=2)),
                            text=sbpo["instance"],
                            hovertemplate="<b>%{text}</b><br>recalculado: %{x:.0f}<br>artigo: %{y:.0f}<extra>"
                                          + name + "</extra>", row=1, col=col)
        lim = [1, max(sbpo[ycol].max(), sbpo[full].max()) * 1.2]
        fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color=MUTED, dash="dot", width=1.5),
                        name="y = x", showlegend=(col == 1), hoverinfo="skip", row=1, col=col)
        fig.update_xaxes(type="log", title_text="valor recalculado neste repositório", row=1, col=col)
        fig.update_yaxes(type="log", title_text="valor reportado no artigo", row=1, col=col)
    return style(fig, "Q2 · As colunas 'Original' e 'Cuthill' usam a mesma métrica que o Q-MCH?", height=480)


def fig_sbpo_reduction(sbpo):
    rows = []
    for cls, g in sbpo.groupby("classe_sbpo"):
        rows.append({
            "classe": cls,
            "Alegado no texto do artigo": SBPO_CLAIMED_REDUCTION.get(cls, np.nan),
            "Artigo, recalculado da Tabela 3 (Q-MCH vs Cuthill)": 100 * (1 - g["art_qmch"] / g["art_cuthill"]).mean(),
            "Artigo, Q-MCH vs (Cuthill−1)/2 (mesma métrica)":
                100 * (1 - g["art_qmch"] / ((g["art_cuthill"] - 1) / 2)).mean(),
            "Seus resultados vs RCM β (mesma métrica)": 100 * (1 - g["run_best"] / g["rcm_bw"]).mean(),
        })
    r = pd.DataFrame(rows)
    fig = go.Figure()
    for i, col in enumerate(r.columns[1:]):
        fig.add_bar(x=r["classe"], y=r[col], name=col, marker_color=SLOTS[i],
                    text=r[col].map(lambda v: "" if pd.isna(v) else f"{v:.1f}%"), textposition="outside",
                    textfont=dict(color=INK2),
                    hovertemplate="<b>%{x}</b><br>" + col + ": %{y:.1f}%<extra></extra>")
    fig.add_hline(y=0, line_color=AXIS)
    fig.update_yaxes(title_text="redução média da banda [%] (positivo = melhor)", ticksuffix="%")
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0, xanchor="left", yanchor="top"))
    return style(fig, "Q3 · Redução média por classe: o que o artigo afirma vs. comparação na mesma métrica",
                 height=520, barmode="group"), r


def count_wtl(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return int((a[ok] < b[ok]).sum()), int((a[ok] == b[ok]).sum()), int((a[ok] > b[ok]).sum())


def fig_sbpo_wins(sbpo):
    comps = [("vs Original (artigo, 2β+1)", sbpo["run_best"], sbpo["art_original"]),
             ("vs Original (β do arquivo)", sbpo["run_best"], sbpo["original_bw"]),
             ("vs Cuthill (artigo, 2β+1)", sbpo["run_best"], sbpo["art_cuthill"]),
             ("vs RCM β (scipy)", sbpo["run_best"], sbpo["rcm_bw"]),
             ("vs Q-MCH (artigo)", sbpo["run_best"], sbpo["art_qmch"]),
             ("Q-MCH artigo vs RCM β", sbpo["art_qmch"], sbpo["rcm_bw"])]
    data = [(name, *count_wtl(a, b)) for name, a, b in comps]
    d = pd.DataFrame(data, columns=["comparação", "vitórias", "empates", "derrotas"])
    fig = go.Figure()
    for col, color in [("vitórias", GOOD), ("empates", NEUTRAL), ("derrotas", BAD)]:
        fig.add_bar(y=d["comparação"], x=d[col], name=col, orientation="h", marker_color=color,
                    text=d[col], textposition="inside", insidetextanchor="middle",
                    textfont=dict(color="#ffffff" if col != "empates" else INK),
                    hovertemplate="<b>%{y}</b><br>" + col + ": %{x} de 26<extra></extra>")
    fig.update_xaxes(title_text="nº de instâncias (26 no total)")
    fig.update_yaxes(autorange="reversed")
    return style(fig, "Q4 · Tabela 4 refeita: em quantas instâncias o Q-MCH é melhor?", height=420,
                 barmode="stack"), d


def wilcoxon_table(sbpo, df):
    def test(name, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        try:
            p = wilcoxon(a, b, zero_method="pratt").pvalue
        except ValueError:
            p = np.nan
        return {"comparação": name, "n": int(ok.sum()), "mediana A": np.median(a), "mediana B": np.median(b),
                "A<B": int((a < b).sum()), "A=B": int((a == b).sum()), "A>B": int((a > b).sum()),
                "p-valor (Wilcoxon, Pratt)": p,
                "conclusão (α=0,05)": "diferença significativa" if p < 0.05 else "sem diferença significativa"}
    rows = [
        test("Q-MCH artigo (A) vs Cuthill artigo (B) — como no SBPO", sbpo["art_qmch"], sbpo["art_cuthill"]),
        test("Q-MCH artigo (A) vs RCM β (B) — mesma métrica", sbpo["art_qmch"], sbpo["rcm_bw"]),
        test("Seus resultados (A) vs Q-MCH artigo (B)", sbpo["run_best"], sbpo["art_qmch"]),
        test("Seus resultados (A) vs RCM β (B) — 26 instâncias SBPO", sbpo["run_best"], sbpo["rcm_bw"]),
        test("Seus resultados (A) vs RCM β (B) — todas as instâncias", df["run_best"], df["rcm_bw"]),
        test("Seus resultados (A) vs ordem aleatória (B) — todas", df["run_best"], df["random_best"]),
    ]
    return pd.DataFrame(rows)


def fig_times(sbpo, df):
    items = [(k, v, "artigo SBPO (Mac M2)") for k, v in SBPO_TABLE5.items()]
    s = sbpo.dropna(subset=["global_time_s"])
    items += [
        ("Seu Q-MCH: execução completa (450 construções + DQN)", s["global_time_s"].mean(), "suas execuções"),
        ("Seu Q-MCH: equivalente a 100 construções (maxiter do artigo)",
         (s["global_time_s"] / s["n_constructions"] * 100).mean(), "suas execuções"),
        ("Seu Q-MCH: uma construção", s["construction_time_mean_s"].mean(), "suas execuções"),
        ("RCM scipy (recalculado)", s["rcm_time_s"].mean(), "recalculado"),
    ]
    d = pd.DataFrame(items, columns=["método", "tempo_s", "fonte"]).sort_values("tempo_s")
    color = {"artigo SBPO (Mac M2)": SLOTS[1], "suas execuções": SLOTS[0], "recalculado": SLOTS[2]}
    fig = go.Figure()
    for src, g in d.groupby("fonte", sort=False):
        fig.add_bar(y=g["método"], x=g["tempo_s"], orientation="h", name=src, marker_color=color[src],
                    text=g["tempo_s"].map(lambda v: f"{v:.3g} s"), textposition="outside",
                    textfont=dict(color=INK2), hovertemplate="<b>%{y}</b><br>%{x:.4g} s<extra>" + src + "</extra>")
    fig.update_xaxes(type="log", title_text="tempo médio por instância [s] (escala log)")
    fig.update_yaxes(categoryorder="array", categoryarray=d["método"].tolist())
    return style(fig, "Q5 · Tabela 5: tempo médio nas 26 instâncias do SBPO", height=480)


# ---------------------------------------------------------------------------
# Figuras - centralidades e aprendizado
# ---------------------------------------------------------------------------
def fig_centrality_box(iters, df):
    it = iters.merge(df[["dataset", "instance", "run_best", "classe"]], on=["dataset", "instance"])
    it["ratio"] = it["bandwidth"] / it["run_best"]
    present = [c for c in CENTRALITY_ORDER if c in set(it["centrality"])]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[DATASET_LABEL["newdata"], DATASET_LABEL["survey"]])
    for c, ds in enumerate(["newdata", "survey"], start=1):
        sub = it[it["dataset"] == ds]
        for cen in present:
            v = sub.loc[sub["centrality"] == cen, "ratio"]
            if v.empty:
                continue
            fig.add_box(y=v, name=cen.replace(" Centrality", ""), marker_color=CENTRALITY_COLOR[cen],
                        boxpoints=False, line_width=1.5, showlegend=False,
                        hovertemplate=cen + "<br>β / melhor β: %{y:.2f}<extra></extra>", row=1, col=c)
    fig.update_yaxes(title_text="β da construção / melhor β da instância (1 = melhor)", range=[0.95, 2.2],
                     row=1, col=1)
    return style(fig, "Q6 · A escolha da centralidade importa? Qualidade de cada construção por centralidade",
                 height=470)


def fig_centrality_best(df, iters):
    # centralidade que atingiu o melhor beta (contando empates em todas que atingiram)
    rows = []
    for (ds, inst), g in iters.groupby(["dataset", "instance"]):
        m = g["bandwidth"].min()
        for cen in g.loc[g["bandwidth"] == m, "centrality"].unique():
            rows.append({"dataset": ds, "instance": inst, "centrality": cen})
    w = pd.DataFrame(rows).merge(df[["dataset", "instance", "classe"]], on=["dataset", "instance"])
    tab = w.pivot_table(index="classe", columns="centrality", values="instance", aggfunc="count", fill_value=0)
    tab = tab[[c for c in CENTRALITY_ORDER if c in tab.columns]]
    totals = w.groupby("classe")["instance"].nunique()
    share = tab.div(totals, axis=0) * 100
    fig = go.Figure(go.Heatmap(
        z=share.values, x=[c.replace(" Centrality", "") for c in share.columns], y=share.index,
        colorscale=[[0, "#f0efec"], [0.5, "#86b6ef"], [1, "#184f95"]], zmin=0, zmax=100,
        text=tab.values, texttemplate="%{text}", textfont=dict(color=INK),
        colorbar=dict(title="% das instâncias", ticksuffix="%"),
        hovertemplate="<b>%{y}</b> · %{x}<br>atingiu o melhor β em %{text} instâncias (%{z:.0f}%)<extra></extra>"))
    return style(fig, "Q6b · Qual centralidade encontrou o melhor β? (nº de instâncias por classe; empates contam para todas)",
                 height=440)


def fig_selection_over_time(iters):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[DATASET_LABEL["newdata"], DATASET_LABEL["survey"]])
    for c, ds in enumerate(["newdata", "survey"], start=1):
        sub = iters[iters["dataset"] == ds]
        # uma escolha do agente por episodio (as 15 construcoes do episodio usam a mesma centralidade)
        ep = sub.groupby(["instance", "episode"])["centrality"].first().reset_index()
        ep["bloco"] = ep["episode"] // 5
        share = ep.groupby(["bloco", "centrality"]).size().unstack(fill_value=0)
        share = share.div(share.sum(axis=1), axis=0) * 100
        labels = [f"{5 * b + 1}–{5 * b + 5}" for b in share.index]
        uniform = 100 / share.shape[1]
        fig.add_scatter(x=labels, y=[uniform] * len(labels), mode="lines", name="escolha uniforme (sem aprendizado)",
                        line=dict(color=INK, dash="dot", width=1.5), legendgroup="unif", showlegend=(c == 2),
                        hovertemplate="uniforme: %{y:.0f}%<extra></extra>", row=1, col=c)
        for cen in [x for x in CENTRALITY_ORDER if x in share.columns]:
            fig.add_scatter(x=labels, y=share[cen], mode="lines+markers", marker=dict(size=8),
                            name=cen.replace(" Centrality", ""),
                            line=dict(color=CENTRALITY_COLOR[cen], width=2), legendgroup=cen,
                            showlegend=(c == 2),  # o survey contem as 7 centralidades
                            hovertemplate="episódios %{x}<br>" + cen + ": %{y:.0f}% das instâncias<extra></extra>",
                            row=1, col=c)
    fig.update_xaxes(title_text="bloco de episódios do agente")
    fig.update_yaxes(title_text="% das instâncias que escolheram a centralidade", ticksuffix="%", row=1, col=1)
    return style(fig, "Q7 · O agente DQN aprende a preferir alguma centralidade ao longo dos episódios?", height=440)


def fig_convergence(iters, df):
    it = iters.merge(df[["dataset", "instance", "run_best"]], on=["dataset", "instance"]).sort_values(
        ["dataset", "instance", "step"])
    it["best_so_far"] = it.groupby(["dataset", "instance"])["bandwidth"].cummin()
    it["ratio"] = it["best_so_far"] / it["run_best"]
    fig = go.Figure()
    for ds in ["newdata", "survey"]:
        q = it[it["dataset"] == ds].groupby("step")["ratio"].quantile([0.25, 0.5, 0.75]).unstack()
        x = q.index + 1
        fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[q[0.75], q[0.25][::-1]], fill="toself",
                        fillcolor=DATASET_COLOR[ds], opacity=0.15, line=dict(width=0), hoverinfo="skip",
                        showlegend=False, legendgroup=ds)
        fig.add_scatter(x=x, y=q[0.5], mode="lines", name=DATASET_LABEL[ds] + " (mediana, faixa IQR)",
                        line=dict(color=DATASET_COLOR[ds], width=2), legendgroup=ds,
                        hovertemplate="construção %{x}<br>mediana: %{y:.3f}× o β final<extra></extra>")
    for k in (100, 200, 300, 400):
        fig.add_vline(x=k, line=dict(color=GRID, width=1))
    fig.add_annotation(x=100, y=1, yref="paper", text="maxiter=100 do artigo", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(color=MUTED, size=11))
    fig.update_xaxes(title_text="nº de construções executadas")
    fig.update_yaxes(title_text="melhor β até aqui / β final")
    return style(fig, "Q8 · Convergência: quanto do resultado final já existe após 100 construções?", height=430)


# ---------------------------------------------------------------------------
# Figuras - Survey
# ---------------------------------------------------------------------------
def survey_coverage(df):
    t = pd.DataFrame(SURVEY_TABLE1, columns=["instance", "grupo", "survey_V", "survey_E_nnz"])
    s = df[df["dataset"] == "survey"].set_index("instance")
    t["lcc_V"] = t["instance"].map(s["lcc_nodes"])
    t["full_E"] = t["instance"].map(s["full_edges"])
    t["run_V"] = t["instance"].map(s["nodes"])
    t["run_E"] = t["instance"].map(s["edges"])
    t["status"] = np.select(
        [t["run_V"].isna(), t["instance"].map(s["reduced_subgraph"]).eq(True)],
        ["não executada", "subgrafo BFS de 2000 nós"], "grafo completo (maior componente)")
    t["fração de V usada"] = 100 * t["run_V"] / t["survey_V"]
    return t


def fig_survey_coverage(cov):
    d = cov.sort_values("survey_V")
    fig = go.Figure()
    fig.add_bar(y=d["instance"], x=d["survey_V"], orientation="h", name="|V| na Tabela 1 do survey",
                marker_color=AXIS, hovertemplate="<b>%{y}</b><br>|V| survey: %{x:,}<extra></extra>")
    st_color = {"grafo completo (maior componente)": SLOTS[0], "subgrafo BFS de 2000 nós": SLOTS[1]}
    for st, color in st_color.items():
        g = d[d["status"] == st]
        fig.add_scatter(y=g["instance"], x=g["run_V"], mode="markers", name=f"|V| usado: {st}",
                        marker=dict(size=11, color=color, line=dict(color=SURFACE, width=2)),
                        customdata=g["fração de V usada"],
                        hovertemplate="<b>%{y}</b><br>|V| usado: %{x:,}<br>%{customdata:.1f}% do grafo<extra></extra>")
    miss = d[d["status"] == "não executada"]
    if len(miss):
        fig.add_scatter(y=miss["instance"], x=miss["survey_V"], mode="markers", name="não executada",
                        marker=dict(symbol="x", size=11, color=BAD), hoverinfo="y+name")
    fig.update_xaxes(type="log", title_text="nº de vértices (escala log)")
    fig.update_yaxes(categoryorder="array", categoryarray=d["instance"].tolist())
    return style(fig, "S1 · Cobertura do benchmark proposto no survey (Tabela 1, 23 matrizes)", height=680,
                 barmode="overlay")


def fig_survey_quality(df):
    s = df[df["dataset"] == "survey"].copy()
    s["label"] = s["instance"] + np.where(s["reduced_subgraph"].astype(bool), " *", "")
    s = s.sort_values("nodes")
    fig = go.Figure()
    for name, col in [("Original", "original_bw"), ("RCM (scipy)", "rcm_bw"),
                      ("Ordem aleatória (VNS-band constr.)", "random_best"), ("Q-MCH (seus resultados)", "run_best")]:
        fig.add_bar(x=s["label"], y=s[col], name=name, marker_color=METHOD_COLOR[name],
                    customdata=s["nodes"],
                    hovertemplate="<b>%{x}</b><br>" + name + ": %{y:,.0f}<br>|V| usado: %{customdata:,}<extra></extra>")
    fig.add_scatter(x=s["label"], y=s["lower_bound"], mode="markers", name="Limite inferior",
                    marker=dict(symbol="line-ew-open", size=16, color=INK, line_width=2),
                    hovertemplate="<b>%{x}</b><br>Limite inferior: %{y:,.0f}<extra></extra>")
    fig.update_yaxes(type="log", title_text="largura de banda β (escala log, menor = melhor)")
    fig.update_xaxes(tickangle=-40)
    fig.add_annotation(text="* rodada em subgrafo BFS de 2000 nós (baselines calculados no mesmo subgrafo)",
                       xref="paper", yref="paper", x=0, y=-0.42, showarrow=False, xanchor="left",
                       font=dict(color=MUTED, size=11))
    return style(fig, "S2 · Instâncias do survey: Q-MCH vs. RCM (estado da prática) vs. ordem aleatória",
                 height=600, barmode="group", margin=dict(l=60, r=30, t=110, b=190))


def fig_ratio_rcm(df):
    d = df.dropna(subset=["vs_rcm_pct"]).sort_values("vs_rcm_pct")
    fig = go.Figure()
    for ds in ["newdata", "survey"]:
        g = d[d["dataset"] == ds]
        fig.add_scatter(x=g["nodes"], y=g["vs_rcm_pct"], mode="markers", name=DATASET_LABEL[ds],
                        marker=dict(size=10, color=DATASET_COLOR[ds], line=dict(color=SURFACE, width=2)),
                        customdata=np.c_[g["instance"], g["run_best"], g["rcm_bw"], g["classe"]],
                        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[3]})<br>|V| = %{x:,}<br>"
                                      "Q-MCH: %{customdata[1]:.0f} · RCM: %{customdata[2]:.0f}<br>"
                                      "diferença: %{y:+.1f}%<extra></extra>")
    fig.add_hline(y=0, line_color=INK2, line_width=1)
    fig.add_annotation(x=1, xref="paper", y=0, text="empate com RCM", showarrow=False, xanchor="right",
                       yanchor="bottom", font=dict(color=MUTED, size=11))
    fig.update_xaxes(type="log", title_text="nº de vértices do grafo executado (escala log)")
    fig.update_yaxes(title_text="(β Q-MCH − β RCM) / β RCM [%]  (abaixo de 0 = Q-MCH melhor)", ticksuffix="%")
    return style(fig, "S3 · Todas as instâncias: o Q-MCH supera o RCM, apontado pelo survey como estado da prática?",
                 height=480)


def fig_random_vs_ours(df):
    d = df.dropna(subset=["random_best"])
    fig = go.Figure()
    for ds in ["newdata", "survey"]:
        g = d[d["dataset"] == ds]
        fig.add_scatter(x=g["random_best"], y=g["run_best"], mode="markers", name=DATASET_LABEL[ds],
                        marker=dict(size=10, color=DATASET_COLOR[ds], line=dict(color=SURFACE, width=2)),
                        customdata=np.c_[g["instance"], g["vs_random_pct"]],
                        hovertemplate="<b>%{customdata[0]}</b><br>aleatório: %{x:.0f}<br>Q-MCH: %{y:.0f}"
                                      "<br>diferença: %{customdata[1]:+.1f}%<extra></extra>")
    lim = [max(1, d[["random_best", "run_best"]].min().min() * 0.8), d[["random_best", "run_best"]].max().max() * 1.2]
    fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color=MUTED, dash="dot", width=1.5), name="y = x",
                    hoverinfo="skip")
    fig.update_xaxes(type="log", title_text="melhor β com ordem aleatória (450 construções)")
    fig.update_yaxes(type="log", title_text="melhor β do Q-MCH (450 construções)")
    return style(fig, "S4 · Centralidades vs. ordem aleatória com o mesmo orçamento (survey cita Maues et al. [74])",
                 height=500)


def fig_runtime_scaling(df):
    fig = go.Figure()
    for ds in ["newdata", "survey"]:
        g = df[df["dataset"] == ds]
        fig.add_scatter(x=g["edges"], y=g["global_time_s"], mode="markers", name=DATASET_LABEL[ds],
                        marker=dict(size=10, color=DATASET_COLOR[ds], line=dict(color=SURFACE, width=2)),
                        customdata=np.c_[g["instance"], g["nodes"]],
                        hovertemplate="<b>%{customdata[0]}</b><br>|V| = %{customdata[1]:,} · |E| = %{x:,}"
                                      "<br>tempo total: %{y:.1f} s<extra></extra>")
    for y, txt in [(SURVEY_TIMEOUT_S, "1500 s — timeout sugerido no survey"),
                   (VNS_BAND_TIMEOUT_S, "500 s — tempo do VNS-band"),
                   (SBPO_TABLE5["Q-MCH"], "0,42 s — Q-MCH no SBPO")]:
        fig.add_hline(y=y, line=dict(color=MUTED, dash="dash", width=1))
        fig.add_annotation(x=0, xref="paper", y=np.log10(y), text=txt, showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(color=MUTED, size=11))
    fig.update_xaxes(type="log", title_text="nº de arestas do grafo executado (escala log)")
    fig.update_yaxes(type="log", title_text="tempo total da execução [s] (escala log)")
    return style(fig, "S5 · Custo computacional vs. tamanho (o survey alerta que o tempo cresce com |E|)", height=480)


def fig_gap_lb(df):
    d = df.dropna(subset=["gap_lb_pct"])
    order = d.groupby("classe")["gap_lb_pct"].median().sort_values().index.tolist()
    fig = go.Figure()
    for i, cls in enumerate(order):
        g = d[d["classe"] == cls]
        fig.add_box(x=g["gap_lb_pct"], name=cls, orientation="h", boxpoints="all", jitter=0.4, pointpos=0,
                    marker=dict(size=7, color=SLOTS[0], line=dict(color=SURFACE, width=1)), line_color=INK2,
                    fillcolor="rgba(0,0,0,0)", showlegend=False, customdata=np.c_[g["instance"], g["lower_bound"],
                                                                                   g["run_best"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>LB = %{customdata[1]:.0f} · Q-MCH = %{customdata[2]:.0f}"
                                  "<br>gap = %{x:.0f}%<extra></extra>")
    fig.update_xaxes(type="log", title_text="gap para o limite inferior [%] (escala log; quanto menor, mais perto do ótimo)",
                     ticksuffix="%")
    return style(fig, "S6 · Distância ao limite inferior (max de Lam et al. usado no SBPO, ⌈Δ/2⌉, ⌈(n−1)/diâm⌉)",
                 height=460)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def table_html(d, fmt=None, max_rows=200):
    fmt = fmt or {}
    d = d.head(max_rows).copy()
    for c in d.columns:
        if c in fmt:
            d[c] = d[c].map(lambda v, f=fmt[c]: "" if pd.isna(v) else f.format(v))
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
                   for r in d.itertuples(index=False))
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def tile(value, label):
    return f'<div class="tile"><div class="v">{value}</div><div class="l">{label}</div></div>'


CSS = f"""
body{{background:{PAGE};color:{INK};font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0}}
main{{max-width:1280px;margin:0 auto;padding:24px 32px 80px}}
h1{{font-size:26px;margin:8px 0 4px}} h2{{font-size:20px;margin:48px 0 8px;border-top:1px solid {GRID};padding-top:24px}}
p,li{{color:{INK2};line-height:1.55;font-size:14.5px}} .src{{color:{MUTED};font-size:13px}}
.card{{background:{SURFACE};border:1px solid rgba(11,11,11,.08);border-radius:10px;padding:8px 12px;margin:14px 0}}
.q{{background:{SURFACE};border-left:3px solid {SLOTS[0]};padding:10px 16px;margin:12px 0;border-radius:4px}}
.q b{{color:{INK}}}
.tiles{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
.tile{{background:{SURFACE};border:1px solid rgba(11,11,11,.08);border-radius:10px;padding:14px 18px;min-width:170px;flex:1}}
.tile .v{{font-size:28px;font-weight:600;color:{INK}}} .tile .l{{font-size:12.5px;color:{INK2};margin-top:4px}}
details{{margin:8px 0 0}} summary{{cursor:pointer;color:{INK2};font-size:13.5px}}
.tbl{{overflow-x:auto;max-height:460px;overflow-y:auto}}
table{{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums;width:100%}}
th,td{{border-bottom:1px solid {GRID};padding:5px 8px;text-align:right;white-space:nowrap}}
th{{position:sticky;top:0;background:{SURFACE};color:{INK};text-align:right}}
td:first-child,th:first-child{{text-align:left}}
"""


def section(title, question, finding, fig=None, table=None, source=None):
    parts = [f"<h3>{title}</h3>" if title else "",
             f'<div class="q"><b>Pergunta.</b> {question}<br><b>O que os dados mostram.</b> {finding}</div>']
    if fig is not None:
        parts.append('<div class="card">' + fig.to_html(full_html=False, include_plotlyjs=False,
                                                         config={"displaylogo": False, "responsive": True}) + "</div>")
    if table is not None:
        parts.append(f"<details><summary>Ver tabela de dados</summary>{table}</details>")
    if source:
        parts.append(f'<p class="src">{source}</p>')
    return "\n".join(parts)


def main():
    os.makedirs(OUT_TABLES, exist_ok=True)
    df, sbpo, iters = build_tables()
    cov = survey_coverage(df)
    wil = wilcoxon_table(sbpo, df)
    f_red, red = fig_sbpo_reduction(sbpo)
    f_wins, wins = fig_sbpo_wins(sbpo)

    df.to_csv(os.path.join(OUT_TABLES, "resumo_por_instancia.csv"), index=False)
    sbpo.to_csv(os.path.join(OUT_TABLES, "sbpo_tabela3_vs_execucoes.csv"), index=False)
    cov.to_csv(os.path.join(OUT_TABLES, "survey_cobertura_tabela1.csv"), index=False)
    wil.to_csv(os.path.join(OUT_TABLES, "wilcoxon.csv"), index=False)

    # --- numeros usados no texto ---
    n_inst = len(df)
    full_match = int((sbpo["art_cuthill"] == sbpo["full_rcm"]).sum())
    near_match = int((np.abs(sbpo["art_cuthill"] - sbpo["full_rcm"]) <= 0.1 * sbpo["full_rcm"]).sum())
    orig_match = int((np.abs(sbpo["art_original"] - sbpo["full_orig"]) <= 2).sum())
    w_art = count_wtl(sbpo["run_best"], sbpo["art_qmch"])
    w_rcm_all = count_wtl(df["run_best"], df["rcm_bw"])
    w_rcm_sbpo = count_wtl(sbpo["art_qmch"], sbpo["rcm_bw"])
    w_rand = count_wtl(df["run_best"], df["random_best"])
    w_rcm_surv = count_wtl(df.loc[df["dataset"] == "survey", "run_best"], df.loc[df["dataset"] == "survey", "rcm_bw"])
    w_rand_surv = count_wtl(df.loc[df["dataset"] == "survey", "run_best"],
                            df.loc[df["dataset"] == "survey", "random_best"])
    med_vs_art = (100 * (sbpo["run_best"] - sbpo["art_qmch"]) / sbpo["art_qmch"]).median()
    s_cov = cov["status"].value_counts()
    surv = df[df["dataset"] == "survey"]
    surv_reduced = sorted(surv.loc[surv["reduced_subgraph"].astype(bool), "instance"])
    centrality_names = sorted(iters["centrality"].dropna().unique())
    n_centralities = len(centrality_names)
    over_timeout = df.loc[df["global_time_s"] > SURVEY_TIMEOUT_S, "instance"].tolist()
    sbpo_time_ratio = (sbpo["global_time_s"] / sbpo["n_constructions"] * 100).mean() / SBPO_TABLE5["Q-MCH"]
    w_rcm_ours_sbpo = count_wtl(sbpo["run_best"], sbpo["rcm_bw"])
    w_rand_sbpo = count_wtl(sbpo["run_best"], sbpo["random_best"])
    red_i = red.set_index("classe")
    red_same = "; ".join(f"{c}: {v:+.1f}%" for c, v in
                         red_i["Artigo, Q-MCH vs (Cuthill−1)/2 (mesma métrica)"].items())
    red_rcm = "; ".join(f"{c}: {v:+.1f}%" for c, v in red_i["Seus resultados vs RCM β (mesma métrica)"].items())
    it_r = iters.merge(df[["dataset", "instance", "run_best"]], on=["dataset", "instance"])
    it_r["ratio"] = it_r["bandwidth"] / it_r["run_best"]
    cen_med = it_r.groupby("centrality")["ratio"].median().sort_values()
    cen_txt = ", ".join(f"{k.replace(' Centrality', '')} {v:.2f}" for k, v in cen_med.items())
    ep = iters.groupby(["dataset", "instance", "episode"])["centrality"].first().reset_index()
    late = ep[ep["episode"] >= 20].groupby("dataset")["centrality"].value_counts(normalize=True)
    max_late = 100 * late.max()
    conv = iters.sort_values(["dataset", "instance", "step"]).copy()
    conv["bsf"] = conv.groupby(["dataset", "instance"])["bandwidth"].cummin()
    at100 = conv[conv["step"] == 99].merge(df[["dataset", "instance", "run_best"]], on=["dataset", "instance"])
    at100_ratio = (at100["bsf"] / at100["run_best"])
    at100_same = int((at100_ratio == 1).sum())
    hub = df[(df["gap_lb_pct"] > 0) & (df["lb_degree"] >= df["lower_bound"])].sort_values("gap_lb_pct").tail(3)

    tiles = "".join([
        tile(f"{n_inst}", "instâncias com resultados (newdata + survey)"),
        tile(f"{w_art[0]}/{w_art[1]}/{w_art[2]}", "vs Q-MCH do artigo: melhor / igual / pior (26 inst.)"),
        tile(f"{w_rcm_all[0]}/{w_rcm_all[1]}/{w_rcm_all[2]}", "vs RCM β: melhor / igual / pior (todas)"),
        tile(f"{full_match + 0}/26", "valores 'Cuthill' do artigo exatamente iguais a 2·β<sub>RCM</sub>+1"),
        tile(f"{s_cov.get('grafo completo (maior componente)', 0)}/23",
             "matrizes do survey rodadas no grafo completo"),
    ])

    html_parts = [f"<!doctype html><html lang='pt-br'><head><meta charset='utf-8'>"
                  f"<title>Comparação Q-MCH × SBPO 2024 × Survey 2025</title><style>{CSS}</style>"
                  "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body><main>",
                  "<h1>Comparação dos resultados com o artigo SBPO 2024 e o survey de heurísticas (2025)</h1>",
                  "<p class='src'>Fontes: <i>Aprendizado por Reforço em uma Heurística Multi-Centralidades para a "
                  "Redução de Largura de Banda em Matrizes Esparsas</i> (SBPO 2024) e S.L. Gonzaga de Oliveira, "
                  "<i>A survey of heuristics for matrix bandwidth reduction</i>, Computer Science Review 56 (2025). "
                  "Resultados: <code>data/newdata/plots</code> e <code>data/survey/plots</code> "
                  "(30 episódios × 15 construções = 450 construções por instância). Baselines recalculados em "
                  "<code>comparison/compute_baselines.py</code> sobre exatamente o mesmo grafo de cada execução.</p>",
                  f'<div class="tiles">{tiles}</div>']

    # ---------------- SBPO ----------------
    html_parts.append("<h2>Parte A · Artigo SBPO 2024 (Q-MCH)</h2>")
    html_parts.append(section(
        "Q1 · Os resultados reproduzem a Tabela 3?",
        "Nas 26 instâncias da Tabela 3 (optimization, structural, thermal), seu Q-MCH chega aos mesmos β do artigo?",
        f"Melhor que o artigo em <b>{w_art[0]}</b>, igual em <b>{w_art[1]}</b> e pior em <b>{w_art[2]}</b> instâncias; "
        f"diferença mediana de <b>{med_vs_art:+.1f}%</b>. As diferenças são de poucas unidades de β, "
        "compatíveis com a aleatoriedade do construtivo (LCR + vértice inicial aleatório): <b>a Tabela 3 é "
        "reproduzida</b>. Porém, as barras verdes (RCM, mesma métrica β) ficam abaixo do Q-MCH em quase todas as "
        f"instâncias: o RCM é melhor que seu Q-MCH em <b>{w_rcm_ours_sbpo[2]}</b> das {sum(w_rcm_ours_sbpo)} instâncias "
        f"com baseline, e a ordem aleatória (rosa) em {w_rand_sbpo[2]}. A diferença é maior em optimization (bp_*) "
        "e thermal (lshp*).",
        fig_sbpo_table3(sbpo),
        table_html(sbpo[["instance", "classe_sbpo", "art_original", "art_cuthill", "art_qmch", "run_best",
                         "hist_best", "run_mean", "rcm_bw", "random_best", "lower_bound"]],
                   {"run_mean": "{:.1f}", "hist_best": "{:.0f}", "random_best": "{:.0f}"}),
        "Tabela 3 do SBPO. 'run_best' = melhor β nas 450 construções da execução atual; 'hist_best' = "
        "best_bandwidth_*.json (melhor histórico entre execuções)."))
    html_parts.append(section(None, "Onde exatamente você ganha ou perde do artigo?",
                              "Cada barra é uma instância; o tooltip mostra os valores absolutos.",
                              fig_sbpo_diff(sbpo)))
    html_parts.append(section(
        "Q2 · A comparação com o Cuthill-McKee usa a mesma métrica?",
        "O survey (Seção 2) lembra que a banda da matriz simétrica é 2β+1, enquanto β é a semi-banda. "
        "Qual das duas o artigo usa em cada coluna?",
        f"A coluna 'Cuthill' do artigo coincide <b>exatamente</b> com 2·β<sub>RCM</sub>+1 em <b>{full_match}</b> das 26 "
        f"instâncias (e fica a ±10% disso em {near_match}); a coluna 'Original' bate com 2·β+1 da rotulação do arquivo em "
        f"{orig_match}. Já o Q-MCH é reportado como semi-banda β. Prova direta: o <b>bcsstk02</b> é um grafo completo "
        "(66 nós, 2145 arestas = 66·65/2), cujo β é obrigatoriamente 65. O artigo lista Cuthill = 131 = 2·65+1 e "
        "Q-MCH = 65. As reduções de ~40% e o p = 2,98×10⁻⁸ comparam, portanto, grandezas diferentes.",
        fig_sbpo_cuthill_metric(sbpo),
        table_html(sbpo[["instance", "art_original", "original_bw", "full_orig", "art_cuthill", "rcm_bw", "full_rcm",
                         "art_qmch"]])))
    html_parts.append(section(
        "Q3 · As reduções de 40,5% (optimization) e 38,5% (structural) se sustentam?",
        "Refazendo a conta com a Tabela 3 do próprio artigo e com a mesma métrica (β vs β), qual é a redução real?",
        "Com a Tabela 3 como publicada, a conta reproduz aproximadamente os valores do texto. Convertendo a coluna "
        f"Cuthill para β ((Cuthill−1)/2), a 'redução' passa a ser <b>{red_same}</b>; seus resultados contra o RCM "
        f"(scipy) ficam em <b>{red_rcm}</b>. Valores negativos indicam que o Q-MCH é <b>pior</b> que o Cuthill/RCM.",
        f_red, table_html(red, {c: "{:.1f}" for c in red.columns[1:]})))
    html_parts.append(section(
        "Q4 · A Tabela 4 (100% de melhoria sobre Original e Cuthill) se mantém?",
        "Em quantas das 26 instâncias o Q-MCH é estritamente melhor que cada referência?",
        f"Contra as colunas publicadas (2β+1) o Q-MCH vence sempre, como no artigo. Contra o RCM na mesma métrica, "
        f"o próprio Q-MCH do artigo vence {w_rcm_sbpo[0]}, empata {w_rcm_sbpo[1]} e perde {w_rcm_sbpo[2]}.",
        f_wins, table_html(wins)))
    html_parts.append(section(
        "Teste de Wilcoxon (como na Seção 5.2 do SBPO)",
        "O teste pareado de Wilcoxon (α = 0,05) continua significativo quando as métricas são homogêneas?",
        f"O artigo reporta p = {SBPO_WILCOXON_P:.2e} para Q-MCH vs Cuthill. A tabela abaixo refaz esse teste e o "
        "repete contra o RCM β, contra o artigo e contra a ordem aleatória.",
        None, table_html(wil, {"p-valor (Wilcoxon, Pratt)": "{:.2e}", "mediana A": "{:.1f}", "mediana B": "{:.1f}"})))
    html_parts.append(section(
        "Q5 · O custo computacional é comparável ao da Tabela 5?",
        "O artigo reporta 0,42 s em média para o Q-MCH (maxiter = 100, Mac M2). Quanto custam suas execuções?",
        f"Sua execução faz 450 construções (30 episódios × 5 × 3) mais o treino do DQN. Normalizando para 100 "
        f"construções, o tempo é ≈ <b>{sbpo_time_ratio:.1f}×</b> o reportado no artigo (hardware diferente). O RCM do "
        "scipy fica abaixo de 0,01 s: segundo o survey, essa é a referência de custo para heurísticas 'low-cost'.",
        fig_times(sbpo, df)))

    html_parts.append("<h2>Parte B · Comportamento do Q-MCH (centralidades e aprendizado)</h2>")
    html_parts.append(section(
        "Q6 · A escolha da centralidade importa?",
        "A premissa do SBPO é que um agente escolha a melhor centralidade por classe. As centralidades produzem "
        "distribuições de β realmente diferentes?",
        f"Mediana de β/β* por centralidade (todas as construções): {cen_txt}. As caixas se sobrepõem quase "
        "totalmente: a centralidade tem efeito pequeno e a variação vem principalmente do multi-start aleatório "
        "(LCR com α = 0,3, em que os 70% restantes da fila são embaralhados, mais o vértice inicial aleatório). "
        "Veja no heatmap abaixo como o melhor β se distribui entre as centralidades em cada classe.",
        fig_centrality_box(iters, df)))
    html_parts.append(section(None, "Qual centralidade encontrou o melhor β em cada classe?",
                              "Leitura por linha: a soma passa de 100% quando há empate entre centralidades.",
                              fig_centrality_best(df, iters)))
    html_parts.append(section(
        "Q7 · O agente aprende?",
        "Com ε inicial = 1 e decaimento de 0,995 por passo, o agente chega a explorar menos e convergir para uma "
        "centralidade em 30 episódios?",
        "Na versão atual de <code>main_survey.py</code>, o decaimento de ε é calculado para que ε chegue a 0,01 "
        "após 80% dos episódios, o estado [passo, gap, recompensa, β] é normalizado antes de entrar na rede e o gap "
        "é medido contra o limite inferior da instância. Nos episódios 21–30, nenhuma centralidade passa de "
        f"{max_late:.0f}% das escolhas. O script continua criando um <b>Agent novo por instância</b> "
        "(não há aprendizado por classe, que é a premissa do SBPO).",
        fig_selection_over_time(iters)))
    html_parts.append(section(
        "Q8 · Quantas construções são necessárias?",
        "O artigo usa maxiter = 100. Quanto o β ainda melhora entre a construção 100 e a 450?",
        f"Após 100 construções, o melhor β acumulado já é, na mediana, {at100_ratio.median():.3f}× o β final "
        f"(90º percentil: {at100_ratio.quantile(0.9):.3f}×), e {at100_same} de {len(at100_ratio)} instâncias já "
        "tinham atingido o β final. As 350 construções extras rendem pouco.",
        fig_convergence(iters, df)))

    # ---------------- Survey ----------------
    html_parts.append("<h2>Parte C · Survey de heurísticas (Gonzaga de Oliveira, 2025)</h2>")
    html_parts.append(section(
        "S1 · O benchmark proposto foi coberto?",
        "A Tabela 1 do survey propõe 23 matrizes (2.427 a 534.388 vértices) como novo benchmark. Quais foram "
        "executadas e em que tamanho?",
        f"<b>{s_cov.get('grafo completo (maior componente)', 0)}</b> rodaram no grafo completo, "
        f"<b>{s_cov.get('subgrafo BFS de 2000 nós', 0)}</b> num subgrafo BFS de 2000 nós (fallback de "
        f"<code>main_survey.py</code>) e <b>{s_cov.get('não executada', 0)}</b> não foi executada "
        "(vsp_befref_fxm_2_4_air02, ausente de data/survey). Resultados em subgrafos <b>não são comparáveis</b> com a "
        "literatura sobre essas matrizes. Observação: o |E| da Tabela 1 conta os não nulos dos dois triângulos, ou "
        "seja, 2× o nº de arestas não direcionadas (ex.: Fashion_MNIST 158.304 = 2 × 79.152).",
        fig_survey_coverage(cov),
        table_html(cov, {"fração de V usada": "{:.1f}", "lcc_V": "{:,.0f}", "full_E": "{:,.0f}", "run_V": "{:,.0f}",
                         "run_E": "{:,.0f}"})))
    html_parts.append(section(
        "S2 · Qualidade nas instâncias do survey",
        "O survey não publica β para essas matrizes; a referência natural é o RCM (George–Liu), apontado como "
        "'state of practice' e implementado no SciPy. Como o Q-MCH se compara?",
        f"Nas {sum(w_rcm_surv)} instâncias do survey, o Q-MCH vence o RCM em <b>{w_rcm_surv[0]}</b>, empata em "
        f"{w_rcm_surv[1]} e perde em <b>{w_rcm_surv[2]}</b>. Contra a ordem aleatória, vence {w_rand_surv[0]}, empata "
        f"{w_rand_surv[1]} e perde {w_rand_surv[2]}. Os ganhos sobre o RCM aparecem em grafos de redes "
        "complexas/VSP, onde o RCM é fraco." + (
            " Instâncias marcadas com * ainda vêm de subgrafos BFS de 2000 nós: " + ", ".join(surv_reduced) + "."
            if surv_reduced else " Todas as instâncias do survey rodaram no grafo completo (maior componente)."),
        fig_survey_quality(df)))
    html_parts.append(section(None, "Visão geral: diferença percentual para o RCM em todas as instâncias.",
                              "Pontos abaixo da linha indicam Q-MCH melhor que RCM.", fig_ratio_rcm(df)))
    html_parts.append(section(
        "S4 · As centralidades superam uma ordem aleatória?",
        "O survey (Seção 4.2.1) relata que Maues et al. [74] avaliaram ordenações por centralidade no construtivo do "
        "VNS-band e que <i>a ordem aleatória deu os melhores resultados</i>. Isso se repete aqui, com orçamento igual "
        "(450 construções)?",
        f"Q-MCH melhor que a ordem aleatória em {w_rand[0]}, igual em {w_rand[1]} e pior em {w_rand[2]} instâncias.",
        fig_random_vs_ours(df)))
    html_parts.append(section(
        "S5 · O custo é aceitável?",
        "O survey destaca que, para acelerar solvers, a heurística precisa ter custo baixo (Seção 3.1), sugere "
        "1500 s como timeout para matrizes grandes e lembra que o tempo depende mais de |E| do que de |V|.",
        ("Execuções acima de 1500 s: " + (", ".join(over_timeout) if over_timeout else "nenhuma") +
         ". O tempo cresce aproximadamente como uma potência de |E| (reta no log-log). Cada execução calcula "
         f"{n_centralities} centralidades ({', '.join(centrality_names)}), com Closeness aproximada por amostragem; "
         "esse cálculo domina o tempo nos grafos maiores. Para comparação, o RCM leva menos de 1 s mesmo no maior "
         "grafo."),
        fig_runtime_scaling(df)))
    html_parts.append(section(
        "S6 · Quão longe do ótimo?",
        "O SBPO usa o limite de Lam et al. (1997) como 'limite dual' no cálculo do gap. Ele é informativo?",
        "Para grafos esparsos, o limite de Lam fica perto de 0. Por isso o limite inferior usado aqui é o máximo entre "
        "Lam, ⌈Δ/2⌉ e ⌈(n−1)/diâmetro⌉ (este último só para n ≤ 5000). Gaps altos (" +
        ", ".join(hub["instance"]) + ") indicam sobretudo que o limite inferior é fraco nessas instâncias, e não "
        "necessariamente que a solução seja ruim. O gap do estado do agente (<code>Env</code>) usa o máximo entre "
        "Lam e ⌈Δ/2⌉ calculado em <code>main_survey.py</code>.",
        fig_gap_lb(df)))
    html_parts.append(section(
        "S7 · Estatísticas recomendadas pelo survey",
        "A Seção 4.3 recomenda reportar média, desvio padrão, melhor resultado e tempo total, com ≥ 3 execuções "
        "independentes por matriz.",
        "Cada instância tem uma única execução (30 episódios). As 450 construções dão média e desvio intra-execução, "
        "mas não substituem as ≥ 3 execuções independentes recomendadas.",
        None,
        table_html(df[["dataset", "classe", "instance", "nodes", "edges", "original_bw", "rcm_bw", "random_best",
                       "run_best", "run_mean", "run_std", "cv_pct", "lower_bound", "gap_lb_pct", "global_time_s",
                       "best_centrality"]].sort_values(["dataset", "classe", "instance"]),
                   {"run_mean": "{:.1f}", "run_std": "{:.1f}", "cv_pct": "{:.1f}", "gap_lb_pct": "{:.0f}",
                    "global_time_s": "{:.1f}", "random_best": "{:.0f}", "rcm_bw": "{:.0f}", "original_bw": "{:.0f}",
                    "lower_bound": "{:.0f}", "nodes": "{:,.0f}", "edges": "{:,.0f}"})))

    html_parts.append("</main></body></html>")
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"[OK] relatório: {OUT_HTML}")
    print(f"[OK] tabelas:   {OUT_TABLES}")


if __name__ == "__main__":
    main()
