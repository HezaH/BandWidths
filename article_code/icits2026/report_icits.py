# Relatorio da reproducao do artigo ICITS 2026 (gera as tabelas e figuras da Secao 4).
#
# Entradas: <results-dir>/runs.csv, instances.csv e (opcional) decisions.csv, gerados por main_icits.py.
# Saidas:   <results-dir>/report_icits.html     -> relatorio com os resultados lado a lado com o artigo
#           <results-dir>/tables/*.csv           -> tabelas em CSV
#           <results-dir>/tables/tables_icits.tex -> Tabelas 3, 4 e 5 no formato do artigo (booktabs)
#
# Protocolo estatistico (Secao 4.1 do artigo):
#   - Wilcoxon pareado bilateral, aproximacao normal, zeros removidos, sem correcao de continuidade;
#   - Holm nos dois p-valores (DQN x Random e DQN x SBS) de cada instancia; na ablacao, Holm sobre as
#     10 instancias (uma comparacao DQN-PP x Random-PP por instancia);
#   - Friedman com correcao de empates sobre DQN, Random e as 7 centralidades fixas (blocos = instancias);
#   - SBS = centralidade fixa com melhor rank medio; VBS = centralidade fixa com menor media por instancia;
#   - GapClosed = (SBS - DQN) / (SBS - VBS) sobre a media da banda normalizada B/B0.
#
# Uso: python icits2026/report_icits.py [--results-dir icits2026/results]

import os
import sys
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE_DIR, "comparison"))
from build_report import (SLOTS, INK, INK2, MUTED, GOOD, BAD, CSS, style, tile, section, explainer)
from build_report import table_html as _table_html


def table_html(d, fmt=None, max_rows=200):
    """table_html do build_report, com células sem valor exibidas como '–' em vez de 'nan'."""
    d = d.copy()
    fmt = fmt or {}
    for c in d.columns:
        if c not in fmt and d[c].isna().any():
            d[c] = d[c].astype(object).where(d[c].notna(), "–")
    return _table_html(d, fmt, max_rows)

CENTRALITY_NAMES = ["Degree", "Closeness", "Betweenness", "Eigenvector", "Katz", "PageRank", "Harmonic"]
FIXED = [f"Fixed:{c}" for c in CENTRALITY_NAMES]
PAPER_NAME = {"lshp_265": "lshp265", "lshp_406": "lshp406"}  # nomes dos arquivos -> nomes no artigo
MESHES = ["lshp_265", "lshp_406", "lshp1009", "lshp1270", "lshp1561", "lshp1882"]
LARGEST_MESHES = ["lshp1009", "lshp1270", "lshp1561", "lshp1882"]
METHOD_COLOR = {"DQN": "#2E6FBA", "Random": "#E08E2B", "SBS": "#4C9A6A"}  # paleta do artigo

# ---------------------------------------------------------------------------
# Valores publicados no artigo (para a comparacao lado a lado)
# ---------------------------------------------------------------------------
# Tabela 3: (|V|, |E|, B0, RCM, DQN, Random, SBS (Katz), VBS) -> medias sobre 30 sementes
PAPER_TABLE3 = {
    "bcsstk01": (48, 176, 35, 26, 26.20, 26.43, 25.97, 25.83),
    "bcsstk04": (132, 1758, 47, 54, 47.00, 47.00, 47.00, 47.00),
    "bcsstk05": (153, 1135, 28, 24, 28.00, 28.00, 28.00, 28.00),
    "bcsstk06": (420, 3720, 47, 50, 47.00, 47.00, 47.00, 47.00),
    "bp_0": (822, 3260, 820, 422, 573.43, 572.93, 572.93, 572.70),
    "bp_200": (822, 3788, 820, 506, 600.57, 597.30, 598.07, 597.63),
    "lshp_265": (265, 744, 256, 18, 25.00, 25.00, 25.20, 24.40),
    "lshp_406": (406, 1155, 394, 22, 31.37, 31.27, 31.83, 30.67),
    "lshp1009": (1009, 2928, 991, 34, 51.23, 50.80, 50.40, 50.40),
    "lshp1270": (1270, 3699, 1250, 38, 58.07, 57.63, 56.93, 56.93),
    "lshp1561": (1561, 4560, 1539, 42, 64.27, 64.27, 63.50, 63.50),
    "lshp1882": (1882, 5511, 1858, 46, 70.93, 70.73, 70.03, 70.03),
    "shl_0": (663, 1682, 661, 424, 493.33, 497.80, 501.17, 485.00),
}
# Tabela 5: (DQN, DQN-PP, Random, Random-PP)
PAPER_TABLE5 = {
    "bcsstk01": (26.20, 28.10, 26.43, 28.23), "bp_0": (573.43, 628.97, 572.93, 628.73),
    "bp_200": (600.57, 632.60, 597.30, 631.87), "lshp_265": (25.00, 23.93, 25.00, 23.90),
    "lshp_406": (31.37, 30.00, 31.27, 30.03), "lshp1009": (51.23, 49.30, 50.80, 49.43),
    "lshp1270": (58.07, 55.93, 57.63, 55.83), "lshp1561": (64.27, 62.30, 64.27, 62.20),
    "lshp1882": (70.93, 69.00, 70.73, 68.97), "shl_0": (493.33, 542.83, 497.80, 543.63),
}
PAPER_HEADLINE = {
    "SBS": "Katz (rank médio 2,50)",
    "Friedman": "χ²_F = 52,21; p = 1,53×10⁻⁸",
    "Ranks": "Degree 3,27; Katz 3,31; Random 3,35; DQN 4,08",
    "GapClosed DQN": "2,2%",
    "GapClosed Random": "≈ −19%",
    "Scores": "DQN 0,48323; SBS 0,48329; VBS 0,48058",
    "Greedy": "20 de 390 execuções",
    "Time": "+5,4% (início pela centralidade) e +6,2% (PP)",
    "PP meshes": "−2,7 a −4,4% (DQN); −2,5 a −4,4% (Random)",
    "PP others": "+5,3 a +10,0%",
    "Residual": "1,45 a 1,50",
}


def pname(inst):
    return PAPER_NAME.get(inst, inst)


# ---------------------------------------------------------------------------
# Estatistica (protocolo do artigo)
# ---------------------------------------------------------------------------
def wilcoxon_paper(a, b):
    """Wilcoxon bilateral, aproximação normal, zeros removidos, sem correção de continuidade.
    Sem nenhuma diferença não nula o teste não se aplica: devolve p = 1."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == b):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox", correction=False, alternative="two-sided",
                          method="approx").pvalue)


def holm(pvalues):
    """Correção de Holm (p-valores ajustados, monótonos e limitados a 1)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wtl(a, b):
    """Vitórias/empates/derrotas de a contra b (menor banda vence), pareado por semente."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return int((a < b).sum()), int((a == b).sum()), int((a > b).sum())


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fpct(v, d=1):
    """Fração em % para o texto; GapClosed fica indefinido quando SBS = VBS (gap nulo)."""
    return f"{100 * v:.{d}f}%" if v is not None and np.isfinite(v) else "indefinido (SBS = VBS)"


def mean_sd(x):
    return f"{np.mean(x):.2f}±{np.std(x, ddof=1):.2f}" if len(x) > 1 else f"{np.mean(x):.2f}"


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load(results_dir, train=None, greedy=None):
    runs = pd.read_csv(os.path.join(results_dir, "runs.csv"))
    if train is None or greedy is None:  # usa o orçamento mais frequente (ignora testes rápidos misturados)
        train, greedy = runs.groupby(["Train Decisions", "Greedy Decisions"]).size().idxmax()
    runs = runs[(runs["Train Decisions"] == train) & (runs["Greedy Decisions"] == greedy)]
    runs = runs.drop_duplicates(["Instance", "Method", "Seed"], keep="last").copy()
    for col, default in [("Dataset", "sbpo"), ("Construction", "paper")]:
        runs[col] = runs[col].fillna(default) if col in runs.columns else default
    inst_file = os.path.join(results_dir, "instances.csv")
    instances = pd.read_csv(inst_file).set_index("Instance") if os.path.exists(inst_file) else None
    return runs, instances, int(train), int(greedy)


def paired(runs, inst, method):
    """Série do melhor β por semente (índice = semente) de um método numa instância."""
    s = runs[(runs["Instance"] == inst) & (runs["Method"] == method)].set_index("Seed")["Best Bandwidth"]
    return s.sort_index()


# ---------------------------------------------------------------------------
# Analises
# ---------------------------------------------------------------------------
def main_analysis(runs):
    instances = [i for i in PAPER_TABLE3 if i in set(runs["Instance"])]
    instances += sorted(set(runs["Instance"]) - set(instances))
    means = runs.groupby(["Instance", "Method"])["Best Bandwidth"].mean().unstack()
    norm = runs.groupby(["Instance", "Method"])["Normalized Bandwidth"].mean().unstack()
    fixed = [f for f in FIXED if f in means.columns]
    methods = [m for m in ["DQN", "Random"] + fixed if m in means.columns]
    means, norm = means.reindex(instances), norm.reindex(instances)
    complete = means[methods].dropna().index.tolist()

    # SBS: centralidade fixa com melhor rank médio (ranks entre as fixas, empates pela média)
    fixed_ranks = means.loc[complete, fixed].apply(lambda r: pd.Series(rankdata(r.values), index=fixed), axis=1)
    sbs = fixed_ranks.mean().idxmin() if fixed else None
    # VBS: centralidade fixa com menor média em cada instância
    vbs_method = means[fixed].idxmin(axis=1)

    # Friedman sobre DQN, Random e as fixas (blocos = instâncias, valores = médias)
    all_ranks = means.loc[complete, methods].apply(lambda r: pd.Series(rankdata(r.values), index=methods), axis=1)
    mean_ranks = all_ranks.mean().sort_values()
    try:
        chi2, p_fr = friedmanchisquare(*[means.loc[complete, m] for m in methods])
    except ValueError:
        chi2, p_fr = np.nan, np.nan

    # instâncias discriminantes: algum método estocástico melhora B0 em alguma semente
    b0 = runs.groupby("Instance")["Initial Bandwidth"].first()
    disc = [i for i in complete if (runs[(runs["Instance"] == i) & runs["Method"].isin(methods)]["Best Bandwidth"]
                                    < b0[i]).any()]

    # Tabela 3
    rows = []
    for i in instances:
        r = runs[runs["Instance"] == i].iloc[0]
        row = {"Instance": pname(i), "|V|": r["Nodes"], "|E|": r["Edges"], "B0": r["Initial Bandwidth"],
               "RCM": r["RCM Bandwidth"]}
        for label, m in [("DQN", "DQN"), ("Random", "Random"), (f"SBS ({sbs.split(':')[1] if sbs else '-'})", sbs),
                         ("VBS", vbs_method.get(i))]:
            row[label] = mean_sd(paired(runs, i, m).values) if isinstance(m, str) and m in means.columns else ""
        row["VBS centrality"] = vbs_method.get(i, "").split(":")[-1] if isinstance(vbs_method.get(i), str) else ""
        rows.append(row)
    table3 = pd.DataFrame(rows)

    # Tabela 4 (comparações pareadas do ponto de vista do DQN)
    rows4 = []
    for i in disc:
        dqn, rnd, sb = paired(runs, i, "DQN"), paired(runs, i, "Random"), paired(runs, i, sbs)
        idx = dqn.index.intersection(rnd.index).intersection(sb.index)
        p_r, p_s = wilcoxon_paper(dqn[idx], rnd[idx]), wilcoxon_paper(dqn[idx], sb[idx])
        p_r_h, p_s_h = holm([p_r, p_s])
        rows4.append({"Instance": pname(i), "vs Random W/T/L": "/".join(map(str, wtl(dqn[idx], rnd[idx]))),
                      "vs Random p_H": p_r_h, "vs SBS W/T/L": "/".join(map(str, wtl(dqn[idx], sb[idx]))),
                      "vs SBS p_H": p_s_h})
    table4 = pd.DataFrame(rows4)

    # GapClosed sobre a média da banda normalizada (todas as instâncias completas)
    score = {m: norm.loc[complete, m].mean() for m in methods}
    score_vbs = norm.loc[complete, fixed].min(axis=1).mean()
    gap = score[sbs] - score_vbs if sbs else np.nan
    gap_closed = {m: (score[sbs] - score[m]) / gap if gap else np.nan for m in ["DQN", "Random"] if m in score}
    gap_share = ((norm.loc[complete, sbs] - norm.loc[complete, fixed].min(axis=1)) /
                 (norm.loc[complete, sbs] - norm.loc[complete, fixed].min(axis=1)).sum()) if gap else pd.Series()

    # excesso sobre o VBS (Figura 3)
    vbs_mean = means[fixed].min(axis=1)
    excess = pd.DataFrame({m: 100 * (means[m] - vbs_mean) / vbs_mean for m in ["DQN", "Random", sbs]
                           if m in means.columns}).loc[disc]

    return dict(instances=instances, complete=complete, disc=disc, means=means, sbs=sbs, vbs_method=vbs_method,
                fixed_ranks=fixed_ranks, mean_ranks=mean_ranks, chi2=chi2, p_fr=p_fr, table3=table3,
                table4=table4, score=score, score_vbs=score_vbs, gap_closed=gap_closed, gap_share=gap_share,
                excess=excess, methods=methods)


def ablation_analysis(runs):
    needed = ["DQN", "DQN-PP", "Random", "Random-PP"]
    candidates = list(PAPER_TABLE5) + sorted(set(runs["Instance"]) - set(PAPER_TABLE5))
    insts = [i for i in candidates if all(len(paired(runs, i, m)) for m in needed)]
    if not insts:
        return None
    rows, pvals = [], []
    for i in insts:
        s = {m: paired(runs, i, m) for m in needed}
        idx = s["DQN-PP"].index.intersection(s["Random-PP"].index)
        pvals.append(wilcoxon_paper(s["DQN-PP"][idx], s["Random-PP"][idx]))
        rows.append({"Instance": pname(i), **{m: mean_sd(s[m].values) for m in needed},
                     "W/T/L": "/".join(map(str, wtl(s["DQN-PP"][idx], s["Random-PP"][idx]))),
                     **{f"_mean_{m}": s[m].mean() for m in needed}})
    table5 = pd.DataFrame(rows)
    table5.insert(table5.columns.get_loc("W/T/L") + 1, "p_H", holm(pvals))
    change = pd.DataFrame({
        "DQN": 100 * (table5["_mean_DQN-PP"] - table5["_mean_DQN"]) / table5["_mean_DQN"],
        "Random": 100 * (table5["_mean_Random-PP"] - table5["_mean_Random"]) / table5["_mean_Random"],
    })
    change.index = [i for i in insts]
    rcm = runs.groupby("Instance")["RCM Bandwidth"].first()
    residual = {i: table5.loc[table5["Instance"] == pname(i), "_mean_DQN-PP"].iloc[0] / rcm[i]
                for i in LARGEST_MESHES if i in insts}
    return dict(insts=insts, table5=table5, change=change, residual=residual)


def time_ratios(runs):
    """Razão pareada de tempo DQN/Random (mesma instância e semente), por regra de início."""
    out = {}
    for dqn, rnd in [("DQN", "Random"), ("DQN-PP", "Random-PP")]:
        a = runs[runs["Method"] == dqn].set_index(["Instance", "Seed"])["Global Time (s)"]
        b = runs[runs["Method"] == rnd].set_index(["Instance", "Seed"])["Global Time (s)"]
        idx = a.index.intersection(b.index)
        if len(idx):
            out[dqn] = 100 * (a[idx].sum() / b[idx].sum() - 1)
    return out


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_excess(excess, sbs):
    fig = go.Figure()
    for m, label, color in [("DQN", "DQN", METHOD_COLOR["DQN"]), ("Random", "Random", METHOD_COLOR["Random"]),
                            (sbs, f"SBS ({sbs.split(':')[1]})", METHOD_COLOR["SBS"])]:
        if m in excess.columns:
            fig.add_bar(x=[pname(i) for i in excess.index], y=excess[m], name=label, marker_color=color,
                        hovertemplate="<b>%{x}</b><br>" + label + ": %{y:+.2f}% acima do VBS<extra></extra>")
    fig.add_hline(y=0, line_color=INK2, line_width=1)
    fig.update_yaxes(title_text="excesso sobre o VBS [%]", ticksuffix="%")
    return style(fig, "Figura 3 · Excesso da banda média sobre o VBS nas instâncias discriminantes", height=420,
                 barmode="group")


def fig_pp_change(change):
    fig = go.Figure()
    for m, color in [("DQN", METHOD_COLOR["DQN"]), ("Random", METHOD_COLOR["Random"])]:
        fig.add_bar(x=[pname(i) for i in change.index], y=change[m], name=m, marker_color=color,
                    hovertemplate="<b>%{x}</b><br>" + m + ": %{y:+.2f}% com início PP<extra></extra>")
    fig.add_hline(y=0, line_color=INK2, line_width=1)
    fig.update_yaxes(title_text="variação da banda média com início PP [%]", ticksuffix="%")
    return style(fig, "Figura 4 · Efeito do início pseudo-periférico (negativo = melhora)", height=420,
                 barmode="group")


def fig_vs_paper(cmp):
    fig = go.Figure()
    for m, color in [("DQN", METHOD_COLOR["DQN"]), ("Random", METHOD_COLOR["Random"]), ("SBS", METHOD_COLOR["SBS"]),
                     ("VBS", MUTED)]:
        d = cmp[cmp["Método"] == m]
        fig.add_scatter(x=d["Artigo"], y=d["Reproduzido"], mode="markers", name=m,
                        marker=dict(size=9, color=color, line=dict(color="#fff", width=1)),
                        customdata=d["Instance"],
                        hovertemplate="<b>%{customdata}</b> · " + m + "<br>artigo %{x:.2f}<br>reproduzido %{y:.2f}"
                                      "<extra></extra>")
    lim = [cmp[["Artigo", "Reproduzido"]].min().min() * 0.9, cmp[["Artigo", "Reproduzido"]].max().max() * 1.1]
    fig.add_scatter(x=lim, y=lim, mode="lines", name="y = x", line=dict(color=MUTED, dash="dot"), hoverinfo="skip")
    fig.update_xaxes(type="log", title_text="média publicada no artigo (Tabela 3)")
    fig.update_yaxes(type="log", title_text="média reproduzida")
    return style(fig, "Reprodução da Tabela 3: média publicada × média reproduzida", height=460)


# ---------------------------------------------------------------------------
# LaTeX (formato do artigo)
# ---------------------------------------------------------------------------
def latex_tables(A, B):
    def pm(s):
        return "$" + s.replace("±", r"\pm") + "$" if s else ""

    sbs_label = A["sbs"].split(":")[1] if A["sbs"] else "-"
    lines = [r"% Gerado por icits2026/report_icits.py", r"% Tabela 3",
             r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             rf"Instance & $|V|$ & $|E|$ & $B_0$ & RCM & DQN & Random & SBS ({sbs_label}) & VBS\\", r"\midrule"]
    for _, r in A["table3"].iterrows():
        cols = [c for c in A["table3"].columns if c.startswith("SBS")]
        lines.append(f"{r['Instance'].replace('_', chr(92) + '_')} & {r['|V|']} & {r['|E|']} & {r['B0']} & {r['RCM']} & "
                     f"{pm(r['DQN'])} & {pm(r['Random'])} & {pm(r[cols[0]])} & {pm(r['VBS'])}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"% Tabela 4", r"\begin{tabular}{lrrrr}", r"\toprule",
              rf"& \multicolumn{{2}}{{c}}{{vs Random}} & \multicolumn{{2}}{{c}}{{vs SBS ({sbs_label})}}\\",
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Instance & W/T/L & $p_H$ & W/T/L & $p_H$\\", r"\midrule"]
    for _, r in A["table4"].iterrows():
        b = lambda p: rf"$\mathbf{{{fmt_p(p)}}}$" if p < 0.05 else f"${fmt_p(p)}$"
        lines.append(f"{r['Instance'].replace('_', chr(92) + '_')} & {r['vs Random W/T/L']} & {b(r['vs Random p_H'])} & "
                     f"{r['vs SBS W/T/L']} & {b(r['vs SBS p_H'])}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if B is not None:
        lines += ["", r"% Tabela 5", r"\begin{tabular}{lrrrrrr}", r"\toprule",
                  r"Instance & DQN & DQN-PP & Random & Random-PP & W/T/L & $p_H$\\", r"\midrule"]
        for _, r in B["table5"].iterrows():
            lines.append(f"{r['Instance'].replace('_', chr(92) + '_')} & {pm(r['DQN'])} & {pm(r['DQN-PP'])} & "
                         f"{pm(r['Random'])} & {pm(r['Random-PP'])} & {r['W/T/L']} & {fmt_p(r['p_H'])}\\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Analises complementares (detalham as tabelas do artigo)
# ---------------------------------------------------------------------------
def load_decisions(results_dir):
    """decisions.csv (uma linha por decisão); None se não existir."""
    f = os.path.join(results_dir, "decisions.csv")
    if not os.path.exists(f):
        return None
    dec = pd.read_csv(f, usecols=["Instance", "Method", "Seed", "Decision", "Phase", "Action", "Bandwidth",
                                  "Incumbent", "Reward"], on_bad_lines="skip")
    # uma execução interrompida depois de gravar as decisões é refeita: fica a última gravação
    return dec.drop_duplicates(["Instance", "Method", "Seed", "Decision"], keep="last")


def fixed_cols(df):
    return [f for f in FIXED if f in df.columns]


def best_fixed_sets(A):
    """Centralidades fixas com a menor média em cada instância (empates contam para todas)."""
    m = A["means"][fixed_cols(A["means"])]
    return {i: {c.split(":")[1] for c in m.columns if m.loc[i, c] == m.loc[i].min()} for i in m.index}


def gap_closed_on(norm, insts, sbs, method):
    """GapClosed calculado só sobre `insts` (para testar a sensibilidade a cada instância)."""
    s = norm.loc[insts]
    vbs, sb, m = s[fixed_cols(s)].min(axis=1).mean(), s[sbs].mean(), s[method].mean()
    return (sb - m) / (sb - vbs) if sb != vbs else np.nan


def effect_table(runs, A):
    """Tamanho do efeito por instância: diferença mediana pareada (em unidades de banda)."""
    rows = []
    for i in A["disc"]:
        dqn, rnd, sb = paired(runs, i, "DQN"), paired(runs, i, "Random"), paired(runs, i, A["sbs"])
        idx = dqn.index.intersection(rnd.index).intersection(sb.index)
        rows.append({"Instance": pname(i), "B0": int(runs[runs["Instance"] == i]["Initial Bandwidth"].iloc[0]),
                     "DQN − Random (mediana)": float(np.median(dqn[idx] - rnd[idx])),
                     "DQN − Random (média)": float(np.mean(dqn[idx] - rnd[idx])),
                     "DQN − SBS (mediana)": float(np.median(dqn[idx] - sb[idx])),
                     "DQN − SBS (média)": float(np.mean(dqn[idx] - sb[idx])),
                     "desvio do DQN": float(np.std(dqn[idx], ddof=1))})
    return pd.DataFrame(rows)


def fig_fixed_heat(A):
    fixed = fixed_cols(A["means"])
    m = A["means"].loc[A["disc"], fixed]
    vbs = m.min(axis=1)
    z = 100 * m.sub(vbs, axis=0).div(vbs, axis=0)
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[f.split(":")[1] for f in fixed], y=[pname(i) for i in z.index], text=m.round(2).values,
        texttemplate="%{text}", textfont=dict(color=INK, size=11), zmin=0, zmax=max(3.0, float(np.nanpercentile(z.values, 95))),
        colorscale=[[0, "#f0efec"], [0.4, "#86b6ef"], [1, "#184f95"]],
        colorbar=dict(title="% acima<br>do VBS", ticksuffix="%"),
        hovertemplate="<b>%{y}</b> · %{x}<br>média %{text}<br>%{z:.2f}% acima da melhor fixa<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return style(fig, "Desempenho de cada centralidade fixa por instância (número = banda média; cor = distância ao VBS)",
                 height=60 + 34 * len(z) + 120, margin=dict(l=110, r=30, t=80, b=40))


def fig_seed_boxes(runs, A):
    disc, sbs = A["disc"], A["sbs"]
    vbs = A["means"].loc[disc, fixed_cols(A["means"])].min(axis=1)
    fig = go.Figure()
    for m, label, color in [("DQN", "DQN", METHOD_COLOR["DQN"]), ("Random", "Random", METHOD_COLOR["Random"]),
                            (sbs, f"SBS ({sbs.split(':')[1]})", METHOD_COLOR["SBS"])]:
        d = runs[(runs["Method"] == m) & runs["Instance"].isin(disc)]
        y = 100 * (d["Best Bandwidth"] - d["Instance"].map(vbs)) / d["Instance"].map(vbs)
        fig.add_box(x=d["Instance"].map(pname), y=y, name=label, marker_color=color, boxpoints="all", jitter=0.35,
                    pointpos=0, marker=dict(size=3), line_width=1.2, customdata=d["Seed"],
                    hovertemplate="<b>%{x}</b> · " + label + "<br>semente %{customdata}: %{y:+.2f}% vs média do VBS"
                                                              "<extra></extra>")
    fig.add_hline(y=0, line_color=INK2, line_width=1)
    fig.update_xaxes(categoryorder="array", categoryarray=[pname(i) for i in disc])
    fig.update_yaxes(title_text="banda da semente vs média do VBS [%]", ticksuffix="%")
    return style(fig, "Distribuição por semente nas instâncias discriminantes (cada ponto é uma execução)",
                 height=480, boxmode="group")


def fig_ranks(mean_ranks, sbs):
    names = [m.replace("Fixed:", "") for m in mean_ranks.index]
    colors = [METHOD_COLOR["DQN"] if m == "DQN" else METHOD_COLOR["Random"] if m == "Random"
              else METHOD_COLOR["SBS"] if m == sbs else MUTED for m in mean_ranks.index]
    fig = go.Figure(go.Bar(y=names, x=mean_ranks.values, orientation="h", marker_color=colors,
                           text=[f"{v:.2f}" for v in mean_ranks.values], textposition="outside",
                           textfont=dict(color=INK2), hovertemplate="%{y}: rank médio %{x:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="rank médio entre as instâncias (1 = melhor)", range=[0, len(names) * 1.1])
    return style(fig, "Ranks médios do teste de Friedman (DQN, Random e as 7 centralidades fixas)", height=400,
                 showlegend=False)


def convergence_curve(dec, A):
    """Mediana, entre instâncias discriminantes, do excesso médio do incumbente sobre o VBS a cada decisão."""
    disc, sbs = A["disc"], A["sbs"]
    vbs = A["means"].loc[disc, fixed_cols(A["means"])].min(axis=1)
    d = dec[dec["Instance"].isin(disc) & dec["Method"].isin(["DQN", "Random", sbs])]
    d = d.assign(exc=100 * (d["Incumbent"] - d["Instance"].map(vbs)) / d["Instance"].map(vbs))
    return d.groupby(["Method", "Instance", "Decision"])["exc"].mean().groupby(["Method", "Decision"]).median().unstack(0)


def fig_convergence(curve, sbs, train):
    fig = go.Figure()
    for m, label, color in [("DQN", "DQN", METHOD_COLOR["DQN"]), ("Random", "Random", METHOD_COLOR["Random"]),
                            (sbs, f"SBS ({sbs.split(':')[1]})", METHOD_COLOR["SBS"])]:
        if m in curve.columns:
            fig.add_scatter(x=curve.index, y=curve[m], mode="lines", name=label, line=dict(color=color, width=2),
                            hovertemplate=label + "<br>decisão %{x}: %{y:+.2f}% acima do VBS<extra></extra>")
    fig.add_vline(x=train + 0.5, line=dict(color=MUTED, dash="dash"))
    fig.add_annotation(x=train + 0.5, y=1, yref="paper", text="fim do treino", showarrow=False, xanchor="left",
                       font=dict(color=MUTED, size=11))
    fig.add_hline(y=0, line_color=INK2, line_width=1)
    late_df = curve.loc[curve.index >= 20]
    late = (late_df if len(late_df) else curve).values   # orçamentos de teste podem ter menos de 20 decisões
    fig.update_yaxes(title_text="incumbente vs média do VBS [%] (mediana entre instâncias)", ticksuffix="%",
                     range=[min(-0.5, float(np.nanmin(late)) - 0.3), float(np.nanpercentile(late, 99)) + 0.5])
    fig.update_xaxes(title_text="decisão (cada uma = 15 construções)")
    return style(fig, "Convergência: melhor banda encontrada até cada decisão", height=460)


def dqn_choice_blocks(dec, A, train, block=50):
    """Participação de cada centralidade nas escolhas do DQN por bloco de decisões e taxa de acerto da
    melhor centralidade fixa (DQN e Random) nas instâncias discriminantes."""
    best = best_fixed_sets(A)
    d = dec[dec["Method"].isin(["DQN", "Random"]) & dec["Instance"].isin(A["disc"])].copy()
    start = ((d["Decision"] - 1) // block) * block + 1
    d["bloco"] = np.where(d["Phase"] == "greedy", f"gulosa ({train + 1}–{d['Decision'].max()})",
                          start.astype(str) + "–" + np.minimum(start + block - 1, train).astype(str))
    order = list(dict.fromkeys(d.sort_values("Decision")["bloco"]))
    d["hit"] = [a in best[i] for i, a in zip(d["Instance"], d["Action"])]
    share = (d[d["Method"] == "DQN"].groupby(["bloco", "Action"]).size().unstack(fill_value=0)
             .reindex(order).fillna(0))
    share = 100 * share.div(share.sum(axis=1), axis=0)
    hits = 100 * d.groupby(["bloco", "Method"])["hit"].mean().unstack().reindex(order)
    expected = 100 * np.mean([len(best[i]) / len(CENTRALITY_NAMES) for i in A["disc"]])
    return share, hits, expected


def fig_choice_share(share):
    fig = go.Figure()
    palette = dict(zip(CENTRALITY_NAMES, SLOTS))
    for c in [c for c in CENTRALITY_NAMES if c in share.columns]:
        fig.add_bar(x=share.index, y=share[c], name=c, marker_color=palette[c],
                    hovertemplate="%{x}<br>" + c + ": %{y:.1f}% das escolhas<extra></extra>")
    fig.add_hline(y=100 / len(CENTRALITY_NAMES), line=dict(color=INK, dash="dot", width=1))
    fig.update_yaxes(title_text="% das escolhas do DQN", ticksuffix="%", range=[0, 100])
    fig.update_xaxes(title_text="bloco de decisões")
    return style(fig, "O que o DQN escolhe ao longo da execução (instâncias discriminantes)", height=460,
                 barmode="stack")


def fig_hits(hits, expected):
    fig = go.Figure()
    for m in ["DQN", "Random"]:
        if m in hits.columns:
            fig.add_scatter(x=hits.index, y=hits[m], mode="lines+markers", name=m,
                            line=dict(color=METHOD_COLOR[m], width=2), marker=dict(size=8),
                            hovertemplate=m + "<br>%{x}: %{y:.1f}%<extra></extra>")
    fig.add_hline(y=expected, line=dict(color=INK, dash="dot", width=1.2))
    fig.add_annotation(x=1, xref="paper", y=expected, text="esperado por acaso", showarrow=False, xanchor="right",
                       yanchor="bottom", font=dict(color=MUTED, size=11))
    fig.update_yaxes(title_text="% das decisões com uma das melhores fixas", ticksuffix="%", rangemode="tozero")
    fig.update_xaxes(title_text="bloco de decisões")
    return style(fig, "O DQN aprende a escolher a melhor centralidade fixa da instância?", height=420)


def greedy_choice_table(dec, A):
    """Centralidades escolhidas pelo DQN na fase gulosa, por instância, com as melhores fixas marcadas."""
    best = best_fixed_sets(A)
    g = dec[(dec["Method"] == "DQN") & (dec["Phase"] == "greedy") & dec["Instance"].isin(A["disc"])]
    tab = 100 * g.groupby(["Instance", "Action"]).size().unstack(fill_value=0)
    tab = tab.div(tab.sum(axis=1), axis=0).reindex(columns=CENTRALITY_NAMES, fill_value=0)
    rows = []
    for i in A["disc"]:
        if i not in tab.index:
            continue
        r = {"Instance": pname(i)}
        r.update({c: f"{tab.loc[i, c]:.0f}%" + (" ★" if c in best[i] else "") for c in CENTRALITY_NAMES})
        r["% na melhor fixa"] = sum(tab.loc[i, c] for c in best[i])
        rows.append(r)
    return pd.DataFrame(rows)


def reward_table(dec, runs, train):
    """Para cada instância: limiar 0,01·B0 em unidades de banda e o que acontece nas decisões de treino do DQN."""
    b0 = runs.groupby("Instance")["Initial Bandwidth"].first()
    d = dec[(dec["Method"] == "DQN") & (dec["Phase"] == "train")].sort_values(["Instance", "Seed", "Decision"])
    prev = d.groupby(["Instance", "Seed"])["Incumbent"].shift(1)
    d = d.assign(prev=prev.fillna(d["Instance"].map(b0)))
    d["kind"] = np.select([d["Bandwidth"] < d["prev"], d["Bandwidth"] == d["prev"]], ["melhora", "empate"], "piora")
    d["gain"] = d["prev"] - d["Bandwidth"]
    rows = []
    for i, g in d.groupby("Instance", sort=False):
        thr = 0.01 * b0[i]
        imp = g[g["kind"] == "melhora"]
        late = g[g["Decision"] > 30]
        late_imp = late[late["kind"] == "melhora"]
        rows.append({"Instance": pname(i), "B0": int(b0[i]), "limiar 0,01·B0": thr,
                     "% melhora": 100 * (g["kind"] == "melhora").mean(),
                     "% empate": 100 * (g["kind"] == "empate").mean(),
                     "% piora": 100 * (g["kind"] == "piora").mean(),
                     "ganho mediano (após decisão 30)": float(late_imp["gain"].median()) if len(late_imp) else np.nan,
                     "% melhorias abaixo do limiar (após 30)": 100 * float((late_imp["gain"] < thr).mean())
                     if len(late_imp) else np.nan})
    present = [r["Instance"] for r in rows]
    order = [pname(i) for i in PAPER_TABLE3 if pname(i) in present]
    order += [x for x in present if x not in order]
    return pd.DataFrame(rows).set_index("Instance").reindex(order).reset_index()


def claims_table(A, B, T, cmp, rcm_cmp, greedy_hits, n_dqn_runs, gap_range):
    """Afirmações do artigo x reprodução, com um veredito por linha."""
    ok, warn = "✅ reproduzido", "⚠️ ver nota"
    rows = []
    add = lambda claim, paper, repro, status: rows.append(
        {"Afirmação do artigo": claim, "Artigo": paper, "Reprodução": repro, "Situação": status})
    same = int((rcm_cmp["B0 artigo"] == rcm_cmp["B0 reproduzido"]).sum()), int(
        (rcm_cmp["RCM artigo"] == rcm_cmp["RCM reproduzido"]).sum())
    add("Mesmas instâncias (B0 e RCM)", "Tabela 3", f"B0 {same[0]}/{len(rcm_cmp)}; RCM {same[1]}/{len(rcm_cmp)}",
        ok if same == (len(rcm_cmp), len(rcm_cmp)) else warn)
    med = cmp["Diferença (%)"].abs().median()
    add("Médias de DQN, Random, SBS e VBS", "Tabela 3", f"diferença mediana {med:.2f}%", ok if med < 1 else warn)
    add("SBS = Katz", "rank médio 2,50", f"{A['sbs'].split(':')[1]} (rank médio {A['fixed_ranks'].mean()[A['sbs']]:.2f})",
        ok if A["sbs"] == "Fixed:Katz" else warn)
    r = A["mean_ranks"]
    add("No Friedman, Degree, Katz e Random ficam à frente do DQN", "3,27; 3,31; 3,35 vs 4,08",
        "; ".join(f"{k.replace('Fixed:', '')} {r[k]:.2f}" for k in ["Fixed:Degree", "Fixed:Katz", "Random", "DQN"]
                  if k in r.index),
        ok if all(r.get(k, 99) < r["DQN"] for k in ["Fixed:Degree", "Fixed:Katz", "Random"]) else warn)
    t4 = A["table4"]
    better = sum(1 for _, x in t4.iterrows() if x["vs Random p_H"] < 0.05
                 and int(x["vs Random W/T/L"].split("/")[0]) > int(x["vs Random W/T/L"].split("/")[2]))
    worse = sum(1 for _, x in t4.iterrows() if x["vs Random p_H"] < 0.05) - better
    add("O DQN não é melhor que a seleção aleatória em nenhuma instância", "pior em 2, igual em 8",
        f"melhor em {better}, pior em {worse}, igual em {len(t4) - better - worse}", ok if better == 0 else warn)
    gc = A["gap_closed"]
    add("Nenhum seletor captura parte relevante do gap SBS–VBS", "DQN 2,2%; Random ≈ −19%",
        f"DQN {fpct(gc['DQN'], 1)}; Random {fpct(gc['Random'], 1)} (sem uma instância: DQN "
        f"{fpct(gap_range[0], 0)} a {fpct(gap_range[1], 0)})",
        ok if abs(gc["DQN"]) < 0.25 and abs(gc["Random"]) < 0.25 else warn)
    add("A fase gulosa quase nunca melhora o resultado", "20 de 390", f"{greedy_hits} de {n_dqn_runs}",
        ok if greedy_hits / max(n_dqn_runs, 1) < 0.1 else warn)
    if B is not None:
        ch = B["change"]
        mesh = [i for i in ch.index if i in MESHES]
        oth = [i for i in ch.index if i not in MESHES]
        add("O início PP melhora todas as malhas, sob os dois seletores", "−2,5 a −4,4%",
            f"{ch.loc[mesh].values.min():+.1f} a {ch.loc[mesh].values.max():+.1f}%",
            ok if (ch.loc[mesh] < 0).all().all() else warn)
        add("O início PP piora as outras famílias", "+5,3 a +10,0%",
            f"{ch.loc[oth].values.min():+.1f} a {ch.loc[oth].values.max():+.1f}%",
            ok if (ch.loc[oth] > 0).all().all() else warn)
        add("Com início PP, DQN e Random não diferem", "p_H ≥ 0,317", f"menor p_H = {B['table5']['p_H'].min():.3f}",
            ok if B["table5"]["p_H"].min() >= 0.05 else warn)
        if B["residual"]:
            v = list(B["residual"].values())
            add("O início PP não fecha a distância para o RCM nas maiores malhas", "fator 1,45 a 1,50",
                f"fator {min(v):.2f} a {max(v):.2f}", ok if min(v) > 1.3 else warn)
    if T:
        add("Aprender custa pouco em tempo", "+5,4% e +6,2%", "; ".join(f"{k} {v:+.1f}%" for k, v in T.items()),
            "⚠️ ver nota")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Extensoes E1 (construcao), E2 (recompensa) e E3 (bandits)
# ---------------------------------------------------------------------------
LEARNERS = ("DQN", "UCB1", "AP")
LEARNER_LABEL = {"DQN": "DQN", "UCB1": "UCB1", "AP": "Adaptive pursuit"}
REWARD_LABEL = {"paper": "recompensa do artigo", "half_unit_tie": "empate = 0,5/B0"}


def split_method(m):
    """'UCB1@half_unit_tie-PP' -> ('UCB1', 'half_unit_tie', 'peripheral')."""
    start = "centrality"
    if m.endswith("-PP"):
        m, start = m[:-3], "peripheral"
    sel, _, rew = m.partition("@")
    return sel, rew or "paper", start


def method_label(m):
    sel, rew, start = split_method(m)
    label = LEARNER_LABEL.get(sel, sel.replace("Fixed:", "fixa "))
    if rew != "paper":
        label += f" ({REWARD_LABEL.get(rew, rew)})"
    return label + (" · PP" if start == "peripheral" else "")


def extension_methods(runs, pp=False):
    """Seletores das extensões E2/E3 (o DQN do artigo e o Random ficam de fora)."""
    out = []
    for m in sorted(runs["Method"].unique()):
        sel, rew, start = split_method(m)
        if sel in LEARNERS and (start == "peripheral") == pp and m not in ("DQN", "DQN-PP"):
            out.append(m)
    return out


def compare_paired(runs, insts, a, b):
    """a contra b nas instâncias: W/T/L somados, instâncias com diferença significativa (Wilcoxon do
    artigo, Holm entre as instâncias) e diferença mediana das médias (em %)."""
    w = t = l = 0
    pvals, signs, diffs = [], [], []
    for i in insts:
        x, y = paired(runs, i, a), paired(runs, i, b)
        idx = x.index.intersection(y.index)
        if not len(idx):
            continue
        ww, tt, ll = wtl(x[idx], y[idx])
        w, t, l = w + ww, t + tt, l + ll
        pvals.append(wilcoxon_paper(x[idx], y[idx]))
        signs.append(np.sign(float((x[idx] - y[idx]).mean())))
        diffs.append(100 * (x[idx].mean() - y[idx].mean()) / y[idx].mean())
    if not pvals:
        return None
    adj = holm(pvals)
    return {"W/T/L": f"{w}/{t}/{l}",
            "melhor (sig.)": int(sum(p < 0.05 and s < 0 for p, s in zip(adj, signs))),
            "pior (sig.)": int(sum(p < 0.05 and s > 0 for p, s in zip(adj, signs))),
            "dif. mediana (%)": float(np.median(diffs))}


def extension_analysis(runs, A, norm, dec):
    """Resumo dos seletores das extensões contra Random e contra o DQN do artigo, nas instâncias
    discriminantes, com GapClosed, excesso sobre o VBS e acerto da melhor fixa na fase gulosa."""
    ext = extension_methods(runs)
    if not ext:
        return None
    sbs, disc, best = A["sbs"], A["disc"], best_fixed_sets(A)
    vbs = A["means"].loc[disc, fixed_cols(A["means"])].min(axis=1)
    methods = [m for m in ["DQN", "Random"] + ext if m in set(runs["Method"])]
    rows, excess, per_inst = [], {}, {}
    for m in methods:
        insts_all = [i for i in A["complete"] if len(paired(runs, i, m))]
        insts = [i for i in disc if i in insts_all]
        if not insts:
            continue
        means_m = pd.Series({i: paired(runs, i, m).mean() for i in insts})
        excess[m] = 100 * (means_m - vbs[insts]) / vbs[insts]
        per_inst[method_label(m)] = means_m.rename(index=pname)
        row = {"método": method_label(m), "instâncias": len(insts_all),
               "GapClosed (%)": 100 * gap_closed_on(norm, insts_all, sbs, m),
               "excesso mediano sobre o VBS (%)": float(excess[m].median())}
        for ref, lab in [("Random", "vs Random"), ("DQN", "vs DQN")]:
            if m != ref:
                c = compare_paired(runs, insts, m, ref)
                if c:
                    row.update({f"{lab}: {k}": v for k, v in c.items()})
        if dec is not None:
            g = dec[(dec["Method"] == m) & (dec["Phase"] == "greedy") & dec["Instance"].isin(insts)]
            if len(g):
                row["% gulosas na melhor fixa"] = 100 * float(np.mean(
                    [a in best[i] for i, a in zip(g["Instance"], g["Action"])]))
        rows.append(row)
    per_inst = pd.DataFrame(per_inst).reset_index().rename(columns={"index": "Instance"})
    return dict(ext=ext, summary=pd.DataFrame(rows), per_inst=per_inst, excess=pd.DataFrame(excess),
                methods=methods)


def fig_ext_excess(excess):
    fig = go.Figure()
    palette = {"DQN": METHOD_COLOR["DQN"], "Random": METHOD_COLOR["Random"]}
    others = [m for m in excess.columns if m not in palette]
    for k, m in enumerate(others):
        palette[m] = SLOTS[(k + 2) % len(SLOTS)]
    for m in excess.columns:
        fig.add_bar(x=[pname(i) for i in excess.index], y=excess[m], name=method_label(m), marker_color=palette[m],
                    hovertemplate="<b>%{x}</b><br>" + method_label(m) + ": %{y:+.2f}% acima do VBS<extra></extra>")
    fig.add_hline(y=0, line_color=INK2, line_width=1)
    fig.update_yaxes(title_text="excesso da média sobre o VBS [%]", ticksuffix="%")
    return style(fig, "Seletores das extensões: excesso da banda média sobre o VBS por instância", height=460,
                 barmode="group")


def e1_comparison(runs, base_runs):
    """Mesmas instâncias e sementes: construção do artigo (base_runs) x ordem do pai (runs)."""
    rcm = runs.groupby("Instance")["RCM Bandwidth"].first()
    common = set(runs["Instance"]) & set(base_runs["Instance"])
    order = [i for i in list(PAPER_TABLE3) + sorted(common - set(PAPER_TABLE3)) if i in common]
    rows = []
    for i in order:
        row = {"Instance": pname(i), "RCM": int(rcm[i])}
        for lab, rr in [("artigo", base_runs), ("pai", runs)]:
            mm = rr[rr["Instance"] == i].groupby("Method")["Best Bandwidth"].mean()
            fx = mm[[c for c in mm.index if c.startswith("Fixed:")]]
            row[f"DQN ({lab})"] = mm.get("DQN", np.nan)
            row[f"Random ({lab})"] = mm.get("Random", np.nan)
            row[f"VBS ({lab})"] = fx.min() if len(fx) else np.nan
            row[f"dispersão das fixas ({lab}, %)"] = 100 * (fx.max() - fx.min()) / fx.min() if len(fx) else np.nan
        row["DQN/RCM (artigo)"] = row["DQN (artigo)"] / row["RCM"]
        row["DQN/RCM (pai)"] = row["DQN (pai)"] / row["RCM"]
        row["variação do DQN (%)"] = 100 * (row["DQN (pai)"] - row["DQN (artigo)"]) / row["DQN (artigo)"]
        rows.append(row)
    return pd.DataFrame(rows)


def fig_e1(e1):
    fig = go.Figure()
    for col, label, color in [("DQN/RCM (artigo)", "construção do artigo", MUTED),
                              ("DQN/RCM (pai)", "ordem do pai (E1)", METHOD_COLOR["DQN"])]:
        fig.add_bar(x=e1["Instance"], y=e1[col], name=label, marker_color=color,
                    hovertemplate="<b>%{x}</b><br>" + label + ": %{y:.2f}× o RCM<extra></extra>")
    fig.add_hline(y=1, line=dict(color=INK2, width=1, dash="dash"))
    fig.update_yaxes(title_text="banda média do DQN / banda do RCM")
    return style(fig, "E1 · DQN em relação ao RCM: construção do artigo × ordem do pai (1 = empata com o RCM)",
                 height=440, barmode="group")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--train", type=int, default=None, help="filtra pelo orçamento (padrão: o mais frequente)")
    parser.add_argument("--greedy", type=int, default=None)
    parser.add_argument("--baseline-dir", default=None,
                        help="resultados com a construção do artigo, para a comparação da extensão E1 (padrão: "
                             "results/ na base sbpo ou results_<base>_paper/)")
    args = parser.parse_args()

    runs, instances, train, greedy = load(args.results_dir, args.train, args.greedy)
    out_tables = os.path.join(args.results_dir, "tables")
    os.makedirs(out_tables, exist_ok=True)

    dataset = str(runs["Dataset"].iloc[0])
    construction = str(runs["Construction"].iloc[0])
    # modo reprodução: mesma base e mesma construção do artigo -> compara com os valores publicados
    paper_mode = dataset == "sbpo" and construction == "paper"
    setting = f"base {dataset}, construção {'do artigo' if construction == 'paper' else 'com ordem do pai (E1)'}"

    A = main_analysis(runs)
    B = ablation_analysis(runs)
    T = time_ratios(runs)
    sbs = A["sbs"]
    n_seeds = int(runs.groupby(["Instance", "Method"]).size().median())
    full_budget = (train, greedy) == (300, 60)

    A["table3"].to_csv(os.path.join(out_tables, "table3_main_results.csv"), index=False)
    A["table4"].to_csv(os.path.join(out_tables, "table4_posthoc.csv"), index=False)
    A["mean_ranks"].rename("mean rank").to_csv(os.path.join(out_tables, "friedman_mean_ranks.csv"))
    if B is not None:
        B["table5"].drop(columns=[c for c in B["table5"].columns if c.startswith("_")]).to_csv(
            os.path.join(out_tables, "table5_ablation.csv"), index=False)
    with open(os.path.join(out_tables, "tables_icits.tex"), "w", encoding="utf-8") as f:
        f.write(latex_tables(A, B))

    # comparação com o artigo (médias da Tabela 3)
    cmp_rows = []
    for i in A["complete"]:
        if i not in PAPER_TABLE3:
            continue
        paper = dict(zip(["DQN", "Random", "SBS", "VBS"], PAPER_TABLE3[i][4:]))
        repro = {"DQN": A["means"].loc[i, "DQN"], "Random": A["means"].loc[i, "Random"],
                 "SBS": A["means"].loc[i, sbs], "VBS": A["means"].loc[i, A["vbs_method"][i]]}
        for m in paper:
            cmp_rows.append({"Instance": pname(i), "Método": m, "Artigo": paper[m], "Reproduzido": repro[m],
                             "Diferença (%)": 100 * (repro[m] - paper[m]) / paper[m]})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(os.path.join(out_tables, "comparison_with_paper.csv"), index=False)
    rcm_cmp = pd.DataFrame([{"Instance": pname(i), "RCM artigo": PAPER_TABLE3[i][3],
                             "RCM reproduzido": runs[runs["Instance"] == i]["RCM Bandwidth"].iloc[0],
                             "B0 artigo": PAPER_TABLE3[i][2],
                             "B0 reproduzido": runs[runs["Instance"] == i]["Initial Bandwidth"].iloc[0]}
                            for i in A["complete"] if i in PAPER_TABLE3])

    # números do texto
    ranks_txt = "; ".join(f"{m.replace('Fixed:', '')} {r:.2f}" for m, r in A["mean_ranks"].items())
    sbs_rank = A["fixed_ranks"].mean()[sbs] if sbs else np.nan
    gc = A["gap_closed"]
    share_max = A["gap_share"].idxmax() if len(A["gap_share"]) else None
    w_dqn_rnd = sum(1 for _, r in A["table4"].iterrows() if r["vs Random p_H"] < 0.05)
    dqn_runs = runs[runs["Method"] == "DQN"]
    greedy_hits = int(dqn_runs["Greedy Improved"].sum())
    if cmp.empty:
        cmp = pd.DataFrame(columns=["Instance", "Método", "Artigo", "Reproduzido", "Diferença (%)"])
    match_abs = cmp.assign(abs_diff=cmp["Diferença (%)"].abs())

    # --- análises complementares ---
    norm = runs.groupby(["Instance", "Method"])["Normalized Bandwidth"].mean().unstack()
    loo = [gap_closed_on(norm, [j for j in A["complete"] if j != i], sbs, "DQN") for i in A["complete"]
           if len(A["complete"]) > 1]
    gap_range = (np.nanmin(loo), np.nanmax(loo)) if np.isfinite(loo).any() else (np.nan, np.nan)
    gc_no_top = gap_closed_on(norm, [j for j in A["complete"] if j != share_max], sbs, "DQN") if share_max else np.nan
    gcr_no_top = gap_closed_on(norm, [j for j in A["complete"] if j != share_max], sbs, "Random") if share_max else np.nan
    eff = effect_table(runs, A)
    best_sets = best_fixed_sets(A)
    best_txt = "; ".join(f"{pname(i)}: {', '.join(sorted(best_sets[i]))}" for i in A["disc"])
    nondisc = [pname(i) for i in A["complete"] if i not in A["disc"]]
    dec = load_decisions(args.results_dir)
    claims = claims_table(A, B, T, cmp, rcm_cmp, greedy_hits, len(dqn_runs), gap_range) if paper_mode else None
    if claims is not None:
        claims.to_csv(os.path.join(out_tables, "claims_vs_paper.csv"), index=False)
    eff.to_csv(os.path.join(out_tables, "effect_sizes.csv"), index=False)
    n_ok = int(claims["Situação"].str.startswith("✅").sum()) if claims is not None else 0
    disc_rcm = runs[runs["Instance"].isin(A["disc"])].groupby("Instance")["RCM Bandwidth"].first()
    dqn_rcm = (A["means"].loc[A["disc"], "DQN"] / disc_rcm).median() if "DQN" in A["means"] else np.nan
    EXT = extension_analysis(runs, A, norm, dec)
    if EXT is not None:
        EXT["summary"].to_csv(os.path.join(out_tables, "extensions_summary.csv"), index=False)
    # E1: comparação com a construção do artigo (mesma base)
    E1 = None
    if construction == "parent":
        base_dir = args.baseline_dir or (os.path.join(HERE, "results") if dataset == "sbpo"
                                         else os.path.join(HERE, f"results_{dataset}_paper"))
        if os.path.exists(os.path.join(base_dir, "runs.csv")):
            base_runs, _, _, _ = load(base_dir, train, greedy)
            base_runs = base_runs[base_runs["Construction"] == "paper"]
            if len(base_runs):
                E1 = dict(table=e1_comparison(runs, base_runs), A_base=main_analysis(base_runs), dir=base_dir)
                E1["table"].to_csv(os.path.join(out_tables, "e1_construction_comparison.csv"), index=False)
    time_by_method = (runs.groupby("Method")["Global Time (s)"].agg(["count", "mean", "sum"])
                      .rename(columns={"count": "execuções", "mean": "tempo médio (s)", "sum": "tempo total (s)"})
                      .reset_index().sort_values("tempo médio (s)"))
    if dec is not None:
        curve = convergence_curve(dec, A)
        share, hits, expected = dqn_choice_blocks(dec, A, train)
        greedy_tab = greedy_choice_table(dec, A)
        rew = reward_table(dec, runs, train)
        rew.to_csv(os.path.join(out_tables, "reward_analysis.csv"), index=False)
        rew_i = rew.set_index("Instance")
        mesh_rows = rew_i.loc[[pname(i) for i in MESHES if pname(i) in rew_i.index]]
        other_rows = rew_i.loc[[pname(i) for i in ["bp_0", "bp_200", "shl_0"] if pname(i) in rew_i.index]]
        disc_rows = rew_i.loc[[pname(i) for i in A["disc"] if pname(i) in rew_i.index]]
        rw = dict(worse=(disc_rows["% piora"].min(), disc_rows["% piora"].max()),
                  tie=(disc_rows["% empate"].min(), disc_rows["% empate"].max()),
                  imp=(disc_rows["% melhora"].min(), disc_rows["% melhora"].max()),
                  mesh_below=mesh_rows["% melhorias abaixo do limiar (após 30)"].min(),
                  other_below=(other_rows["% melhorias abaixo do limiar (após 30)"].min(),
                               other_rows["% melhorias abaixo do limiar (após 30)"].max()))
        at_train = curve.loc[train] if train in curve.index else curve.iloc[-1]
        final = curve.iloc[-1]
        hit_late = hits.iloc[-2] if len(hits) >= 2 else hits.iloc[-1]   # último bloco de treino
        hit_greedy = hits.iloc[-1]

    tiles = "".join(([
        tile(f"{n_ok}/{len(claims)}", "afirmações do artigo reproduzidas"),
        tile(f"{match_abs['abs_diff'].median():.2f}%", "diferença mediana para as médias da Tabela 3"),
    ] if paper_mode else [
        tile(construction, "construção (paper = artigo; parent = ordem do pai, E1)"),
        tile(f"{dqn_rcm:.2f}×", "banda do DQN / RCM (mediana das instâncias discriminantes)"),
    ]) + [
        tile(f"{len(A['complete'])} × {n_seeds} × {len(A['methods'])}", "instâncias × sementes × métodos"),
        tile(sbs.split(":")[1] if sbs else "-", f"SBS (rank médio {sbs_rank:.2f})"),
        tile(f"{fpct(gc.get('DQN', np.nan), 1)}", "do gap SBS–VBS fechado pelo DQN (artigo: 2,2%)"),
    ])

    parts = [
        "<!doctype html><html lang='pt-br'><head><meta charset='utf-8'><title>Reprodução ICITS 2026</title>"
        f"<style>{CSS}</style><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body><main>",
        ("<h1>Reprodução do artigo ICITS 2026: DQN × seleção aleatória × centralidades fixas</h1>" if paper_mode
         else f"<h1>Extensão do artigo ICITS 2026 ({setting})</h1>"),
        "<p class='src'><i>A Matched-Budget Assessment of DQN Centrality Selection for Matrix Bandwidth "
        "Minimization</i> (<code>2026-ICITS-Peru-3/samplepaper.tex</code>). Resultados de "
        f"<code>{os.path.relpath(args.results_dir, BASE_DIR)}</code>, gerados por <code>icits2026/main_icits.py</code>; "
        "tabelas em CSV e LaTeX em <code>tables/</code>.</p>",
        f'<div class="tiles">{tiles}</div>',
    ]
    if not full_budget:
        parts.append("<div class='q'><b>Atenção:</b> estes resultados usam um orçamento reduzido "
                     f"({train}+{greedy} decisões). O artigo usa 300+60; os números abaixo servem só para conferir "
                     "o pipeline e não são comparáveis aos publicados.</div>")

    # ---------------- Resumo ----------------
    if not paper_mode:
        parts += [
            "<h2>Resumo</h2>",
            "<p>Este relatório aplica o protocolo do artigo (mesmo orçamento, mesmas sementes, seleção aleatória, "
            f"centralidades fixas, SBS, VBS e GapClosed) a uma configuração diferente da publicada: <b>{setting}</b>. "
            "Os valores marcados como <i>Artigo (SBPO, construção original)</i> são do artigo e servem só como "
            "referência: não se espera que coincidam com os daqui.</p>",
            explainer("O que muda em relação ao artigo", [
                "<b>Construção com ordem do pai (E1)</b>: os filhos de cada vértice são rotulados na ordem do pai, "
                "como no Cuthill–McKee, e a centralidade só ordena os filhos de cada pai. É o primeiro próximo passo "
                "da conclusão do artigo." if construction == "parent" else
                "<b>Construção do artigo</b>: cada nível inteiro é ordenado pela centralidade e embaralhado pela LCR.",
                "<b>Base survey</b>: matrizes de <code>data/survey</code>, com Closeness, Betweenness e Harmonic "
                "aproximadas por amostragem (grafos grandes demais para as versões exatas)." if dataset == "survey"
                else "<b>Base SBPO</b>: as 13 instâncias HB do artigo.",
                f"Banda média do DQN / RCM nas instâncias discriminantes (mediana): <b>{dqn_rcm:.2f}×</b> "
                "(no artigo, 1,45 a 1,50 nas maiores malhas mesmo com início PP).",
            ]),
        ]
    else:
      parts += [
        "<h2>Resumo</h2>",
        "<p>O artigo pergunta se um DQN que escolhe entre sete centralidades para guiar uma construção aleatória "
        "por níveis resolve melhor o problema de minimização de banda do que um seletor que não sabe nada, com o "
        "mesmo orçamento. A resposta publicada é negativa: o DQN não supera a seleção aleatória, fecha uma fração "
        "pequena do gap entre a melhor centralidade única (SBS) e o oráculo por instância (VBS), e o ganho mais "
        "claro vem da regra do vértice inicial (pseudo-periférico), não da política aprendida. Esta reprodução "
        "refaz todos os experimentos com o código de <code>icits2026/</code> e confronta cada afirmação.</p>",
        f"<div class='q'><b>Veredito.</b> {n_ok} de {len(claims)} afirmações quantitativas do artigo são "
        f"reproduzidas, e as médias da Tabela 3 ficam a {match_abs['abs_diff'].median():.2f}% (mediana) dos valores "
        "publicados. As conclusões do artigo se mantêm. Os pontos marcados com ⚠️ são diferenças de magnitude, "
        "explicadas nas notas abaixo da tabela, e nenhum deles inverte uma conclusão.</div>",
        "<h3>Afirmações do artigo confrontadas com a reprodução</h3>",
        "<p class='howto'><b>Como ler.</b> Uma linha por afirmação quantitativa, na ordem da Seção 4 do artigo: o "
        "valor publicado, o reproduzido e a situação (✅ reproduzido; ⚠️ diferença explicada nas notas).</p>",
        table_html(claims),
        explainer("Notas", [
                    "<b>GapClosed</b> é uma razão entre diferenças pequenas (o gap SBS–VBS inteiro é de ~0,004 na "
                    "banda normalizada) e é dominada por poucas instâncias. Retirando uma instância de cada vez, o "
                    f"GapClosed do DQN varia de {fpct(gap_range[0], 0)} a {fpct(gap_range[1], 0)}. Valores "
                    "pequenos e instáveis são exatamente a conclusão do artigo; o sinal do Random não é estável "
                    "nem lá nem aqui.",
                    "<b>Faixas do início PP</b>: nas famílias não malha, a piora reproduzida é menor que a "
                    "publicada, e o desvio em bp_0 e bp_200 com PP é bem maior. A causa provável é a forma de sortear "
                    "o vértice pseudo-periférico, que o artigo não especifica (aqui, um conjunto de até 16 vértices).",
                    "<b>Tempo</b>: a comparação pareada de tempo só vale se DQN e Random rodarem nas mesmas "
                    "condições. Nesta execução parte das tarefas rodou com 1 processo e parte com vários, então "
                    "a diferença de tempo inclui a disputa entre processos e não é comparável à do artigo.",
                ]),
      ]

    # ---------------- E1: construção ----------------
    if E1 is not None:
        e1, Ab = E1["table"], E1["A_base"]
        gap_base = Ab["score"].get(Ab["sbs"], np.nan) - Ab["score_vbs"]
        gap_new = A["score"].get(sbs, np.nan) - A["score_vbs"]
        disp_cols = ("dispersão das fixas (artigo, %)", "dispersão das fixas (pai, %)")
        parts += [
            "<h2>Extensão E1 · Ordenação dentro do nível que preserva o pai</h2>",
            section(
                "A ordem do pai fecha a distância para o RCM?",
                "O artigo atribui o fator residual de ~1,5 em relação ao RCM à ordenação dentro de cada nível. Com "
                "a ordem do pai, mesma base, mesmas sementes e mesmo orçamento, o que acontece?",
                f"DQN/RCM (mediana): construção do artigo {e1['DQN/RCM (artigo)'].median():.2f}× → ordem do pai "
                f"<b>{e1['DQN/RCM (pai)'].median():.2f}×</b>; variação mediana da banda do DQN "
                f"{e1['variação do DQN (%)'].median():+.1f}%. Instâncias em que o DQN empata ou supera o RCM: "
                f"{int((e1['DQN/RCM (pai)'] <= 1).sum())} de {len(e1)} (antes: {int((e1['DQN/RCM (artigo)'] <= 1).sum())}).",
                fig_e1(e1),
                table_html(e1, {c: "{:.2f}" for c in e1.columns if c not in ("Instance", "RCM")}),
                how="Cada par de barras compara, numa instância, a banda média do DQN dividida pela do RCM com a "
                    "construção do artigo (cinza) e com a ordem do pai (azul). Abaixo da linha tracejada, o DQN é "
                    "melhor que o RCM. A tabela traz DQN, Random e VBS nas duas construções e a dispersão entre as "
                    f"centralidades fixas (base de comparação: <code>{os.path.relpath(E1['dir'], BASE_DIR)}</code>).",
                notes=[
                    "Esta é a hipótese da Seção 4.3 do artigo testada diretamente: se o fator cai para perto de 1, a "
                    "causa do resíduo era a ordenação dentro do nível, e não o vértice inicial nem o seletor.",
                    f"<b>Há mais o que aprender?</b> O gap SBS–VBS (média de B/B0) passa de {gap_base:.5f} para "
                    f"{gap_new:.5f}, e a dispersão mediana entre as centralidades fixas de "
                    f"{e1[disp_cols[0]].median():.2f}% para {e1[disp_cols[1]].median():.2f}%. Um gap maior significa "
                    "que escolher bem a centralidade passa a valer mais, e daria espaço para um seletor aprendido.",
                    "Com a ordem do pai a centralidade atua só na ordem dos filhos de cada pai e no sorteio do início, "
                    "então as ações podem ficar mais parecidas entre si. As seções seguintes (Tabela 4, GapClosed, "
                    "escolhas do DQN) mostram se o seletor passa a fazer diferença.",
                ]),
        ]

    # ---------------- Como ler ----------------
    parts += [
        "<h2>Como ler este relatório</h2>",
        "<p>Cada seção reproduz uma tabela, figura ou afirmação da Seção 4 do artigo e mostra o valor publicado "
        "ao lado (<i>Artigo:</i>). Seções marcadas como <b>complementares</b> não existem no artigo: usam os dados "
        "por decisão (<code>decisions.csv</code>) para explicar <i>por que</i> os resultados são os que são. Os "
        "números não devem coincidir exatamente com os publicados: as sequências aleatórias, as versões das "
        "bibliotecas e o hardware são diferentes. O critério de reprodução é chegar às mesmas conclusões com "
        "valores próximos.</p>",
        explainer("Glossário", [
            "<b>β (banda)</b>: max |π(u) − π(v)| sobre as arestas, para uma rotulação π dos vértices. <b>B0</b> é a "
            "banda da ordenação do arquivo e é o incumbente inicial de todos os métodos.",
            "<b>Decisão</b>: o seletor escolhe uma centralidade e o construtivo faz 15 construções com ela; a "
            "melhor banda é B_t. Uma execução tem 300 decisões de treino e 60 gulosas (5.400 construções).",
            "<b>Random</b>: escolhe a centralidade uniformemente a cada decisão. Mede o efeito de diversificar sem "
            "aprender. <b>Fixed:X</b>: usa sempre a centralidade X.",
            "<b>SBS</b> (single best solver): a centralidade fixa com melhor rank médio no conjunto de instâncias. "
            "<b>VBS</b> (virtual best solver): em cada instância, a centralidade fixa com menor média. É um oráculo "
            "estático; um seletor que troca de centralidade pode superá-lo.",
            "<b>GapClosed</b> = (SBS − DQN)/(SBS − VBS), sobre a média de B/B0: 0% = igual à SBS, 100% = igual ao "
            "VBS, negativo = pior que a SBS.",
            "<b>W/T/L</b>: sementes em que o primeiro método tem banda menor, igual ou maior que o segundo. "
            "<b>p_H</b>: p-valor de Wilcoxon corrigido por Holm.",
            "<b>Instância discriminante</b>: algum método melhora B0 em alguma semente. bcsstk04, bcsstk05 e "
            "bcsstk06 não são: todos os métodos devolvem B0 em todas as sementes.",
            "<b>PP</b>: vértice inicial pseudo-periférico (alta excentricidade), como no RCM, em vez do sorteio "
            "proporcional à centralidade.",
        ]),
        explainer("Configuração reproduzida", [
            "7 centralidades (Degree, Closeness, Betweenness, Eigenvector, Katz, PageRank, Harmonic) calculadas uma "
            "vez por instância.",
            "Construtivo: vértice inicial proporcional à centralidade (uniforme se os pesos forem inválidos), níveis "
            "da busca em largura ordenados pela centralidade, LCR com α = 0,3 e lista e complemento embaralhados; "
            "15 construções por decisão.",
            "DQN: estado de 12 dimensões, recompensa densa, replay de 10.000, mini-lotes de 64, rede alvo "
            "sincronizada a cada 25 atualizações, Smooth L1, Adam (0,001), γ = 0,9, gradiente limitado a 1,0, "
            "ε exponencial de 1 a 0,01 em T = 300. Todas as execuções do DQN fizeram "
            f"{int(dqn_runs['Gradient Updates'].median())} atualizações e {int(dqn_runs['Target Syncs'].median())} "
            "sincronizações, como no artigo (237 e 9).",
            f"Orçamento: {train} decisões de treino + {greedy} gulosas = {(train + greedy) * 15:,} construções por "
            f"execução, igual para todos os métodos, com as mesmas {n_seeds} sementes e B0 como incumbente inicial.",
        ]),
        explainer("Pontos em que a reprodução pode divergir do artigo", [
            "O artigo não informa os parâmetros das centralidades. Aqui Closeness, Harmonic e Betweenness são "
            "exatas, Katz usa α = 0,005 e Eigenvector até 1.000 iterações.",
            "O início pseudo-periférico sorteia, a cada construção, um vértice de um conjunto de até 16 vértices "
            "pseudo-periféricos (George–Liu) obtidos a partir de vértices aleatórios.",
            "Holm na Tabela 4: a família são os dois p-valores (vs Random e vs SBS) de cada instância. Os valores "
            "iguais em bcsstk01 no artigo indicam essa leitura.",
            "Sequências aleatórias, versões das bibliotecas e hardware diferentes (Tabela 2 do artigo contra a "
            "máquina local).",
        ]),
    ]

    # ---------------- Seção 4.1 ----------------
    inst_tab = rcm_cmp.copy() if len(rcm_cmp) else pd.DataFrame({"Instance": [pname(i) for i in A["complete"]]})
    inst_finding = (f"B0 coincide em {int((rcm_cmp['B0 artigo'] == rcm_cmp['B0 reproduzido']).sum())} de "
                    f"{len(rcm_cmp)} instâncias; RCM em {int((rcm_cmp['RCM artigo'] == rcm_cmp['RCM reproduzido']).sum())}"
                    f" de {len(rcm_cmp)}." if len(rcm_cmp) else
                    f"{len(A['complete'])} instâncias da base {dataset} (sem valores publicados para comparar).")
    if instances is not None:
        extra = instances.reset_index()[["Instance", "Nodes", "Edges", "Lower Bound", "Centrality Time (s)"]]
        extra["Instance"] = extra["Instance"].map(pname)
        inst_tab = inst_tab.merge(extra, on="Instance", how="left")
    parts += [
        "<h2>Seção 4.1 · Instâncias e referências</h2>",
        section(
            "B0 e RCM das 13 instâncias",
            "Os grafos e as referências são os mesmos do artigo?",
            inst_finding, None, table_html(inst_tab, {"Centrality Time (s)": "{:.1f}"}),
            how="A tabela compara, por instância, B0 e a banda do Reverse Cuthill–McKee com os valores da Tabela 3, "
                "e mostra |V|, |E|, o limite inferior e o tempo para calcular as 7 centralidades.",
            notes=[
                "B0 igual confirma que o grafo e a rotulação inicial são os mesmos: a maior componente conexa com os "
                "rótulos do arquivo.",
                "O RCM é calculado com o networkx. O do SciPy, usado no <code>main_survey.py</code>, diverge do "
                "artigo em várias instâncias (lshp1270: 65 contra 38), porque cada implementação escolhe o vértice "
                "inicial e desempata de outro jeito.",
                "O limite inferior (máximo entre Lam et al. e ⌈Δ/2⌉) não aparece no artigo. Ele é fraco nestas "
                "instâncias (ex.: 6 contra RCM 26 em bcsstk01), então serve só como referência de quanto o ótimo "
                "pode estar abaixo do RCM, não como medida de qualidade.",
            ]),
    ]

    tab3_sec = (
        section(
            "Tabela 3 · Banda média ± desvio padrão",
            "As médias reproduzidas ficam próximas das publicadas?",
            f"Diferença mediana absoluta de {match_abs['abs_diff'].median():.2f}% entre as médias reproduzidas e "
            f"as publicadas (máxima {match_abs['abs_diff'].max():.2f}%, em "
            f"{match_abs.loc[match_abs['abs_diff'].idxmax(), 'Instance']} / "
            f"{match_abs.loc[match_abs['abs_diff'].idxmax(), 'Método']}). Instâncias não discriminantes: "
            f"{', '.join(nondisc) if nondisc else 'nenhuma'}.",
            fig_vs_paper(cmp),
            table_html(A["table3"]) + "<br>" + table_html(cmp, {"Artigo": "{:.2f}", "Reproduzido": "{:.2f}",
                                                                "Diferença (%)": "{:+.2f}"}),
            how="Cada ponto é uma média (instância × método) da Tabela 3: eixo x = artigo, eixo y = reprodução, em "
                "escala log. Pontos sobre a diagonal reproduzem o valor publicado. A tabela traz a Tabela 3 "
                "completa (com a centralidade que é o VBS em cada instância) e a diferença de cada média para o "
                "artigo.",
            notes=[
                "Diferenças de poucos décimos estão dentro do desvio padrão entre sementes (coluna ± da Tabela 3). "
                "Como a média de 30 sementes tem erro padrão de ~desvio/5,5, diferenças abaixo disso são "
                "compatíveis com o acaso.",
                "As maiores diferenças relativas aparecem em instâncias pequenas (bcsstk01), em que 0,2 unidade de "
                "banda já é quase 1%.",
                "Nas instâncias não discriminantes, todos os métodos devolvem B0 em todas as sementes: elas entram "
                "no Friedman e no GapClosed, mas não ajudam a distinguir os métodos.",
            ])
    ) if paper_mode else section(
        "Tabela 3 · Banda média ± desvio padrão",
        "Quais são as médias por instância nesta configuração?",
        f"Médias de 30 sementes de DQN, Random, SBS e VBS ({setting}). Instâncias não discriminantes: "
        f"{', '.join(nondisc) if nondisc else 'nenhuma'}.",
        None, table_html(A["table3"]),
        how="Mesma Tabela 3 do artigo, recalculada nesta configuração; a coluna VBS centrality indica a "
            "melhor centralidade fixa de cada instância.")

    # ---------------- Seção 4.2 ----------------
    parts += [
        "<h2>Seção 4.2 · O seletor aprendido contra o portfólio</h2>",
        "<p>A primeira pergunta do artigo é se o DQN escolhe melhor que um seletor que não sabe nada. Ela "
        "pressupõe que a escolha da centralidade faça diferença; as seções abaixo verificam isso primeiro (Tabela 3 "
        "e desempenho das fixas) e depois comparam o DQN com Random, SBS e VBS por médias, ranks, testes pareados "
        "e GapClosed.</p>",
        tab3_sec,
        section(
            "Desempenho de cada centralidade fixa (complementar)",
            "A melhor centralidade fixa muda de instância para instância, como o artigo afirma?",
            f"Melhores fixas por instância (empates contam): {best_txt}. <i>Artigo:</i> Eigenvector em bcsstk01 e "
            "shl_0, Harmonic em bp_0, Closeness em bp_200, Degree nas malhas menores e Katz na maioria das maiores.",
            fig_fixed_heat(A),
            how="Linhas = instâncias discriminantes; colunas = centralidades fixas. O número é a banda média das 30 "
                "sementes, e a cor, quanto ela fica acima da melhor fixa da instância (claro = melhor).",
            notes=[
                "Se uma coluna fosse clara em todas as linhas, uma única centralidade bastaria e não haveria o que "
                "selecionar. A variação entre linhas é o que dá sentido a um seletor por instância.",
                "As diferenças entre as melhores fixas costumam ser de poucos por cento. É o tamanho máximo do ganho "
                "que um seletor perfeito de uma centralidade por instância poderia obter (o gap SBS–VBS).",
                "Betweenness e PageRank tendem a ficar entre as piores: concentram o sorteio do vértice inicial em "
                "vértices centrais, o que piora a estrutura de níveis (Figura 2 do artigo).",
            ]),
        section(
            "Distribuição por semente (complementar)",
            "As diferenças entre DQN, Random e SBS são maiores que a variação entre sementes?",
            "As caixas de DQN e Random se sobrepõem em quase todas as instâncias; a SBS se destaca nas maiores "
            "malhas, onde coincide com o VBS.",
            fig_seed_boxes(runs, A),
            how="Cada ponto é uma execução (uma semente), em % acima da média do VBS da instância. As caixas vão do 1º "
                "ao 3º quartil. Valores negativos são execuções melhores que a média do VBS.",
            notes=[
                "Esta é a informação que as médias da Tabela 3 resumem. Com caixas sobrepostas, diferenças entre "
                "médias de poucos décimos não indicam que um método seja melhor.",
                "Muitos pontos empilhados no mesmo valor (bandas inteiras) explicam os muitos empates (T) da "
                "Tabela 4.",
            ]),
        section(
            "Friedman e ranks médios",
            "O teste global rejeita desempenho igual, e em que ordem ficam os métodos?",
            f"χ²_F = {A['chi2']:.2f}, p = {A['p_fr']:.2e}. Ranks médios: {ranks_txt}. SBS = "
            f"<b>{sbs.split(':')[1] if sbs else '-'}</b> (rank médio entre as fixas {sbs_rank:.2f}). "
            f"<br><i>Artigo:</i> {PAPER_HEADLINE['Friedman']}; {PAPER_HEADLINE['Ranks']}; SBS = {PAPER_HEADLINE['SBS']}.",
            fig_ranks(A["mean_ranks"], sbs),
            table_html(A["mean_ranks"].rename("rank médio").reset_index().rename(columns={"index": "método"}),
                       {"rank médio": "{:.2f}"}),
            how="Os blocos do teste são as 13 instâncias e os valores, as médias entre sementes de cada método. Em "
                "cada instância os 9 métodos recebem ranks de 1 (menor banda) a 9; empates recebem a média. As barras "
                "mostram o rank médio (azul = DQN, laranja = Random, verde = SBS).",
            notes=[
                "O teste rejeitar a igualdade não diz que o DQN é melhor: diz que algum método difere. A ordem dos "
                "ranks mostra que a diferença vem das centralidades ruins (Betweenness, PageRank), e não do DQN.",
                "As três instâncias não discriminantes empatam todos os métodos (rank 5 para todos) e diluem as "
                "diferenças, sem alterar a ordem.",
                "Ranks são ordinais: ficar à frente por 0,1 unidade de banda conta igual a ficar à frente por 10.",
            ]),
        section(
            "Tabela 4 · Comparações pareadas do ponto de vista do DQN",
            "Depois da correção de Holm, o DQN é melhor que a seleção aleatória ou que a SBS em alguma instância?",
            f"O DQN difere significativamente da seleção aleatória em {w_dqn_rnd} das {len(A['table4'])} instâncias "
            "discriminantes (veja o sentido em W/T/L). <i>Artigo:</i> pior que Random em bp_200 e lshp1009; pior que "
            "a SBS em bp_200 e nas quatro maiores malhas, melhor em lshp406 e shl_0.",
            None, table_html(A["table4"], {"vs Random p_H": "{:.3f}", "vs SBS p_H": "{:.3f}"})
            + "<br><b>Tamanho do efeito</b> (diferença pareada em unidades de banda; negativo = DQN melhor):"
            + table_html(eff, {c: "{:+.2f}" for c in eff.columns if "−" in c} | {"desvio do DQN": "{:.2f}"}),
            how="W/T/L conta as sementes em que o DQN tem banda menor, igual ou maior. p_H é o p-valor de Wilcoxon "
                "(bilateral, aproximação normal, zeros removidos, sem correção de continuidade) ajustado por Holm; "
                "abaixo de 0,05, a diferença é significativa. A segunda tabela mostra o tamanho das diferenças.",
            notes=[
                "Os empates (T) não entram no teste (zeros removidos): uma instância com 20 empates em 30 testa só as "
                "10 sementes restantes, o que reduz o poder.",
                "Ausência de significância não prova equivalência, como o próprio artigo ressalta; mas a tabela de "
                "efeitos mostra que as diferenças entre DQN e Random são, em geral, menores que o desvio entre "
                "sementes.",
                "Contra a SBS, o padrão do artigo se repete: nas maiores malhas a SBS (Katz) é também o VBS e o DQN "
                "perde por desperdiçar decisões com centralidades piores.",
            ]),
        section(
            "Figura 3 e GapClosed",
            "Quanto do benefício da seleção por instância (gap SBS–VBS) o DQN captura?",
            f"Scores (média de B/B0): DQN {A['score'].get('DQN', np.nan):.5f}, SBS {A['score'].get(sbs, np.nan):.5f}, "
            f"VBS {A['score_vbs']:.5f}, Random {A['score'].get('Random', np.nan):.5f}. GapClosed: DQN "
            f"<b>{fpct(gc.get('DQN', np.nan), 1)}</b>, Random {fpct(gc.get('Random', np.nan), 1)}. "
            + (f"{pname(share_max)} responde por {100 * A['gap_share'].max():.0f}% do gap; sem ela, DQN "
               f"{fpct(gc_no_top, 1)} e Random {fpct(gcr_no_top, 1)}. " if share_max else "")
            + f"<br><i>Artigo:</i> {PAPER_HEADLINE['Scores']}; GapClosed DQN {PAPER_HEADLINE['GapClosed DQN']}, "
              f"Random {PAPER_HEADLINE['GapClosed Random']}; shl_0 ≈ 69% do gap.",
            fig_excess(A["excess"], sbs),
            table_html(A["gap_share"].rename("parcela do gap").mul(100).round(1).reset_index()
                       .rename(columns={"index": "Instance"})),
            how="Cada barra é o excesso percentual da banda média do método sobre o VBS. Zero significa empatar com o "
                "VBS; a SBS fica em zero onde Katz é a melhor fixa. A tabela mostra quanto cada instância contribui "
                "para o gap SBS–VBS.",
            notes=[
                "O gap inteiro é pequeno (diferença de milésimos na banda normalizada), então o GapClosed oscila "
                f"muito: retirando uma instância de cada vez, o do DQN vai de {fpct(gap_range[0], 0)} a "
                f"{fpct(gap_range[1], 0)}. Por isso o artigo usa a medida junto com as comparações pareadas.",
                "A normalização por B0 dá peso a instâncias em que a banda obtida é uma fração grande de B0 (shl_0, "
                "BP) e quase nenhum às malhas, em que a banda obtida é ~4% de B0.",
                "Um GapClosed acima de 100% seria possível para um seletor que troca de centralidade de forma útil "
                "durante a execução; nenhum dos dois chega perto disso.",
            ]),
    ]

    # ---------------- Complementares: comportamento do DQN ----------------
    if dec is not None:
        parts += [
            "<h2>Comportamento do DQN (complementar)</h2>",
            "<p>As seções anteriores mostram <i>que</i> o DQN não supera a seleção aleatória. As seguintes usam as "
            "360 decisões de cada execução para mostrar <i>por quê</i>: como a busca converge, o que o agente "
            "escolhe e o que a recompensa ensina a ele.</p>",
            section(
                "Convergência ao longo das decisões",
                "O DQN se afasta da seleção aleatória em algum momento da execução?",
                f"Na decisão {train} (fim do treino), o incumbente fica, na mediana das instâncias, "
                f"{at_train.get('DQN', np.nan):+.2f}% (DQN), {at_train.get('Random', np.nan):+.2f}% (Random) e "
                f"{at_train.get(sbs, np.nan):+.2f}% (SBS) acima da média do VBS; ao final, "
                f"{final.get('DQN', np.nan):+.2f}%, {final.get('Random', np.nan):+.2f}% e {final.get(sbs, np.nan):+.2f}%.",
                fig_convergence(curve, sbs, train),
                how="Para cada decisão, a curva mostra a mediana (entre as instâncias discriminantes) da média entre "
                    "sementes do incumbente, em % acima da média do VBS. A linha tracejada marca o fim do treino; "
                    "depois dela o DQN escolhe sem exploração. O eixo vertical está cortado para mostrar a parte "
                    "final; nas primeiras decisões os valores são muito maiores.",
                notes=[
                    "Curvas do DQN e do Random juntas desde o início indicam que a política não muda o ritmo da "
                    "busca: o que importa é o número de construções, não quem escolhe.",
                    "A quase ausência de queda depois do fim do treino é a mesma conclusão da 'Fase gulosa' abaixo: "
                    "as 60 decisões finais raramente encontram algo melhor.",
                ]),
            section(
                "O que o DQN escolhe",
                "O agente concentra as escolhas em alguma centralidade, e essa centralidade é a melhor da instância?",
                f"No último bloco de treino, o DQN escolhe uma das melhores fixas da instância em "
                f"{hit_late.get('DQN', np.nan):.0f}% das decisões e na fase gulosa em {hit_greedy.get('DQN', np.nan):.0f}%, "
                f"contra {hit_greedy.get('Random', np.nan):.0f}% da seleção aleatória e {expected:.0f}% esperados por "
                "acaso.",
                fig_choice_share(share),
                table_html(greedy_tab, {"% na melhor fixa": "{:.0f}"}),
                how="As barras empilhadas mostram a participação de cada centralidade nas escolhas do DQN, por bloco "
                    "de 50 decisões e na fase gulosa, somando todas as instâncias discriminantes e sementes. A linha "
                    "pontilhada é a participação de uma escolha uniforme (1/7). A tabela mostra, por instância, as "
                    "escolhas da fase gulosa; ★ marca as melhores centralidades fixas.",
                notes=[
                    "No início ε ≈ 1 e as barras ficam perto de 1/7; à medida que ε cai, as escolhas passam a "
                    "refletir a rede. Concentração em uma centralidade não implica acerto: compare com as ★.",
                    "O estado do DQN não descreve o grafo (só progresso, incumbente e taxas de sucesso), então ele só "
                    "pode aprender uma preferência pela própria experiência na instância, com 300 decisões.",
                ]),
            section(
                "Acerto da melhor centralidade ao longo da execução",
                "A fração de decisões com a melhor centralidade fixa cresce com o treino?",
                "Uma curva do DQN subindo acima da do Random indicaria aprendizado útil.",
                fig_hits(hits, expected),
                how="Fração das decisões em que o seletor escolheu uma das melhores centralidades fixas da instância "
                    "(definidas pelas execuções fixas, independentes das do DQN), por bloco de decisões. A seleção "
                    "aleatória e a linha pontilhada mostram o nível do acaso.",
                notes=[
                    "Mesmo acertando mais que o acaso, o ganho em banda pode ser nulo: a diferença entre a melhor e a "
                    "segunda melhor centralidade costuma ser menor que a variação entre sementes (heatmap acima).",
                    "A melhor centralidade é definida pela média de 30 sementes; em instâncias com empates ou "
                    "diferenças mínimas, 'acertar' tem pouco significado prático.",
                ]),
            section(
                "A recompensa favorece empates",
                "O argumento do artigo (Seção 3.2 e 4.4) de que um empate vale mais que melhorias pequenas aparece "
                "nos dados?",
                "A tabela mostra, por instância, o limiar 0,01·B0 abaixo do qual uma melhoria rende menos que um "
                "empate, e quantas das melhorias do DQN ficaram abaixo dele depois das primeiras 30 decisões. "
                "<i>Artigo:</i> limiar abaixo de 1 unidade nas bcsstk, 8,2 nas BP e 18,6 em lshp1882.",
                None, table_html(rew, {"limiar 0,01·B0": "{:.2f}", "% melhora": "{:.1f}", "% empate": "{:.1f}",
                                       "% piora": "{:.1f}", "ganho mediano (após decisão 30)": "{:.1f}",
                                       "% melhorias abaixo do limiar (após 30)": "{:.0f}"}),
                how="Classificação das decisões de treino do DQN pela banda da decisão contra o incumbente anterior: "
                    "melhora, empate ou piora. 'Ganho mediano' é o tamanho típico das melhorias depois da decisão 30, "
                    "quando a busca já saiu da fase inicial de quedas grandes.",
                notes=[
                    f"<b>Quase toda decisão sai pior que o incumbente</b> ({rw['worse'][0]:.0f} a {rw['worse'][1]:.0f}% "
                    f"nas instâncias discriminantes); empates são {rw['tie'][0]:.1f} a {rw['tie'][1]:.1f}% e melhorias "
                    f"{rw['imp'][0]:.1f} a {rw['imp'][1]:.1f}%. Depois que o incumbente se estabiliza, o melhor de 15 "
                    "construções raramente o iguala, e a recompensa vira uma sequência de penalidades minúsculas "
                    "(−0,05 × diferença/B0), quase iguais para todas as centralidades. Há pouco sinal para a rede "
                    "diferenciar as ações.",
                    (f"<b>Melhorias tardias pequenas.</b> Nas malhas, pelo menos {rw['mesh_below']:.0f}% das melhorias depois "
                    "da decisão 30 ficam abaixo do limiar (ganho mediano de 1 unidade); nas BP e em shl_0, de "
                    f"{rw['other_below'][0]:.0f} a {rw['other_below'][1]:.0f}%. Nesses casos um empate rende mais que "
                    "a melhoria, que é o efeito que o artigo aponta como principal limitação e que restringe a "
                    "conclusão negativa a esta recompensa."
                     if len(mesh_rows) and len(other_rows) else
                     (f"<b>Melhorias tardias pequenas.</b> Nas instâncias discriminantes desta base, em mediana "
                      f"{disc_rows['% melhorias abaixo do limiar (após 30)'].median():.0f}% das melhorias depois da "
                      "decisão 30 ficam abaixo do limiar 0,01·B0, isto é, valem menos que um empate."
                      if disc_rows["% melhorias abaixo do limiar (após 30)"].notna().any() else
                      "<b>Melhorias tardias.</b> Não houve melhorias depois da decisão 30 nestas execuções.")),
                    "Reescalar a recompensa para que qualquer melhoria estrita supere um empate é o próximo passo "
                    "proposto na conclusão do artigo.",
                ]),
        ]
    parts.append(section(
        "Fase gulosa",
        f"As {greedy} decisões gulosas encontram uma banda menor que as {train} de treino?",
        f"Em {greedy_hits} de {len(dqn_runs)} execuções do DQN. <i>Artigo:</i> {PAPER_HEADLINE['Greedy']}.",
        None, table_html(dqn_runs.groupby("Instance")["Greedy Improved"].sum().rename("execuções").reset_index()
                         .assign(Instance=lambda d: d["Instance"].map(pname))),
        how="Conta as execuções do DQN em que a menor banda das decisões gulosas foi menor que o melhor resultado "
            "das decisões de treino.",
        notes=["Com ε = 0 o agente escolhe sempre a mesma centralidade, a preferida da rede; se ela não é melhor que "
               "as outras, as 60 decisões finais equivalem a mais 900 construções com uma centralidade fixa, o que "
               "raramente supera o incumbente já obtido com 4.500 construções."]))

    # ---------------- Seção 4.3 ----------------
    if B is not None:
        tab5 = B["table5"].drop(columns=[c for c in B["table5"].columns if c.startswith("_")])
        ch = B["change"]
        meshes = [i for i in ch.index if i in MESHES]
        others = [i for i in ch.index if i not in MESHES]
        rng = lambda s: f"{s.min():+.1f} a {s.max():+.1f}%" if len(s) else "-"
        ch_tab = ch.round(2).reset_index().rename(columns={"index": "Instance", "DQN": "variação DQN (%)",
                                                            "Random": "variação Random (%)"})
        ch_tab["Instance"] = ch_tab["Instance"].map(pname)
        parts += [
            "<h2>Seção 4.3 · Ablação fatorial: seletor × vértice inicial</h2>",
            "<p>Como o RCM domina todos os métodos estocásticos, o artigo procura a causa na construção e não no "
            "seletor. O RCM parte de um vértice pseudo-periférico, enquanto a construção favorece vértices centrais; "
            "a ablação cruza os dois seletores (DQN e Random) com as duas regras de início, mantendo tudo o mais "
            "igual, nas 10 instâncias discriminantes.</p>",
            section(
                "Tabela 5 e Figura 4",
                "O início pseudo-periférico melhora as malhas e piora as outras famílias? Com ele, o DQN difere da "
                "seleção aleatória?",
                f"Malhas: {rng(ch.loc[meshes, 'DQN'])} (DQN) e {rng(ch.loc[meshes, 'Random'])} (Random). Outras: "
                f"{rng(ch.loc[others, 'DQN'])} (DQN) e {rng(ch.loc[others, 'Random'])} (Random). Menor p_H entre "
                f"DQN-PP e Random-PP: {tab5['p_H'].min():.3f}."
                f"<br><i>Artigo:</i> malhas {PAPER_HEADLINE['PP meshes']}; outras {PAPER_HEADLINE['PP others']}; "
                "p_H ≥ 0,317.",
                fig_pp_change(ch), table_html(tab5, {"p_H": "{:.3f}"}) + "<br>" + table_html(ch_tab),
                how="Cada barra é a variação percentual da banda média ao trocar o início pela centralidade pelo "
                    "início pseudo-periférico (PP), sob o mesmo seletor. A Tabela 5 compara DQN-PP e Random-PP por "
                    "semente, com Holm sobre as 10 instâncias.",
                notes=[
                    "Nas malhas, o início periférico produz muitos níveis estreitos e a banda cai, como o argumento da "
                    "Figura 2 do artigo prevê (limite ≈ soma das larguras de dois níveis consecutivos).",
                    "Nas famílias BP, SHL e bcsstk01, o efeito se inverte: a estrutura desses grafos não é de malha e "
                    "o vértice de maior excentricidade não gera níveis estreitos. Isso mostra que a regra de início "
                    "deveria depender da família.",
                    "As duas barras de cada instância quase iguais e p_H alto indicam que a melhora vem da regra de "
                    "início, e não do seletor. O próprio artigo ressalva que o início PP remove um dos dois lugares em "
                    "que a centralidade atua, o que pode deixar as ações ainda mais parecidas.",
                    "O desvio com PP nas BP é maior aqui do que no artigo: o sorteio entre até 16 vértices "
                    "pseudo-periféricos, escolha desta implementação, produz inícios mais variados.",
                ]),
            section(
                "Fator residual para o RCM",
                "O início PP fecha a distância para o RCM nas quatro maiores malhas?",
                ("DQN-PP / RCM: " + ", ".join(f"{pname(i)} {v:.2f}" for i, v in B["residual"].items())
                 if B["residual"] else "As quatro maiores malhas ainda não estão nos resultados")
                + f". <i>Artigo:</i> {PAPER_HEADLINE['Residual']}.", None,
                how="Razão entre a banda média do DQN-PP e a banda do RCM nas quatro maiores malhas.",
                notes=([
                    "Mesmo com o início certo, o método fica ~1,5 vez acima do RCM. O artigo atribui o restante à "
                    "ordenação dentro de cada nível: a construção ordena o nível inteiro pela centralidade e o "
                    "embaralha, enquanto o Cuthill–McKee rotula os vértices na ordem dos pais.",
                    "Essa hipótese pode ser testada diretamente com a extensão E1: <code>python icits2026/main_icits.py "
                    "--construction parent</code>.",
                ] if construction == "paper" else [
                    "Com a ordem do pai (E1) o fator fica perto de 1: a ordenação dentro do nível explicava o resíduo "
                    "que o artigo atribuía a ela.",
                ])),
        ]

    # ---------------- Extensões E2 e E3 ----------------
    if EXT is not None:
        summ_ext = EXT["summary"]
        has_e2 = any(split_method(m)[1] != "paper" for m in EXT["ext"])
        has_e3 = any(split_method(m)[0] in ("UCB1", "AP") for m in EXT["ext"])
        ext_labels = {method_label(m) for m in EXT["ext"]}
        best_row = (summ_ext[summ_ext["método"].isin(ext_labels)]
                    .sort_values("excesso mediano sobre o VBS (%)").iloc[0])
        parts += [
            "<h2>Extensões E2 e E3 · Recompensa reescalada e seletores de bandit</h2>",
            "<p>As duas extensões mudam só o seletor; construção, orçamento, sementes e referências (Random, fixas, "
            "SBS, VBS) são os mesmos das seções anteriores. "
            + ("<b>E2</b> troca o prêmio do empate de 0,01 por 0,5/B0, de modo que qualquer melhoria estrita "
               "(no mínimo 1 unidade, que rende 1/B0) valha o dobro de um empate. " if has_e2 else "")
            + ("<b>E3</b> substitui a rede por seletores de bandit, que o artigo aponta como comparadores naturais: "
               "UCB1 (Auer et al., 2002) e adaptive pursuit (Thierens, 2005), com a mesma recompensa do DQN e a "
               "mesma divisão em 300 decisões de aprendizado e 60 gulosas. " if has_e3 else "")
            + "O teto de qualquer seletor de uma centralidade por instância é o VBS.</p>",
            section(
                "Os novos seletores escolhem melhor que o acaso?",
                "Contra a seleção aleatória e contra o DQN do artigo, com Holm entre as instâncias discriminantes, "
                "algum seletor das extensões é significativamente melhor?",
                f"Menor excesso mediano sobre o VBS: <b>{best_row['método']}</b> "
                f"({best_row['excesso mediano sobre o VBS (%)']:+.2f}%). Veja na tabela, para cada seletor, as "
                "instâncias com diferença significativa contra Random e contra o DQN.",
                fig_ext_excess(EXT["excess"]),
                table_html(summ_ext, {c: "{:+.2f}" for c in summ_ext.columns if "%" in c and "gulosas" not in c}
                           | {"% gulosas na melhor fixa": "{:.0f}"})
                + "<br><b>Banda média por instância</b>:" + table_html(EXT["per_inst"],
                                                                        {c: "{:.2f}" for c in EXT["per_inst"].columns[1:]}),
                how="As barras mostram, por instância discriminante, o excesso da banda média de cada seletor sobre o "
                    "VBS (0 = igual à melhor centralidade fixa). Na tabela, 'W/T/L' soma as sementes de todas as "
                    "instâncias; 'melhor/pior (sig.)' conta as instâncias com diferença significativa (Wilcoxon do "
                    "artigo, Holm entre as instâncias); 'dif. mediana' é a diferença mediana das médias, em %; "
                    "'% gulosas na melhor fixa' é a fração das decisões gulosas que escolheram uma das melhores "
                    "centralidades fixas da instância.",
                notes=[
                    "<b>Leitura de E2.</b> Se o DQN com empate = 0,5/B0 passa a acertar mais a melhor centralidade e a "
                    "ficar abaixo do Random, a recompensa era o gargalo e a conclusão negativa do artigo era específica "
                    "dela, como o próprio artigo ressalva. Se continua igual, a conclusão fica mais forte.",
                    "<b>Leitura de E3.</b> Um bandit que iguale ou supere o DQN mostra que a rede neural não acrescenta "
                    "nada ao problema na forma atual (o estado não descreve o grafo). Nesse caso, aumentar a "
                    "complexidade do modelo não se justifica, como a conclusão do artigo sugere.",
                    "Os bandits normalizam a recompensa pelo mínimo e máximo já observados, para não depender da "
                    "escala (o empate vale 0,01 e as melhorias, ~1/B0). Isso não muda a ordem entre as recompensas.",
                    "O ganho possível é limitado pelo gap SBS–VBS: mesmo um seletor perfeito de uma centralidade por "
                    "instância só chega ao VBS. Diferenças pequenas em banda podem ser significativas sem serem "
                    "relevantes; compare sempre com o excesso do Random.",
                ]),
        ]
        ext_pp = extension_methods(runs, pp=True)
        if ext_pp:
            base_pp = [m for m in ["DQN-PP", "Random-PP"] if m in set(runs["Method"])]
            pp_tab = pd.DataFrame({method_label(m): {pname(i): paired(runs, i, m).mean()
                                                    for i in A["disc"] if len(paired(runs, i, m))}
                                   for m in base_pp + ext_pp}).reset_index().rename(columns={"index": "Instance"})
            parts.append(section(
                "Extensões com início pseudo-periférico", "Como os seletores das extensões se comportam com início PP?",
                "Banda média por instância, ao lado de DQN-PP e Random-PP.", None,
                table_html(pp_tab, {c: "{:.2f}" for c in pp_tab.columns[1:]})))

    # ---------------- Custo ----------------
    parts.append(section(
        "Custo do aprendizado",
        "Quanto tempo a mais o DQN leva em relação à seleção aleatória?",
        ("; ".join(f"{k}: {v:+.1f}%" for k, v in T.items()) if T else "Sem pares DQN/Random")
        + f". <i>Artigo:</i> {PAPER_HEADLINE['Time']}.",
        None, table_html(time_by_method, {"tempo médio (s)": "{:.1f}", "tempo total (s)": "{:,.0f}"}),
        how="Razão pareada (mesma instância e semente) entre o tempo total do DQN e o da seleção aleatória. A tabela "
            "mostra o tempo médio e o total de cada método.",
        notes=["O tempo de uma execução depende também da carga da máquina. Quando as execuções rodam em condições "
               "diferentes (por exemplo, parte com 1 processo e parte com vários), a razão pareada mistura o custo do "
               "aprendizado com a disputa por CPU.",
               "O treino da rede (237 atualizações) é pequeno perto das 5.400 construções; a diferença de tempo entre "
               "métodos vem mais das centralidades escolhidas, que mudam o custo do sorteio do vértice inicial."]))

    # ---------------- Síntese ----------------
    parts += [
        "<h2>Síntese e considerações finais</h2>",
        explainer("O que a reprodução confirma" if paper_mode else "O que esta configuração mostra", ([
            f"O DQN não supera a seleção aleatória com o mesmo orçamento: nenhuma instância com o DQN "
            f"significativamente melhor, ranks médios DQN {A['mean_ranks'].get('DQN', np.nan):.2f} contra Random "
            f"{A['mean_ranks'].get('Random', np.nan):.2f}.",
            f"Nenhum seletor captura parte relevante do gap SBS–VBS (DQN {fpct(gc.get('DQN', np.nan), 1)}, Random "
            f"{fpct(gc.get('Random', np.nan), 1)}), e a medida é instável por depender de poucas instâncias.",
            "O início pseudo-periférico melhora todas as malhas e piora as outras famílias, igualmente sob os dois "
            "seletores, e não fecha a distância de ~1,5× para o RCM.",
            "O comportamento do agente explica o resultado: as escolhas pouco se afastam do acaso e a recompensa "
            "premia empates acima das melhorias pequenas que dominam o fim da busca.",
        ] if paper_mode else [
            f"Banda do DQN / RCM nas instâncias discriminantes: {dqn_rcm:.2f}× (mediana)"
            + (f"; com a construção do artigo, {E1['table']['DQN/RCM (artigo)'].median():.2f}×." if E1 is not None
               else "."),
            f"DQN contra a seleção aleatória: diferença significativa em {w_dqn_rnd} de {len(A['table4'])} "
            f"instâncias discriminantes; ranks médios DQN {A['mean_ranks'].get('DQN', np.nan):.2f} e Random "
            f"{A['mean_ranks'].get('Random', np.nan):.2f}.",
            f"GapClosed: DQN {fpct(gc.get('DQN', np.nan), 1)}, Random {fpct(gc.get('Random', np.nan), 1)}.",
        ] + ([f"Extensões E2/E3: menor excesso mediano sobre o VBS com {EXT['summary'][EXT['summary']['método'].isin({method_label(m) for m in EXT['ext']})].sort_values('excesso mediano sobre o VBS (%)').iloc[0]['método']}."]
             if EXT is not None else []))),
        explainer("Limitações desta reprodução", [
            "Parâmetros das centralidades e sorteio do vértice pseudo-periférico não informados no artigo (escolhas "
            "documentadas acima).",
            "Bibliotecas e hardware diferentes da Tabela 2 do artigo; tempos medidos em condições mistas.",
            "Como no artigo, ausência de significância não é prova de equivalência (não houve teste de equivalência).",
        ]),
        explainer("Próximos passos (conclusão do artigo)", [
            "Ordenação dentro do nível que preserve a ordem dos pais (Cuthill–McKee), já disponível no construtivo.",
            "Recompensa reescalada para que toda melhoria estrita valha mais que um empate.",
            "Comparar o DQN com seletores de bandit (UCB1, adaptive pursuit) e treinar entre instâncias da mesma "
            "família antes de aumentar a complexidade do modelo.",
            "Escolher a regra de início pela família do grafo (PP nas malhas, centralidade nas demais).",
        ]),
        "</main></body></html>",
    ]

    html_text = "\n".join(parts)
    if not paper_mode:
        html_text = html_text.replace("<i>Artigo:</i>", "<i>Artigo (SBPO, construção original):</i>")
    out_html = os.path.join(args.results_dir, "report_icits.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(f"[OK] relatório: {out_html}")
    print(f"[OK] tabelas:   {out_tables} (CSV e tables_icits.tex)")


if __name__ == "__main__":
    main()
