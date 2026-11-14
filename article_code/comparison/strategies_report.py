# Relatorio comparando as estrategias do main_survey.py:
#   condicao (o que o agente escolhe: centralidade, alpha da LCR ou estrategia de inicio)
#   x busca local do VNS-band (com / sem)
#
# Entradas: result_summary_<dataset>_<tag>.csv e result_output_<dataset>_<tag>.csv
#           (gerados por: python main_survey.py --condition all --local-search both --dataset <dataset>)
# Saida:    comparison/report_estrategias_<dataset>.html e comparison/tabelas/estrategias_<dataset>_*.csv
#
# Uso: python comparison/strategies_report.py --dataset newdata

import os
import glob
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import friedmanchisquare

from build_report import (HERE, BASE_DIR, OUT_TABLES, FOLDER_CLASS, SLOTS, INK, INK2, MUTED, AXIS, SURFACE,
                          GOOD, BAD, NEUTRAL, CSS, style, table_html, tile, section, explainer)

CONDITION_LABEL = {"centrality": "Centralidade", "alpha": "α da LCR", "start": "Estratégia de início"}
CONDITION_COLOR = {"centrality": SLOTS[0], "alpha": SLOTS[1], "start": SLOTS[2]}
CONDITION_ORDER = ["centrality", "alpha", "start"]
# ordem fixa das ações (a cor segue a ação, não a posição entre as ações presentes)
ACTION_ORDER = {
    "centrality": ["Degree", "Closeness", "Betweenness", "Eigenvector", "PageRank", "Katz Centrality", "Harmonic Centrality"],
    "alpha": ["0.0", "0.1", "0.3", "0.5", "1.0"],
    "start": ["centrality", "inverse_centrality", "uniform", "peripheral"],
}


def strategy_label(condition, local_search):
    return f"{CONDITION_LABEL[condition]} · {'com' if local_search else 'sem'} busca local"


def bar_style(condition, local_search):
    """Cor = condição; preenchimento = busca local (sólido com, hachurado sem)."""
    color = CONDITION_COLOR[condition]
    if local_search:
        return dict(color=color)
    return dict(color="rgba(0,0,0,0)", pattern=dict(shape="/", fgcolor=color, size=6, solidity=0.45),
                line=dict(color=color, width=1.5))


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load(dataset, base_dir=BASE_DIR):
    summaries, outputs = [], []
    # so as condicoes de estrategia (o oraculo tem relatorio proprio: oracle_report.py)
    is_strategy = lambda f: "_oracle" not in os.path.basename(f)
    for f in filter(is_strategy, sorted(glob.glob(os.path.join(base_dir, f"result_summary_{dataset}_*.csv")))):
        summaries.append(pd.read_csv(f))
    for f in filter(is_strategy, sorted(glob.glob(os.path.join(base_dir, f"result_output_{dataset}_*.csv")))):
        outputs.append(pd.read_csv(f, usecols=lambda c: c in {
            "bandwidth", "construction_bandwidth", "Episode", "Run", "condition", "action", "local_search",
            "Instance", "local_time"}))
    if not summaries:
        raise SystemExit(f"Nenhum result_summary_{dataset}_*.csv encontrado em {base_dir}")
    summ = pd.concat(summaries, ignore_index=True)
    out = pd.concat(outputs, ignore_index=True)
    summ = summ[summ["Condition"].isin(CONDITION_ORDER)].reset_index(drop=True)
    out = out[out["condition"].isin(CONDITION_ORDER)].reset_index(drop=True)
    summ["Local Search"] = summ["Local Search"].astype(bool)
    out["local_search"] = out["local_search"].astype(bool)
    summ["estrategia"] = [strategy_label(c, ls) for c, ls in zip(summ["Condition"], summ["Local Search"])]
    out["estrategia"] = [strategy_label(c, ls) for c, ls in zip(out["condition"], out["local_search"])]

    # classe de cada instância (pasta do .mtx)
    classes = {}
    data_dir = os.path.join(base_dir, "data", dataset)
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if not d.startswith("plots")]
        for name in files:
            if name.endswith(".mtx"):
                classes.setdefault(name[:-4], FOLDER_CLASS.get(os.path.basename(root), os.path.basename(root)))
    summ["classe"] = summ["Instance"].map(classes).fillna(dataset)
    if dataset == "survey":
        summ["classe"] = "survey"

    # melhor resultado por (instância, estratégia) considerando todas as execuções
    best = (summ.groupby(["Instance", "classe", "Condition", "Local Search", "estrategia"], as_index=False)
                .agg(best=("Best Bandwidth", "min"), construction_best=("Best Construction Bandwidth", "min"),
                     mean=("Mean Bandwidth", "mean"), std=("Std Bandwidth", "mean"),
                     time=("Global Time (s)", "mean"), rcm=("RCM Bandwidth", "first"),
                     lb=("Lower Bound", "first"), original=("Initial Bandwidth", "first"),
                     nodes=("Nodes", "first"), best_action=("Best Action", "first")))
    best["vs_rcm_pct"] = 100 * (best["best"] - best["rcm"]) / best["rcm"]
    best["ls_gain_pct"] = 100 * (best["construction_best"] - best["best"]) / best["construction_best"]
    inst_min = best.groupby("Instance")["best"].transform("min")
    best["vs_best_pct"] = 100 * (best["best"] - inst_min) / inst_min
    return summ, out, best


def ordered_strategies(best):
    present = set(best["estrategia"])
    return [strategy_label(c, ls) for c in CONDITION_ORDER for ls in (False, True) if strategy_label(c, ls) in present]


def strategy_parts(label):
    for c in CONDITION_ORDER:
        for ls in (False, True):
            if strategy_label(c, ls) == label:
                return c, ls
    raise KeyError(label)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_vs_rcm_by_class(best, strategies):
    classes = best.groupby("classe")["vs_rcm_pct"].median().sort_values().index.tolist()
    fig = go.Figure()
    for s in strategies:
        c, ls = strategy_parts(s)
        g = best[best["estrategia"] == s].groupby("classe")["vs_rcm_pct"].median().reindex(classes)
        n = best[best["estrategia"] == s].groupby("classe")["Instance"].count().reindex(classes)
        fig.add_bar(x=classes, y=g.values, name=s, marker=bar_style(c, ls), customdata=n.values,
                    hovertemplate="<b>%{x}</b><br>" + s + "<br>mediana: %{y:+.1f}% vs RCM<br>%{customdata} instâncias"
                                  "<extra></extra>")
    fig.add_hline(y=0, line_color=INK2, line_width=1)
    fig.update_yaxes(title_text="mediana de (β − β_RCM) / β_RCM  [%]   (negativo = melhor que RCM)", ticksuffix="%")
    return style(fig, "E1 · Qualidade por classe: diferença para o RCM (mediana das instâncias)", height=500,
                 barmode="group")


def fig_ranks(best, strategies):
    piv = best.pivot_table(index="Instance", columns="estrategia", values="best").dropna()[strategies]
    ranks = piv.rank(axis=1, method="average")
    mean_rank = ranks.mean().reindex(strategies)
    wins = piv.eq(piv.min(axis=1), axis=0).sum().reindex(strategies)
    stat, p = friedmanchisquare(*[piv[s] for s in strategies]) if len(strategies) >= 3 else (np.nan, np.nan)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.32,
                        subplot_titles=("Rank médio (1 = melhor)", "Instâncias em que atinge o melhor β (empates contam)"))
    for s in strategies:
        c, ls = strategy_parts(s)
        fig.add_bar(y=[s], x=[mean_rank[s]], orientation="h", marker=bar_style(c, ls), showlegend=False,
                    text=[f"{mean_rank[s]:.2f}"], textposition="outside", textfont=dict(color=INK2),
                    hovertemplate=s + "<br>rank médio: %{x:.2f}<extra></extra>", row=1, col=1)
        fig.add_bar(y=[s], x=[wins[s]], orientation="h", marker=bar_style(c, ls), showlegend=False,
                    text=[f"{wins[s]}/{len(piv)}"], textposition="outside", textfont=dict(color=INK2),
                    hovertemplate=s + "<br>melhor em %{x} instâncias<extra></extra>", row=1, col=2)
    fig.update_yaxes(categoryorder="array", categoryarray=strategies[::-1])
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(range=[0, len(strategies) * 1.18], row=1, col=1)   # folga para os rótulos
    fig.update_xaxes(range=[0, len(piv) * 1.2], row=1, col=2)
    title = f"E2 · Ranking das estratégias em {len(piv)} instâncias (Friedman p = {p:.1e})"
    return style(fig, title, height=420), piv, mean_rank, wins, p


def fig_heatmap(best, strategies):
    piv = best.pivot_table(index="Instance", columns="estrategia", values="vs_best_pct")[strategies]
    cls = best.drop_duplicates("Instance").set_index("Instance")["classe"]
    piv = piv.assign(_c=cls.reindex(piv.index)).sort_values(["_c"]).drop(columns="_c")
    best_val = best.pivot_table(index="Instance", columns="estrategia", values="best")[strategies].reindex(piv.index)
    zmax = max(5, np.nanpercentile(piv.values, 95))
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=[s.replace(" · ", "<br>") for s in strategies],
        y=[f"{i} · {cls[i]}" for i in piv.index], customdata=best_val.values,
        colorscale=[[0, "#f0efec"], [0.35, "#86b6ef"], [1, "#184f95"]], zmin=0, zmax=zmax,
        colorbar=dict(title="% acima do<br>melhor β", ticksuffix="%"),
        hovertemplate="<b>%{y}</b><br>%{x}<br>β = %{customdata:.0f} (%{z:.1f}% acima do melhor)<extra></extra>"))
    fig.update_xaxes(side="top")
    return style(fig, "E3 · Por instância: quão longe cada estratégia fica do melhor β encontrado (0% = melhor)",
                 height=max(500, 15 * len(piv) + 160), margin=dict(l=210, r=30, t=150, b=30))


def fig_actions(out, best):
    """Para cada condição: escolha do agente ao longo dos episódios e qualidade por valor da ação."""
    present = [c for c in CONDITION_ORDER if c in set(out["condition"])]
    fig = make_subplots(rows=len(present), cols=2, horizontal_spacing=0.08, vertical_spacing=0.12,
                        subplot_titles=[t for c in present for t in (
                            f"{CONDITION_LABEL[c]}: % das escolhas do agente por bloco de episódios",
                            f"{CONDITION_LABEL[c]}: β da construção / melhor β da instância")])
    inst_best = out.groupby("Instance")["bandwidth"].transform("min")
    out = out.assign(ratio=out["bandwidth"] / inst_best)
    for r, c in enumerate(present, start=1):
        sub = out[out["condition"] == c].assign(action=lambda d: d["action"].astype(str))
        present_actions = set(sub["action"].astype(str).unique())
        order = ACTION_ORDER[c] + sorted(present_actions - set(ACTION_ORDER[c]))
        actions = [a for a in order if a in present_actions]
        # nº de ações possíveis (para a referência de escolha uniforme); centralidades dependem da configuração
        n_actions = len(actions) if c == "centrality" else len(order)
        ep =sub.groupby(["Instance", "local_search", "Run", "Episode"])["action"].first().reset_index()
        ep["bloco"] = (ep["Episode"] - 1) // 5
        share = ep.groupby(["bloco", "action"]).size().unstack(fill_value=0)
        share = share.div(share.sum(axis=1), axis=0) * 100
        labels = [f"{5 * b + 1}–{5 * b + 5}" for b in share.index]
        for a in actions:
            color = SLOTS[order.index(a) % len(SLOTS)]
            fig.add_scatter(x=labels, y=share.get(a, pd.Series(0, index=share.index)).values, mode="lines+markers",
                            name=f"{CONDITION_LABEL[c]} = {a}", legendgroup=c, legendgrouptitle_text=CONDITION_LABEL[c],
                            line=dict(color=color, width=2), marker=dict(size=7),
                            hovertemplate=f"{c}={a}<br>episódios %{{x}}: %{{y:.0f}}%<extra></extra>", row=r, col=1)
            v = sub.loc[sub["action"] == a, "ratio"]
            fig.add_box(y=v, name=str(a), marker_color=color, boxpoints=False, line_width=1.5, showlegend=False,
                        legendgroup=c, hovertemplate=f"{c}={a}<br>β/β*: %{{y:.2f}}<extra></extra>", row=r, col=2)
        fig.add_scatter(x=labels, y=[100 / max(n_actions, 1)] * len(labels), mode="lines", showlegend=(r == 1),
                        name="escolha uniforme", line=dict(color=INK, dash="dot", width=1.2), hoverinfo="skip",
                        row=r, col=1)
        fig.update_yaxes(ticksuffix="%", row=r, col=1)
        fig.update_yaxes(range=[0.95, 2.0], row=r, col=2)
    style(fig, "E4 · O que o agente escolhe em cada condição e quanto cada valor da ação rende",
          height=330 * len(present) + 120, margin=dict(l=60, r=250, t=90, b=50))
    # legenda vertical à direita (aplicada depois do style, que define a legenda horizontal padrão)
    fig.update_layout(legend=dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top", groupclick="toggleitem"))
    return fig


def fig_local_search(best):
    d = best.copy()
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
        "Ganho da busca local sobre a melhor construção (por instância)", "Custo: tempo total por execução"))
    for c in [x for x in CONDITION_ORDER if x in set(d["Condition"])]:
        g = d[(d["Condition"] == c) & (d["Local Search"])]
        fig.add_box(y=g["ls_gain_pct"], name=CONDITION_LABEL[c], marker_color=CONDITION_COLOR[c], boxpoints="all",
                    jitter=0.4, pointpos=0, marker=dict(size=5), line_width=1.5, showlegend=False,
                    customdata=g["Instance"], hovertemplate="<b>%{customdata}</b><br>ganho: %{y:.1f}%<extra></extra>",
                    row=1, col=1)
        for ls in (False, True):
            g2 = d[(d["Condition"] == c) & (d["Local Search"] == ls)]
            if g2.empty:
                continue
            fig.add_box(y=g2["time"], x=[CONDITION_LABEL[c]] * len(g2), name=f"{'com' if ls else 'sem'} busca local",
                        marker_color=CONDITION_COLOR[c] if ls else MUTED, boxpoints=False, line_width=1.5,
                        legendgroup=str(ls), showlegend=(c == "centrality"), offsetgroup=str(ls),
                        hovertemplate="%{y:.1f} s<extra></extra>", row=1, col=2)
    fig.update_yaxes(title_text="redução de β pela busca local [%]", ticksuffix="%", row=1, col=1)
    fig.update_yaxes(type="log", title_text="tempo por execução [s] (log)", row=1, col=2)
    fig.update_layout(boxmode="group")
    return style(fig, "E5 · Busca local do VNS-band: quanto melhora e quanto custa", height=480)


def fig_convergence(out, strategies):
    o = out.copy()
    o["idx"] = o.groupby(["Instance", "estrategia", "Run"]).cumcount() + 1
    o["bsf"] = o.groupby(["Instance", "estrategia", "Run"])["bandwidth"].cummin()
    final = o.groupby(["Instance", "estrategia", "Run"])["bsf"].transform("min")
    inst_best = o.groupby("Instance")["bandwidth"].transform("min")
    o["ratio"] = o["bsf"] / inst_best
    fig = go.Figure()
    for s in strategies:
        c, ls = strategy_parts(s)
        q = o[o["estrategia"] == s].groupby("idx")["ratio"].median()
        fig.add_scatter(x=q.index, y=q.values, mode="lines", name=s,
                        line=dict(color=CONDITION_COLOR[c], width=2, dash="solid" if ls else "dash"),
                        hovertemplate=s + "<br>construção %{x}: %{y:.3f}× o melhor β<extra></extra>")
    fig.update_xaxes(title_text="nº de construções executadas")
    fig.update_yaxes(title_text="melhor β até aqui / melhor β da instância (mediana)", range=[0.98, 1.4])
    return style(fig, "E6 · Convergência por estratégia (linha cheia = com busca local; tracejada = sem)", height=460)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["newdata", "survey"], default="newdata")
    parser.add_argument("--results-dir", default=BASE_DIR, help="pasta com os result_*.csv (padrão: article_code)")
    args = parser.parse_args()

    summ, out, best = load(args.dataset, args.results_dir)
    strategies = ordered_strategies(best)
    os.makedirs(OUT_TABLES, exist_ok=True)
    best.to_csv(os.path.join(OUT_TABLES, f"estrategias_{args.dataset}_melhor_por_instancia.csv"), index=False)

    f_rank, piv, mean_rank, wins, p_friedman = fig_ranks(best, strategies)
    agg = best.groupby("estrategia").agg(
        instancias=("Instance", "nunique"),
        mediana_vs_rcm=("vs_rcm_pct", "median"),
        vence_rcm=("vs_rcm_pct", lambda v: int((v < 0).sum())),
        empata_rcm=("vs_rcm_pct", lambda v: int((v == 0).sum())),
        perde_rcm=("vs_rcm_pct", lambda v: int((v > 0).sum())),
        ganho_busca_local=("ls_gain_pct", "median"),
        tempo_medio_s=("time", "mean"),
    ).reindex(strategies)
    agg["rank_medio"] = mean_rank
    agg["melhor_em"] = wins
    agg = agg.reset_index()
    agg.to_csv(os.path.join(OUT_TABLES, f"estrategias_{args.dataset}_resumo.csv"), index=False)

    top = agg.sort_values("rank_medio").iloc[0]
    # ganho e custo da busca local: so faz sentido nas estrategias com busca local
    ls_rows = agg[[strategy_parts(e)[1] for e in agg["estrategia"]]]
    no_ls_rows = agg[[not strategy_parts(e)[1] for e in agg["estrategia"]]]
    ls_med = ls_rows["ganho_busca_local"].median() if len(ls_rows) else np.nan
    t_ratio = (ls_rows["tempo_medio_s"].mean() / no_ls_rows["tempo_medio_s"].mean()
               if len(ls_rows) and len(no_ls_rows) else np.nan)
    runs = summ.groupby(["Instance", "estrategia"]).size()
    n_multi = int((runs > 1).sum())
    # alpha = 0 e alpha = 1 sao equivalentes no codigo atual (get_LCR embaralha a LCR e o complemento)
    inst_best_all = out.groupby("Instance")["bandwidth"].transform("min")
    ratio_all = out["bandwidth"] / inst_best_all
    act = out["action"].astype(str)
    is_alpha = out["condition"] == "alpha"
    med_a0 = ratio_all[is_alpha & (act == "0.0")].median()
    med_a1 = ratio_all[is_alpha & (act == "1.0")].median()
    n_a0, n_a1 = int((is_alpha & (act == "0.0")).sum()), int((is_alpha & (act == "1.0")).sum())
    # comparacao pareada (mesma instancia e mesma busca local), antes da busca local
    pair = (out[is_alpha & act.isin(["0.0", "1.0"])].assign(a=act)
               .groupby(["Instance", "local_search", "a"])["construction_bandwidth"].median()
               .unstack("a").dropna())
    n_pairs = len(pair)
    pair_ratio = (pair["1.0"] / pair["0.0"]).median() if n_pairs else np.nan
    centralities_used = sorted(out.loc[out["condition"] == "centrality", "action"].astype(str).unique())
    tiles = "".join([
        tile(f"{best['Instance'].nunique()}", f"instâncias ({args.dataset})"),
        tile(f"{len(strategies)}", "estratégias (condição × busca local)"),
        tile(top["estrategia"].replace(" · ", "<br>"), f"melhor rank médio ({top['rank_medio']:.2f})"),
        tile(f"{top['vence_rcm']}/{top['instancias']}", "instâncias em que a melhor estratégia vence o RCM"),
        tile(f"{ls_med:.1f}%", "ganho mediano da busca local (estratégias com busca local)"),
    ])

    best_action_tab = (best.groupby(["estrategia", "best_action"]).size().rename("instâncias").reset_index()
                           .sort_values(["estrategia", "instâncias"], ascending=[True, False]))

    n_inst = best["Instance"].nunique()
    parts = [f"<!doctype html><html lang='pt-br'><head><meta charset='utf-8'><title>Estratégias Q-MCH — {args.dataset}</title>"
             f"<style>{CSS}</style><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body><main>",
             f"<h1>Comparação das estratégias do Q-MCH ({args.dataset})</h1>",
             "<p class='src'>Cada <b>condição</b> define o que o agente DQN escolhe a cada episódio; os demais parâmetros "
             "ficam fixos (centralidade = Degree, α = 0,3, início = proporcional à centralidade). "
             "<b>Cor</b> = condição (azul: centralidade, laranja: α da LCR, verde: estratégia de início); "
             "<b>preenchimento</b> = busca local do VNS-band (sólido = com, hachurado/tracejado = sem). "
             "Gerado a partir de <code>result_summary_*</code>/<code>result_output_*</code> do main_survey.py.</p>",
             f'<div class="tiles">{tiles}</div>',
             "<h2>Como ler este relatório</h2>",
             f"<p>Este relatório compara {len(strategies)} estratégias do Q-MCH: três <b>condições</b> (o que o agente "
             "DQN escolhe a cada episódio) combinadas com a <b>busca local</b> do VNS-band (com ou sem). Todas rodam "
             f"nas mesmas {n_inst} instâncias, no mesmo grafo (maior componente conexa) e com o mesmo orçamento "
             "(30 episódios × 15 construções = 450 construções), e são comparadas com o RCM recalculado em cada "
             "instância. Cada seção traz a pergunta, a resposta curta, como ler o gráfico e as considerações: "
             "hipóteses, limitações e o que o resultado permite ou não concluir.</p>",
             explainer("As três condições", [
                 "<b>Centralidade</b> (Q-MCH original): o agente escolhe entre as centralidades "
                 f"({', '.join(centralities_used)}), com α = 0,3 e vértice inicial sorteado com probabilidade "
                 "proporcional à centralidade.",
                 "<b>α da LCR</b>: o agente escolhe α ∈ {0; 0,1; 0,3; 0,5; 1}, com centralidade fixa (Degree). α é a "
                 "fração dos filhos de cada vértice que forma a lista restrita de candidatos (os mais centrais), "
                 "rotulados antes dos demais.",
                 "<b>Estratégia de início</b>: o agente escolhe como sortear o vértice inicial. As opções são "
                 "<code>centrality</code> (proporcional à centralidade), <code>inverse_centrality</code> "
                 "(proporcional a 1/centralidade, que favorece vértices periféricos), <code>uniform</code> (sorteio "
                 "uniforme, como no VNS-band) e <code>peripheral</code> (vértice pseudo-periférico de George–Liu, "
                 "como no RCM e no GPS). Centralidade e α ficam fixos em Degree e 0,3.",
                 "O número de ações difere entre as condições (centralidades, 5 valores de α, 4 inícios). Em 30 "
                 "episódios, quanto mais ações, menos vezes o agente experimenta cada uma, o que dificulta o "
                 "aprendizado.",
             ]),
             explainer("Busca local do VNS-band", [
                 "É aplicada depois de cada construção. Para cada vértice crítico (extremo de uma aresta de tamanho "
                 "β), tenta trocar o rótulo dele com o de um vértice cujo rótulo esteja mais perto do ponto médio "
                 "dos rótulos vizinhos. A troca é aceita se nenhuma aresta passar de β e o número de arestas "
                 "críticas diminuir (Mladenović et al., 2010, baseada em Lim et al., 2006).",
                 "É uma intensificação local: nunca piora a construção, mas o ganho é limitado quando a construção "
                 "já é boa, e o custo por construção sobe bastante (ver E5).",
             ]),
             explainer("Métricas usadas", [
                 "<b>best</b>: melhor β final da estratégia na instância (depois da busca local, quando houver). "
                 "<b>construction_best</b>: melhor β das construções antes da busca local.",
                 "<b>vs_rcm_pct</b> = (best − β<sub>RCM</sub>) / β<sub>RCM</sub>. Negativo significa melhor que o "
                 "RCM.",
                 "<b>vs_best_pct</b>: quanto a estratégia fica acima do melhor β encontrado na instância por "
                 "qualquer uma das estratégias (0% = foi a melhor).",
                 "<b>ls_gain_pct</b> = (construction_best − best) / construction_best: redução obtida pela busca "
                 "local. É 0 por definição nas estratégias sem busca local.",
                 "<b>rank</b>: posição da estratégia em cada instância (1 = menor β; empates recebem a média das "
                 "posições).",
             ]),
             explainer("Cuidados gerais", [
                 (f"<b>Número de execuções.</b> {n_multi} combinações instância × estratégia têm mais de uma "
                  "execução nos CSVs (reexecuções acumuladas). Como 'best' é o mínimo entre execuções, essas "
                  "combinações levam vantagem. Para a dissertação, regenere os CSVs com o mesmo N_RUNS para todas."
                  if n_multi else
                  "<b>Número de execuções.</b> Todas as combinações instância × estratégia têm o mesmo número de "
                  "execuções. Com uma execução, diferenças pequenas podem estar dentro da variação entre sementes."),
                 "<b>α = 0 e α = 1 são equivalentes no código atual.</b> <code>get_LCR</code> embaralha tanto a "
                 "lista restrita quanto o complemento. Com α = 0 todos os filhos vão para o complemento, e com α = 1 "
                 "todos vão para a lista restrita: nos dois casos a ordem dos filhos é aleatória. O comentário de "
                 "<code>CONDITIONS</code> em <code>main_survey.py</code> ('1 = ordena todos pela centralidade') "
                 "não corresponde ao código. Na prática, a condição α tem 4 ações distintas, não 5. Nos dados, "
                 f"comparando as mesmas instâncias ({n_pairs} pares instância × busca local), a mediana de "
                 f"β(α = 1) / β(α = 0) é {pair_ratio:.3f}. As medianas agregadas diferem ({med_a0:.3f} contra "
                 f"{med_a1:.3f}) principalmente porque o agente escolheu α = 1 raramente ({n_a1} construções, contra {n_a0} com "
                 "α = 0) e em outras instâncias.",
                 "<b>Orçamento por construções, não por tempo.</b> Com busca local, cada execução leva em média "
                 f"{t_ratio:.1f}× mais tempo. As comparações de qualidade abaixo não descontam esse custo.",
             ]),
             section("E1 · Qual estratégia supera o RCM em cada classe?",
                     "O RCM (estado da prática segundo o survey) é a referência. Qual combinação condição × busca local "
                     "fica mais abaixo dele?",
                     "Barras abaixo de zero: a estratégia é melhor que o RCM na mediana da classe.",
                     fig_vs_rcm_by_class(best, strategies),
                     table_html(agg, {"mediana_vs_rcm": "{:+.1f}", "ganho_busca_local": "{:.1f}",
                                      "tempo_medio_s": "{:.1f}", "rank_medio": "{:.2f}"}),
                     how="As classes estão no eixo x, ordenadas pela mediana geral. Há uma barra por estratégia em "
                         "cada classe (cor = condição; sólida = com busca local; hachurada = sem). A altura é a "
                         "mediana, entre as instâncias da classe, da diferença percentual para o RCM. O tooltip "
                         "mostra quantas instâncias a classe tem.",
                     notes=[
                         "Comparar barras da mesma cor (sólida × hachurada) mostra o efeito da busca local. Comparar "
                         "cores com o mesmo preenchimento mostra o efeito da condição.",
                         "A mediana por classe esconde a dispersão. Em classes com poucas instâncias (ver o tooltip), "
                         "uma única instância pode mudar a barra.",
                         "A tabela agregada ('Ver tabela de dados') traz, por estratégia: vitórias, empates e derrotas "
                         "contra o RCM, ganho mediano da busca local, tempo médio, rank médio e em quantas "
                         "instâncias a estratégia atinge o melhor β.",
                     ]),
             section("E2 · Ranking geral",
                     "Ordenando as estratégias em cada instância (1 = menor β), qual tem o melhor rank médio? "
                     "A diferença é estatisticamente significativa (Friedman, α = 0,05)?",
                     f"Friedman p = {p_friedman:.2e} "
                     f"({'há' if p_friedman < 0.05 else 'não há'} diferença significativa entre as estratégias).",
                     f_rank,
                     how="À esquerda, o rank médio de cada estratégia (1 = seria sempre a melhor; "
                         f"{len(strategies)} = sempre a pior). À direita, em quantas instâncias a estratégia "
                         "atinge o melhor β entre todas (empates contam para todas as empatadas). O título traz o "
                         "p-valor do teste de Friedman.",
                     notes=[
                         "<b>O Friedman só diz que existe diferença.</b> Ele testa se pelo menos uma estratégia "
                         "difere das outras, mas não diz quais pares diferem. Para isso, aplique um teste post-hoc "
                         "(Nemenyi, ou Wilcoxon pareado com correção de Holm) e apresente um diagrama de diferença "
                         "crítica, que é o padrão na literatura de heurísticas.",
                         "O rank é ordinal: ficar em 1º por 1 unidade de β conta igual a ficar em 1º por 20%. Leia "
                         "junto com E3, que mostra a distância.",
                         "Em instâncias pequenas, várias estratégias costumam empatar no melhor β, e cada empate "
                         "conta para todas. Por isso a coluna da direita pode somar mais que o número de "
                         "instâncias.",
                     ]),
             section("E3 · Onde cada estratégia vence",
                     "Em quais instâncias cada estratégia encontra o melhor β, e quão longe fica nas demais?",
                     "Células claras = melhor β da instância; o tooltip mostra o β absoluto.",
                     fig_heatmap(best, strategies),
                     how="As linhas são instâncias, agrupadas por classe, e as colunas, estratégias. A cor indica "
                         "quanto o β da estratégia fica acima do melhor β encontrado na instância por qualquer "
                         "estratégia: claro significa que atingiu o melhor, e escuro, que ficou longe. A escala "
                         "satura no percentil 95, para que poucos valores extremos não apaguem o resto.",
                     notes=[
                         "Procure padrões por classe (blocos de linhas). Uma estratégia que só vence numa classe "
                         "sugere que a escolha deveria depender da classe, que é exatamente a premissa do Q-MCH.",
                         "Uma coluna clara quase inteira indica uma estratégia robusta. Uma coluna com poucas "
                         "células escuras, mas muito escuras, indica falhas ocasionais, o que importa quando o "
                         "método precisa funcionar bem em qualquer instância.",
                     ]),
             section("E4 · Escolhas do agente",
                     "Em cada condição, o agente concentra as escolhas em algum valor da ação? Esse valor é de fato "
                     "o que produz construções melhores?",
                     "À esquerda, a evolução das escolhas (linha pontilhada = escolha uniforme). À direita, a qualidade "
                     "de todas as construções feitas com cada valor da ação.",
                     fig_actions(out, best),
                     table_html(best_action_tab),
                     how="Há uma linha de painéis por condição. À esquerda está a porcentagem das escolhas do agente "
                         "em cada valor da ação, por bloco de 5 episódios, com todas as instâncias e execuções "
                         "juntas; a linha pontilhada é a escolha uniforme. À direita está a distribuição de β da "
                         "construção dividido pelo melhor β da instância, para todas as construções feitas com cada "
                         "valor (eixo cortado entre 0,95 e 2). A tabela mostra em quantas instâncias cada valor "
                         "estava em uso quando o melhor β apareceu.",
                     notes=[
                         "<b>Concentrar não é aprender.</b> A concentração das escolhas só indica aprendizado se o "
                         "valor escolhido também produz construções melhores (caixa mais baixa à direita). Se o "
                         "agente concentra num valor cuja caixa é igual às outras, a preferência é arbitrária.",
                         "<b>Teste de sanidade com α.</b> α = 0 e α = 1 fazem o mesmo no código atual (razão pareada "
                         f"{pair_ratio:.3f}; ver 'Cuidados gerais'). Qualquer preferência do agente entre os dois é "
                         "ruído e dá uma ideia do tamanho das diferenças que podem surgir por acaso. Note também que "
                         "as caixas da direita misturam instâncias diferentes para cada valor, conforme o agente "
                         "escolheu: compare valores na mesma instância antes de concluir que um é melhor.",
                         "A agregação entre instâncias pode esconder convergência individual: agentes que convergem "
                         "para valores diferentes somam algo parecido com uniforme.",
                         "Com busca local, o melhor β pode ter vindo mais da busca local que da ação escolhida. A "
                         "tabela de 'best_action' deve ser lida com isso em mente.",
                     ]),
             section("E5 · Busca local",
                     "Quanto a busca local do VNS-band reduz β em cada condição, e qual é o custo em tempo?",
                     "Ganho = redução do melhor β final em relação à melhor construção sem a busca local, na mesma execução.",
                     fig_local_search(best),
                     how="À esquerda, para cada condição com busca local, há um ponto por instância com a redução de "
                         "β obtida pela busca local em relação à melhor construção da mesma execução. À direita "
                         "está o tempo total por execução, com e sem busca local (eixo log).",
                     notes=[
                         f"<b>Custo-benefício.</b> Ganho mediano de {ls_med:.1f}% com um tempo médio "
                         f"{t_ratio:.1f}× maior. Para saber se compensa, compare com gastar o mesmo tempo extra em "
                         "mais construções sem busca local (orçamento por tempo).",
                         "<b>O ganho é medido dentro da execução com busca local.</b> Como a busca local muda o β que "
                         "o agente observa (e, portanto, as recompensas e as escolhas), a trajetória não é a mesma "
                         "da execução sem busca local. A comparação justa entre 'com' e 'sem' é a de E1 e E2 (melhor "
                         "β final de cada estratégia).",
                         "Pontos em 0% são instâncias em que a busca local não conseguiu melhorar a melhor "
                         "construção, por exemplo porque ela já estava no limite inferior ou num ótimo local da "
                         "vizinhança de trocas.",
                     ]),
             section("E6 · Convergência",
                     "Quantas construções cada estratégia precisa para chegar perto do melhor β?",
                     "Mediana entre instâncias do melhor β acumulado dividido pelo melhor β da instância.",
                     fig_convergence(out, strategies),
                     how="O eixo x é o número de construções já feitas. O eixo y é a mediana, entre instâncias, do "
                         "melhor β até ali dividido pelo melhor β da instância entre todas as estratégias (1 = já "
                         "atingiu o melhor de todas). Linha cheia = com busca local; tracejada = sem; cor = "
                         "condição. O eixo y está cortado em 1,4.",
                     notes=[
                         "O eixo x conta construções, não tempo. Com busca local, cada construção custa várias vezes "
                         "mais (E5); numa comparação por tempo, as curvas tracejadas avançariam muito mais para a "
                         "direita no mesmo intervalo.",
                         "A referência é o melhor β entre todas as estratégias. Uma curva que estabiliza acima de 1 "
                         "indica que a estratégia converge para um resultado pior, não que lhe falta orçamento.",
                     ]),
             section("Tabela completa", "Melhor resultado por instância e estratégia.", "",
                     None, table_html(best[["Instance", "classe", "estrategia", "best", "construction_best", "mean", "std",
                                            "best_action", "rcm", "lb", "vs_rcm_pct", "ls_gain_pct", "time"]]
                                      .sort_values(["classe", "Instance", "estrategia"]),
                                      {"mean": "{:.1f}", "std": "{:.1f}", "vs_rcm_pct": "{:+.1f}",
                                       "ls_gain_pct": "{:.1f}", "time": "{:.1f}"}, max_rows=5000),
                     how="Uma linha por instância × estratégia. As colunas estão descritas em 'Métricas usadas'; "
                         "mean e std são a média e o desvio do β das construções, rcm e lb são o β do RCM e o "
                         "limite inferior, e time é o tempo médio por execução, em segundos. O arquivo também é "
                         f"gravado em <code>comparison/tabelas/estrategias_{args.dataset}_melhor_por_instancia.csv</code>."),
             "<h2>Síntese e considerações finais</h2>",
             explainer("O que os dados sustentam", [
                 f"<b>Melhor estratégia no conjunto.</b> {top['estrategia']} tem o melhor rank médio "
                 f"({top['rank_medio']:.2f}) e vence o RCM em {top['vence_rcm']} de {top['instancias']} instâncias. "
                 f"Friedman p = {p_friedman:.1e}.",
                 f"<b>Busca local.</b> Reduz o melhor β em {ls_med:.1f}% na mediana, ao custo de ~{t_ratio:.1f}× o tempo.",
                 "<b>Condição.</b> Compare em E1 e E2 as três cores com o mesmo preenchimento. Se a ordem das "
                 "condições muda entre classes (E3), a escolha da condição deveria depender da classe.",
             ]),
             explainer("O que ainda falta para concluir", [
                 "Um teste post-hoc par a par (Nemenyi ou Wilcoxon com correção de Holm) para dizer quais "
                 "estratégias diferem de fato.",
                 "Mesmo número de execuções (N_RUNS ≥ 3) para todas as combinações, reportando média ± desvio.",
                 "Uma comparação com o mesmo limite de tempo, já que a busca local muda bastante o custo por "
                 "construção.",
                 "Corrigir ou documentar a equivalência α = 0 / α = 1 antes de interpretar a condição α.",
             ]),
             "</main></body></html>"]
    out_html = os.path.join(HERE, f"report_estrategias_{args.dataset}.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"[OK] relatório: {out_html}")


if __name__ == "__main__":
    main()
