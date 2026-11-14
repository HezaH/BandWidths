# Relatorio do experimento de referencia ("oraculo") do main_survey.py.
#
# Para cada instancia, as MESMAS sementes rodam com politicas diferentes de escolha da centralidade:
#   dqn               -> o agente do Q-MCH (como na condicao "centrality")
#   random            -> centralidade sorteada uniformemente a cada episodio
#   fixed:<centr.>    -> sempre a mesma centralidade
# Pergunta central: existe algo a aprender? Se a melhor centralidade fixa nao for melhor que
# escolher ao acaso, nenhum agente consegue ganhar escolhendo a centralidade.
#
# Entradas: result_summary_<dataset>_<tag>.csv e result_output_<dataset>_<tag>.csv (tag = oracle | oracle_ls)
#           gerados por: python main_survey.py --condition oracle --dataset <dataset> [--oracle-runs 3]
# Saida:    comparison/report_<tag>_<dataset>.html e comparison/tabelas/<tag>_<dataset>_*.csv
#
# Uso: python comparison/oracle_report.py --dataset newdata [--tag oracle_ls]

import os
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import wilcoxon

from build_report import (HERE, BASE_DIR, OUT_TABLES, FOLDER_CLASS, SLOTS, INK, INK2, MUTED, AXIS, SURFACE,
                          GOOD, BAD, NEUTRAL, CSS, CENTRALITY_COLOR, style, table_html, tile, section, explainer,
                          count_wtl)

CATEGORY_ORDER = ["acima do ruído", "dentro do ruído", "sem ganho"]
CATEGORY_COLOR = {"acima do ruído": GOOD, "dentro do ruído": NEUTRAL, "sem ganho": BAD}


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def instance_classes(dataset, base_dir):
    classes = {}
    for root, dirs, files in os.walk(os.path.join(base_dir, "data", dataset)):
        dirs[:] = [d for d in dirs if not d.startswith("plots")]
        for name in files:
            if name.endswith(".mtx"):
                classes.setdefault(name[:-4], FOLDER_CLASS.get(os.path.basename(root), os.path.basename(root)))
    return classes


def load(dataset, tag, base_dir):
    summary_file = os.path.join(base_dir, f"result_summary_{dataset}_{tag}.csv")
    if not os.path.exists(summary_file):
        raise SystemExit(f"Arquivo não encontrado: {summary_file}\n"
                         f"Rode antes: python main_survey.py --condition oracle --dataset {dataset}"
                         + (" --local-search on" if tag.endswith("_ls") else ""))
    summ = pd.read_csv(summary_file)
    out_file = os.path.join(base_dir, f"result_output_{dataset}_{tag}.csv")
    out = (pd.read_csv(out_file, usecols=lambda c: c in {"Instance", "policy", "Run", "Episode", "action"})
           if os.path.exists(out_file) else None)
    classes = instance_classes(dataset, base_dir)
    summ["classe"] = "survey" if dataset == "survey" else summ["Instance"].map(classes).fillna(dataset)
    return summ, out


# ---------------------------------------------------------------------------
# Metricas por instancia
# ---------------------------------------------------------------------------
def per_instance(summ):
    """Uma linha por instância com a média (entre sementes) do melhor β de cada política e as
    medidas do experimento: ganho potencial, ruído entre sementes e variância explicada."""
    rows = []
    for inst, g in summ.groupby("Instance"):
        piv = g.pivot_table(index="Seed", columns="Policy", values="Best Bandwidth").dropna()
        fixed = [c for c in piv.columns if str(c).startswith("fixed:")]
        if "random" not in piv.columns or not fixed or piv.empty:
            continue
        means = piv.mean()
        rnd = means["random"]
        best_val, worst_val = means[fixed].min(), means[fixed].max()
        best_arms = [c.split(":", 1)[1] for c in fixed if means[c] == best_val]

        # melhor fixa "honesta": escolhe a centralidade pelas outras sementes e avalia na semente deixada de fora
        honest = np.nan
        if len(piv) >= 2:
            vals = []
            for s in piv.index:
                arm = piv.drop(index=s)[fixed].mean().idxmin()
                vals.append(piv.loc[s, arm])
            honest = float(np.mean(vals))

        # ruído: desvio padrão entre sementes com a centralidade fixa (média quadrática entre centralidades)
        noise = float(np.sqrt(piv[fixed].var(ddof=1).mean())) if len(piv) >= 2 else np.nan

        # fração da variância do melhor β explicada pela centralidade (eta², só centralidades fixas)
        cells = piv[fixed].values
        grand = cells.mean()
        ss_tot = ((cells - grand) ** 2).sum()
        ss_arm = len(piv) * ((cells.mean(axis=0) - grand) ** 2).sum()
        eta2 = ss_arm / ss_tot if ss_tot > 0 else np.nan

        def rel(x):
            return 100 * (x - rnd) / rnd if rnd else np.nan

        first = g.iloc[0]
        row = {
            "Instance": inst, "classe": first["classe"], "nodes": first["Nodes"], "edges": first["Edges"],
            "seeds": len(piv), "rcm": first["RCM Bandwidth"], "lower_bound": first["Lower Bound"],
            "random": rnd, "dqn": means.get("dqn", np.nan), "best_fixed": best_val, "worst_fixed": worst_val,
            "best_fixed_honest": honest, "best_arms": ", ".join(best_arms), "n_best_arms": len(best_arms),
            "n_arms": len(fixed),
            "gain_hindsight_pct": -rel(best_val),          # positivo = melhor fixa supera a aleatória
            "gain_honest_pct": -rel(honest),
            "dqn_vs_random_pct": rel(means.get("dqn", np.nan)),  # negativo = DQN melhor que a aleatória
            "spread_pct": 100 * (worst_val - best_val) / rnd if rnd else np.nan,
            "noise_pct": 100 * noise / rnd if rnd else np.nan,
            "eta2": eta2,
        }
        for c in fixed:
            row[c] = means[c]
            row[f"rel_{c}"] = rel(means[c])
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab["categoria"] = np.select(
        [tab["gain_honest_pct"] <= 0, tab["gain_honest_pct"] < tab["noise_pct"]],
        ["sem ganho", "dentro do ruído"], "acima do ruído")
    return tab


def arms_of(tab):
    return [c.split(":", 1)[1] for c in tab.columns if c.startswith("fixed:")]


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_policies_vs_random(tab, arms):
    series = [("DQN", "dqn_vs_random_pct", SLOTS[0])]
    series += [(f"fixa: {a}", f"rel_fixed:{a}", CENTRALITY_COLOR.get(a, MUTED)) for a in arms]
    series += [("melhor fixa (honesta)", None, GOOD), ("melhor fixa (retrospectiva)", None, SLOTS[5 % len(SLOTS)]),
               ("pior fixa", None, BAD)]
    extra = {"melhor fixa (honesta)": -tab["gain_honest_pct"], "melhor fixa (retrospectiva)": -tab["gain_hindsight_pct"],
             "pior fixa": 100 * (tab["worst_fixed"] - tab["random"]) / tab["random"]}
    fig = go.Figure()
    for name, col, color in series:
        y = extra[name] if col is None else tab[col]
        fig.add_box(y=y, name=name, marker_color=color, boxpoints="all", jitter=0.4, pointpos=0,
                    marker=dict(size=4), line_width=1.5, customdata=tab["Instance"],
                    hovertemplate="<b>%{customdata}</b><br>" + name + ": %{y:+.1f}% vs aleatória<extra></extra>")
    fig.add_hline(y=0, line_color=INK2, line_width=1)
    fig.update_yaxes(title_text="(β da política − β aleatória) / β aleatória [%]  (negativo = melhor)", ticksuffix="%")
    return style(fig, "O1 · Cada política comparada com a escolha aleatória da centralidade (média das sementes)",
                 height=520, showlegend=False)


def fig_gain_vs_noise(tab):
    fig = go.Figure()
    for cat in CATEGORY_ORDER:
        g = tab[tab["categoria"] == cat]
        fig.add_scatter(x=g["noise_pct"], y=g["gain_honest_pct"], mode="markers", name=f"{cat} ({len(g)})",
                        marker=dict(size=9, color=CATEGORY_COLOR[cat], line=dict(color=SURFACE, width=1.5)),
                        customdata=np.c_[g["Instance"], g["classe"], g["best_arms"]],
                        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>ruído: %{x:.1f}%"
                                      "<br>ganho honesto: %{y:+.1f}%<br>melhor fixa: %{customdata[2]}<extra></extra>")
    lim = max(1.0, float(np.nanmax(np.r_[tab["noise_pct"], tab["gain_honest_pct"]])) * 1.1)
    fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", name="ganho = ruído",
                    line=dict(color=MUTED, dash="dot", width=1.5), hoverinfo="skip")
    fig.add_hline(y=0, line_color=INK2, line_width=1)
    fig.update_xaxes(title_text="ruído: desvio padrão entre sementes com a centralidade fixa [% do β aleatório]",
                     rangemode="tozero")
    fig.update_yaxes(title_text="ganho da melhor fixa (honesta) sobre a aleatória [%]", ticksuffix="%")
    return style(fig, "O2 · Há o que aprender? Ganho potencial da melhor centralidade vs. ruído entre sementes",
                 height=520)


def fig_best_arm_heat(tab, arms):
    rows = []
    for cls, g in tab.groupby("classe"):
        counts = {a: int(g["best_arms"].str.split(", ").apply(lambda xs: a in xs).sum()) for a in arms}
        rows.append({"classe": cls, "n": len(g), **counts})
    d = pd.DataFrame(rows).set_index("classe")
    share = d[arms].div(d["n"], axis=0) * 100
    fig = go.Figure(go.Heatmap(
        z=share.values, x=arms, y=[f"{c} ({n})" for c, n in d["n"].items()], text=d[arms].values,
        texttemplate="%{text}", textfont=dict(color=INK), zmin=0, zmax=100,
        colorscale=[[0, "#f0efec"], [0.5, "#86b6ef"], [1, "#184f95"]],
        colorbar=dict(title="% das<br>instâncias", ticksuffix="%"),
        hovertemplate="<b>%{y}</b> · %{x}<br>melhor fixa em %{text} instâncias (%{z:.0f}%)<extra></extra>"))
    return style(fig, "O3 · Qual centralidade fixa é a melhor em cada classe? (empates contam para todas)",
                 height=max(380, 34 * len(d) + 160))


def fig_eta2(tab):
    order = tab.groupby("classe")["eta2"].median().sort_values().index.tolist()
    fig = go.Figure()
    for cls in order:
        g = tab[tab["classe"] == cls]
        fig.add_box(x=100 * g["eta2"], name=f"{cls} ({len(g)})", orientation="h", boxpoints="all", jitter=0.4,
                    pointpos=0, marker=dict(size=6, color=SLOTS[0]), line=dict(color=SLOTS[0], width=1.5),
                    showlegend=False, customdata=g["Instance"],
                    hovertemplate="<b>%{customdata}</b><br>η² = %{x:.0f}%<extra></extra>")
    fig.update_xaxes(title_text="η²: % da variância do melhor β explicada pela centralidade fixa", range=[0, 100],
                     ticksuffix="%")
    return style(fig, "O4 · Quanto da variação do resultado vem da centralidade (e quanto da semente)?",
                 height=max(380, 34 * len(order) + 160))


def dqn_choice_share(out, tab):
    """Fração dos episódios em que a política escolheu uma das melhores centralidades fixas
    (definidas pelas execuções fixas, independentes da política avaliada)."""
    best = tab.set_index("Instance")["best_arms"].str.split(", ")
    ep = (out[out["policy"].isin(["dqn", "random"])]
          .groupby(["Instance", "policy", "Run", "Episode"])["action"].first().reset_index())
    ep = ep[ep["Instance"].isin(best.index)]
    ep["hit"] = [a in best[i] for i, a in zip(ep["Instance"], ep["action"].astype(str))]
    ep["bloco"] = (ep["Episode"] - 1) // 5
    share = ep.groupby(["policy", "bloco"])["hit"].mean().unstack(0) * 100
    expected = 100 * (tab["n_best_arms"] / tab["n_arms"]).mean()
    return share, expected


def fig_dqn_choices(share, expected):
    labels = [f"{5 * b + 1}–{5 * b + 5}" for b in share.index]
    fig = go.Figure()
    for pol, name, color in [("dqn", "DQN", SLOTS[0]), ("random", "aleatória", MUTED)]:
        if pol in share.columns:
            fig.add_scatter(x=labels, y=share[pol], mode="lines+markers", name=name,
                            line=dict(color=color, width=2), marker=dict(size=8),
                            hovertemplate=name + "<br>episódios %{x}: %{y:.0f}%<extra></extra>")
    fig.add_scatter(x=labels, y=[expected] * len(labels), mode="lines", name="esperado por acaso",
                    line=dict(color=INK, dash="dot", width=1.5), hoverinfo="skip")
    fig.update_xaxes(title_text="bloco de episódios")
    fig.update_yaxes(title_text="% dos episódios com uma das melhores centralidades", ticksuffix="%", rangemode="tozero")
    return style(fig, "O5 · O DQN passa a escolher a melhor centralidade ao longo dos episódios?", height=440)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def wilcoxon_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    try:
        return wilcoxon(a[ok], b[ok], zero_method="pratt").pvalue
    except ValueError:
        return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["newdata", "survey"], default="newdata")
    parser.add_argument("--tag", choices=["oracle", "oracle_ls"], default="oracle",
                        help="oracle = sem busca local; oracle_ls = com busca local")
    parser.add_argument("--results-dir", default=BASE_DIR, help="pasta com os result_*.csv (padrão: article_code)")
    args = parser.parse_args()

    summ, out = load(args.dataset, args.tag, args.results_dir)
    tab = per_instance(summ)
    if tab.empty:
        raise SystemExit("Nenhuma instância com as políticas 'random' e 'fixed:*' nos resultados.")
    arms = arms_of(tab)
    os.makedirs(OUT_TABLES, exist_ok=True)
    tab.to_csv(os.path.join(OUT_TABLES, f"{args.tag}_{args.dataset}_por_instancia.csv"), index=False)

    # --- números usados no texto ---
    n_inst = len(tab)
    seeds = int(tab["seeds"].median())
    cat = tab["categoria"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    share_above = 100 * cat["acima do ruído"] / n_inst
    med_gain_h = tab["gain_honest_pct"].median()
    med_gain_r = tab["gain_hindsight_pct"].median()
    med_noise = tab["noise_pct"].median()
    med_spread = tab["spread_pct"].median()
    med_eta2 = 100 * tab["eta2"].median()
    med_dqn = tab["dqn_vs_random_pct"].median()
    w_dqn = count_wtl(tab["dqn"], tab["random"])
    w_hon = count_wtl(tab["best_fixed_honest"], tab["random"])
    w_dqn_best = count_wtl(tab["dqn"], tab["best_fixed_honest"])
    p_dqn = wilcoxon_p(tab["dqn"], tab["random"])
    p_hon = wilcoxon_p(tab["best_fixed_honest"], tab["random"])
    arm_best_total = {a: int(tab["best_arms"].str.split(", ").apply(lambda xs: a in xs).sum()) for a in arms}
    top_arm = max(arm_best_total, key=arm_best_total.get)
    times = summ.groupby("Policy")["Global Time (s)"].mean()
    t_dqn_ratio = times.get("dqn", np.nan) / times[[p for p in times.index if p.startswith("fixed:")]].mean()
    has_out = out is not None and not out.empty
    if has_out:
        share, expected = dqn_choice_share(out, tab)
        late = share.iloc[-2:].mean() if len(share) >= 2 else share.iloc[-1]
        dqn_late = late.get("dqn", np.nan)
        rnd_late = late.get("random", np.nan)

    pol_time = (summ.groupby("Policy").agg(execucoes=("Run", "size"), tempo_medio_s=("Global Time (s)", "mean"),
                                          beta_medio=("Best Bandwidth", "mean"))
                .reset_index().sort_values("Policy"))

    # regra de decisão para a síntese
    learnable = share_above >= 25
    dqn_uses = (w_dqn[0] > w_dqn[2]) and (p_dqn < 0.05 if not np.isnan(p_dqn) else False)
    if not learnable:
        verdict = ("<b>Há pouco a aprender com as ações atuais.</b> Em só "
                   f"{cat['acima do ruído']} de {n_inst} instâncias ({share_above:.0f}%) a melhor centralidade fixa "
                   "supera a escolha aleatória por mais que o ruído entre sementes. Nessa situação nenhum agente, "
                   "por melhor que seja, consegue ganhar escolhendo a centralidade. O próximo passo é fazer as ações "
                   "diferirem (ações não redundantes, centralidade crescente no estilo Cuthill–McKee, LCR sem "
                   "embaralhar a parte restrita, vértice inicial independente da centralidade) e repetir este "
                   "experimento.")
    elif not dqn_uses:
        verdict = ("<b>Há o que aprender, mas o DQN não aproveita.</b> Em "
                   f"{cat['acima do ruído']} de {n_inst} instâncias ({share_above:.0f}%) a melhor centralidade fixa "
                   "supera a aleatória por mais que o ruído, mas o DQN não fica sistematicamente melhor que a "
                   "aleatória. O problema está no aprendizado: recompensa contínua e relativa, mais episódios ou "
                   "transferência entre instâncias da classe, ou um bandit (UCB/Thompson) como linha de base.")
    else:
        verdict = ("<b>Há o que aprender e o DQN aproveita parte disso.</b> A melhor centralidade fixa supera a "
                   f"aleatória acima do ruído em {share_above:.0f}% das instâncias, e o DQN fica melhor que a "
                   "aleatória. Vale medir quanto do ganho possível ele captura (DQN contra a melhor fixa honesta) e "
                   "testar a generalização entre instâncias da classe.")

    tiles = "".join([
        tile(f"{n_inst}", f"instâncias × {seeds} sementes × {2 + len(arms)} políticas"),
        tile(f"{cat['acima do ruído']}/{n_inst}", "instâncias em que a melhor fixa supera a aleatória acima do ruído"),
        tile(f"{med_gain_h:+.1f}%", "ganho mediano da melhor fixa (honesta) sobre a aleatória"),
        tile(f"{w_dqn[0]}/{w_dqn[1]}/{w_dqn[2]}", "DQN vs aleatória: melhor / igual / pior"),
        tile(f"{med_eta2:.0f}%", "variância mediana explicada pela centralidade (η²)"),
    ])

    label_ls = "com busca local" if args.tag.endswith("_ls") else "sem busca local"
    parts = [
        f"<!doctype html><html lang='pt-br'><head><meta charset='utf-8'><title>Oráculo Q-MCH — {args.dataset}</title>"
        f"<style>{CSS}</style><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body><main>",
        f"<h1>Experimento de referência: o agente tem o que aprender? ({args.dataset}, {label_ls})</h1>",
        "<p class='src'>Gerado a partir de <code>result_summary_" + f"{args.dataset}_{args.tag}" + ".csv</code> e "
        "<code>result_output_" + f"{args.dataset}_{args.tag}" + ".csv</code> "
        "(<code>python main_survey.py --condition oracle</code>).</p>",
        f'<div class="tiles">{tiles}</div>',
        "<h2>Como ler este relatório</h2>",
        "<p>Os relatórios anteriores mostraram que o agente do Q-MCH não aprende uma preferência clara de "
        "centralidade. Há duas explicações possíveis, que pedem soluções diferentes: <b>(1)</b> as centralidades "
        "produzem resultados equivalentes, e então não há o que aprender; ou <b>(2)</b> há uma centralidade melhor, "
        "mas o agente não consegue descobri-la. Este experimento separa os dois casos. Em cada instância, as "
        "<b>mesmas sementes</b> rodam com políticas diferentes de escolha da centralidade, todas com o mesmo "
        "orçamento do Q-MCH (30 episódios × 15 construções = 450 construções).</p>",
        explainer("Políticas comparadas", [
            "<b>DQN</b>: o agente do Q-MCH, exatamente como na condição <code>centrality</code>.",
            "<b>Aleatória</b>: a centralidade é sorteada uniformemente a cada episódio. É a referência: representa "
            "um agente que não aprendeu nada. O sorteio usa um gerador próprio e não altera a aleatoriedade do "
            "construtivo.",
            f"<b>Fixa</b>: a mesma centralidade nos 30 episódios ({', '.join(arms)}). A melhor delas é o que um "
            "agente perfeito conseguiria se escolhesse uma única centralidade por instância.",
        ]),
        explainer("Medidas", [
            "<b>β de uma execução</b>: o melhor β das 450 construções. Para cada política e instância, usa-se a "
            f"média entre as {seeds} sementes.",
            "<b>Melhor fixa (retrospectiva)</b>: a centralidade fixa com menor β médio. É otimista, porque escolhe "
            "a melhor olhando os mesmos dados em que é avaliada: com várias centralidades e poucas sementes, alguma "
            "sempre parece melhor por acaso.",
            "<b>Melhor fixa (honesta)</b>: validação deixando uma semente de fora. A centralidade é escolhida pelas "
            "outras sementes e avaliada na semente deixada de fora, repetindo para cada semente. É a estimativa "
            "usada nas conclusões.",
            "<b>Ganho potencial</b> = (β aleatória − β melhor fixa honesta) / β aleatória. Positivo significa que "
            "escolher bem a centralidade compensa.",
            "<b>Ruído</b>: desvio padrão do β entre sementes com a centralidade fixa, em % do β aleatório. Indica "
            "quanto o resultado muda só por trocar a semente.",
            "<b>Categoria da instância</b>: <i>acima do ruído</i> se o ganho potencial supera o ruído; <i>dentro do "
            "ruído</i> se é positivo, mas menor que o ruído; <i>sem ganho</i> se a melhor fixa não supera a "
            "aleatória.",
            "<b>η²</b>: fração da variação do β, entre todas as execuções com centralidade fixa, explicada pela "
            "troca de centralidade. O restante vem das sementes.",
        ]),
        explainer("Cuidados gerais", [
            "<b>Mesmas sementes em todas as políticas.</b> Isso reduz o ruído da comparação, mas não o elimina: "
            "centralidades diferentes consomem a sequência aleatória de formas diferentes, então as construções "
            "divergem logo nas primeiras chamadas.",
            f"<b>Poucas sementes ({seeds}).</b> O ruído é estimado com poucas observações por instância, e as "
            "categorias de instâncias próximas da fronteira podem mudar com mais sementes "
            "(<code>--oracle-runs</code>).",
            "<b>Critério conservador.</b> O ganho compara médias entre sementes, mas é contrastado com o desvio de "
            "uma execução isolada, não com o erro padrão da média. Isso exige um ganho maior para considerar que "
            "há o que aprender.",
            "<b>Só a escolha da centralidade.</b> α, vértice inicial e busca local ficam fixos. O experimento não "
            "diz nada sobre ações de outro tipo (ver o relatório de estratégias).",
        ]),
        "<h2>Resultados</h2>",
        section(
            "O1 · Alguma política supera a escolha aleatória?",
            "Comparando cada política com a escolha aleatória na mesma instância, quem fica sistematicamente "
            "abaixo de zero?",
            f"Mediana em relação à aleatória: DQN <b>{med_dqn:+.1f}%</b> (melhor/igual/pior: "
            f"{w_dqn[0]}/{w_dqn[1]}/{w_dqn[2]}, Wilcoxon p = {p_dqn:.2g}); melhor fixa honesta "
            f"<b>{-med_gain_h:+.1f}%</b> ({w_hon[0]}/{w_hon[1]}/{w_hon[2]}, p = {p_hon:.2g}); melhor fixa "
            f"retrospectiva {-med_gain_r:+.1f}%. A diferença mediana entre a pior e a melhor centralidade fixa é de "
            f"{med_spread:.1f}% do β.",
            fig_policies_vs_random(tab, arms),
            how="Cada caixa reúne todas as instâncias, e cada ponto é uma instância. O eixo vertical é a diferença "
                "percentual do β médio da política para o β médio da escolha aleatória na mesma instância. Abaixo "
                "de zero, a política é melhor que escolher ao acaso. As três últimas caixas não são políticas "
                "executáveis: escolhem a melhor ou a pior centralidade fixa de cada instância.",
            notes=[
                "A distância entre 'melhor fixa (retrospectiva)' e 'melhor fixa (honesta)' mede o otimismo de "
                "escolher a melhor olhando os próprios dados. Se as duas ficam longe, boa parte da 'melhor "
                "centralidade' era sorte.",
                "A 'pior fixa' mostra o outro lado: se ela também fica perto de zero, todas as centralidades são "
                "equivalentes nesta configuração do construtivo.",
                "O DQN só pode ficar abaixo da melhor fixa honesta se mudar de centralidade de forma útil durante a "
                "execução. Ficar perto da aleatória indica que ele não está escolhendo melhor que um sorteio.",
            ]),
        section(
            "O2 · Há o que aprender em cada instância?",
            "O ganho da melhor centralidade fixa é maior que a variação causada só pela troca de semente?",
            f"Acima do ruído: <b>{cat['acima do ruído']}</b>; dentro do ruído: <b>{cat['dentro do ruído']}</b>; "
            f"sem ganho: <b>{cat['sem ganho']}</b> (de {n_inst}). Ruído mediano de {med_noise:.1f}% contra ganho "
            f"mediano de {med_gain_h:+.1f}%.",
            fig_gain_vs_noise(tab),
            how="Cada ponto é uma instância. O eixo x é o ruído (desvio entre sementes) e o eixo y, o ganho "
                "potencial da melhor fixa honesta sobre a aleatória, ambos em % do β aleatório. Pontos acima da "
                "diagonal pontilhada são as instâncias em que escolher a centralidade compensa mais que o acaso; "
                "pontos abaixo da linha horizontal 0 são aquelas em que a melhor fixa nem supera a aleatória.",
            notes=[
                "Este é o gráfico que responde à pergunta do experimento. Se a maioria dos pontos fica abaixo da "
                "diagonal, um agente perfeito ganharia pouco, e esforços no aprendizado (recompensa, rede, "
                "episódios) não vão aparecer nos resultados.",
                "Instâncias com ruído zero (todas as sementes dão o mesmo β) ficam no eixo vertical. Em geral são "
                "instâncias pequenas, em que o construtivo atinge o mesmo valor sempre.",
                "As instâncias acima da diagonal são as candidatas para avaliar um agente melhorado. Vale olhar se "
                "se concentram em alguma classe (passe o mouse ou veja a tabela).",
            ]),
        section(
            "O3 · Qual centralidade fixa ganha, e isso depende da classe?",
            "A premissa do SBPO é que a melhor centralidade depende da classe de matriz. Isso aparece nas "
            "execuções fixas?",
            f"No total, <b>{top_arm}</b> é a melhor fixa (ou empata como melhor) em {arm_best_total[top_arm]} "
            f"de {n_inst} instâncias. Contagem por centralidade: "
            + ", ".join(f"{a} {n}" for a, n in sorted(arm_best_total.items(), key=lambda kv: -kv[1])) + ".",
            fig_best_arm_heat(tab, arms),
            how="As linhas são classes (com o número de instâncias entre parênteses) e as colunas, centralidades. A "
                "célula mostra em quantas instâncias da classe aquela centralidade teve o menor β médio, com a cor "
                "em % da classe. Empates contam para todas as centralidades empatadas, por isso a linha pode somar "
                "mais de 100%.",
            notes=[
                "Uma linha com uma coluna dominante sugere que a classe tem uma centralidade preferida, o que daria "
                "a um agente treinado por classe algo concreto para aprender.",
                "Linhas espalhadas, ou dominadas por empates, indicam que a melhor centralidade muda de instância "
                "para instância por acaso. Confira com O2 se essas instâncias estão dentro do ruído.",
                "Esta contagem usa a melhor fixa retrospectiva, a mais exposta à sorte. Interprete as colunas "
                "junto com a categoria de cada instância.",
            ]),
        section(
            "O4 · Quanto do resultado vem da centralidade e quanto da semente?",
            "Entre todas as execuções com centralidade fixa, qual fração da variação do β é explicada por trocar "
            "a centralidade?",
            f"η² mediano de <b>{med_eta2:.0f}%</b>. O restante ({100 - med_eta2:.0f}%) vem das diferenças entre "
            "sementes.",
            fig_eta2(tab),
            how="Há uma caixa por classe, e cada ponto é uma instância. 0% significa que a centralidade não muda "
                "nada (toda a variação vem da semente); 100% significa que a semente não muda nada (toda a "
                "variação vem da centralidade).",
            notes=[
                "η² baixo é a forma quantitativa da conclusão dos relatórios anteriores: a repetição e o sorteio do "
                "vértice inicial pesam mais que a escolha da centralidade.",
                f"Com {seeds} sementes e {len(arms)} centralidades, η² tem bastante variação por instância. Mesmo "
                "sem efeito real, ele fica em torno de (k−1)/(N−1) por acaso (k = centralidades, N = execuções). "
                f"Aqui isso dá ≈ {100 * (len(arms) - 1) / max(len(arms) * seeds - 1, 1):.0f}%. Valores perto disso "
                "não indicam efeito.",
                "Instâncias com ruído e diferenças nulos (todas as execuções iguais) ficam sem η² e não aparecem.",
            ]),
    ]
    if has_out:
        parts.append(section(
            "O5 · O DQN aprende a escolher a melhor centralidade?",
            "Ao longo dos episódios, o DQN escolhe com mais frequência uma das melhores centralidades fixas da "
            "instância?",
            f"Nos dois últimos blocos de episódios, o DQN escolhe uma das melhores centralidades em "
            f"<b>{dqn_late:.0f}%</b> dos episódios, contra {rnd_late:.0f}% da política aleatória e {expected:.0f}% "
            "esperados por acaso.",
            fig_dqn_choices(share, expected),
            how="O eixo x são blocos de 5 episódios, e o eixo y, a porcentagem dos episódios em que a política "
                "escolheu uma das melhores centralidades fixas da instância (definidas pelas execuções fixas, que "
                "são independentes das execuções do DQN). A política aleatória e a linha pontilhada mostram o "
                "nível do acaso. Aprendizado aparece como a curva do DQN subindo acima delas nos últimos blocos.",
            notes=[
                "Diferente da Q7 do relatório de comparação, esta medida não é afetada pela agregação entre "
                "instâncias: cada episódio é comparado com a melhor centralidade <i>da própria instância</i>.",
                "Se a melhor centralidade foi definida por sorte (instâncias dentro do ruído), nem um agente "
                "perfeito subiria a curva. Leia junto com O2.",
                "Com ε chegando a 0,01 por volta do episódio 24, os últimos blocos refletem o que a rede considera "
                "melhor, não a exploração.",
            ]))
    parts += [
        section(
            "Custo das políticas",
            "Quanto tempo cada política leva por execução?",
            f"O DQN leva em média {t_dqn_ratio:.2f}× o tempo das políticas fixas. A diferença vem do treino da rede "
            "e da centralidade escolhida (centralidades com sorteio do vértice inicial mais concentrado ou com "
            "mais empates mudam o custo do construtivo).",
            None, table_html(pol_time, {"tempo_medio_s": "{:.2f}", "beta_medio": "{:.1f}"}),
            how="A tabela (abra 'Ver tabela de dados') mostra, por política, o número de execuções, o tempo médio "
                "por execução e o β médio de todas as execuções. O β médio mistura instâncias de tamanhos muito "
                "diferentes: use O1 para comparar qualidade."),
        section(
            "Tabela por instância", "Valores médios por política e medidas do experimento.", "",
            None, table_html(tab[["Instance", "classe", "nodes", "seeds", "random", "dqn", "best_fixed",
                                  "best_fixed_honest", "worst_fixed", "best_arms", "gain_honest_pct", "noise_pct",
                                  "categoria", "eta2", "dqn_vs_random_pct", "rcm"]]
                             .sort_values(["classe", "Instance"]),
                             {"random": "{:.1f}", "dqn": "{:.1f}", "best_fixed": "{:.1f}",
                              "best_fixed_honest": "{:.1f}", "worst_fixed": "{:.1f}", "gain_honest_pct": "{:+.1f}",
                              "noise_pct": "{:.1f}", "eta2": "{:.2f}", "dqn_vs_random_pct": "{:+.1f}",
                              "nodes": "{:,.0f}"}, max_rows=5000),
            how="Uma linha por instância. random, dqn, best_fixed, best_fixed_honest e worst_fixed são β médios "
                "entre sementes; gain_honest_pct, noise_pct e dqn_vs_random_pct estão em % do β aleatório. O "
                f"arquivo também é gravado em <code>comparison/tabelas/{args.tag}_{args.dataset}_por_instancia.csv"
                "</code>."),
        "<h2>Síntese e considerações finais</h2>",
        f"<div class='q'>{verdict}</div>",
        explainer("O que os dados sustentam", [
            f"Ganho potencial mediano de {med_gain_h:+.1f}% (melhor fixa honesta sobre a aleatória), com ruído "
            f"mediano de {med_noise:.1f}% entre sementes.",
            f"A centralidade explica, na mediana, {med_eta2:.0f}% da variação do β entre execuções com centralidade "
            "fixa.",
            f"DQN contra a aleatória: melhor/igual/pior em {w_dqn[0]}/{w_dqn[1]}/{w_dqn[2]} instâncias (mediana "
            f"{med_dqn:+.1f}%, Wilcoxon p = {p_dqn:.2g}). DQN contra a melhor fixa honesta: "
            f"{w_dqn_best[0]}/{w_dqn_best[1]}/{w_dqn_best[2]}.",
        ]),
        explainer("Próximos passos", [
            "Para dar mais peso ao agente, primeiro aumente a diferença entre as ações e rode este experimento de "
            "novo. O ganho potencial (O2) é o teto do que qualquer agente consegue.",
            "Com ganho potencial acima do ruído, compare o DQN com um bandit simples (UCB1 ou Thompson Sampling) no "
            "mesmo orçamento. Se o bandit capturar o ganho e o DQN não, o problema é a formulação do aprendizado.",
            "Para testar a premissa do SBPO (aprender por classe), use as classes em que O3 mostra uma centralidade "
            "dominante e instâncias acima do ruído em O2.",
            "Mais sementes (<code>--oracle-runs 5</code>) estreitam a fronteira entre 'acima' e 'dentro do ruído'.",
        ]),
        "</main></body></html>",
    ]
    out_html = os.path.join(HERE, f"report_{args.tag}_{args.dataset}.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"[OK] relatório: {out_html}")


if __name__ == "__main__":
    main()
