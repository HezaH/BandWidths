# Pipeline dos experimentos do artigo ICITS 2026: roda em sequencia as etapas de pipeline_icits.json
# (reproducao e extensoes E2, E3 e E1) e gera os relatorios no fim.
#
# Cada etapa chama o main_icits.py com os argumentos da etapa, num processo proprio, e roda em paralelo
# com os nucleos livres no momento menos 2 (--workers auto). As etapas sao executadas uma apos a outra
# para que cada uma use toda a maquina. Como o main_icits.py pula as execucoes ja gravadas, o pipeline
# pode ser interrompido e retomado a qualquer momento: basta rodar o mesmo comando de novo.
#
# Uso (a partir de article_code/):
#   python icits2026/pipeline_icits.py --dry-run           -> etapas, execucoes pendentes e tempo estimado
#   python icits2026/pipeline_icits.py                     -> roda todas as etapas habilitadas + relatorios
#   python icits2026/pipeline_icits.py --only e2_recompensa e3_bandits
#   python icits2026/pipeline_icits.py --from e1_ordem_do_pai
#   python icits2026/pipeline_icits.py --config outro_pipeline.json --dataset survey

if __name__ == "__main__":
    print("[INFO] Iniciando o pipeline_icits.py", flush=True)

import os
import sys
import json
import time
import argparse
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))   # article_code/icits2026
BASE_DIR = os.path.dirname(HERE)                     # article_code
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import main_icits as mi  # noqa: E402  (reaproveita o plano, a retomada e o cálculo de núcleos livres)

DEFAULT_CONFIG = os.path.join(HERE, "pipeline_icits.json")
MAIN_SCRIPT = os.path.join(HERE, "main_icits.py")
REPORT_SCRIPT = os.path.join(HERE, "report_icits.py")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stage_settings(stage, defaults, dataset):
    """Junta os padrões do pipeline com os argumentos da etapa."""
    s = {**defaults, **stage.get("args", {})}
    s["dataset"] = dataset
    s.setdefault("construction", "paper")
    s.setdefault("experiment", "all")
    s.setdefault("reward", ["paper"])
    return s


def stage_argv(s):
    """Converte as configurações da etapa nos argumentos de linha de comando do main_icits.py."""
    argv = ["--dataset", s["dataset"], "--construction", s["construction"], "--experiment", s["experiment"],
            "--seeds", str(s["seeds"]), "--train", str(s["train"]), "--greedy", str(s["greedy"]),
            "--workers", str(s["workers"]), "--reserve-cores", str(s["reserve_cores"]), "--reward", *s["reward"]]
    for key, flag in [("selectors", "--selectors"), ("instances", "--instances"), ("methods", "--methods")]:
        if s.get(key):
            argv += [flag, *s[key]]
    if s.get("results_dir"):
        argv += ["--results-dir", s["results_dir"]]
    if s.get("no_decisions"):
        argv.append("--no-decisions")
    return argv


def stage_results_dir(s):
    return s.get("results_dir") or mi.default_results_dir(s["dataset"], s["construction"])


def stage_pending(s):
    """(execuções planejadas, pendentes) da etapa, pela mesma lógica de retomada do main_icits.py."""
    plan = mi.build_plan(s["dataset"], s["experiment"], s.get("selectors"), s["reward"], s.get("instances"),
                         s.get("methods"))
    pending = mi.pending_jobs(plan, s["seeds"], stage_results_dir(s), s["train"], s["greedy"], s["dataset"],
                              s["construction"])
    return len(plan) * s["seeds"], pending


def time_per_instance():
    """Tempo médio por execução de cada instância, a partir de todos os runs.csv já gravados (para estimar)."""
    frames = []
    for d in os.listdir(HERE):
        f = os.path.join(HERE, d, "runs.csv")
        if d.startswith("results") and os.path.exists(f):
            frames.append(pd.read_csv(f, usecols=["Instance", "Global Time (s)"]))
    if not frames:
        return {}
    return pd.concat(frames).groupby("Instance")["Global Time (s)"].mean().to_dict()


def estimate_hours(pending, times, workers):
    """Estimativa simples: soma dos tempos médios das execuções pendentes dividida pelos processos.
    Instâncias sem histórico não entram (a estimativa fica marcada como parcial)."""
    known = [times[i] for i, _, _ in pending if i in times]
    partial = len(known) < len(pending)
    return sum(known) / max(workers, 1) / 3600, partial


def run_reports(results_dirs, workers, logger):
    """Gera os relatórios das pastas tocadas pelo pipeline, em paralelo (um processo por relatório)."""
    procs, pending = [], list(results_dirs)
    while pending or procs:
        while pending and len(procs) < max(1, workers):
            d = pending.pop(0)
            logger.info("Relatorio: %s", d)
            procs.append((d, subprocess.Popen([sys.executable, REPORT_SCRIPT, "--results-dir", d],
                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                              encoding="utf-8", errors="replace")))
        for d, p in list(procs):
            if p.poll() is not None:
                out = p.stdout.read()
                procs.remove((d, p))
                ok = p.returncode == 0
                logger.info("Relatorio %s: %s", "gerado" if ok else "FALHOU", d)
                if not ok:
                    logger.info(out[-2000:])
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="Pipeline dos experimentos ICITS 2026 (etapas em sequência, "
                                                 "cada uma em paralelo)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="arquivo JSON com as etapas")
    parser.add_argument("--dataset", choices=list(mi.DATASET_KEY), default=None,
                        help="base de todas as etapas (padrão: a do JSON; se lá estiver null, pergunta no terminal)")
    parser.add_argument("--only", nargs="+", default=None, help="roda só estas etapas")
    parser.add_argument("--from", dest="start", default=None, help="começa nesta etapa (pula as anteriores)")
    parser.add_argument("--skip", nargs="+", default=[], help="pula estas etapas")
    parser.add_argument("--dry-run", action="store_true", help="só mostra etapas, pendências e tempo estimado")
    parser.add_argument("--no-reports", action="store_true", help="não gera os relatórios no fim")
    parser.add_argument("--continue-on-error", action="store_true", help="segue para a próxima etapa se uma falhar")
    args = parser.parse_args()

    config = load_config(args.config)
    defaults = config.get("defaults", {})
    dataset_opt = args.dataset or defaults.get("dataset")
    dataset = mi.DATASET_KEY[dataset_opt] if dataset_opt else mi.ask_dataset()

    stages = [st for st in config["stages"] if st.get("enabled", True)]
    if args.only:
        stages = [st for st in stages if st["name"] in args.only]
    if args.start:
        names = [st["name"] for st in stages]
        if args.start not in names:
            raise SystemExit(f"Etapa '{args.start}' não encontrada. Etapas: {names}")
        stages = stages[names.index(args.start):]
    stages = [st for st in stages if st["name"] not in args.skip]
    if not stages:
        raise SystemExit("Nenhuma etapa selecionada.")

    os.makedirs(mi.LOGS_DIR, exist_ok=True)
    logger = mi.configure_logger(os.path.join(mi.LOGS_DIR, f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    workers_now, how = mi.resolve_workers(defaults.get("workers", "auto"), defaults.get("reserve_cores", 2), 10 ** 6)
    times = time_per_instance()
    logger.info("Pipeline: %s | base: %s | %s etapa(s) | paralelismo agora: %s", os.path.basename(args.config),
                dataset, len(stages), how)

    # ------------------------------------------------------------
    # PLANO: pendências e estimativa de tempo por etapa
    # ------------------------------------------------------------
    rows = []
    for st in stages:
        s = stage_settings(st, defaults, dataset)
        planned, pending = stage_pending(s)
        hours, partial = estimate_hours(pending, times, workers_now)
        rows.append({"etapa": st["name"], "construção": s["construction"], "experimento": s["experiment"],
                     "seletores": " ".join(s.get("selectors") or ["(padrão do artigo)"]),
                     "recompensa": " ".join(s["reward"]), "planejadas": planned, "pendentes": len(pending),
                     "estimativa (h)": f"{hours:.1f}{'+' if partial else ''}",
                     "pasta": os.path.relpath(stage_results_dir(s), BASE_DIR)})
    plan_df = pd.DataFrame(rows)
    logger.info("Plano do pipeline (estimativa com %s processo(s); '+' = instâncias sem histórico de tempo):\n%s",
                workers_now, plan_df.to_string(index=False))
    logger.info("Observação: as pendências das etapas seguintes podem diminuir quando uma etapa anterior grava "
                "execuções que elas compartilham.")
    if args.dry_run:
        return 0

    # ------------------------------------------------------------
    # EXECUCAO: uma etapa após a outra; cada etapa roda em paralelo dentro do main_icits.py
    # ------------------------------------------------------------
    summary, touched = [], []
    for st in stages:
        s = stage_settings(st, defaults, dataset)
        _, pending = stage_pending(s)
        results_dir = stage_results_dir(s)
        if results_dir not in touched:
            touched.append(results_dir)
        if not pending:
            logger.info("=== Etapa %s: nada pendente, pulando", st["name"])
            summary.append({"etapa": st["name"], "situação": "já concluída", "execuções": 0, "duração (min)": 0.0})
            continue
        logger.info("=== Etapa %s (%s): %s execucoes pendentes", st["name"], st.get("description", ""), len(pending))
        t0 = time.time()
        code = subprocess.call([sys.executable, MAIN_SCRIPT, *stage_argv(s)])
        minutes = (time.time() - t0) / 60
        status = "concluída" if code == 0 else f"FALHOU (código {code})"
        summary.append({"etapa": st["name"], "situação": status, "execuções": len(pending),
                        "duração (min)": round(minutes, 1)})
        logger.info("=== Etapa %s: %s em %.1f min", st["name"], status, minutes)
        if code != 0 and not args.continue_on_error:
            logger.info("Pipeline interrompido. Corrija o problema e rode o mesmo comando para retomar.")
            break

    if config.get("reports", True) and not args.no_reports and touched:
        workers_rep, _ = mi.resolve_workers("auto", defaults.get("reserve_cores", 2), len(touched))
        run_reports(touched, workers_rep, logger)

    logger.info("Resumo do pipeline:\n%s", pd.DataFrame(summary).to_string(index=False))
    return 0 if all(r["situação"] != "" and not r["situação"].startswith("FALHOU") for r in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
