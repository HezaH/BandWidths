# Reprodução do artigo ICITS 2026

*A Matched-Budget Assessment of DQN Centrality Selection for Matrix Bandwidth Minimization*
(manuscrito em `../2026-ICITS-Peru-3/samplepaper.tex`).

Esta pasta implementa a configuração descrita no artigo sem alterar o pipeline do `main_survey.py`.
O código segue a organização, os nomes e as funções auxiliares do `main_survey.py` e reaproveita os
módulos de `article_code/` (grafo, construtivo, leitura das instâncias).

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `main_icits.py` | Execução dos experimentos (baseado no `main_survey.py`) |
| `agent_dqn.py` | DQN do artigo: replay, rede alvo, Smooth L1, gradiente limitado, ε exponencial |
| `environment_icits.py` | Ambiente: estado de 12 dimensões e recompensa densa (e a recompensa da extensão E2) |
| `bandits.py` | Seletores de bandit da extensão E3: UCB1 e adaptive pursuit |
| `pipeline_icits.py` / `pipeline_icits.json` | Pipeline: roda as etapas (reprodução, E2, E3, E1) em sequência e gera os relatórios |
| `report_icits.py` | Tabelas 3, 4 e 5, Friedman, SBS/VBS/GapClosed, Figuras 3 e 4, comparação com o artigo |
| `requirements.txt` | Dependências do pipeline (inclui `../requirements.txt`, do `main_survey.py`) |
| `cache/` | Grafo e centralidades por instância (gerado na 1ª execução) |
| `results/` | `instances.csv`, `runs.csv`, `decisions.csv`, `report_icits.html` e `tables/` |
| `logs/` | Log de cada execução do `main_icits.py` |

## Instalação (máquina virtual Linux)

Requer **Python 3.11 ou mais novo** (o networkx 3.6.1, versão em que o RCM reproduz o artigo, exige). A partir da
pasta `article_code`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # torch só CPU (~200 MB em vez de ~2 GB com CUDA)
pip install -r icits2026/requirements.txt                            # inclui ../requirements.txt (main_survey)
python icits2026/pipeline_icits.py --dry-run                         # confere a instalação e mostra o plano
```

Na VM, use `python` (do venv) no lugar de `C:/Python311/python`. As pastas `cache/`, `logs/` e o
`decisions.csv` não vão para o git: a VM recalcula as centralidades na primeira execução. Para continuar na VM
experimentos começados em outra máquina, copie a pasta `results*/` correspondente (o `runs.csv` basta para a
retomada; o `decisions.csv` alimenta as análises por decisão do relatório).

Testado com dois conjuntos de versões (numpy 1.24 / pandas 2.2 / torch 2.13 e numpy 2.4 / pandas 3.1 / torch 2.15),
com resultados idênticos execução a execução.

## Como rodar

A partir da pasta `article_code` (use o Python que tem networkx e torch, ex.: `C:/Python311/python`):

```bash
# teste rápido do pipeline (orçamento reduzido, pasta separada)
python icits2026/main_icits.py --train 20 --greedy 4 --seeds 2 --instances bcsstk01 lshp_265 \
       --results-dir icits2026/results_teste --fresh

# experimento completo do artigo: principal (13 instâncias x 9 métodos x 30 sementes) + ablação
python icits2026/main_icits.py

# ou por partes
python icits2026/main_icits.py --experiment main
python icits2026/main_icits.py --experiment ablation

# relatório
python icits2026/report_icits.py
```

A execução pode ser interrompida e retomada: execuções já gravadas em `runs.csv` com o mesmo orçamento são
puladas. `--fresh` apaga os resultados anteriores.

## Paralelismo

Tudo roda em paralelo, sempre. Com `--workers auto` (padrão), o número de processos é calculado no início de
cada execução como **núcleos lógicos × (1 − uso médio da CPU nos últimos 3 s) − 2**, no mínimo 1:

- a preparação das instâncias (centralidades, B0, RCM) e as execuções usam o mesmo pool de processos;
- as execuções maiores são enviadas primeiro, para que nenhuma fique sozinha no fim;
- `--reserve-cores N` muda a folga (padrão 2) e `--workers N` fixa a quantidade, se preferir.

O log mostra a decisão, por exemplo: `12 nucleos logicos, uso atual 14% (~10 livres), reserva de 2 -> 8 processo(s)`.
A medição acontece uma vez, no início: se outro programa estiver pesado nesse momento (o OneDrive sincronizando a
pasta, por exemplo), a execução usa menos processos até o fim.

## Pipeline

`pipeline_icits.py` roda em sequência as etapas de `pipeline_icits.json`, cada uma em paralelo, e no fim gera os
relatórios de todas as pastas tocadas. As etapas padrão seguem a ordem da conclusão do artigo:

| Etapa | O que roda |
|---|---|
| `reproducao` | Protocolo do artigo (principal + ablação), construção do artigo |
| `e2_recompensa` | DQN com empate = 0,5/B0 |
| `e3_bandits` | UCB1 e adaptive pursuit, com as duas recompensas |
| `e1_ordem_do_pai` | Protocolo completo com a ordem do pai |
| `e1_com_e2_e3` | E2 e E3 sobre a construção E1 |

```bash
python icits2026/pipeline_icits.py --dry-run          # etapas, execuções pendentes e tempo estimado
python icits2026/pipeline_icits.py                    # roda tudo (pergunta a base, se o JSON tiver dataset null)
python icits2026/pipeline_icits.py --only e2_recompensa e3_bandits
python icits2026/pipeline_icits.py --from e1_ordem_do_pai
python icits2026/pipeline_icits.py --dataset survey   # todas as etapas na base survey
```

- **Retomada:** cada etapa pula as execuções já gravadas. Se o pipeline for interrompido, o mesmo comando
  continua de onde parou, e etapas concluídas aparecem como "nada pendente".
- **Falhas:** por padrão o pipeline para na primeira etapa que falhar; `--continue-on-error` segue para a próxima.
- **Configuração:** para mudar etapas, sementes, instâncias ou pastas, edite o JSON (ou crie outro e passe com
  `--config`). Cada etapa aceita os mesmos campos dos argumentos do `main_icits.py` (`construction`,
  `experiment`, `selectors`, `reward`, `instances`, `results_dir`...).
- O log do pipeline fica em `logs/pipeline_<data>.log`, com o plano no início e o resumo das etapas no fim.

**Custo.** Uma execução (5.400 construções) leva de ~4 s (bcsstk01) a ~55 s (lshp1882). O experimento
principal tem 3.510 execuções e a ablação 600, cerca de 18 h de CPU no total; com 8 processos, ~2 a 3 h.
`pipeline_icits.py --dry-run` estima o tempo de cada etapa a partir dos tempos já medidos.

## Base de dados

Sem `--dataset`, o script pergunta no terminal qual base rodar (como o `main_survey.py`):

- **SBPO / artigo** (`--dataset sbpo`): as 13 instâncias HB do artigo, em `data/newdata`.
- **Survey** (`--dataset survey`): as matrizes de `data/survey`. Closeness, Betweenness e Harmonic usam versões
  aproximadas por amostragem (as exatas levariam dias), e cada execução pode levar horas nos grafos maiores:
  use `--instances` e `--seeds`.

## Extensões propostas na conclusão do artigo

Os padrões reproduzem o artigo. As opções abaixo implementam os três próximos passos da conclusão:

| Extensão | Opção | O que muda | O que roda de novo |
|---|---|---|---|
| **E1** ordem do pai | `--construction parent` | Filhos de cada vértice rotulados na ordem do pai (Cuthill–McKee); a centralidade só ordena os filhos de cada pai | Tudo (a construção muda para todos os métodos) |
| **E2** recompensa reescalada | `--reward half_unit_tie` | Empate vale 0,5/B0, abaixo de qualquer melhoria estrita (que rende ≥ 1/B0) | Só os seletores que aprendem |
| **E3** bandits | `--selectors UCB1 AP` | UCB1 e adaptive pursuit, com a mesma recompensa e a mesma divisão 300 + 60 do DQN | Só os novos seletores |

```bash
# E1: protocolo completo com a ordem do pai (resultados em results_sbpo_parent/)
python icits2026/main_icits.py --dataset sbpo --construction parent

# E2: só o DQN com a recompensa reescalada (Random e fixas de results/ são reaproveitados)
python icits2026/main_icits.py --dataset sbpo --experiment main --selectors DQN --reward half_unit_tie

# E3: bandits com a recompensa do artigo e com a reescalada
python icits2026/main_icits.py --dataset sbpo --experiment main --selectors UCB1 AP --reward paper half_unit_tie

# E2 e E3 também sobre a construção E1
python icits2026/main_icits.py --dataset sbpo --construction parent --experiment main --selectors DQN UCB1 AP \
       --reward half_unit_tie
```

Os nomes dos métodos seguem `<seletor>[@<recompensa>][-PP]`, por exemplo `DQN@half_unit_tie` ou `UCB1-PP`.
Cada combinação de base e construção tem a sua pasta de resultados (`results/` para SBPO + artigo, senão
`results_<base>_<construção>/`). E2 e E3 gravam na mesma pasta da construção usada, ao lado das referências.

**Relatório.** `python icits2026/report_icits.py --results-dir <pasta>` detecta a configuração:

- `results/` (SBPO + construção do artigo): reprodução, com cada valor ao lado do publicado.
- Outras pastas: modo extensão; os valores do artigo aparecem como referência
  (*Artigo (SBPO, construção original)*). Com `--construction parent`, há uma seção E1 comparando com a pasta
  da construção do artigo (`--baseline-dir` para outra pasta).
- Quando há seletores de E2/E3 na pasta, aparece a seção "Extensões E2 e E3" (contra Random e contra o DQN do
  artigo, GapClosed, excesso sobre o VBS e acerto da melhor centralidade na fase gulosa).

## Correspondência com o artigo

| Artigo | Implementação |
|---|---|
| 7 centralidades | `CENTRALITIES` em `main_icits.py` |
| Início proporcional à centralidade, uniforme se os pesos forem inválidos | `valid_start_weights` + `start_strategy` |
| Nível ordenado pela centralidade, LCR α = 0,3, lista e complemento embaralhados | `PARENT_ORDER = False`, `ALPHA = 0.3` |
| 15 construções por decisão | `N_CALLS = 5` × `ITER_MAX = 3` |
| Estado de 12 dimensões, recompensa densa | `EnvICITS.get_state`, `EnvICITS.reward` |
| DQN 2×128 ReLU, replay 10.000, lotes de 64, alvo a cada 25, Smooth L1, Adam 0,001, γ 0,9, clip 1,0 | `DQNAgent` |
| ε exponencial de 1 a 0,01 em T = 300; 300 treino + 60 gulosas | `DQNAgent.epsilon_at`, `TRAIN_DECISIONS`, `GREEDY_DECISIONS` |
| 237 atualizações e 9 sincronizações por execução | conferido em `runs.csv` (`Gradient Updates`, `Target Syncs`) |
| Random, fixas, mesmas 30 sementes, B0 como incumbente | `MAIN_METHODS`, `N_SEEDS`, `EnvICITS` |
| 13 instâncias HB (sem bcsstk02 e bcsstk07) | `INSTANCES` |
| RCM | `rcm_bandwidth_networkx` (reproduz 11 de 13 valores da Tabela 3) |
| Ablação 2×2 nas 10 instâncias discriminantes | `ABLATION_METHODS`, `ABLATION_INSTANCES` |
| Wilcoxon (aprox. normal, zeros removidos, sem correção), Holm, Friedman, GapClosed | `report_icits.py` |

## O que o artigo não especifica (escolhas desta implementação)

- **Parâmetros das centralidades**: Closeness, Harmonic e Betweenness exatas; Katz com α = 0,005; Eigenvector
  com até 1.000 iterações (configurações do `main_survey.py`).
- **Início pseudo-periférico**: a cada construção, um vértice sorteado de um conjunto de até 16 vértices
  pseudo-periféricos (George–Liu), como na estratégia `peripheral` do construtivo.
- **Holm da Tabela 4**: família = os dois p-valores (vs Random, vs SBS) de cada instância.
- **RCM em bp_0 e bp_200**: o desempate do RCM depende da ordem dos vértices e difere do artigo.

Os números não devem coincidir exatamente com os publicados (sequências aleatórias, bibliotecas e hardware
diferentes). O relatório mostra cada valor reproduzido ao lado do publicado.
