#include "grafo.h"
#include <cstdio>  // Necessário para usar fprintf
#include <cmath>    // para sqrt
#include <limits>   // para std::numeric_limits

Grafo::Grafo(int vertices) : V(vertices), percorrido(vertices, false) {
    adj.resize(V);
}

void Grafo::adicionarAresta(int v1, int v2) {
    adj[v1].push_back(v2);
    fprintf(stderr, "Adicionando aresta de %d para %d\n", v1, v2);
    // adj[v2].push_back(v1);
}

void Grafo::mostrarGrafo() {
    for (int i = 0; i < V; ++i) {
        // cout << "Vertice " << i << ": ";
        for (int vizinho : adj[i]) {
            cout << vizinho << " ";
        }
        cout << endl;
    }
}

void Grafo::buscaEmLargura(int inicio) {
    vector<int> distancia(V, -1);
    vector<int> predecessor(V, -1);
    queue<int> fila;

    percorrido[inicio] = true;
    distancia[inicio] = 0;
    fila.push(inicio);

    cout << "\nOrdem de visita dos vertices (BFS):\n";

    while (!fila.empty()) {
        int w = fila.front();
        fila.pop();
        cout << "Visitando vertice " << w << " (distancia = " << distancia[w] << ")\n";

        for (int u : adj[w]) {
            if (!percorrido[u]) {
                distancia[u] = distancia[w] + 1;
                fila.push(u);
                predecessor[u] = w;
                percorrido[u] = true;
            }
        }
    }

    // cout << "\nResumo final das distancias:\n";
    // for (int i = 0; i < V; i++) {
    //     // cout << "Vertice " << i << ": distancia = " << distancia[i];
    //     if (predecessor[i] != -1)
    //         cout << ", predecessor = " << predecessor[i];
    //     cout << endl;
    // }
}
int Grafo::contarArestas() const {
    int total = 0;
    for (const auto& vizinhos : adj) {
        total += vizinhos.size();
    }
    return total;
}



void Grafo::exportarParaDot(const string& nomeArquivo) {
    ofstream file(nomeArquivo);
    if (!file.is_open()) {
        cerr << "Erro ao criar arquivo DOT.\n";
        return;
    }

    file << "graph G {\n";
    for (int i = 0; i < V; ++i) {
        for (int j : adj[i]) {
            if (i < j)
                file << "    " << i << " -- " << j << ";\n";
        }
    }
    file << "}\n";
    file.close();
    cout << "Arquivo DOT gerado: " << nomeArquivo << endl;
}

void Grafo::metricasMatriz() {
    int total = 0;
    int diagonal = 0;
    int abaixoDiagonal = 0;
    int acimaDiagonal = 0;
    int assimetricas = 0;

    vector<vector<bool>> matriz(V, vector<bool>(V, false));

    for (int i = 0; i < V; ++i) {
        for (int j : adj[i]) {
            total++;
            matriz[i][j] = true;

            if (i == j)
                diagonal++;
            else if (i > j)
                abaixoDiagonal++;
            else
                acimaDiagonal++;
        }
    }

    // Contando todas as arestas assimétricas
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            bool valor1 = matriz[i][j];
            bool valor2 = matriz[j][i];
            
            if (valor1 && !valor2) {
                assimetricas++;
            }
        }
    }

    std::cout << "\nMétricas da matriz de adjacência:\n";
    std::cout << "Total de arestas: " << total << "\n";
    std::cout << "Arestas na diagonal (auto-laços): " << diagonal << "\n";
    std::cout << "Arestas abaixo da diagonal: " << abaixoDiagonal << "\n";
    std::cout << "Arestas acima da diagonal: " << acimaDiagonal << "\n";
    std::cout << "A - A' (arestas assimétricas): " << assimetricas << "\n";
}



void Grafo::calcularLarguraDeBanda() {
    int menor = std::numeric_limits<int>::max();
    int maior = 0;
    double soma = 0.0;
    std::vector<int> diferencas;

    for (int i = 0; i < V; ++i) {
        for (int j : adj[i]) {
            if (i == j) continue;  // ignorar diagonal

            int diff = std::abs(i - j);
            diferencas.push_back(diff);
            soma += diff;

            if (diff < menor) menor = diff;
            if (diff > maior) maior = diff;
        }
    }

    double media = diferencas.empty() ? 0.0 : soma / diferencas.size();

    double somaQuadrados = 0.0;
    for (int d : diferencas) {
        somaQuadrados += std::pow(d - media, 2);
    }

    double desvio = diferencas.empty() ? 0.0 : std::sqrt(somaQuadrados / diferencas.size());

    std::cout << "\n==== Informações de Largura de Banda ====\n";
    std::cout << "Lower: " << menor << "\n";
    std::cout << "Upper: " << maior << "\n";
    std::cout << "Average |i - j|: " << media << "\n";
    std::cout << "Standard deviation: " << desvio << "\n";
}
