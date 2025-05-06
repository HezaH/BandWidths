#include "grafo.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <stdexcept>
#include <string>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Grafo::Grafo(int vertices) : V(vertices), percorrido(vertices, false) {
    adj.resize(V);
}

void Grafo::adicionarAresta(int v1, int v2) {
    adj[v1].push_back(v2);
}

void Grafo::buscaEmLargura(int v) {
    std::vector<int> distancia(V, -1), predecessor(V, -1);
    std::queue<int> fila;

    percorrido[v] = true;
    distancia[v] = 0;
    fila.push(v);

    while (!fila.empty()) {
        int w = fila.front();
        fila.pop();

        for (int u : adj[w]) {
            if (!percorrido[u]) {
                fila.push(u);
                distancia[u] = distancia[w] + 1;
                predecessor[u] = w;
                percorrido[u] = true;
            }
        }
    }
}

std::vector<std::vector<int>> Grafo::gerarMatrizAdjacencia() const {
    std::vector<std::vector<int>> M(V, std::vector<int>(V, 0));
    for (int i = 0; i < V; ++i) {
        for (int j : adj[i]) {
            M[i][j] = 1;
        }
    }
    return M;
}

void Grafo::exportarMatrizAdjComoJPEG(const std::string &filename, int quality) const {
    auto M = gerarMatrizAdjacencia();
    std::vector<unsigned char> buffer(V * V);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            buffer[i * V + j] = M[i][j] ? 0 : 255;
        }
    }
    if (!stbi_write_jpg(filename.c_str(), V, V, 1, buffer.data(), quality)) {
        throw std::runtime_error("Falha ao escrever a imagem JPEG");
    }
}

std::vector<int> Grafo::Cuthill_McKee(int start) {
    std::vector<int> S(V, 0), order;
    std::vector<bool> percorrido(V, false);

    S[start] = 1;
    order.push_back(start);
    percorrido[start] = true;

    int i = 1, j = 0;

    while (i < V && j < static_cast<int>(order.size())) {
        int u = order[j++];
        std::vector<int> vizinhosNaoRotulados;

        for (int w : adj[u]) {
            if (!percorrido[w]) vizinhosNaoRotulados.push_back(w);
        }

        std::sort(vizinhosNaoRotulados.begin(), vizinhosNaoRotulados.end(),
                  [this](int a, int b) { return adj[a].size() < adj[b].size(); });

        for (int w : vizinhosNaoRotulados) {
            if (!percorrido[w]) {
                percorrido[w] = true;
                S[w] = ++i;
                order.push_back(w);
            }
        }
    }
    return S;
}

std::vector<std::vector<int>> Grafo::reordenarGrafo(const std::vector<int>& S) const {
    std::vector<std::vector<int>> newAdj(V);
    std::vector<int> inv(V);

    for (int v = 0; v < V; ++v) {
        inv[S[v] - 1] = v;
    }

    for (int new_v = 0; new_v < V; ++new_v) {
        int old_v = inv[new_v];
        for (int w : adj[old_v]) {
            newAdj[new_v].push_back(S[w] - 1);
        }
        std::sort(newAdj[new_v].begin(), newAdj[new_v].end());
    }
    return newAdj;
}

void Grafo::setAdjacencias(const std::vector<std::vector<int>>& newAdj) {
    if (newAdj.size() != static_cast<size_t>(V)) {
        throw std::runtime_error("Tamanho inadequado para as adjacências");
    }
    adj = newAdj;
}
