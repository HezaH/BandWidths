#include "grafo.h"
#include <cstdio>  // Necessário para usar fprintf
#include <cmath>    // para sqrt
#include <limits>   // para std::numeric_limits
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>
#include <algorithm>

Grafo::Grafo(int vertices) : V(vertices), percorrido(vertices, false) {
    adj.resize(V);
}

void Grafo::adicionarAresta(int v1, int v2) {
    adj[v1].push_back(v2);
}

void Grafo::buscaEmLargura(int v) {
    vector<int> distancia(V, -1);
    vector<int> predecessor(V, -1);
    queue<int> fila;

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
    // Inicializa uma matriz V×V com zeros
    std::vector<std::vector<int>> M(V, std::vector<int>(V, 0));

    // Para cada vértice i, marca 1 nas colunas j adjacentes
    for (int i = 0; i < V; ++i) {
        for (int j : adj[i]) {
            M[i][j] = 1;
        }
    }

    return M;
}

void Grafo::exportarMatrizAdjComoJPEG(const std::string &filename, int quality) const {
    auto M = gerarMatrizAdjacencia();
    // Buffer de pixels em escala de cinza
    std::vector<unsigned char> buffer(V * V);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            buffer[i * V + j] = M[i][j] ? 0 : 255;
        }
    }

    // Escreve a imagem JPEG usando stb_image_write
    if (!stbi_write_jpg(filename.c_str(), V, V, 1, buffer.data(), quality)) {
        throw std::runtime_error("Falha ao escrever a imagem JPEG");
    }
}

std::vector<int> Grafo::Cuthill_McKee(int start) {
    std::vector<int> S(V, 0); // Vetor S: nova numeração dos vértices
    std::vector<int> order;   // Sequência de vértices conforme a numeração
    std::vector<bool> percorrido(V, false); // Vetor para marcar vértices visitados

    S[start] = 1;
    order.push_back(start);
    percorrido[start] = true;

    int i = 1;  // Quantidade de vértices numerados
    int j = 0;  // Índice na lista "order"

    while (i < V && j < static_cast<int>(order.size())) {
        int u = order[j];

        // Coleta os vizinhos não rotulados
        std::vector<int> vizinhosNaoRotulados;
        for (int w : adj[u]) {
            if (!percorrido[w])
                vizinhosNaoRotulados.push_back(w);
        }

        // Ordena os vizinhos por grau crescente
        std::sort(vizinhosNaoRotulados.begin(), vizinhosNaoRotulados.end(),
                  [this](int a, int b) {
                      return adj[a].size() < adj[b].size();
                  });

        // Atribui rótulos aos vizinhos
        for (int w : vizinhosNaoRotulados) {
            if (!percorrido[w]) {
                percorrido[w] = true;
                i++;
                S[w] = i;
                order.push_back(w);
            }
        }
        j++;
    }
    return S;
}

// Remapeia a lista de adjacência segundo a nova numeração S
// S é 1-based: se S[v] = j, o vértice original 'v' agora terá índice j-1.
std::vector<std::vector<int>> Grafo::reordenarGrafo(const std::vector<int>& S) const {
    std::vector<std::vector<int>> newAdj(V, std::vector<int>());
    
    // Cria o vetor inverso para S: se S[v] = j, então o vértice original v ocupa a posição (j-1) na nova ordem.
    std::vector<int> inv(V); 
    for (int v = 0; v < V; ++v) {
        int new_idx = S[v] - 1;
        inv[new_idx] = v;
    }
    
    // Para cada novo índice v (0 a V-1), identifique o vértice original e remapeie suas arestas.
    for (int new_v = 0; new_v < V; ++new_v) {
        int old_v = inv[new_v];  // vértice original que foi reordenado para a posição new_v
        for (int w : adj[old_v]) {
            int new_w = S[w] - 1;
            newAdj[new_v].push_back(new_w);
        }
        // Opcional: ordena a lista de vizinhos para visualização
        std::sort(newAdj[new_v].begin(), newAdj[new_v].end());
    }
    return newAdj;
}

// Permite atualizar a lista de adjacência do grafo com a nova ordem.
void Grafo::setAdjacencias(const std::vector<std::vector<int>>& newAdj) {
    if (newAdj.size() != static_cast<size_t>(V)) {
        throw std::runtime_error("Tamanho inadequado para as adjacências");
    }
    adj = newAdj;
}




