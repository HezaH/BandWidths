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

Grafo::Grafo(int vertices) : V(vertices), percorrido(vertices, false) {
    adj.resize(V);
}

void Grafo::adicionarAresta(int v1, int v2) {
    adj[v1].push_back(v2);
    // fprintf(stderr, "Adicionando aresta de %d para %d\n", v1, v2);
    // adj[v2].push_back(v1);
}

void Grafo::buscaEmLargura(int inicio) {
    vector<int> distancia(V, -1);
    vector<int> predecessor(V, -1);
    queue<int> fila;

    percorrido[inicio] = true;
    distancia[inicio] = 0;
    fila.push(inicio);

    // cout << "\nOrdem de visita dos vertices (BFS):\n";

    while (!fila.empty()) {
        int w = fila.front();
        fila.pop();
        // cout << "Visitando vertice " << w << " (distancia = " << distancia[w] << ")\n";

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