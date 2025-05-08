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

std::vector<std::vector<int>> Grafo::buscaEmLarguraNivel(int v) const {
    // Vetor para marcar os vertices visitados localmente.
    std::vector<bool> visitado(V, false);
    // Vetor para armazenar a distância (ou nivel) de cada vertice a partir de v.
    std::vector<int> distancia(V, -1);
    // Fila para implementar a busca em largura.
    std::queue<int> fila;
    // Estrutura que armazenará os niveis; cada nivel e representado por um vetor de vertices.
    std::vector<std::vector<int>> niveis;

    // Inicializando com o vertice de partida:
    visitado[v] = true;
    distancia[v] = 0;
    fila.push(v);

    // Enquanto houver vertices para processar:
    while (!fila.empty()) {
        // Determina a quantidade de vertices no nivel atual.
        int levelSize = fila.size();
        std::vector<int> nivelAtual;
        
        // Processa todos os vertices que pertencem ao nivel corrente.
        for (int i = 0; i < levelSize; i++) {
            int w = fila.front();
            fila.pop();
            nivelAtual.push_back(w);
            
            // Para cada vizinho de w, se nao visitado, marca-o e coloca-o na fila.
            for (int u : adj[w]) {
                if (!visitado[u]) {
                    visitado[u] = true;
                    distancia[u] = distancia[w] + 1;
                    fila.push(u);
                }
            }
        }
        
        // Adiciona o nivel atual à estrutura de niveis.
        niveis.push_back(nivelAtual);
    }
    
    return niveis;
}

// Implementaçao em Class_Graph.cpp
std::vector<int> Grafo::filtrarVerticesGrauMinimo(const std::vector<int>& vertices) const {
    std::vector<int> resultado;
    if (vertices.empty()) {
        return resultado;
    }
    
    // Primeiro, determine o grau minimo entre os vertices
    int minGrau = calcularGrau(vertices[0]);
    for (int v : vertices) {
        int grauAtual = calcularGrau(v);
        if (grauAtual < minGrau) {
            minGrau = grauAtual;
        }
    }
    
    // Filtra os vertices cujo grau e igual ao grau minimo
    for (int v : vertices) {
        if (calcularGrau(v) == minGrau) {
            resultado.push_back(v);
        }
    }
    
    return resultado;
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
    std::vector<int> S(V, 0); // Vetor para armazenar a nova ordem dos vertices (inicia com zeros)
    std::vector<int> order; // Vetor para manter a ordem de visita dos vertices
    std::vector<bool> percorrido(V, false); // Vetor para marcar os vertices já visitados

    S[start] = 1; // Define a posiçao do vertice inicial 'start' como 1 na nova ordem
    order.push_back(start); // Adiciona o vertice inicial à ordem de visita
    percorrido[start] = true; // Marca o vertice inicial como visitado

    int i = 1, j = 0; // 'i' rastreia a próxima posiçao disponivel em 'S', 'j' rastreia o próximo vertice a ser processado em 'order'

    while (i < V && j < static_cast<int>(order.size())) { // Enquanto houver vertices nao ordenados e vertices a serem processados
        int u = order[j++]; // Obtem o próximo vertice a ser processado e incrementa 'j'
        std::vector<int> vizinhosNaoRotulados; // Vetor para armazenar os vizinhos nao visitados de 'u'
        std::cout << "Processando o vertice: " << u << std::endl;
        for (int w : adj[u]) { // Para cada vizinho 'w' de 'u'
            
            if (!percorrido[w]) {
                // std::cout << w << std::endl;
                vizinhosNaoRotulados.push_back(w);
             } // Se 'w' nao foi visitado, adiciona-o à lista de vizinhos nao visitados
        }

        std::sort(vizinhosNaoRotulados.begin(), vizinhosNaoRotulados.end(),
        [this](int a, int b) -> bool {
            // Imprime os valores do tamanho das listas de adjacência dos vertices a e b.
            std::cout << "Comparando vertice " << a << " (tamanho: " 
                      << adj[a].size() << ") com vertice " << b 
                      << " (tamanho: " << adj[b].size() << ")" << std::endl;
            return adj[a].size() < adj[b].size();
        });
    
                  // Para exibir a lista completa no terminal:
        std::cout << "Vizinhos nao rotulados ordenados: ";
        for (int viz : vizinhosNaoRotulados) {
            std::cout << viz << " ";
        }
        std::cout << std::endl;

        for (int w : vizinhosNaoRotulados) { // Para cada vizinho nao visitado 'w'
            if (!percorrido[w]) { // Se 'w' ainda nao foi visitado
                percorrido[w] = true; // Marca 'w' como visitado
                S[w] = ++i; // Define a posiçao de 'w' na nova ordem e incrementa 'i'
                order.push_back(w); // Adiciona 'w' à ordem de visita
            }
        }
    }
    return S; // Retorna o vetor 'S' contendo a nova ordem dos vertices
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

// Metodo para calcular o grau de um vertice
int Grafo::calcularGrau(int v) const {
    if (v < 0 || v >= V) {
        throw std::out_of_range("Vertice inválido");
    }
    return adj[v].size();
}

// Metodo para encontrar o vertice de grau minimo
int Grafo::verticeGrauMinimo(const std::vector<int>& vertices) const {
    if (vertices.empty()) {
        throw std::invalid_argument("A lista de vertices está vazia");
    }

    int verticeMin = vertices[0];
    int grauMin = calcularGrau(verticeMin);

    for (int v : vertices) {
        int grauAtual = calcularGrau(v);
        if (grauAtual < grauMin) {
            grauMin = grauAtual;
            verticeMin = v;
        }
    }

    return verticeMin;
}

int Grafo::GeorgeLiu(int v) const {
    // Calcula a estrutura de niveis (BFS) a partir de v
    std::vector<std::vector<int>> Lv = buscaEmLarguraNivel(v);

    // Declara 'verticesMinimos' fora do if para que possa ser usado posteriormente
    std::vector<int> verticesMinimos;
    if (!Lv.empty()) {
        // Filtra o último nivel para obter os vertices com o menor grau
        verticesMinimos = filtrarVerticesGrauMinimo(Lv.back());
        
        // Exibe os vertices filtrados
        std::cout << "Vertices do último nivel com grau minimo:" << std::endl;
        for (int candidate : verticesMinimos) {
            std::cout << "Vertice " << candidate << " (grau: " << calcularGrau(candidate) << ")" << std::endl;
        }
    }
    
    int u;
    do {
        // Escolhe o vertice com o menor grau dentre os candidatos filtrados
        u = verticeGrauMinimo(verticesMinimos);
        // Calcula a nova estrutura de niveis a partir do candidato u
        std::vector<std::vector<int>> Lu = buscaEmLarguraNivel(u);
        
        // Verifica se a nova estrutura possui mais niveis que a atual, usando (por exemplo) o tamanho do nivel u comparado com o de v.
        // OBS.: Certifique-se de que a forma de indexar Lv com u e v está de acordo com o seu algoritmo,
        // pois Lv e um vetor de niveis (cada nivel e um vetor) e nao necessariamente indexado pelo id do vertice.
        if (Lv[u].size() > Lv[v].size()){
            std::cout << "Nivel: " << u << " Size: " << Lv[u].size() << std::endl;
            std::cout << "Nivel: " << v << " Size: " << Lv[v].size() << std::endl;
            v = u;
            Lv = Lu;
            // Atualiza o conjunto de candidatos com base no novo Lv (por exemplo, filtrando o último nivel)
            if (!Lv.empty()) {
                verticesMinimos = filtrarVerticesGrauMinimo(Lv.back());
            }
        }
    } while (u != v);  // Use '!=' para comparaçao, nao '='
    
    return v;
}
