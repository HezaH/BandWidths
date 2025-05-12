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

// 1. Estrutura de nivel enraizada via BFS.
// Retorna um vetor de niveis: cada nivel e um vetor com os vertices daquela distância da raiz.
std::vector<std::vector<int>> Grafo:: buscaEmLarguraNivel(int v) const {
    std::vector<bool> visitado(V, false);
    std::vector<int> distancia(V, -1);
    std::queue<int> fila;
    std::vector<std::vector<int>> niveis;

    // Inicializa o vertice de partida
    visitado[v] = true;
    distancia[v] = 0;
    fila.push(v);

    while (!fila.empty()) {
        int levelSize = fila.size();
        std::vector<int> nivelAtual;
        for (int i = 0; i < levelSize; i++) {
            int w = fila.front();
            fila.pop();
            nivelAtual.push_back(w);
            // Para cada vizinho de w, se não visitado, marca e enfileira
            for (int u : adj[w]) {
                if (!visitado[u]) {
                    visitado[u] = true;
                    distancia[u] = distancia[w] + 1;
                    fila.push(u);
                }
            }
        }
        niveis.push_back(nivelAtual);
    }
    return niveis;
}

// 2. Filtro para extrair, de um conjunto de vertices, aqueles com grau minimo.
std::vector<int> Grafo::filtrarVerticesGrauMinimo(const std::vector<int>& vertices) const {
    std::vector<int> resultado;
    if (vertices.empty()) {
        return resultado;
    }
    
    int minGrau = calcularGrau(vertices[0]);
    // Determina o grau minimo entre os vertices fornecidos
    for (int v = 0; v < V; ++v) {
        int grauAtual = calcularGrau(v);
        std::cout << "Grau do vertice " << v << ": " << grauAtual << std::endl;
        if (grauAtual < minGrau) {
            minGrau = grauAtual;
        }
    }
    // Filtra os vertices que possuem exatamente esse grau minimo
    for (int v = 0; v < V; ++v) {
        if (calcularGrau(v) == minGrau) {
            std::cout << "Vertice com grau minimo: " << v << std::endl;
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
int Grafo:: calcularGrau(int v) const {
    if (v < 0 || v >= V) {
        throw std::out_of_range("Vertice inválido");
    }
    return adj[v].size();
}

int Grafo::GeorgeLiu(int v) const {
    std::vector<std::vector<int>> Lv = buscaEmLarguraNivel(v); //Estrutura de nivel enraizada
    int currentLevels = Lv.size();
    int u;
   
    // Enquanto houver candidatos que possam melhorar a profundidade da estrutura
    do {
        // Pega o último nivel da estrutura
        const std::vector<int>& lastLevel = Lv.back();
        std::cout << "Ultimo nivel: ";
        for (int l : lastLevel) {
            std::cout << l << " ";
        }
        std::cout << std::endl;
        
        // u = v;

        // Filtra os vertices do último nivel com o menor grau
        std::vector<int> candidatos = filtrarVerticesGrauMinimo(lastLevel);
        std::cout << "Candidatos: ";
        for (int c : candidatos) {
            std::cout << c << " ";
        }
        std::cout << std::endl;

        // Calcula a nova estrutura de niveis a partir de u
        u = candidatos[0];
        std::vector<std::vector<int>> Lu = buscaEmLarguraNivel(u);
        int newLevels = Lu.size();

        // Debug: exibe a comparação entre estruturas
        std::cout << "Comparando: candidato u = " << u
                  << " (niveis = " << newLevels << ") vs. v = " << v
                  << " (niveis = " << currentLevels << ")" << std::endl;
        
        // Se a nova estrutura (a partir de u) for mais profunda, atualiza o candidato
        if (newLevels > currentLevels) {
            v = u;
            Lv = Lu;
            currentLevels = newLevels;
        } else {
            // Se não houver melhoria, encerra o loop
            break;
        }
    } while (u=v);
    return v;
}

// Metodo que retorna os vertices em ordem crescente de grau.
std::vector<int> Grafo::ordenarVerticesPorGrau() const {
    // Cria um vetor com todos os vertices do grafo.
    // Supondo que 'V' e o número total de vertices do grafo:
    std::vector<int> vertices;
    for (int i = 0; i < V; ++i) {
        vertices.push_back(i);
    }

    // Ordena o vetor 'vertices' em ordem crescente, de acordo com o grau de cada vertice.
    std::sort(vertices.begin(), vertices.end(), [this](int a, int b) {
        return calcularGrau(a) < calcularGrau(b);
    });

    // Exibe o resultado:
    std::cout << "Vertices ordenados por grau (crescente):\n";
    for (int v : vertices) {
        std::cout << "Vertice " << v << " (grau = " << calcularGrau(v) << ")\n";
    }
    
    return vertices;
}


int Grafo::VerticePseudoPeriferico_GPS() const {
    int v;
    int w;
    std::vector<int> S(V, 0); // Vetor para armazenar a nova ordem dos vertices (inicia com zeros)
    v = verticeGrauMinimo(S);
    std::cout << "Vertice v: " << v << std::endl;
    std::vector<std::vector<int>> Lv = buscaEmLarguraNivel(v);
    int currentLevels = Lv.size();

    do{
        std::vector<int> verticesOrdenados = ordenarVerticesPorGrau();
        
        do{
            w = verticesOrdenados[0];
            std::cout << "Vertice w: " << w << std::endl;
            // Remove o primeiro vertice da lista de candidatos
            verticesOrdenados.erase(verticesOrdenados.begin());

            std::vector<std::vector<int>> Lw = buscaEmLarguraNivel(w);
            int newLevels = Lw.size();

            // Debug: exibe a comparação entre estruturas
                std::cout << "Comparando: candidato w = " << w
                << " (niveis = " << newLevels << ") vs. v = " << v
                << " (niveis = " << currentLevels << ")" << std::endl;
            // Se a nova estrutura (a partir de u) for mais profunda, atualiza o candidato
            if (newLevels > currentLevels) {
                v = w;
                Lv = Lw;
                currentLevels = newLevels;
            } else {
                // Se não houver melhoria, encerra o loop
                break;
            }

        }
       while((v != w) && !verticesOrdenados.empty());
        
    }while((v != w));
    
    return v;
}

int Grafo::verticeGrauMinimo(const std::vector<int>& vertices) const {
    if (vertices.empty()) {
        throw std::invalid_argument("A lista de vertices está vazia");
    }
    
    int verticeMin = vertices[0];
    int grauMin = calcularGrau(vertices[0]);
    
    for (int v = 0; v < V; ++v) {
        
        int grauAtual = calcularGrau(v);
        std::cout << "Vertice " << v << " Grau " << grauAtual << ::endl;
        if (grauAtual < grauMin) {
            grauMin = grauAtual;
            verticeMin = v;
        }
    }
    
    return verticeMin;
}
