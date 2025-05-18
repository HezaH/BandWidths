#include "grafo.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <stdexcept>
#include <string>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <climits>

Grafo::Grafo(int vertices) : V(vertices), percorrido(vertices, false) {
    adj.resize(V);
}

void Grafo::adicionarAresta(int v1, int v2) {
    adj[v1].push_back(v2);
}

std::vector<std::vector<int>> Grafo::buscaEmLarguraNivel(int v) const {
    vector<bool> visitado(V, false);              // Marca os vertices visitados
    queue<int> fila;                              // Fila para a BFS
    vector<vector<int>> niveis;                     // Armazena os niveis
     
    // Inicializa com o vertice de partida
    visitado[v] = true;
    fila.push(v);
    
    // Enquanto houver vertices na fila...
    while (!fila.empty()) {
        int levelSize = fila.size();              // Número de vertices para o nivel atual
        vector<int> nivelAtual;                   // Vetor para armazenar os vertices deste nivel
        
        // Processa exatamente os vertices do nivel atual
        for (int i = 0; i < levelSize; i++) {
            int atual = fila.front();
            fila.pop();
            nivelAtual.push_back(atual);
            
            // cout << "Percurso pelo vertice: " << atual << endl;
            
            // Enfileira os vizinhos não visitados deste vertice
            for (int vizinho : adj[atual]) {
                if (!visitado[vizinho]) {
                    visitado[vizinho] = true;
                    fila.push(vizinho);
                    // cout << vizinho << " ";
                }
            }
            // cout << endl;
        }
        
        // Armazena o nivel atual na estrutura de niveis
        niveis.push_back(nivelAtual);
        // cout << endl;
    }
    
    for (size_t i = 0; i < niveis.size(); i++) {
        cout << "Nivel " << i << ": ";
        for (int vert : niveis[i]) {
            cout << vert << " ";
        }
        cout << endl;
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
    std::vector<bool> percorrido(V, false); // Vetor para marcar os vertices ja visitados

    S[start] = 1; // Define a posicao do vertice inicial 'start' como 1 na nova ordem
    order.push_back(start); // Adiciona o vertice inicial à ordem de visita
    percorrido[start] = true; // Marca o vertice inicial como visitado

    int i = 1, j = 0; // 'i' rastreia a próxima posicao disponivel em 'S', 'j' rastreia o próximo vertice a ser processado em 'order'

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
            // Imprime os valores do tamanho das listas de adjacencia dos vertices a e b.
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
                S[w] = ++i; // Define a posicao de 'w' na nova ordem e incrementa 'i'
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
        throw std::runtime_error("Tamanho inadequado para as adjacencias");
    }
    adj = newAdj;
}

// Metodo para calcular o grau de um vertice
int Grafo:: calcularGrau(int v) const {
    if (v < 0 || v >= V) {
        throw std::out_of_range("Vertice invalido");
    }
    return adj[v].size();
}

int Grafo::GeorgeLiu(int v) const {
    std::vector<std::vector<int>> Lv = buscaEmLarguraNivel(v); //Estrutura de nivel enraizada
    size_t n1 = 1;
    int currentLevels = Lv[n1].size();
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

        // Filtra os vertices do último nivel com o menor grau
        int minGrau = INT_MAX;
        int minVer = -1;
        // Determina o grau minimo entre os vertices fornecidos
        for (int l :lastLevel) {
            int grauAtual = adj[l].size();
            std::cout << "Grau do vertice " << l << ": " << grauAtual << std::endl;
            if (grauAtual < minGrau) {
                minGrau = grauAtual;
                minVer = l;
            }
        }

        // Calcula a nova estrutura de niveis a partir de u
        u = minVer;
        std::vector<std::vector<int>> Lu = buscaEmLarguraNivel(u);
        int newLevels = Lu[n1].size();

        // Debug: exibe a comparacao entre estruturas
        std::cout << "Comparando: candidato u = " << u
                  << " (niveis = " << newLevels << ") vs. v = " << v
                  << " (niveis = " << currentLevels << ")" << std::endl;
        
        // Se a nova estrutura (a partir de u) for mais profunda, atualiza o candidato
        if (newLevels > currentLevels) {
            v = u;
            Lv = Lu;
            currentLevels = newLevels;
        } 
        else {
            // Se nao houver melhoria, encerra o loop
            break;
        }
    } while (u==v);
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

            // Debug: exibe a comparacao entre estruturas
                std::cout << "Comparando: candidato w = " << w
                << " (niveis = " << newLevels << ") vs. v = " << v
                << " (niveis = " << currentLevels << ")" << std::endl;
            // Se a nova estrutura (a partir de u) for mais profunda, atualiza o candidato
            if (newLevels > currentLevels) {
                v = w;
                Lv = Lw;
                currentLevels = newLevels;
            } else {
                // Se nao houver melhoria, encerra o loop
                break;
            }

        }
       while((v != w) && !verticesOrdenados.empty());
        
    }while((v != w));
    
    return v;
}

int Grafo::verticeGrauMinimo(const std::vector<int>& vertices) const {
    if (vertices.empty()) {
        throw std::invalid_argument("A lista de vertices esta vazia");
    }
    
    int verticeMin = vertices[0];
    int grauMin = adj[vertices[0]].size();
    
    for (int v = 0; v < V; ++v) {
        
        int grauAtual = adj[v].size();
        std::cout << "Vertice " << v << " Grau " << grauAtual << ::endl;
        if (grauAtual < grauMin) {
            grauMin = grauAtual;
            verticeMin = v;
        }
    }
    
    return verticeMin;
}

int Grafo::VerticeMenorLargura(const std::vector<int>& vertices) const {
    int u = -1;
    int largura = INT_MAX;
    // Itera pelos vertices do vetor 'vertices' (por exemplo, o último nivel da BFS)
    for (int w : vertices) {
        std::cout << "Verificando vertice: " << w << std::endl;
        std::vector<std::vector<int>> Lw = buscaEmLarguraNivel(w);
        int b_lw = algo8(Lw);
        if (b_lw < largura) {
            u = w;
            largura = b_lw;
            std::cout << "Vertice u: " << u << std::endl;
        }
    }
    return u;
}

int Grafo::algo8(std::vector<std::vector<int>> & Lw) const {
    int l = 1;
    // Aqui a funcao parece retornar o número de niveis.
    // (A lógica pode ser simplificada, mas manteremos o exemplo.)
    for (size_t i = 1; i < Lw.size(); ++i) {
        if (Lw.size() > static_cast<size_t>(l)) {
            l = Lw.size();
        }
    }
    return l;
}