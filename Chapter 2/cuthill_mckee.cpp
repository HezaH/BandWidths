// Algoritmo 3: Cuthill-McKee.
// Entrada: grafo G =(V,E) conexo; vertice inicial v ∈ V;
// Saída: renumeracao S com |V| entradas;

#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <algorithm> // Necessário para sort

using namespace std;

class Grafo {
private:
    int V; // Número de vértices
    vector<vector<int>> adj; // Lista de adjacência (guarda os vértices conectados a cada vértice)
    vector<bool> percorrido; // Marca se um vértice foi visitado (rotulado)

public:
    // Construtor: Inicializa "percorrido" como false para todos os vértices
    Grafo(int vertices) : V(vertices), percorrido(vertices, false) {
        adj.resize(V); // Ajusta o tamanho da lista de adjacência
    }

    // Método para adicionar uma aresta entre dois vértices (grafo não direcionado)
    void adicionarAresta(int v1, int v2) {
        adj[v1].push_back(v2);
        adj[v2].push_back(v1);
    }

    // Exibe o grafo mostrando as listas de adjacência de cada vértice
    void mostrarGrafo() {
        for (int i = 0; i < V; ++i) {
            cout << "Vertice " << i << ": ";
            for (int vizinho : adj[i]) {
                cout << vizinho << " ";
            }
            cout << endl;
        }
    }

    // Implementa o algoritmo Cuthill-McKee para reordenar os vértices
    void CutHill_McKee(int start) {
        vector<int> S(V, 0); // Vetor S: guarda a nova numeração dos vértices (inicialmente 0, não numerado)
        vector<int> order; // Vetor "order": guarda a sequência de vértices conforme a numeração atribuída

        // Define o vértice inicial com rótulo 1
        S[start] = 1;
        order.push_back(start);
        percorrido[start] = true;  // Marca o vértice inicial como visitado

        int i = 1;  // Quantidade de vértices numerados até agora
        int j = 0;  // Índice na lista "order" para processar os vértices

        // Laço principal do algoritmo: enquanto houver vértices para numerar
        while (i < V && j < (int)order.size()) {
            int u = order[j];  // Pega o próximo vértice da sequência para processar

            // Coleta os vértices vizinhos ainda não rotulados
            vector<int> vizinhosNaoRotulados;
            for (int w : adj[u]) {
                if (!percorrido[w])
                    vizinhosNaoRotulados.push_back(w);
            }

            // Ordena os vértices vizinhos por grau crescente (menor grau primeiro)
            sort(vizinhosNaoRotulados.begin(), vizinhosNaoRotulados.end(),
                 [this](int a, int b) {
                     return adj[a].size() < adj[b].size();
                 });

            // Atribui rótulos aos vértices vizinhos
            for (int w : vizinhosNaoRotulados) {
                if (!percorrido[w]) {
                    percorrido[w] = true;  // Marca como rotulado
                    i++;                   // Incrementa o contador de vértices numerados
                    S[w] = i;              // Atribui o próximo rótulo
                    order.push_back(w);    // Adiciona w à sequência de processamento
                }
            }
            j++;  // Passa para o próximo vértice na sequência
        }

        // Exibe a renumeração final dos vértices
        cout << "\nRenumeracao (S):" << endl;
        for (int v = 0; v < V; v++) {
            cout << "Vertice " << v << " recebe rotulo " << S[v] << endl;
        }

        // Exibe também a ordem dos vértices (a função inversa da renumeração)
        cout << "\nOrdem dos vertices (s^-1):" << endl;
        for (int vertice : order) {
            cout << vertice << " ";
        }
        cout << "\n";
    }
};

// Gera um grafo conexo aleatório com "numVertices" vértices
Grafo gerarGrafoConexo(int numVertices) {
    Grafo g(numVertices);
    srand(time(0)); // Inicializa a semente para gerar valores aleatórios

    // Conecta os vértices em cadeia para garantir que o grafo é conexo
    for (int i = 0; i < numVertices - 1; ++i) {
        g.adicionarAresta(i, i + 1);
    }

    // Adiciona algumas arestas aleatórias para tornar o grafo mais denso
    for (int i = 0; i < numVertices / 2; ++i) {
        int v1 = rand() % numVertices;
        int v2 = rand() % numVertices;
        if (v1 != v2) {
            g.adicionarAresta(v1, v2);
        }
    }
    return g;
}

int main() {
    int numVertices = 6; // Define o número de vértices do grafo
    Grafo g = gerarGrafoConexo(numVertices); // Gera um grafo conexo
    
    cout << "Grafo gerado:\n";
    g.mostrarGrafo(); // Exibe o grafo gerado
    
    // Executa o algoritmo de Cuthill-McKee a partir do vértice 0
    g.CutHill_McKee(0);
    
    return 0;
}
