// Algoritmo 3: Cuthill-McKee.
// Entrada: grafo G =(V,E) conexo; vertice inicial v ∈ V;
// Saida: renumeracao S com |V| entradas;

#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <algorithm> // Necessário para sort

using namespace std;

class Grafo {
private:
    int V; // Número de vertices
    vector<vector<int>> adj; // Lista de adjacência (guarda os vertices conectados a cada vertice)
    vector<bool> percorrido; // Marca se um vertice foi visitado (rotulado)

public:
    // Construtor: Inicializa "percorrido" como false para todos os vertices
    Grafo(int vertices) : V(vertices), percorrido(vertices, false) {
        adj.resize(V); // Ajusta o tamanho da lista de adjacência
    }

    // Metodo para adicionar uma aresta entre dois vertices (grafo nao direcionado)
    void adicionarAresta(int v1, int v2) {
        adj[v1].push_back(v2);
        adj[v2].push_back(v1);
    }

    // Exibe o grafo mostrando as listas de adjacência de cada vertice
    void mostrarGrafo() {
        for (int i = 0; i < V; ++i) {
            cout << "Vertice " << i << ": ";
            for (int vizinho : adj[i]) {
                cout << vizinho << " ";
            }
            cout << endl;
        }
    }

    // Implementa o algoritmo Cuthill-McKee para reordenar os vertices
    void CutHill_McKee(int start) {
        vector<int> S(V, 0); // Vetor S: guarda a nova numeraçao dos vertices (inicialmente 0, nao numerado)
        vector<int> order; // Vetor "order": guarda a sequência de vertices conforme a numeraçao atribuida

        // Define o vertice inicial com rótulo 1
        S[start] = 1;
        order.push_back(start);
        percorrido[start] = true;  // Marca o vertice inicial como visitado

        int i = 1;  // Quantidade de vertices numerados ate agora
        int j = 0;  // indice na lista "order" para processar os vertices

        // Laço principal do algoritmo: enquanto houver vertices para numerar
        while (i < V && j < (int)order.size()) {
            int u = order[j];  // Pega o próximo vertice da sequência para processar

            // Coleta os vertices vizinhos ainda nao rotulados
            vector<int> vizinhosNaoRotulados;
            for (int w : adj[u]) {
                if (!percorrido[w])
                    vizinhosNaoRotulados.push_back(w);
            }

            // Ordena os vertices vizinhos por grau crescente (menor grau primeiro)
            sort(vizinhosNaoRotulados.begin(), vizinhosNaoRotulados.end(),
                 [this](int a, int b) {
                     return adj[a].size() < adj[b].size();
                 });

            // Atribui rótulos aos vertices vizinhos
            for (int w : vizinhosNaoRotulados) {
                if (!percorrido[w]) {
                    percorrido[w] = true;  // Marca como rotulado
                    i++;                   // Incrementa o contador de vertices numerados
                    S[w] = i;              // Atribui o próximo rótulo
                    order.push_back(w);    // Adiciona w à sequência de processamento
                }
            }
            j++;  // Passa para o próximo vertice na sequência
        }

        // Exibe a renumeraçao final dos vertices
        cout << "\nRenumeracao (S):" << endl;
        for (int v = 0; v < V; v++) {
            cout << "Vertice " << v << " recebe rotulo " << S[v] << endl;
        }

        // Exibe tambem a ordem dos vertices (a funçao inversa da renumeraçao)
        cout << "\nOrdem dos vertices (s^-1):" << endl;
        for (int vertice : order) {
            cout << vertice << " ";
        }
        cout << "\n";
    }
};

// Gera um grafo conexo aleatório com "numVertices" vertices
Grafo gerarGrafoConexo(int numVertices) {
    Grafo g(numVertices);
    srand(time(0)); // Inicializa a semente para gerar valores aleatórios

    // Conecta os vertices em cadeia para garantir que o grafo e conexo
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
    int numVertices = 6; // Define o número de vertices do grafo
    Grafo g = gerarGrafoConexo(numVertices); // Gera um grafo conexo
    
    cout << "Grafo gerado:\n";
    g.mostrarGrafo(); // Exibe o grafo gerado
    
    // Executa o algoritmo de Cuthill-McKee a partir do vertice 0
    g.CutHill_McKee(0);
    
    return 0;
}
