

// Algoritmo 2: Busca em largura.
// Entrada: grafo G =(V,E) nao direcionado e conexo; vertice v ∈ V inicial;

// Saida: determina a distancia de todos os vertices a partir do vertice v ∈V;
#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>

using namespace std;

class Grafo {
private:
    int V; // Número de vertices
    vector<vector<int>> adj; // Lista de adjacência
    vector<bool> percorrido; // Marca se um vertice foi visitado na busca

public:
    // Construtor: Inicializa percorrido como false para todos os vertices
    Grafo(int vertices) : V(vertices), percorrido(vertices, false) {
        adj.resize(V);
    }

    void adicionarAresta(int v1, int v2) {
        adj[v1].push_back(v2);
        adj[v2].push_back(v1); // Grafo nao direcionado
    }

    void mostrarGrafo() {
        for (int i = 0; i < V; ++i) {
            cout << "Vertice " << i << ": ";
            for (int vizinho : adj[i]) {
                cout << vizinho << " ";
            }
            cout << endl;
        }
    }

    // Busca em Largura (BFS)
    void buscaEmLargura(int inicio) {
        vector<int> distancia(V, -1); // Inicializa distancias como -1 (nao alcançados)
        vector<int> predecessor(V, -1); // -1 significa que ainda nao foi rotulado
        queue<int> fila;

        // Marca o vertice inicial como percorrido
        percorrido[inicio] = true;
        distancia[inicio] = 0;
        fila.push(inicio);

        while (!fila.empty()) {
            int w = fila.front(); // Atribui o vertice da frente da fila a v
            fila.pop(); // Remove o vertice da frente da fila w

            // Visita os vizinhos do vertice atual
            for (int u : adj[w]) {

                if (!percorrido[u]) {//se o vertice u nao foi percorrido
                    distancia[u] = distancia[w] + 1;
                    fila.push(u);
                    predecessor[u] = w;  // Rótulo: quem descobriu u
                    percorrido[u] = true;
                }
            }
        }

        // Exibir distancias
        cout << "\nDistancias a partir do vertice " << inicio << ":\n";
        for (int i = 0; i < V; i++) {
            cout << "Vertice " << i << " -> Distancia: " << distancia[i] << endl;
        }
    }
};

// Gera um grafo conexo aleatório
Grafo gerarGrafoConexo(int numVertices) {
    Grafo g(numVertices);
    srand(time(0));

    // Garante a conexao minima entre os vertices formando uma cadeia
    for (int i = 0; i < numVertices - 1; ++i) {
        g.adicionarAresta(i, i + 1);
    }

    // Adiciona arestas aleatórias para tornar o grafo mais denso
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
    int numVertices = 6;
    Grafo g = gerarGrafoConexo(numVertices);
    
    cout << "Grafo gerado:\n";
    g.mostrarGrafo();
    
    // Executar BFS a partir do vertice 0
    g.buscaEmLargura(0);

    return 0;
}
