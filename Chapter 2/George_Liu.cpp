//  Algoritmo 4: George-Liu.
//  Entrada: grafo G =(V,E);
//  Saida: vertice pseudoperiferico v ∈ V 

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

    // Metodo para adicionar uma aresta entre dois vertices (grafo não direcionado)
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

    // Busca em Largura (BFS)
    vector<int> buscaEmLargura(int inicio) {
        vector<int> distancia(V, -1); // Inicializa distancias como -1 (não alcançados)
        vector<int> predecessor(V, -1); // -1 significa que ainda não foi rotulado
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

                if (!percorrido[u]) {//se o vertice u não foi percorrido
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
        return distancia;
    }
    
    //Algoritmo 5: VerticeGrauMinimo.
    // Entrada: grafo G =(V,E); conjunto de vertices V ⊆ V;
    // Saida: vertice de grau minimo v ∈ V ;

    // Funcao para calcular o vertice de grau minimo.
    int VerticeGrauMin() {
        // Se não houver vertices, pode retornar um valor indicativo (como -1).
        if (V == 0)
            return -1;

        int minGrau = std::numeric_limits<int>::max();
        int verticeMin = 0;

        for (int i = 0; i < V; i++) {
            if (adj[i].size() < minGrau) {
                minGrau = adj[i].size();
                verticeMin = i;
            }
        }
        cout << "Vertice de grau minimo: " << verticeMin 
             << " (grau: " << minGrau << ")" << endl;
        return verticeMin;
    }

    // Funcao para calcular a estrutura de niveis enraizada a partir de um vertice "inicio".
    // A estrutura de niveis e representada como um vetor de vetores, onde
    // o i-esimo vetor contem todos os vertices cujo nivel (distância ate o "inicio") e i.
    vector<vector<int>> estruturaDeNiveis(int inicio) {
        // Inicializa vetor de distâncias; -1 indica vertice não alcançado
        vector<int> distancia(V, -1);
        // Vetor local para controle de visitacao
        vector<bool> visitado(V, false);
        queue<int> fila;

        // Inicia a busca a partir do vertice "inicio"
        visitado[inicio] = true;
        distancia[inicio] = 0;
        fila.push(inicio);

        // Executa a BFS
        while (!fila.empty()) {
            int atual = fila.front();
            fila.pop();

            // Para cada vizinho não visitado, atualiza a distância e o coloca na fila
            for (int vizinho : adj[atual]) {
                if (!visitado[vizinho]) {
                    visitado[vizinho] = true;
                    distancia[vizinho] = distancia[atual] + 1;
                    fila.push(vizinho);
                }
            }
        }

        // Determina o nivel máximo encontrado (para redimensionar a estrutura de niveis)
        int maxNivel = 0;
        for (int d : distancia) {
            if (d > maxNivel)
                maxNivel = d;
        }

        // Agrupa os vertices por nivel
        vector<vector<int>> niveis(maxNivel + 1);
        for (int i = 0; i < V; i++) {
            if (distancia[i] != -1) { // Só considera os vertices alcançados
                niveis[distancia[i]].push_back(i);
            }
        }

        // Exibe a estrutura de niveis (opcional)
        cout << "\nEstrutura de niveis enraizada a partir do vertice " << inicio << ":\n";
        for (int nivel = 0; nivel < niveis.size(); nivel++) {
            cout << "Nivel " << nivel << ": ";
            for (int vertice : niveis[nivel]) {
                cout << vertice << " ";
            }
            cout << endl;
        }

        return niveis;
    }
    // Funcao para calcular a excentricidade de um vertice
    int calcularExcentricidade(int v) {//const {
        vector<int> dist;
        dist = buscaEmLargura(v);
        int excentricidade = 0;
        for (int d : dist) {
            // Ignoramos vertices inalcancáveis (caso existam) representados por -1
            if (d != -1 && d > excentricidade)
                excentricidade = d;
        }
        return excentricidade;
    }
        
    // Implementa o algoritmo Cuthill-McKee para reordenar os vertices
    int GeorgeLiu(int start) {
        vector<int> S(V, 0); // Vetor S: guarda a nova numeracao dos vertices (inicialmente 0, não numerado)
        vector<int> order; // Vetor "order": guarda a sequência de vertices conforme a numeracao atribuida

        int v = start; // Define o vertice inicial arbitrario
        vector<vector<int>> Lv = estruturaDeNiveis(v); // Calcula a estrutura de niveis enraizada a partir de v
        
        int u;
        do {
            u = VerticeGrauMin();
            vector<vector<int>> Lu = estruturaDeNiveis(u);
            int lu = calcularExcentricidade(u);
            int lv = calcularExcentricidade(v);
            if (lu > lv) {
                Lv = Lu;
                v = u;
            }
        }while (u != v);
        return v;
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
    
    // Executa o algoritmo de Cuthill-McKee a partir de um vertice arbitrario
    int VertArb = (rand() % numVertices) + 1;
    int v = g.GeorgeLiu(VertArb);

    cout << "Vertice pseudoperiferico: " << v << endl;
    
    return 0;
}
