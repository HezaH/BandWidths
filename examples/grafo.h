#ifndef GRAFO_H
#define GRAFO_H

#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <string>
using namespace std;

class Grafo {
private:
    int V;
    vector<vector<int>> adj;
    vector<bool> percorrido;

public:
    Grafo(int vertices);
    void adicionarAresta(int v1, int v2);
    void mostrarGrafo();
    void buscaEmLargura(int inicio);
    int contarArestas() const;
    void exportarParaDot(const string& nomeArquivo);
    void metricasMatriz();
    void calcularLarguraDeBanda();
};

#endif // GRAFO_H
