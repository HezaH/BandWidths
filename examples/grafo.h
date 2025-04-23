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
    void buscaEmLargura(int inicio);
    std::vector<std::vector<int>> gerarMatrizAdjacencia() const;
    void exportarMatrizAdjComoJPEG(const std::string &filename, int quality) const; // Adicione esta linha
};

#endif // GRAFO_H
