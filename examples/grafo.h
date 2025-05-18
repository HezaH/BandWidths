#ifndef GRAFO_H
#define GRAFO_H

#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <string>
#include <stdexcept> // For runtime_error exceptions
using namespace std;

class Grafo {
private:
    int V;
    std::vector<std::vector<int>> adj;
    std::vector<bool> percorrido;

public:

    Grafo(int V_);

    // Metodos para manipulaçao do grafo
    void adicionarAresta(int u, int v);

    // Adicione a seguinte declaraçao para busca em largura:
    std::vector<std::vector<int>> buscaEmLarguraNivel(int v) const;

    // Algoritmo Cuthill-McKee (retorna nova numeraçao S; S[v] e 1-based)
    std::vector<int> Cuthill_McKee(int start);

    // Metodo para reordenar a lista de adjacência conforme a nova numeraçao S
    // Esse metodo utiliza a estrutura interna 'adj'
    std::vector<std::vector<int>> reordenarGrafo(const std::vector<int>& S) const;

    // Se necessário, permite atualizar a lista de adjacência com a nova ordem
    void setAdjacencias(const std::vector<std::vector<int>>& newAdj);

    // Funçao para gerar a matriz de adjacência a partir da lista de adjacência
    std::vector<std::vector<int>> gerarMatrizAdjacencia() const;

    // Exporta a matriz de adjacência como uma imagem JPEG (usa stb_image_write)
    void exportarMatrizAdjComoJPEG(const std::string &filename, int quality) const;

    // Metodo para calcular o grau de um vertice
    int calcularGrau(int v) const;

    // Metodo para encontrar o vertice de grau minimo
    int verticeGrauMinimo(const std::vector<int>& vertices) const;

    // Getters públicos
    const std::vector<std::vector<int>>& getAdjacencias() const { return adj; }
    int GeorgeLiu(int v) const;

    // Metodo para filtrar vertices com grau minimo
    std::vector<int> filtrarVerticesGrauMinimo(const std::vector<int>& vertices) const;

    // Altere a declaraçao para:
    int VerticePseudoPeriferico_GPS() const;
    
    std::vector<int> ordenarVerticesPorGrau() const;

    // Altere a assinatura para receber um vetor de inteiros:
    int VerticeMenorLargura(const std::vector<int>& vertices) const;


    int algo8(std::vector<std::vector<int>> & Lw) const;
};

#endif // GRAFO_H
