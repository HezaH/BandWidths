#include <iostream>     // Biblioteca padrao do C++ para entrada e saida (std::cout, std::cerr, etc.)
#include <cstdio>       // Biblioteca padrao do C para manipulacao de arquivos (fopen, fscanf, etc.)
#include <cstring>      // Ensure compatibility for perror and other C functions
#include <string>       // For std::string conversion
#include "mmio.h"       // Biblioteca que permite ler arquivos no formato Matrix Market (.mtx)
#include "grafo.h"      // Header personalizado que define a classe Grafo (estrutura de dados para o grafo)
#include <cmath>        // Biblioteca para funcões matematicas como std::sqrt
#include <algorithm>    // std::min_element, std::max_element
#include <numeric>      // std::accumulate

int main() {
    MM_typecode matcode;   // Estrutura que armazena o tipo da matriz (ex: real, simetrica, etc.)
    FILE *f;               // Ponteiro para o arquivo que sera lido
    int M, N, nz;          // M: numero de linhas (vertices), N: numero de colunas, nz: numero de entradas nao-nulas

    // Abre o arquivo Matrix Market "ck400.mtx" no modo leitura ("r")
    f = fopen("ck400.mtx", "r");
    if (f == NULL) {
        perror("Erro ao abrir o arquivo");  // Exibe mensagem de erro do sistema se nao conseguir abrir
        return 1;  // Encerra o programa com codigo de erro
    }

    // Le o cabecalho do arquivo e as dimensões da matriz
    // mm_read_banner: le o tipo da matriz (ex: simetrica, real, etc.)
    // mm_read_mtx_crd_size: le o numero de linhas, colunas e elementos nao nulos
    if (mm_read_banner(f, &matcode) != 0 || mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        std::cerr << "Erro ao ler arquivo Matrix Market.\n";  // Mensagem de erro personalizada
        return 2;  // Encerra com outro codigo de erro
    }

    Grafo g(M);  // Cria um objeto da classe Grafo com M vertices (assumindo matriz quadrada)

    // Loop para ler cada uma das nz entradas nao nulas da matriz
    for (int i = 0; i < nz; i++) {
        int r, c;       // r: linha da entrada, c: coluna
        double val;     // val: valor da entrada (pode ser usado como peso da aresta)
                
        // Le a proxima entrada do arquivo no formato "linha coluna valor"
        // Espera que cada linha tenha tres valores: inteiros r, c e numero val (real ou inteiro)
        if (fscanf(f, "%d %d %lg\n", &r, &c, &val) != 3) {
            std::cerr << "Erro ao ler linha " << i + 1 << " do arquivo.\n";  // Mensagem de erro detalhada
            return 3;  // Codigo de erro especifico para erro de leitura
        }

        r--; c--;  // Ajusta os indices de 1-based (formato .mtx) para 0-based (indices em C++ comecam do zero)
        if (val != 0.0) {
            g.adicionarAresta(r, c);  // Adiciona uma aresta no grafo (de r para c, grafo direcionado)
        }
    }

    fclose(f);  // Fecha o arquivo apos a leitura


    int GeoLiu = g.GeorgeLiu(23);
    int v = g.VerticePseudoPeriferico_GPS();
    std::vector<std::vector<int>> Lv = g.buscaEmLarguraNivel(v);
    if (!Lv.empty()) {
        // Se, por exemplo, voce deseja usar o último nivel na funcao VerticeMenorLargura:
        int u = g.VerticeMenorLargura(Lv.back());
        std::cout << "Vertice com menor largura do ultimo nivel: " << u << std::endl;
    }

        // --- Etapa 1: Obter a nova numeracao dos vertices com Cuthill-McKee ---
    std::vector<int> S = g.Cuthill_McKee(0);
    std::cout << "Nova numeracao (S):" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << "Vertice " << i << " -> " << S[i] << std::endl;
    }
    
    // --- Etapa 2: Reordenar a lista de adjacencia conforme a nova numeracao ---
    std::vector<std::vector<int>> newAdj = g.reordenarGrafo(S);
    
    // Opcional: imprimir a nova lista de adjacencia para verificacao
    std::cout << "\nNova lista de adjacencia (apos reordenacao):" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << "Vertice (novo indice) " << i << ": ";
        for (int viz : newAdj[i]) {
            std::cout << viz << " ";
        }
        std::cout << std::endl;
    }
    
    // Atualiza a estrutura do grafo com a nova lista de adjacencia
    g.setAdjacencias(newAdj);

    // --- Etapa 3: Exportar a matriz de adjacencia como JPEG ---
    try {
        g.exportarMatrizAdjComoJPEG("grafo_reordenado.jpg", 90);
        std::cout << "\nA imagem 'grafo_reordenado.jpg' foi gerada com sucesso." << std::endl;
    } catch (const std::runtime_error &e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}


