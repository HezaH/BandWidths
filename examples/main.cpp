#include <iostream>     // Biblioteca padrao do C++ para entrada e saida (std::cout, std::cerr, etc.)
#include <cstdio>       // Biblioteca padrao do C para manipulacao de arquivos (fopen, fscanf, etc.)
#include "mmio.h"       // Biblioteca que permite ler arquivos no formato Matrix Market (.mtx)
#include "grafo.h"      // Header personalizado que define a classe Grafo (estrutura de dados para o grafo)
#include <cmath>        // Biblioteca para funcões matemáticas como std::sqrt

int main() {
    MM_typecode matcode;   // Estrutura que armazena o tipo da matriz (ex: real, simetrica, etc.)
    FILE *f;               // Ponteiro para o arquivo que será lido
    int M, N, nz;          // M: número de linhas (vertices), N: número de colunas, nz: número de entradas nao-nulas
    int total = 0;
    int diagonal = 0;
    int belowDiagonal = 0;
    int aboveDiagonal = 0;
    int assimetricas = 0;
    double soma = 0.0;
    //Column
    int minCol = nz;     // inicializa com o máximo possivel
    int maxCol = 0;
    int shortestCol = 0;
    int longestCol = 0;
    double sumCol = 0.0;
    double varianceCol = 0.0;
    //Rows
    int minRow = nz;     // inicializa com o máximo possivel
    int maxRow = 0;
    int shortestRow = 0;
    int longestRow = 0;
    double sumRow = 0.0;
    double varianceRow = 0.0;

    // Abre o arquivo Matrix Market "ck400.mtx" no modo leitura ("r")
    f = fopen("ck400.mtx", "r");
    if (f == NULL) {
        perror("Erro ao abrir o arquivo");  // Exibe mensagem de erro do sistema se nao conseguir abrir
        return 1;  // Encerra o programa com código de erro
    }

    // Lê o cabecalho do arquivo e as dimensões da matriz
    // mm_read_banner: lê o tipo da matriz (ex: simetrica, real, etc.)
    // mm_read_mtx_crd_size: lê o número de linhas, colunas e elementos nao nulos
    if (mm_read_banner(f, &matcode) != 0 || mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        std::cerr << "Erro ao ler arquivo Matrix Market.\n";  // Mensagem de erro personalizada
        return 2;  // Encerra com outro código de erro
    }

    Grafo g(M);  // Cria um objeto da classe Grafo com M vertices (assumindo matriz quadrada)
    Grafo gT(M);  // Cria um grafo transposto (nao utilizado no código atual, mas pode ser útil para outras operacões)
    
    std::vector<std::vector<double>> A(M, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> AT(N, std::vector<double>(M, 0.0)); // Transposta deve ter dimensões invertidas
    std::vector<int> nnzPerColumn(N, 0); // Conta de nao nulos por coluna
    std::vector<int> nnzPerRow(M, 0); // Conta de nao nulos por coluna

    // Loop para ler cada uma das nz entradas nao nulas da matriz
    for (int i = 0; i < nz; i++) {
        int r, c;       // r: linha da entrada, c: coluna
        double val;     // val: valor da entrada (pode ser usado como peso da aresta)
        total++;
                
        // Lê a próxima entrada do arquivo no formato "linha coluna valor"
        // Espera que cada linha tenha três valores: inteiros r, c e número val (real ou inteiro)
        if (fscanf(f, "%d %d %lg\n", &r, &c, &val) != 3) {
            std::cerr << "Erro ao ler linha " << i + 1 << " do arquivo.\n";  // Mensagem de erro detalhada
            return 3;  // Código de erro especifico para erro de leitura
        }

        r--; c--;  // Ajusta os indices de 1-based (formato .mtx) para 0-based (indices em C++ comecam do zero) 
        if (r == c) {
            diagonal++;
        } else if (r > c) {      // If the row index is greater than the column index: BELOW the diagonal.
            belowDiagonal++;
        } else {                 // Otherwise (r < c): ABOVE the diagonal.
            aboveDiagonal++;
        }
        
        if (val != 0.0) {
            A[r][c] = val;
            AT[c][r] = val;  
            nnzPerColumn[c]++;
            nnzPerRow[r]++;
            std::cout << r + 1 << " " << c + 1 << " " << val << "\n";
            g.adicionarAresta(r, c);  // Adiciona uma aresta no grafo (de r para c, grafo direcionado)
            gT.adicionarAresta(c, r);  // Adiciona a aresta no grafo transposto (de c para r)
        }
    }
    
    std::cout << "\n";
    std::cout << "Matriz A - A^T:\n";

    for (int i = 0; i < M; ++i) {
        sumRow += nnzPerRow[i];
        for (int j = 0; j < N; ++j) {
            double val = A[i][j] - AT[i][j];
            soma += val * val;
        }
        
        if (nnzPerRow[i] < minRow) {
            minRow = nnzPerRow[i];
            shortestRow = i;
        }
        if (nnzPerRow[i] > maxRow) {
            maxRow = nnzPerRow[i];
            longestRow = i;
        }
    }
    
    std::cout << "\n";

    for (int i = 0; i < N; ++i) {
        sumCol += nnzPerColumn[i];
        if (nnzPerColumn[i] < minCol) {
            minCol = nnzPerColumn[i];
            shortestCol = i;
        }
        if (nnzPerColumn[i] > maxCol) {
            maxCol = nnzPerColumn[i];
            longestCol = i;
        }
    }

    double norma = std::sqrt(soma);
    double averageCol = sumCol / N;
    double averageRow = sumRow / N;

    for (int i = 0; i < N; ++i) {
        double diff = nnzPerColumn[i] - averageCol;
        varianceCol += diff * diff;
    }
    
    for (int i = 0; i < N; ++i) {
        double diff = nnzPerRow[i] - averageRow;
        varianceRow += diff * diff;
    }
    varianceCol /= N;
    varianceRow /= N;

    double stddevRow = std::sqrt(varianceRow);
    double stddevCol = std::sqrt(varianceCol);

    std::cout << "\nNorma de Frobenius de A - A^T: " << norma << "\n";

    // Exibe informacões sobre a matriz lida
    std::cout << "\n==== Informacões da Matriz ====\n";
    std::cout << "Total de arestas: " << total << "\n";
    std::cout << "Arestas na diagonal (auto-lacos): " << diagonal << "\n";
    std::cout << "Below diagonal: " << belowDiagonal << "\n";
    std::cout << "Above diagonal: " << aboveDiagonal << "\n";

    // Exibicao dos resultados
    std::cout << "\n==== Estatisticas por Coluna ====\n";
    std::cout << "Media de nao nulos por coluna: " << averageCol << "\n";
    std::cout << "Desvio padrao: " << stddevCol << "\n";
    std::cout << "Coluna com menos nao nulos: " << shortestCol + 1 << " (" << minCol << " valores)\n";
    std::cout << "Coluna com mais nao nulos: " << longestCol + 1 << " (" << maxCol << " valores)\n";

    std::cout << "\n==== Estatisticas por Linha ====\n";
    std::cout << "Media de nao nulos por Linha: " << averageRow << "\n";
    std::cout << "Desvio padrao: " << stddevRow << "\n";
    std::cout << "Linha com menos nao nulos: " << shortestRow + 1 << " (" << minRow << " valores)\n";
    std::cout << "Linha com mais nao nulos: " << longestRow + 1 << " (" << maxRow << " valores)\n";


    fclose(f);  // Fecha o arquivo após a leitura

    // Exibe o número total de arestas no grafo
    std::cout << "Total de arestas no grafo (direcionado): " << g.contarArestas() << std::endl;

    // Mostra a estrutura do grafo no console (geralmente com listas de adjacência)
    // std::cout << "Grafo carregado do arquivo ck400.mtx:\n";
    // g.mostrarGrafo();  // Chama o metodo que imprime os vertices e suas conexões

    // Executa busca em largura (BFS) a partir do vertice 0
    g.buscaEmLargura(0);

    // Exporta a estrutura do grafo para um arquivo .dot, para visualizacao com Graphviz
    g.exportarParaDot("grafo.dot");

    // Chama a funcao que calcula e exibe as metricas da matriz de adjacência
    g.metricasMatriz();

    g.calcularLarguraDeBanda();


    return 0;  // Retorna 0 indicando que o programa terminou com sucesso
}

