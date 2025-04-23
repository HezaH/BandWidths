#include <iostream>     // Biblioteca padrao do C++ para entrada e saida (std::cout, std::cerr, etc.)
#include <cstdio>       // Biblioteca padrao do C para manipulacao de arquivos (fopen, fscanf, etc.)
#include "mmio.h"       // Biblioteca que permite ler arquivos no formato Matrix Market (.mtx)
#include "grafo.h"      // Header personalizado que define a classe Grafo (estrutura de dados para o grafo)
#include <cmath>        // Biblioteca para funcões matemáticas como std::sqrt
#include <algorithm>  // std::min_element, std::max_element
#include <numeric>  // std::accumulate

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

    // Profile Storage
    double min_lowerbandwidth = 0;
    double min_upperbandwidth = 0;
    double max_lowerbandwidth = 0;
    double max_upperbandwidth = 0;

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

    std::vector<int> lowerbandwidth(N, 0); // Conta de nao nulos por coluna
    std::vector<int> upperbandwidth(M, 0); // Conta de nao nulos por coluna

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

    // Para cada linha i, encontre os índices mínimo e máximo de coluna com elemento não nulo.
    for (int i = 0; i < M; ++i) {
    int j_min = N;  // inicializa com um valor maior que qualquer coluna possível
    int j_max = -1; // inicializa com um valor menor que qualquer coluna possível
    for (int j = 0; j < N; ++j) {
        if (A[i][j] != 0.0) {  // se o elemento for não nulo
            if (j < j_min) {
                j_min = j;
            }
            if (j > j_max) {
                j_max = j;
            }
        }
    }
    // Se a linha i tem pelo menos um nonzero:
    if (j_min != N && j_max != -1) {
        lowerbandwidth[i] = i - j_min;
        upperbandwidth[i] = j_max - i;
    } else {
        // Se não existem nonzeros na linha, a bandwidth pode ser considerada zero.
        lowerbandwidth[i] = 0;
        upperbandwidth[i] = 0;
    }
    }

    int min_lower = *std::min_element(lowerbandwidth.begin(), lowerbandwidth.end());
    int max_lower = *std::max_element(lowerbandwidth.begin(), lowerbandwidth.end());
    int min_upper = *std::min_element(upperbandwidth.begin(), upperbandwidth.end());
    int max_upper = *std::max_element(upperbandwidth.begin(), upperbandwidth.end());
    double sum_lower = std::accumulate(lowerbandwidth.begin(), lowerbandwidth.end(), 0.0);
    double avg_lower = sum_lower / lowerbandwidth.size();
    double sum_upper = std::accumulate(upperbandwidth.begin(), upperbandwidth.end(), 0.0);
    double avg_upper = sum_upper / upperbandwidth.size();
    double variance_lower = 0.0;

    for (int i = 0; i < lowerbandwidth.size(); i++) {
        double diff = lowerbandwidth[i] - avg_lower;
        variance_lower += diff * diff;
    }
    variance_lower /= lowerbandwidth.size();

    double stddev_lower = std::sqrt(variance_lower);
    double variance_upper = 0.0;

    for (int i = 0; i < upperbandwidth.size(); i++) {
        double diff = upperbandwidth[i] - avg_upper;
        variance_upper += diff * diff;
    }
    variance_upper /= upperbandwidth.size();

    double stddev_upper = std::sqrt(variance_upper);
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

    // Exibe os resultados
    std::cout << "\n==== Profiles Storage ====\n";
    std::cout << "Lower Bandwidth: " << min_lower << " - " << max_lower << "\n";
    std::cout << "Upper Bandwidth: " << min_upper << " - " << max_upper << "\n";
    std::cout << "Average Lower Bandwidth: " << avg_lower << "\n";
    std::cout << "Average Upper Bandwidth: " << avg_upper << "\n";
    std::cout << "StdDev Lower Bandwidth: " << stddev_lower << "\n";
    std::cout << "StdDev Upper Bandwidth: " << stddev_upper << "\n";

    fclose(f);  // Fecha o arquivo após a leitura

    g.exportarMatrizAdjComoJPEG("adj_matrix.jpg", 90);

    std::cout << "Imagem PGM salva em 'adj_matrix.pgm'\n";

    return 0;  // Retorna 0 indicando que o programa terminou com sucesso
}

