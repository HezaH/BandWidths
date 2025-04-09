#include <iostream>     // Biblioteca padrão do C++ para entrada e saída (std::cout, std::cerr, etc.)
#include <cstdio>       // Biblioteca padrão do C para manipulação de arquivos (fopen, fscanf, etc.)
#include "mmio.h"       // Biblioteca que permite ler arquivos no formato Matrix Market (.mtx)
#include "grafo.h"      // Header personalizado que define a classe Grafo (estrutura de dados para o grafo)

int main() {
    MM_typecode matcode;   // Estrutura que armazena o tipo da matriz (ex: real, simétrica, etc.)
    FILE *f;               // Ponteiro para o arquivo que será lido
    int M, N, nz;          // M: número de linhas (vértices), N: número de colunas, nz: número de entradas não-nulas

    // Abre o arquivo Matrix Market "ck400.mtx" no modo leitura ("r")
    f = fopen("ck400.mtx", "r");
    if (f == NULL) {
        perror("Erro ao abrir o arquivo");  // Exibe mensagem de erro do sistema se não conseguir abrir
        return 1;  // Encerra o programa com código de erro
    }

    // Lê o cabeçalho do arquivo e as dimensões da matriz
    // mm_read_banner: lê o tipo da matriz (ex: simétrica, real, etc.)
    // mm_read_mtx_crd_size: lê o número de linhas, colunas e elementos não nulos
    if (mm_read_banner(f, &matcode) != 0 || mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        std::cerr << "Erro ao ler arquivo Matrix Market.\n";  // Mensagem de erro personalizada
        return 2;  // Encerra com outro código de erro
    }

    Grafo g(M);  // Cria um objeto da classe Grafo com M vértices (assumindo matriz quadrada)
    int total = 0;
    int diagonal = 0;
    int belowDiagonal = 0;
    int aboveDiagonal = 0;
    int assimetricas = 0;

    // Loop para ler cada uma das nz entradas não nulas da matriz
    for (int i = 0; i < nz; i++) {
        int r, c;       // r: linha da entrada, c: coluna
        double val;     // val: valor da entrada (pode ser usado como peso da aresta)
   
        total++;
                

        
        // Lê a próxima entrada do arquivo no formato "linha coluna valor"
        // Espera que cada linha tenha três valores: inteiros r, c e número val (real ou inteiro)
        if (fscanf(f, "%d %d %lg\n", &r, &c, &val) != 3) {
            std::cerr << "Erro ao ler linha " << i + 1 << " do arquivo.\n";  // Mensagem de erro detalhada
            return 3;  // Código de erro específico para erro de leitura
        }

        r--; c--;  // Ajusta os índices de 1-based (formato .mtx) para 0-based (índices em C++ começam do zero)
            
        if (r == c){
            diagonal++;
        }else if (c < r){
            belowDiagonal++;
        }else{
            aboveDiagonal++;
        }
        
        if (val != 0.0) {
            std::cout << r + 1 << " " << c + 1 << " " << val << "\n";
            g.adicionarAresta(c , r);  // Adiciona uma aresta no grafo (de r para c, grafo direcionado)
        }
    }
    // Exibe informações sobre a matriz lida
    std::cout << "\n==== Informações da Matriz ====\n";
    std::cout << "Total de arestas: " << total << "\n";
    std::cout << "Arestas na diagonal (auto-laços): " << diagonal << "\n";
    std::cout << "Below diagonal: " << belowDiagonal << "\n";
    std::cout << "Above diagonal: " << aboveDiagonal << "\n";

    fclose(f);  // Fecha o arquivo após a leitura

    // Exibe o número total de arestas no grafo
    std::cout << "Total de arestas no grafo (direcionado): " << g.contarArestas() << std::endl;

    // Mostra a estrutura do grafo no console (geralmente com listas de adjacência)
    // std::cout << "Grafo carregado do arquivo ck400.mtx:\n";
    // g.mostrarGrafo();  // Chama o método que imprime os vértices e suas conexões

    // Executa busca em largura (BFS) a partir do vértice 0
    g.buscaEmLargura(0);

    // Exporta a estrutura do grafo para um arquivo .dot, para visualização com Graphviz
    g.exportarParaDot("grafo.dot");

    // Chama a função que calcula e exibe as métricas da matriz de adjacência
    g.metricasMatriz();

    g.calcularLarguraDeBanda();


    return 0;  // Retorna 0 indicando que o programa terminou com sucesso
}

