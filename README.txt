g++ -g VerticePseudo_GPS.cpp -o a.exe

g++ -g main.cpp Class_Graph.cpp mmio.c -o main.exe
gdb main.exe       
dot -Tpng grafo.dot -o grafo.png
