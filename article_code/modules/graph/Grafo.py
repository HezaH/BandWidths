import math
import os
import subprocess
import random
from collections import defaultdict

""" Version: v2.5 (livro Jayme) João Vitor Maués Dias Carneiro """


class Grafo(object):
	"""
	Classe base para as classes GrafoListaAdj e GrafoMatrizAdj
	"""	
	def __init__(self, orientado = False):
		"""
		Grafo se orientado=False ou Digrafo se orientado=True.
		"""
		self.n, self.m, self.orientado = None, None, orientado
		
	def DefinirN(self, n):
		"""
		Define o número n de vértices.
		"""	
		self.n, self.m = n, 0

	def V(self):
		"""
		Retorna a lista de vértices.
		"""
		for i in range(1,self.n+1):
			yield i

	def E(self, IterarSobreNo=False):
		"""
		Retorna lista de arestas uv, onde u é um inteiro e v é um inteiro se o grafo é GrafoMatrizAdj
		ou IterarSobreNo=False; v é GrafoListaAdj.NoAresta, caso contrário.
		"""
		for v in self.V():
			for w in self.N(v, Tipo = "+" if self.orientado else "*", IterarSobreNo=IterarSobreNo):
				enumerar = True
				if not self.orientado:
					wint = w if isinstance(w, int) else w.Viz
					enumerar = v < wint
				if enumerar:
					yield (v, w)
	
	

class GrafoListaAdj(Grafo):
	
	class NoAresta(object):
		"""
		Objeto nó da lista de adjacência.
		Atributos:
		- Viz (vizinho)
		- e (Aresta)
	        - Tipo (+/-)
		- Prox (próxima aresta na lista de adjacência)
		- Ant (aresta anterior na lista de adjacência (se a lista for duplamente encadeada))
		"""	

		def __init__(self):
			self.Viz = None
			self.e = None
			self.Prox = None
			self.Rotulo = None
  
	class Aresta(object):
		"""
		Objeto único para representar a aresta.
		Atributos:
		- v1, No1 (um dos vértices desta aresta e seu respectivo nó, isto é, v1 == No1.Viz)
		- v2, No2 (análogo em relação ao segundo vértice)
		"""
		def __init__(self):
			self.v1, self.No1 = None, None
			self.v2, self.No2 = None, None

	def DefinirN(self, n, VizinhancaDuplamenteLigada=False):
		"""
		Define o número n de vértices.
		Se VizinhancaDuplamenteLigada=True, a lista encadeada dos vizinhos de um vértice é duplamente ligada (permitindo remoção de arestas de tempo constante). 
		"""	
		super(GrafoListaAdj, self).DefinirN(n)
		self.L = [None]*(self.n+1)
		for i in range(1,self.n+1):
			self.L[i] = GrafoListaAdj.NoAresta() #nó cabeça
		self.VizinhancaDuplamenteLigada = VizinhancaDuplamenteLigada

	def AdicionarAresta(self, u, v):
		"""
		Adiciona aresta uv.
		"""
		def AdicionarLista(u,v,e,Tipo):
			No = GrafoListaAdj.NoAresta()
			No.Viz, No.e, No.Prox, self.L[u].Prox = v, e, self.L[u].Prox, No
			if self.VizinhancaDuplamenteLigada:
				self.L[u].Prox.Ant = self.L[u]
				if self.L[u].Prox.Prox != None:				
					self.L[u].Prox.Prox.Ant = self.L[u].Prox
			if self.orientado:
				No.Tipo = Tipo
			return No

		e = GrafoListaAdj.Aresta()
		e.v1, e.v2 = u, v
		e.No1 = AdicionarLista(u,v,e,"+")
		e.No2 = AdicionarLista(v,u,e,"-")
		self.m = self.m+1
		return e


	def RemoverAresta(self, uv):
		"""
		Remove a aresta uv.
		"""
		def RemoverLista(No):
			No.Ant.Prox = No.Prox
			if No.Prox != None:
				No.Prox.Ant = No.Ant
		RemoverLista(uv.No1)
		RemoverLista(uv.No2)

	def SaoAdj(self, u, v):
		"""
		Retorna True se uv é uma aresta e False, caso contrário.
		"""
		Tipo = "+" if self.orientado else "*"
		for w in self.N(u, Tipo):
			if w == v:
				return True
		return False

	def N(self, v, Tipo = "*", Fechada=False, IterarSobreNo=False):
		"""
		Retorna lista de Grafo.NoAresta representando os vizinhos do vértice v. 
		Se Fechada=True, o próprio v é incluido na lista.
		Tipo="*" significa listar todas as arestas incidentes em v. Se G é orientado, 
		Tipo="+" (resp. "-") significa listar apenas as arestas de saída (resp. de entrada) de v.
		IterarSobreNo=False indica que a lista de vizinhos deve constituir da lista de vértices. Caso
		contrário, a lista é dos nós da lista encadeada de vizinhos (NoAresta).
		"""
		if Fechada:
			No = GrafoListaAdj.NoAresta()
			No.Viz, No.e, No.Prox = v, None, None
			yield No if IterarSobreNo else No.Viz
		w = self.L[v].Prox

		while w != None:
			if Tipo == "*" or w.Tipo == Tipo:
				yield w if IterarSobreNo else w.Viz
			w = w.Prox


class RedutorGrafo(object):
	"""Extrai um subgrafo menor a partir de uma lista de arestas.

	Uso tipico (com arestas 1-based):
		k, edges_k, mapa, inv = RedutorGrafo.ReduzirArestas(n, edges, max_nodes=2000, estrategia='bfs')
		g_k = RedutorGrafo.ConstruirGrafoListaAdj(k, edges_k)

	Retorna tambem o mapeamento no_original->no_novo para rastrear correspondencias.
	"""

	@staticmethod
	def ReduzirArestas(n, edges, max_nodes, estrategia="bfs", start_node=None, seed=0, garantir_conexo=True):
		if max_nodes is None:
			mapa = {i: i for i in range(1, int(n) + 1)}
			inv = {i: i for i in range(1, int(n) + 1)}
			return int(n), list(edges), mapa, inv
		if int(max_nodes) <= 0:
			raise ValueError("max_nodes deve ser > 0")
		if int(n) <= int(max_nodes):
			mapa = {i: i for i in range(1, int(n) + 1)}
			inv = {i: i for i in range(1, int(n) + 1)}
			return int(n), list(edges), mapa, inv

		# vertices presentes nas arestas
		vertices = set()
		for (u, v) in edges:
			vertices.add(int(u))
			vertices.add(int(v))

		if len(vertices) == 0:
			k = min(int(n), int(max_nodes))
			mapa = {i: i for i in range(1, k + 1)}
			inv = {i: i for i in range(1, k + 1)}
			return k, [], mapa, inv

		rng = random.Random(int(seed))
		estrategia = (estrategia or "").strip().lower()

		# Monta adjacencia e componentes conexos (somente vertices com arestas)
		adj = defaultdict(list)
		for (u, v) in edges:
			u = int(u)
			v = int(v)
			adj[u].append(v)
			adj[v].append(u)

		def _componente_do_no(no_inicio):
			comp = set()
			pilha = [int(no_inicio)]
			while len(pilha) > 0:
				u = pilha.pop()
				if u in comp:
					continue
				comp.add(u)
				for w in adj.get(u, []):
					if w not in comp:
						pilha.append(w)
			return comp

		# Escolhe o componente alvo
		if start_node is not None:
			start_node = int(start_node)
			if start_node not in vertices:
				start_node = None

		if start_node is not None:
			componente_alvo = _componente_do_no(start_node)
		else:
			visitados = set()
			componente_alvo = None
			for vtx in vertices:
				if vtx in visitados:
					continue
				comp = _componente_do_no(vtx)
				visitados.update(comp)
				if (componente_alvo is None) or (len(comp) > len(componente_alvo)):
					componente_alvo = comp
			# start_node para BFS dentro do maior componente
			start_node = rng.choice(sorted(componente_alvo))

		# Quantos nos podemos pegar mantendo conectividade
		if garantir_conexo:
			max_nodes_eff = min(int(max_nodes), len(componente_alvo))
		else:
			max_nodes_eff = int(max_nodes)

		# Seleciona nos
		if estrategia == "random" and not garantir_conexo:
			ordenados = sorted(vertices)
			k = min(max_nodes_eff, len(ordenados))
			escolhidos = set(rng.sample(ordenados, k=k))
		else:
			# BFS (conexo) - se estrategia='random' e garantir_conexo=True, faz BFS com vizinhos embaralhados
			escolhidos = set()
			fila = [int(start_node)]
			while len(fila) > 0 and len(escolhidos) < max_nodes_eff:
				u = fila.pop(0)
				if u in escolhidos:
					continue
				if garantir_conexo and (u not in componente_alvo):
					continue
				escolhidos.add(u)
				vizinhos = list(adj.get(u, []))
				if estrategia == "random":
					rng.shuffle(vizinhos)
				for w in vizinhos:
					if w not in escolhidos:
						fila.append(w)

			# Se por algum motivo nao encheu (ex: componente pequeno), mantem conectado e retorna menor
			# (nao completa com nos desconexos)

		# arestas do subgrafo induzido
		edges_sub = []
		for (u, v) in edges:
			u = int(u)
			v = int(v)
			if (u != v) and (u in escolhidos) and (v in escolhidos):
				edges_sub.append((u, v))

		# re-rotular para 1..k
		ordenados = sorted(escolhidos)
		mapa = {}
		inv = {}
		k = 0
		for old in ordenados:
			k += 1
			mapa[old] = k
			inv[k] = old

		edges_relabel = set()
		for (u, v) in edges_sub:
			ru = mapa[u]
			rv = mapa[v]
			if ru == rv:
				continue
			if ru < rv:
				edges_relabel.add((ru, rv))
			else:
				edges_relabel.add((rv, ru))

		return k, sorted(edges_relabel), mapa, inv

	@staticmethod
	def ConstruirGrafoListaAdj(n, edges, VizinhancaDuplamenteLigada=True):
		g = GrafoListaAdj()
		g.DefinirN(int(n), VizinhancaDuplamenteLigada=VizinhancaDuplamenteLigada)
		for (u, v) in edges:
			g.AdicionarAresta(int(u), int(v))
		return g