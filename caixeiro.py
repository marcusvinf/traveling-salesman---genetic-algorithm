import numpy as np



__author__ = "Marcus Vinicius"

def iniciar_populacao(_cidades, matriz_de_adjacencia, n_populacao):
        return Population(
        np.asarray([np.random.permutation(_cidades) for _ in range(n_populacao)]), 
        matriz_de_adjacencia
         )
def swap(cromossomo):
    a, b = np.random.choice(len(cromossomo), 2)
    cromossomo[a], cromossomo[b] = (
        cromossomo[b],
        cromossomo[a],
    )
    return cromossomo

def algoritimo_genetico(
_cidades,
matriz_de_adjacencia,
n_populacao=5,
n_iter=20,
selecionaridade=0.15,
p_cross=0.5,
p_mut=0.1,
print_interval=100,
return_historico=False,
verbose=False,
):
    pop = iniciar_populacao(_cidades, matriz_de_adjacencia, n_populacao)
    melhor_populacao = pop.melhor_populacao
    score = float("inf")
    historico = []
    for i in range(n_iter):
        pop.selecionar(n_populacao * selecionaridade)
        historico.append(pop.score)
        if verbose:
            print(f"Geracão {i}: {pop.score}")
        elif i % print_interval == 0:
            print(f"Geracão {i}: {pop.score}")
        if pop.score < score:
            melhor_populacao = pop.melhor_populacao
            score = pop.score
        filhos = pop.mutate(p_cross, p_mut)
        pop = Population(filhos, pop.matriz_de_adjacencia)
    if return_historico:
        return melhor_populacao, historico
    return melhor_populacao

class Cidades:

    def __init__(self, n_cidades, fator=10):
        self.n_cidades = n_cidades
        self.fator = fator

    def gerar_cidades(self):
        return np.random.rand(self.n_cidades, 2) * self.n_cidades * self.fator
    
    
    def gerar_matriz(self, coordenadascidade_coordenadas):
        result = [
            [self.obter_distancia(cidade1, cidade2) for cidade2 in coordenadascidade_coordenadas]
        for cidade1 in coordenadascidade_coordenadas
    ]
        return np.asarray(result)

    def obter_distancia(self, cidade1, cidade2):
        return np.sqrt((cidade1[0] - cidade2[0])**2 + (cidade1[1] - cidade2[1])**2)

class Population():
    def __init__(self, populat, matriz_de_adjacencia):
        self.populat = populat
        self.pais = []
        self.score = 0
        self.melhor_populacao = None
        self.matriz_de_adjacencia = matriz_de_adjacencia
    
    def mutate(self, p_cross=0.1, p_mut=0.1):
        proxima_pop = []
        filhos = self.crossover(p_cross)
        for filho in filhos:
            if np.random.rand() < p_mut:
                proxima_pop.append(swap(filho))
            else:
                proxima_pop.append(filho)
        return proxima_pop
    
    
    
    def fitness(self, cromossomo):
        return sum(
        [
            self.matriz_de_adjacencia[cromossomo[i], cromossomo[i + 1]]
            for i in range(len(cromossomo) - 1)
        ]
    )  
    def avaliar(self):
        distancias = np.asarray(
        [self.fitness(cromossomo) for cromossomo in self.populat]
    )
        self.score = np.min(distancias)
        self.melhor_populacao = self.populat[distancias.tolist().index(self.score)]
        self.pais.append(self.melhor_populacao)
        if False in (distancias[0] == distancias):
            distancias = np.max(distancias) - distancias
        return distancias / np.sum(distancias)
    
    def selecionar(self, k=4):
        fit = self.avaliar()
        while len(self.pais) < k:
            idx = np.random.randint(0, len(fit))
            if fit[idx] > np.random.rand():
                self.pais.append(self.populat[idx])
        self.pais = np.asarray(self.pais)


    
    def crossover(self, p_cross=0.1):
        filhos = []
        count, size = self.pais.shape
        for _ in range(len(self.populat)):
            if np.random.rand() > p_cross:
                filhos.append(
                    list(self.pais[np.random.randint(count, size=1)[0]])
                )
            else:
                pai1, pai2 = self.pais[
                    np.random.randint(count, size=2), :
                ]
                idx = np.random.choice(range(size), size=2, replace=False)
                start, end = min(idx), max(idx)
                filho = [None] * size
                for i in range(start, end + 1, 1):
                    filho[i] = pai1[i]
                indicie = 0
                for i in range(size):
                    if filho[i] is None:
                        while pai2[indicie] in filho:
                            indicie += 1
                        filho[i] = pai2[indicie]
                filhos.append(filho)
        return filhos




_cidades = range(100)
cada_cidade = Cidades(len(_cidades))
cidade_coordenadas = cada_cidade.gerar_cidades()
matriz_de_adjacencia = cada_cidade.gerar_matriz(cidade_coordenadas)
melhor_populacao, historico = algoritimo_genetico(
    _cidades, matriz_de_adjacencia, n_populacao=20, n_iter=1000, verbose=True, return_historico=True
)
