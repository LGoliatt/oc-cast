import pandas as pd

# Carregar os arquivos CSV
te_df = pd.read_csv("./SedimentDatabase_TE.csv")
me_nut_df = pd.read_csv("./SedimentDatabase_ME_Nut.csv")
minerals_df = pd.read_csv("./SedimentDatabase_Minerals.csv")

# Verificar as colunas comuns para merge
merge_keys = ['Sample_ID', 'Location_ID', 'Observationtype', 'Sampletype', 'Basin_ID']

# Realizar merges progressivos usando as colunas de referência
merged_df = te_df.merge(me_nut_df, on=merge_keys, how='outer', suffixes=('_TE', '_ME'))
merged_df = merged_df.merge(minerals_df, on=merge_keys, how='outer', suffixes=('', '_Minerals'))

# Mostrar as primeiras linhas do dataframe resultante
#import ace_tools as tools; tools.display_dataframe_to_user(name="Merged GloRiSe Dataset", dataframe=merged_df)

# Exibir o número de amostras combinadas
print(merged_df.shape)

merged_df.to_csv("Merged_GloRiSe_Dataset.csv", index=False)

#%%
# Caminho para o arquivo CSV
caminho_arquivo = "Merged_GloRiSe_Dataset.csv"

# Lê o arquivo CSV
df = pd.read_csv(caminho_arquivo)

# Define as colunas de entrada (features) e saída (target)
colunas_entrada = ['K2O_wt', 
                   'Fe2O3T_wt', 
                   'SiO2_wt', 
                   'Al2O3_wt', 
                   'P2O5_tot_wt',                    
                   'Illite',
                   ]
coluna_saida = 'C_org_wt'

# Verifica se as colunas existem no DataFrame
colunas_existentes = [col for col in colunas_entrada + [coluna_saida] if col in df.columns]

# Cria novo DataFrame apenas com as colunas desejadas
df_modelo = df[colunas_existentes]

# Remove linhas com valores ausentes
df_modelo = df_modelo.dropna()

# Exibe o tamanho do DataFrame final
print("Tamanho do conjunto de dados (linhas, colunas):", df_modelo.shape)

# Exibe as primeiras linhas (opcional)
print(df_modelo.shape)
df=df_modelo
#%%

import pandas as pd
from itertools import combinations

# Carrega o CSV
df = pd.read_csv("Merged_GloRiSe_Dataset.csv")
df.dropna(axis=0, how='all', inplace=True)
df.dropna(axis=1, how='all', inplace=True)

# Seleciona colunas numéricas
colunas_numericas = df.select_dtypes(include=['number']).columns.tolist()


# # Percorre combinações de 2 até 10 colunas numéricas (pode ajustar esse limite)
# for i in range(2, min(11, len(colunas_numericas))):
#     # Inicializa variáveis para armazenar o melhor resultado
#     melhor_conjunto = []
#     maior_tamanho = 0
#     for cols in combinations(colunas_numericas, i):
#         if 'C_org_wt' in cols:
#             df_temp = df[list(cols)].dropna()
#             tamanho = len(df_temp)
#             if tamanho > maior_tamanho:
#                 maior_tamanho = tamanho
#                 melhor_conjunto = cols
#                 print(i, maior_tamanho, melhor_conjunto)
#%%
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# Filtra colunas numéricas e garante que C_org_wt esteja presente
colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
colunas_numericas = [col for col in colunas_numericas if col != 'C_org_wt']
colunas_numericas.insert(0, 'C_org_wt')  # garante que sempre será a primeira

n = len(colunas_numericas)

# Função objetivo: minimizar o número de NaNs (maximizar amostras válidas)
def objetivo(mask):
    mask_bin = np.round(mask).astype(bool)
    mask_bin[0] = True  # garante que 'C_org_wt' sempre estará presente
    colunas_escolhidas = np.array(colunas_numericas)[mask_bin]
    df_temp = df[list(colunas_escolhidas)].dropna()
    return len(df_temp) /sum(mask_bin) # negativo porque vamos maximizar com sinal invertido

# Bounds: valores entre 0 e 1 para cada coluna (serão arredondados para 0 ou 1)
bounds = [(0, 1)] * n
x=np.random.rand(n).round()
# Executa otimização
result = differential_evolution(objetivo, bounds, strategy='best1bin', maxiter=100, popsize=20, polish=False, disp=True)

# Extrai melhores colunas
mask_result = np.round(result.x).astype(bool)
mask_result[0] = True
melhores_colunas = list(np.array(colunas_numericas)[mask_result])

# Resultado final
print("Melhor conjunto de colunas:", melhores_colunas)
print("Amostras completas:", -result.fun)

df_temp = df[list(melhores_colunas)].dropna()
print("Tamanho :", len(df_temp))

#%%
import numpy as np
import pandas as pd
import random

def simulated_annealing(df, colunas_numericas, max_iter=5000, temp=100.0, cooling=0.99):
    colunas_numericas = [col for col in colunas_numericas if col != 'C_org_wt']
    colunas_numericas.insert(0, 'C_org_wt')  # Garante C_org_wt no início
    n = len(colunas_numericas)

    def fitness(cols_ativadas):
        cols = [col for i, col in enumerate(colunas_numericas) if cols_ativadas[i]]
        if 'C_org_wt' not in cols or len(cols) < 2:
            return 0
        return len(df[cols].dropna())

    def gerar_vizinho(sol):
        nova = sol.copy()
        idx = random.randint(1, len(sol)-1)  # nunca remove 'C_org_wt'
        nova[idx] = 1 - nova[idx]
        return nova

    # Estado inicial
    estado_atual = [1] + [random.randint(0, 1) for _ in range(n - 1)]
    melhor_estado = estado_atual.copy()
    melhor_fitness = fitness(melhor_estado)

    for _ in range(max_iter):
        vizinho = gerar_vizinho(estado_atual)
        f_atual = fitness(estado_atual)
        f_vizinho = fitness(vizinho)
        delta = f_vizinho - f_atual

        if delta > 0 or random.random() < np.exp(delta / temp):
            estado_atual = vizinho.copy()
            if f_vizinho > melhor_fitness:
                melhor_fitness = f_vizinho
                melhor_estado = vizinho.copy()

        temp *= cooling  # resfriamento

    # Resultado final
    melhores_colunas = [col for i, col in enumerate(colunas_numericas) if melhor_estado[i]]
    return melhores_colunas, melhor_fitness

cleaned_df=df
colunas_boas, amostras_validas = simulated_annealing(cleaned_df, cleaned_df.select_dtypes(include=[np.number]).columns.tolist())
print("Melhores colunas:", colunas_boas)
print("Linhas completas:", amostras_validas)

#%%
import pandas as pd
import random
import time
from typing import List, Tuple

# Carregando o arquivo CSV enviado pelo usuário
file_path = "./Merged_GloRiSe_Dataset.csv"
df = pd.read_csv(file_path)
numeric_columns = df.select_dtypes(include=['number']).columns
non_numeric_columns = df.select_dtypes(exclude=['number']).columns

# Lista de colunas a serem removidas
cols_to_remove = ['Location_ID', 'Observationtype', 'Basin_ID', 'Sampletype', 'Sample_ID']

# Remover duplicadas da lista antes de tentar excluir
cols_to_remove = list(set(cols_to_remove))

# Remover as colunas do DataFrame
df = df.drop(columns=cols_to_remove, errors='ignore')

# Exibir as primeiras linhas e o nome das colunas
df.head(), df.columns.tolist()

# Parâmetros do algoritmo evolutivo
TIME_LIMIT = 300  # segundos
POP_SIZE = 100
MAX_COLS = 30
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.3

# Lista de colunas válidas (removendo as Unnamed)
valid_columns = [col for col in df.columns if not col.startswith('Unnamed')]

def fitness(columns: List[str]) -> int:
    """Fitness = número de linhas sem NaN nas colunas selecionadas"""
    return df[columns].dropna().shape[0]

def generate_individual() -> List[str]:
    """Cria um indivíduo com até MAX_COLS colunas aleatórias"""
    num_cols = random.randint(1, MAX_COLS)
    return random.sample(valid_columns, num_cols)

def tournament_selection(population: List[List[str]], scores: List[int]) -> List[str]:
    """Seleciona o melhor entre dois indivíduos aleatórios"""
    i1, i2 = random.sample(range(len(population)), 2)
    return population[i1] if scores[i1] > scores[i2] else population[i2]

def crossover(parent1: List[str], parent2: List[str]) -> List[str]:
    """Combina colunas dos dois pais"""
    child = list(set(parent1[:len(parent1)//2] + parent2[len(parent2)//2:]))
    if len(child) > MAX_COLS:
        child = random.sample(child, MAX_COLS)
    return child

def mutate(individual: List[str]) -> List[str]:
    """Aplica mutação no indivíduo"""
    if random.random() < 0.5 and len(individual) > 1:
        individual.remove(random.choice(individual))  # remove uma coluna
    else:
        new_col = random.choice(valid_columns)
        if new_col not in individual and len(individual) < MAX_COLS:
            individual.append(new_col)  # adiciona nova coluna
    return individual


# Reimplementação com foco multiobjetivo: max colunas e max linhas completas

def dominates(ind1: Tuple[int, int], ind2: Tuple[int, int]) -> bool:
    """Retorna True se ind1 domina ind2"""
    return (ind1[0] >= ind2[0] and ind1[1] > ind2[1]) or (ind1[0] > ind2[0] and ind1[1] >= ind2[1])

# Ajustando novo tempo e limites
MAX_COLS_MO = 30
start_time = time.time()
pareto_front = []

# Inicializar população
population = [generate_individual() for _ in range(POP_SIZE)]

def evaluate(ind: List[str]) -> Tuple[int, int]:
    return len(ind), df[ind].dropna().shape[0]

# Avaliar população
evaluations = [evaluate(ind) for ind in population]

# Inicializar Pareto front
pareto_front = []
pareto_solutions = []

for ind, eval in zip(population, evaluations):
    dominated = False
    for _, other_eval in zip(pareto_solutions, pareto_front):
        if dominates(other_eval, eval):
            dominated = True
            break
    if not dominated:
        pareto_solutions.append(ind)
        pareto_front.append(eval)

# Loop evolutivo com critério de tempo
while time.time() - start_time < TIME_LIMIT:
    new_population = []
    while len(new_population) < POP_SIZE:
        p1 = tournament_selection(population, [e[1] for e in evaluations])
        p2 = tournament_selection(population, [e[1] for e in evaluations])

        if random.random() < CROSSOVER_RATE:
            child = crossover(p1, p2)
        else:
            child = p1[:]

        if random.random() < MUTATION_RATE:
            child = mutate(child)

        new_population.append(child)

    population = new_population
    evaluations = [evaluate(ind) for ind in population]

    for ind, eval in zip(population, evaluations):
        dominated = False
        to_remove = []
        for i, other_eval in enumerate(pareto_front):
            if dominates(other_eval, eval):
                dominated = True
                break
            elif dominates(eval, other_eval):
                to_remove.append(i)
        if not dominated:
            for i in sorted(to_remove, reverse=True):
                pareto_front.pop(i)
                pareto_solutions.pop(i)
            pareto_front.append(eval)
            pareto_solutions.append(ind)


import matplotlib.pyplot as plt

# Separar os pontos da fronteira de Pareto
n_colunas = [p[0] for p in pareto_front]
n_linhas = [p[1] for p in pareto_front]

# Plotar a fronteira de Pareto
plt.figure(figsize=(10, 6))
plt.scatter(n_colunas, n_linhas, color='blue', s=60)
plt.title('Fronteira de Pareto - Número de Colunas vs Linhas Completas')
plt.xlabel('Número de Colunas')
plt.ylabel('Número de Linhas Completas')
plt.grid(True)
plt.show()


#%%

for c in pareto_solutions:
    if 'C_tot_wt' in c or 'C_org_wt' in c:
        print(df[c].dropna().shape, c)
        print()

