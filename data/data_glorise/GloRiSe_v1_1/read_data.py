import pandas as pd

# Caminho para o arquivo CSV
caminho_arquivo = "SedimentDatabase_ME_Nut.csv"

# Lê o arquivo CSV
df = pd.read_csv(caminho_arquivo)

# Define as colunas de entrada (features) e saída (target)
colunas_entrada = ['LOI_wt','Silt_perc', 'Clay_perc',
                   'Fe2O3T_wt', 'Al2O3_wt', 'TiO2_wt',
    #'SiO2_wt', 'Al2O3_wt', 'Fe2O3T_wt', 'Clay_perc', 'AvgGrainSize_mum'
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
#%%
import pandas as pd
from itertools import combinations

# Carrega o CSV
df = pd.read_csv("SedimentDatabase_ME_Nut.csv")

# Seleciona colunas numéricas
colunas_numericas = df.select_dtypes(include=['number']).columns.tolist()


# Percorre combinações de 2 até 10 colunas numéricas (pode ajustar esse limite)
for i in range(2, min(11, len(colunas_numericas))):
    # Inicializa variáveis para armazenar o melhor resultado
    melhor_conjunto = []
    maior_tamanho = 0
    for cols in combinations(colunas_numericas, i):
        if 'C_org_wt' in cols:
            df_temp = df[list(cols)].dropna()
            tamanho = len(df_temp)
            if tamanho > maior_tamanho:
                maior_tamanho = tamanho
                melhor_conjunto = cols
                print(maior_tamanho, melhor_conjunto)

# Exibe o resultado
print(f"Maior conjunto de dados possui {maior_tamanho} linhas")
print(f"Colunas selecionadas: {melhor_conjunto}")


#%%