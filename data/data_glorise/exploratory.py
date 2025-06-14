import pandas as pd
import folium
from folium import plugins
import matplotlib.cm as cm
import matplotlib.colors as colors

# === 1. Ler o CSV ===
df = pd.read_csv("df_nut_lat_long.csv")

# === 2. Remover pontos sem coordenadas ou C_org_wt ===
df = df.dropna(subset=["Lat_deg", "Lon_deg", "C_org_wt"])

# === 3. Normalizar os valores de C_org_wt para 0–1 ===
norm = colors.Normalize(vmin=df["C_org_wt"].min(), vmax=df["C_org_wt"].max())
colormap = cm.get_cmap('YlOrRd')  # cores tipo calor

# === 4. Criar o mapa ===
m = folium.Map(location=[df["Lat_deg"].mean(), df["Lon_deg"].mean()], zoom_start=2, tiles="CartoDB positron")

# === 5. Adicionar os círculos ===
for _, row in df.iterrows():
    color = colors.rgb2hex(colormap(norm(row["C_org_wt"])))
    folium.CircleMarker(
        location=[row["Lat_deg"], row["Lon_deg"]],
        radius=5,
        fill=True,
        color=color,
        fill_opacity=0.8,
        popup=f"C_org_wt: {row['C_org_wt']:.2f}<br>Região: {row['Region']}<br>País: {row['Country']}"
    ).add_to(m)

# === 6. Adicionar legenda de cor ===
# from branca.colormap import LinearColormap
# color_scale = LinearColormap(['#ffffcc', '#fd8d3c', '#800026'],
#                              vmin=df["C_org_wt"].min(), vmax=df["C_org_wt"].max(),
#                              caption='Carbono Orgânico (C_org_wt)')
# color_scale.add_to(m)

# === 7. Mostrar ou salvar ===
m.save("mapa_C_org_wt.html")
m


#%%
# Read the Excel file
excel_path = "./df_nut_lat_long.xlsx"
df_excel = pd.read_excel(excel_path)

# Define input features and target
features = ['K2O_wt', 'Fe2O3T_wt', 'SiO2_wt', 'Al2O3_wt', 'P2O5_tot_wt']
target = 'C_org_wt'

# Split into training and test sets based on the 'Train' column
X_train = df_excel[df_excel['Train'] == 1][features]
y_train = df_excel[df_excel['Train'] == 1][target]

X_test = df_excel[df_excel['Train'] != 1][features]
y_test = df_excel[df_excel['Train'] != 1][target]

# Show shapes of the datasets
X_train.shape, X_test.shape, y_train.shape, y_test.shape


#%%
import optuna
from flaml import AutoML
from sklearn.metrics import mean_squared_error
import numpy as np
from permetrics.regression import RegressionMetric
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold

random_state=1

# Função objetivo do Optuna
def objective(trial):
    selected_features = [feature for feature in X_train.columns 
                         if trial.suggest_categorical(feature, [True, False])]
    
    if not selected_features:
        return float('inf')

    X_sub = X_train[selected_features]
    y_sub = y_train

    # Criar modelo FLAML com tempo reduzido para cada trial
    automl = AutoML()
    automl.fit(X_sub, y_sub, task="regression", time_budget=30, verbose=0)

    # CrossValPredict com KFold (3 dobras)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    y_pred_cv = cross_val_predict(automl.model.estimator, X_sub, y_sub, cv=kf)

    mse = mean_squared_error(y_sub, y_pred_cv)

    if mse < objective.best_mse:
        objective.best_mse = mse
        objective.best_features = selected_features
        objective.best_params = automl.best_config
        objective.best_model = automl

    return mse + 0.01 * len(selected_features)  # penalidade leve

# Inicializar armazenamento externo
objective.best_mse = float('inf')
objective.best_features = []
objective.best_model = None

# Rodar Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Acesso aos resultados finais
final_features = objective.best_features
final_params = objective.best_params
final_estimator = objective.best_model.best_estimator
final_model = objective.best_model.model

# Previsão no teste
y_pred = final_model.predict(X_test[final_features])
metrics = RegressionMetric(list(y_test), y_pred)

# Resultados
results = {
    "RMSE": metrics.RMSE(),
    "MAE": metrics.MAE(),
    "R2": metrics.R2(),
    "R": metrics.R(),
    "MAPE": metrics.MAPE(),
    "A10": metrics.A10(),
    "COLS": final_features,
    "PARAMS": final_params,
    "MODEL": "FLAML-OPTUNA-FS-CV"
}
print("Melhores features:", final_features)
print("Melhores parâmetros:", final_params)

# Impressão formatada
A = []
A.append(results)
metricas = ["RMSE", "MAE", "R2", "R", "MAPE", "A10"]
run = 1
for m, value in results.items():
    if m in metricas:
        print(f"{run} - {m}: {value:.4f}\t", end='')
print()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.8, edgecolors='k', s=80)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valores Reais (y_test)", fontsize=12)
plt.ylabel("Valores Preditos (y_pred)", fontsize=12)
plt.title("Dispersão: y_test vs y_pred", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()

#%%
import optuna
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error

def objective(trial):
    selected_features = [f for f in X_train.columns if trial.suggest_categorical(f, [True, False])]
    if not selected_features:
        return float('inf')
    X_sub = X_train[selected_features]
    y_sub = y_train
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Pesos do ensemble
    weights = {
        'rf': trial.suggest_float("w_rf", 0.0, 1.0),
        'lgbm': trial.suggest_float("w_lgbm", 0.0, 1.0),
        'enet': trial.suggest_float("w_enet", 0.0, 1.0),
        'svr': trial.suggest_float("w_svr", 0.0, 1.0),
        'knn': trial.suggest_float("w_knn", 0.0, 1.0),
        'ann': trial.suggest_float("w_ann", 0.0, 1.0)
    }
    total_weight = sum(weights.values())
    if total_weight == 0:
        return float('inf')
    for k in weights:
        weights[k] /= total_weight

    # Otimizar hiperparâmetros dos modelos
    rf = RandomForestRegressor(
        n_estimators=trial.suggest_int("rf_n_estimators", 50, 200),
        max_depth=trial.suggest_int("rf_max_depth", 3, 20),
        random_state=random_state
    )

    lgbm = LGBMRegressor(
        num_leaves=trial.suggest_int("lgbm_num_leaves", 15, 100),
        learning_rate=trial.suggest_float("lgbm_lr", 0.01, 0.3),
        n_estimators=100,
        random_state=random_state
    )

    enet = ElasticNet(
        alpha=trial.suggest_float("enet_alpha", 0.0001, 1.0, log=True),
        l1_ratio=trial.suggest_float("enet_l1_ratio", 0.0, 1.0),
        random_state=random_state
    )

    svr = SVR(
        C=trial.suggest_float("svr_C", 0.1, 10),
        epsilon=trial.suggest_float("svr_epsilon", 0.01, 1.0),
        kernel=trial.suggest_categorical("svr_kernel", ["rbf", "linear"])
    )

    knn = KNeighborsRegressor(
        n_neighbors=trial.suggest_int("knn_n", 2, 20),
        weights=trial.suggest_categorical("knn_weights", ["uniform", "distance"])
    )

    ann = MLPRegressor(
        hidden_layer_sizes=trial.suggest_categorical("ann_hidden", [(50,), (50,25), (100,), (100,50)]),
        alpha=trial.suggest_float("ann_alpha", 1e-5, 1e-1, log=True),
        max_iter=1000,
        random_state=random_state
    )

    models = {'rf': rf, 'lgbm': lgbm, 'enet': enet, 'svr': svr, 'knn': knn, 'ann': ann}
    y_ensemble = np.zeros(len(X_sub))

    for name, model in models.items():
        y_model = cross_val_predict(model, X_sub, y_sub, cv=kf, n_jobs=-1)
        y_ensemble += weights[name] * y_model

    mse = mean_squared_error(y_sub, y_ensemble)
    penalty = 0.01 * len(selected_features)

    if mse < objective.best_mse:
        objective.best_mse = mse
        objective.best_features = selected_features
        objective.best_weights = weights.copy()
        objective.best_hyperparams = {k: trial.params[k] for k in trial.params}

    return mse + penalty

# Inicializar variáveis globais
objective.best_mse = float('inf')
objective.best_features = []
objective.best_weights = {}
objective.best_hyperparams = {}

# Otimização
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Exibir os melhores parâmetros
print("Melhores features:", objective.best_features)
print("Melhores pesos do ensemble:", objective.best_weights)
print("Melhores hiperparâmetros:", objective.best_hyperparams)


# Dados com melhores features
X_final = X_train[objective.best_features]
X_test_final = X_test[objective.best_features]
params = objective.best_hyperparams
weights = objective.best_weights

# Recriar modelos com hiperparâmetros otimizados
models_final = {
    'rf': RandomForestRegressor(
        n_estimators=params['rf_n_estimators'],
        max_depth=params['rf_max_depth'],
        random_state=random_state
    ),
    'lgbm': LGBMRegressor(
        num_leaves=params['lgbm_num_leaves'],
        learning_rate=params['lgbm_lr'],
        n_estimators=100,
        random_state=random_state
    ),
    'enet': ElasticNet(
        alpha=params['enet_alpha'],
        l1_ratio=params['enet_l1_ratio'],
        random_state=random_state
    ),
    'svr': SVR(
        C=params['svr_C'],
        epsilon=params['svr_epsilon'],
        kernel=params['svr_kernel']
    ),
    'knn': KNeighborsRegressor(
        n_neighbors=params['knn_n'],
        weights=params['knn_weights']
    ),
    'ann': MLPRegressor(
        hidden_layer_sizes=params['ann_hidden'],
        alpha=params['ann_alpha'],
        max_iter=1000,
        random_state=random_state
    )
}

# Treinar modelos
for name, model in models_final.items():
    model.fit(X_final, y_train)

# Ensemble final (média ponderada)
y_pred = sum(weights[name] * models_final[name].predict(X_test_final)
             for name in models_final)

# Métricas finais
metrics = RegressionMetric(list(y_test), y_pred)
results = {
    "RMSE": metrics.RMSE(),
    "MAE": metrics.MAE(),
    "R2": metrics.R2(),
    "R": metrics.R(),
    "MAPE": metrics.MAPE(),
    "A10": metrics.A10(),
    "COLS": objective.best_features,
    "WEIGHTS": weights,
    "PARAMS": params,
    "MODEL": "OPTUNA-ENSEMBLE-6MODELS-HPTUNED"
}

print("Resultados finais:", results)


# Impressão formatada
A = []
A.append(results)
metricas = ["RMSE", "MAE", "R2", "R", "MAPE", "A10"]
run = 1
for m, value in results.items():
    if m in metricas:
        print(f"{run} - {m}: {value:.4f}\t", end='')
print()

# Calcular as principais métricas
r2 = metrics.R2()
rmse = metrics.RMSE()
mae = metrics.MAE()

# Plot com métricas no título
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.8, edgecolors='k', s=80)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Values (y_test)", fontsize=12)
plt.ylabel("Predicted Values (y_pred)", fontsize=12)
plt.title(f"y_test vs y_pred | R²={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()

#%%
# Lista de modelos que deseja usar no ensemble
models_to_use = ['rf', 'lgbm', 'enet', 'svr', 'knn', 'ann']  # Exemplo: ['rf', 'enet', 'svr']
#models_to_use = ['knn',]

import optuna
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error
from permetrics.regression import RegressionMetric
import matplotlib.pyplot as plt

# Escolha do sampler: "TPE" para monobjetivo ou "NSGAII" para multiobjetivo
sampler_name = "TPE"  # ou "NSGAII"

np.random.seed(random_state)

# def objective(trial):
#     selected_features = [f for f in X_train.columns if trial.suggest_categorical(f, [True, False])]
#     if not selected_features:
#         return float('inf')
#     X_sub = X_train[selected_features]
#     y_sub = y_train
#     kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

#     # Pesos apenas para modelos selecionados
#     weights = {name: trial.suggest_float(f"w_{name}", 0.0, 1.0) for name in models_to_use}
    
#     total_weight = sum(weights.values())
#     if total_weight == 0:
#         return float('inf')
#     for k in weights:
#         weights[k] /= total_weight

#     # Zerar pesos muito baixos
#     for k in weights:
#         if weights[k] < 0.05:
#             weights[k] = 0.0
    
#     total_weight = sum(weights.values())
#     if total_weight == 0:
#         return float('inf')
#     for k in weights:
#         weights[k] /= total_weight

#     # Criar apenas os modelos escolhidos com hiperparâmetros otimizados
#     models = {}

#     if 'rf' in models_to_use:
#         models['rf'] = RandomForestRegressor(
#             n_estimators=trial.suggest_int("rf_n_estimators", 50, 200),
#             max_depth=trial.suggest_int("rf_max_depth", 3, 20),
#             random_state=random_state
#         )
#     if 'lgbm' in models_to_use:
#         models['lgbm'] = LGBMRegressor(
#             num_leaves=trial.suggest_int("lgbm_num_leaves", 15, 100),
#             learning_rate=trial.suggest_float("lgbm_lr", 0.01, 0.3),
#             n_estimators=trial.suggest_int("lgbm_n_estimators", 50, 300),
#             random_state=random_state
#         )
#     if 'enet' in models_to_use:
#         models['enet'] = ElasticNet(
#             alpha=trial.suggest_float("enet_alpha", 0.0001, 1.0, log=True),
#             l1_ratio=trial.suggest_float("enet_l1_ratio", 0.0, 1.0),
#             random_state=random_state
#         )
#     if 'svr' in models_to_use:
#         models['svr'] = SVR(
#             C=trial.suggest_float("svr_C", 0.1, 10),
#             epsilon=trial.suggest_float("svr_epsilon", 0.01, 1.0),
#             kernel=trial.suggest_categorical("svr_kernel", ["rbf", "linear"])
#         )
#     if 'knn' in models_to_use:
#         models['knn'] = KNeighborsRegressor(
#             n_neighbors=trial.suggest_int("knn_n", 2, 20),
#             weights=trial.suggest_categorical("knn_weights", ["uniform", "distance"])
#         )
#     if 'ann' in models_to_use:
#         hidden_units = trial.suggest_int("ann_hidden", 1, 50)
#         models['ann'] = MLPRegressor(
#             hidden_layer_sizes=(hidden_units,),
#             alpha=trial.suggest_float("ann_alpha", 1e-5, 1e-1, log=True),
#             max_iter=1000,
#             random_state=random_state,
#         )

#     y_ensemble = np.zeros(len(X_sub))
#     for name, model in models.items():
#         y_model = cross_val_predict(model, X_sub, y_sub, cv=kf, n_jobs=-1)
#         y_ensemble += weights[name] * y_model

#     mse = mean_squared_error(y_sub, y_ensemble)
#     penalty = 0.01 * len(selected_features)

#     if mse < objective.best_mse:
#         objective.best_mse = mse
#         objective.best_features = selected_features
#         objective.best_weights = weights.copy()
#         objective.best_hyperparams = {k: trial.params[k] for k in trial.params}

#     return mse + penalty

def objective(trial):
    selected_features = [f for f in X_train.columns if trial.suggest_categorical(f, [True, False])]
    if not selected_features:
        return float('inf') if sampler_name == "TPE" else (float('inf'), float('inf'))

    X_sub = X_train[selected_features]
    y_sub = y_train
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Pesos do ensemble
    weights = {name: trial.suggest_float(f"w_{name}", 0.0, 1.0) for name in models_to_use}
    
    # Zerar pesos baixos
    for k in weights:
        if weights[k] < 0.05:
            weights[k] = 0.0

    total_weight = sum(weights.values())
    if total_weight == 0:
        return float('inf') if sampler_name == "TPE" else (float('inf'), float('inf'))
    for k in weights:
        weights[k] /= total_weight

    # Modelos
    models = {}

    if 'rf' in models_to_use:
        models['rf'] = RandomForestRegressor(
            n_estimators=trial.suggest_int("rf_n_estimators", 50, 200),
            max_depth=trial.suggest_int("rf_max_depth", 3, 20),
            random_state=random_state
        )
    if 'lgbm' in models_to_use:
        models['lgbm'] = LGBMRegressor(
            num_leaves=trial.suggest_int("lgbm_num_leaves", 15, 100),
            learning_rate=trial.suggest_float("lgbm_lr", 0.01, 0.3),
            n_estimators=trial.suggest_int("lgbm_n_estimators", 50, 300),
            random_state=random_state
        )
    if 'enet' in models_to_use:
        models['enet'] = ElasticNet(
            alpha=trial.suggest_float("enet_alpha", 0.0001, 1.0, log=True),
            l1_ratio=trial.suggest_float("enet_l1_ratio", 0.0, 1.0),
            random_state=random_state
        )
    if 'svr' in models_to_use:
        models['svr'] = SVR(
            C=trial.suggest_float("svr_C", 0.1, 10),
            epsilon=trial.suggest_float("svr_epsilon", 0.01, 1.0),
            kernel=trial.suggest_categorical("svr_kernel", ["rbf", "linear"])
        )
    if 'knn' in models_to_use:
        models['knn'] = KNeighborsRegressor(
            n_neighbors=trial.suggest_int("knn_n", 2, 20),
            weights=trial.suggest_categorical("knn_weights", ["uniform", "distance"])
        )
    if 'ann' in models_to_use:
        hidden_units = trial.suggest_int("ann_hidden", 1, 50)
        models['ann'] = MLPRegressor(
            hidden_layer_sizes=(hidden_units,),
            alpha=trial.suggest_float("ann_alpha", 1e-5, 1e-1, log=True),
            max_iter=1000,
            random_state=random_state,
        )

    y_ensemble = np.zeros(len(X_sub))
    for name, model in models.items():
        y_model = cross_val_predict(model, X_sub, y_sub, cv=kf, n_jobs=-1)
        y_ensemble += weights[name] * y_model

    mse = mean_squared_error(y_sub, y_ensemble)
    complexity = len(selected_features)
    penalty = 0.01 * complexity

    # Armazenar os melhores
    if sampler_name == "TPE":
        if mse + penalty < objective.best_mse:
            objective.best_mse = mse + penalty
            objective.best_features = selected_features
            objective.best_weights = weights.copy()
            objective.best_hyperparams = {k: trial.params[k] for k in trial.params}
        return mse + penalty
    else:  # NSGAII
        if (mse, complexity) < (objective.best_mse, objective.best_complexity):
            objective.best_mse = mse
            objective.best_complexity = complexity
            objective.best_features = selected_features
            objective.best_weights = weights.copy()
            objective.best_hyperparams = {k: trial.params[k] for k in trial.params}
        return mse, complexity


# Variáveis globais
# objective.best_mse = float('inf')
# objective.best_features = []
# objective.best_weights = {}
# objective.best_hyperparams = {}
objective.best_mse = float('inf')
objective.best_features = []
objective.best_weights = {}
objective.best_hyperparams = {}

# Apenas se usar NSGAII:
if sampler_name == "NSGAII":
    objective.best_complexity = float('inf')
    
    
# Otimizar

# Criação do estudo com base no sampler escolhido
if sampler_name == "TPE":
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=random_state),
        direction="minimize"
    )
elif sampler_name == "NSGAII":
    study = optuna.create_study(
        directions=["minimize", "minimize"],  # Ex: minimizar MSE e complexidade
        sampler=optuna.samplers.NSGAIISampler(seed=random_state)
    )
else:
    raise ValueError("Sampler inválido. Use 'TPE' ou 'NSGAII'.")
    
study.optimize(objective, n_trials=50, n_jobs=1)

# === Reconstrução final com os melhores modelos ===
X_final = X_train[objective.best_features]
X_test_final = X_test[objective.best_features]
params = objective.best_hyperparams
weights = objective.best_weights

# Montar modelos finais com hiperparâmetros otimizados
models_final = {}
if 'rf' in models_to_use:
    models_final['rf'] = RandomForestRegressor(
        n_estimators=params['rf_n_estimators'],
        max_depth=params['rf_max_depth'],
        random_state=random_state
    )
if 'lgbm' in models_to_use:
    models_final['lgbm'] = LGBMRegressor(
        num_leaves=params['lgbm_num_leaves'],
        learning_rate=params['lgbm_lr'],
        n_estimators=params['lgbm_n_estimators'],
        random_state=random_state
    )
if 'enet' in models_to_use:
    models_final['enet'] = ElasticNet(
        alpha=params['enet_alpha'],
        l1_ratio=params['enet_l1_ratio'],
        random_state=random_state
    )
if 'svr' in models_to_use:
    models_final['svr'] = SVR(
        C=params['svr_C'],
        epsilon=params['svr_epsilon'],
        kernel=params['svr_kernel']
    )
if 'knn' in models_to_use:
    models_final['knn'] = KNeighborsRegressor(
        n_neighbors=params['knn_n'],
        weights=params['knn_weights']
    )
if 'ann' in models_to_use:
    models_final['ann'] = MLPRegressor(
        hidden_layer_sizes=params['ann_hidden'],
        alpha=params['ann_alpha'],
        max_iter=1000,
        random_state=random_state
    )

# Treinar modelos
for name, model in models_final.items():
    model.fit(X_final, y_train)

# Ensemble
y_pred = sum(weights[name] * models_final[name].predict(X_test_final) for name in models_final)

# Métricas
metrics = RegressionMetric(list(y_test), y_pred)

results = {
    "RMSE": metrics.RMSE(),
    "MAE": metrics.MAE(),
    "R2": metrics.R2(),
    "R": metrics.R(),
    "MAPE": metrics.MAPE(),
    "A10": metrics.A10(),
    "COLS": objective.best_features,
    "WEIGHTS": weights,
    "PARAMS": params,
    "MODEL": f"OPTUNA-ENSEMBLE-{models_to_use}"
}
print("Resultados finais:", results)

# Plot com métricas no título
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.8, edgecolors='k', s=80)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Values (y_test)", fontsize=12)
plt.ylabel("Predicted Values (y_pred)", fontsize=12)
plt.title(f"y_test vs y_pred | R²={results['R2']:.3f} | RMSE={results['RMSE']:.3f} | MAE={results['MAE']:.3f}", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()

#%%