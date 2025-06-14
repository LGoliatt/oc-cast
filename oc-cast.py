import numpy as np
import pandas as pd
import optuna
import torch
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import ElasticNet, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from ngboost import NGBRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from deepforest import CascadeForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import VotingRegressor
from permetrics.regression import RegressionMetric
import warnings
warnings.filterwarnings("ignore")

# === Load your dataset ===
def load_and_split_data(excel_path, use_random_split=True, test_size=0.2, random_state=42):
    df = pd.read_excel(excel_path)
    features = ['K2O_wt', 'Fe2O3T_wt', 'SiO2_wt', 'Al2O3_wt', 'P2O5_tot_wt']
    target = 'C_org_wt'
    X = df[features]
    y = df[target]
    if use_random_split:
        return train_test_split(X, y, test_size=test_size, random_state=random_state), features
    else:
        if 'Train' not in df.columns:
            raise ValueError("Column 'Train' not found in dataset.")
        return (
            df[df['Train'] == 1][features],
            df[df['Train'] != 1][features],
            df[df['Train'] == 1][target],
            df[df['Train'] != 1][target]
        ), features

# === Get Model Factory ===
def get_model(name, trial, prefix, random_state):
    if name == "en":
        return ElasticNet(
            alpha=trial.suggest_float(f"{prefix}en_alpha", 1e-4, 1.0, log=True),
            l1_ratio=trial.suggest_float(f"{prefix}en_l1_ratio", 0.0, 1.0),
            random_state=random_state
        )
    elif name == "knn":
        return KNeighborsRegressor(
            n_neighbors=trial.suggest_int(f"{prefix}knn_n", 2, 20),
            weights=trial.suggest_categorical(f"{prefix}knn_weights", ["uniform", "distance"])
        )
    elif name == "ann":
        return MLPRegressor(
            hidden_layer_sizes=(trial.suggest_int(f"{prefix}ann_hidden", 5, 100),),
            alpha=trial.suggest_float(f"{prefix}ann_alpha", 1e-5, 1e-1, log=True),
            max_iter=1000,
            random_state=random_state
        )
    elif name == "lgbm":
        return LGBMRegressor(
            num_leaves=trial.suggest_int(f"{prefix}lgbm_num_leaves", 15, 100),
            learning_rate=trial.suggest_float(f"{prefix}lgbm_lr", 0.01, 0.3),
            n_estimators=trial.suggest_int(f"{prefix}lgbm_n_estimators", 50, 300),
            random_state=random_state
        )
    elif name == "ngb":
        return NGBRegressor(
            n_estimators=trial.suggest_int(f"{prefix}ngb_n_estimators", 50, 300),
            learning_rate=trial.suggest_float(f"{prefix}ngb_lr", 0.001, 0.2),
            random_state=random_state,
            verbose=False
        )
    elif name == "xgb":
        return XGBRegressor(
            n_estimators=trial.suggest_int(f"{prefix}xgb_n_estimators", 50, 300),
            learning_rate=trial.suggest_float(f"{prefix}xgb_lr", 0.01, 0.3),
            max_depth=trial.suggest_int(f"{prefix}xgb_max_depth", 3, 10),
            subsample=trial.suggest_float(f"{prefix}xgb_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float(f"{prefix}xgb_colsample", 0.6, 1.0),
            random_state=random_state,
            verbosity=0
        )    
    elif name == "cat":
            return CatBoostRegressor(
                iterations=trial.suggest_int(f"{prefix}cat_iterations", 50, 300),
                learning_rate=trial.suggest_float(f"{prefix}cat_lr", 0.01, 0.3),
                depth=trial.suggest_int(f"{prefix}cat_depth", 4, 10),
                verbose=0,
                random_state=random_state
            )    
    elif name == "tabnet":
        return TabNetRegressor(
            n_d=trial.suggest_int(f"{prefix}tabnet_n_d", 8, 64, step=8),
            n_a=trial.suggest_int(f"{prefix}tabnet_n_a", 8, 64, step=8),
            n_steps=trial.suggest_int(f"{prefix}tabnet_n_steps", 3, 10),
            gamma=trial.suggest_float(f"{prefix}tabnet_gamma", 1.0, 2.0),
            lambda_sparse=trial.suggest_float(f"{prefix}tabnet_lambda_sparse", 1e-6, 1e-3, log=True),
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=trial.suggest_float(f"{prefix}tabnet_lr", 1e-4, 1e-2, log=True)),
            verbose=0,
            seed=random_state
        )    
    elif name == "deepforest":
        return CascadeForestRegressor(
            n_estimators=trial.suggest_int(f"{prefix}df_n_estimators", 50, 300),
            max_layers=trial.suggest_int(f"{prefix}df_max_layers", 2, 10),
            n_trees=trial.suggest_int(f"{prefix}df_n_trees", 50, 300),
            random_state=random_state
        )        
    elif name == "svr":
        return LinearSVR(
            C=trial.suggest_float(f"{prefix}svr_C", 0.01, 10.0, log=True),
            epsilon=trial.suggest_float(f"{prefix}svr_epsilon", 0.001, 1.0, log=True),
            random_state=random_state
        )
    elif name == "gpr":
        matern_nu = trial.suggest_categorical(f"{prefix}gpr_nu", [0.5, 1.5, 2.5])  # Smoothness
        #matern_nu = trial.suggest_float(f"{prefix}gpr_nu", 0.1, 5.0)
        kernel = C(
            trial.suggest_float(f"{prefix}gpr_const", 0.1, 10.0)
        ) * Matern(
            length_scale=trial.suggest_float(f"{prefix}gpr_length_scale", 0.1, 10.0),
            nu=matern_nu
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=trial.suggest_float(f"{prefix}gpr_alpha", 1e-10, 1e-2, log=True),
            normalize_y=True,
            random_state=random_state
        )
    elif name == "lr":
        return RidgeCV()
    elif name == "rf":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int(f"{prefix}rf_n_estimators", 10, 200),
            max_depth=trial.suggest_int(f"{prefix}rf_max_depth", 2, 20),
            random_state=random_state
        )
    elif name == "et":
        return ExtraTreesRegressor(
            n_estimators=trial.suggest_int(f"{prefix}et_n_estimators", 50, 300),
            max_depth=trial.suggest_int(f"{prefix}et_max_depth", 3, 20),
            min_samples_split=trial.suggest_int(f"{prefix}et_min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int(f"{prefix}et_min_samples_leaf", 1, 5),
            bootstrap=trial.suggest_categorical(f"{prefix}et_bootstrap", [True, False]),
            random_state=random_state
        )

# === Rebuild model using best params ===
def rebuild_model(best_params, models_to_use, feature_names, random_state):
    selected_features = [f for f in feature_names if best_params.get(f"use_feat_{f}", False)]
    idxs = [feature_names.index(f) for f in selected_features]
    model_params = {}
    stacking_used = len(models_to_use) > 1
    final_estimator_name = None
    final_estimator_params = {}
    selected_base=[]

    if not stacking_used:
        def fake_trial(): return optuna.trial.FixedTrial(params=best_params)
        model_name = models_to_use[0]
        model = get_model(model_name, fake_trial(), "", random_state)
        model_params[model_name] = {k: v for k, v in best_params.items() if k.startswith(model_name)}
        final_estimator_name = model_name
        final_estimator_params = model_params[model_name]
    else:
        selected_base = [m for m in models_to_use if best_params.get(f"use_{m}", False)]
        def fake_trial(): return optuna.trial.FixedTrial(params=best_params)
        estimators = []
        for m in selected_base:
            estimators.append((m, get_model(m, fake_trial(), '', random_state)))
            model_params[m] = {k: v for k, v in best_params.items() if k.startswith(m)}
        final_estimator_name = best_params.get("final_estimator")
        final_est = get_model(final_estimator_name, fake_trial(), "final_", random_state)
        final_estimator_params = {k: v for k, v in best_params.items() if k.startswith(f"final_{final_estimator_name}")}
        model = StackingRegressor(estimators=estimators, final_estimator=final_est)
        model_params[final_estimator_name] = final_estimator_params

    return model, selected_features, selected_base, model_params, stacking_used, len(model_params), final_estimator_name, final_estimator_params

#%%
# === Main Logic ===
def optimize_stacking_model(excel_path, models_to_use, use_random_split, use_feature_selection, random_state=42, run=0):
    #%%
    (X_train, X_test, y_train, y_test), feature_names = load_and_split_data(
        excel_path, use_random_split=use_random_split, test_size=0.3, random_state=random_state
    )

    X_train_np, y_train_np = X_train.values, y_train.values
    X_test_np, y_test_np = X_test.values, y_test.values
    
    baseline_model = (models_to_use[0].upper() if len(models_to_use) == 1 else 'stk'.upper()) + ('-FS' if use_feature_selection else '')

    def objective(trial):
        if use_feature_selection:
            selected_features = [f for f in feature_names if trial.suggest_categorical(f"use_feat_{f}", [True, False])]
            if len(selected_features) < 2:
                return float("inf")
        else:
            selected_features = [f for f in feature_names if trial.suggest_categorical(f"use_feat_{f}", [True, ])]
            
        idxs = [feature_names.index(f) for f in selected_features]
        X_sel = X_train_np[:, idxs]

        if len(models_to_use) == 1:
            model = get_model(models_to_use[0], trial, "", random_state)
        else:
            selected_base = [m for m in models_to_use if trial.suggest_categorical(f"use_{m}", [True, False])]
            if len(selected_base) < 2:
                return float("inf")
            estimators = [(m, get_model(m, trial, '', random_state)) for m in selected_base]
            final_model = trial.suggest_categorical("final_estimator", models_to_use)
            final_est = get_model(final_model, trial, "final_", random_state)
            model = StackingRegressor(estimators=estimators, final_estimator=final_est)
            
        kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
        score = cross_val_score(model, X_sel, y_train_np, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=1)
        return -np.mean(score)

    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=random_state),
        direction="minimize"
    )
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best_params = study.best_params
    model, selected_features, selected_models, model_params, stacking_used, n_models, final_estimator_name, final_estimator_params = rebuild_model(
        best_params, models_to_use, feature_names, random_state)
    stacked_model = False if final_estimator_name==None else True
    idxs = [feature_names.index(f) for f in selected_features]

    X_train_sel = X_train_np[:, idxs]
    X_test_sel = X_test_np[:, idxs]

    model.fit(X_train_sel, y_train_np)
    y_pred = model.predict(X_test_sel)

    #== uncertainty
    err_e =np.array(y_test) - np.array(y_pred)
    err_m=err_e.mean()
    err_s = np.sqrt(sum((err_e - err_m)**2)/(len(err_e)-1))
      
       
    n_outcomes=120_000
    data=np.random.uniform( low=X_test_sel.min(axis=0), high=X_test_sel.max(axis=0), size=(n_outcomes, X_test_sel.shape[1]) )
    data=pd.DataFrame(data,columns=selected_features)
    #data=np.random.normal( loc=X_test_.mean(axis=0), scale=X_test_.std(axis=0), size=(n_outcomes, X_test_.shape[1]) )
    predict = model.predict(data)
    median = np.median(predict)
    mad=np.abs(predict - median).mean()
    uncertainty = 100*mad/median
    
    r = permutation_importance(model, X_test_sel, y_test.ravel(),
                            n_repeats=100,
                            scoring='r2',
                            random_state=0)
    fi=dict(zip(feature_names, r['importances_mean']))
            
    
    # Save to JSON
    output = {
        "y_test": list(y_test),
        "y_pred": list(y_pred),
        "selected_features": selected_features,
        "selected_models": selected_models,
        "stacking_used": stacking_used,
        "stacked_model":stacked_model,
        "number_of_models": n_models,
        "final_estimator": final_estimator_name,
        "final_estimator_params": final_estimator_params,
        "model_params": model_params,
        'feature_names':feature_names,
        "selected_features": selected_features,
        "use_random_split": use_random_split,
        "use_feature_selection": use_feature_selection,
        "model_params": model_params,
        'model_name':baseline_model,
        'error_mean': err_m,
        'error_sdev': err_s,
        'uncertainty_median': median,
        'uncertainty_mad': mad,
        'uncertainty_measure': uncertainty,
        'feature_importance':fi,
        'run':run,
    }    
    #%%
    return output

#%%
def convert_np(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj
    
# === Experimental Setup ===
folder_path = "./json-files/"
os.system('mkdir -p '+folder_path)

excel_path="./data/data_glorise/df_nut_lat_long.xlsx"
all_models = ['et'] + ['lr', 'rf', 'svr', 'knn', 'ann', 'lgbm'] +['gpr'] + ['xgb']# + ['cat'] # ['ngb'] 
#model_sets = [all_models] #+ [[m] for m in all_models]  # all + individual
model_sets = [[m] for m in all_models]  #+ [all_models] # individual + all 
random_splits = [True]  # random split or not
feature_selections = [ True, False]
n_runs = 50

# === Execute all combinations
for run in range(n_runs):
    for use_random_split in random_splits:
        for use_feature_selection in feature_selections:
            for models_to_use in model_sets:
                random_state = run
                np.random.seed(random_state)
                #print(f"\n{'='*60}")
                #print(f"Run #{run+1} | Models: {models_to_use} | Random Split: {use_random_split} | FS: {use_feature_selection}")
                #print(f"{'='*60}")
                output= optimize_stacking_model(
                    excel_path=excel_path,
                    models_to_use=models_to_use,
                    use_random_split=use_random_split,
                    use_feature_selection=use_feature_selection,
                    random_state=random_state,
                    run=run,
                )
                
                metrics = RegressionMetric(output['y_test'],output['y_pred'])
                print(f"{run:3d} | {output['model_name']: <15} | R2: {metrics.R2():.3f} | RMSE: {metrics.RMSE():.3f} | MAE: {metrics.MAE():.3f} | A10: {metrics.A10():.3f}")
                
                json_filename = f"model_run{run}_split{int(use_random_split)}_fs{int(use_feature_selection)}_{output['model_name']:_>15}.json".lower()
                json_filename = folder_path +json_filename
                with open(json_filename, "w") as f:
                    json.dump(output, f, indent=2, default=convert_np)
        
#%%