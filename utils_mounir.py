import pandas as pd
from collections import defaultdict
from config_mounir import modalities
import numpy as np

def get_modality(appen):
    result = []
    for modality in modalities:
        if modality in appen:
            result.append(modality)
    
    return "_".join(result)

def create_file(prefix, appen):
    # Names the csv file and fills the header
    
    modality = get_modality(appen)
    
    add_drop_modality = ["drop_modality"] if modality else []        
    add_comb = "comb_" if "comb" in appen else ""
    
    file_header = ["backbone", "method", "dataset", "layer_proc"] + \
                        add_drop_modality + ["fpr95", "auroc", "id_acc", "exec_time"]
    backbone = "baseline" if "baseline" in appen else "a2d_npmix"
    
    if "ash" in appen:
        layer_proc = "ash"
    elif "react" in appen:
        layer_proc = "react"
    else:
        layer_proc = ""
    
    filename = prefix + f'{add_comb}{layer_proc}.csv'
    
    with open(filename, "w") as f:
        f.write(",".join(file_header)+"\n")
        
    return filename


def save_results_gen(backbone, method, dataset, layer_proc, fpr95, auroc, id_acc, exec_time, filename, appen=""):
    if "ash" in layer_proc:
        layer_proc = "ash"
    elif "react" in layer_proc:
        layer_proc = "react"
    else:
        layer_proc = "none"
    
    backbone = "baseline" if "baseline" in backbone else "a2d_npmix"
    
    configuration = [backbone, method, dataset, layer_proc]
    #ajouter quelle modalité a été processée
    if any(x in appen for x in ["video", "flow", "sound"]):
        configuration.append(appen.replace(f"baseline_best_{layer_proc}_", "")[:-1])
    
    metrics = [fpr95, auroc, id_acc, exec_time]
    
    modality = get_modality(appen)
    if modality:
        configuration.append(modality)

    with open(filename, "a") as f:           
        f.write(",".join(configuration)+
                ",{:.4f},{:.4f},{:.4f},{}\n".format(*metrics))  
        
########################################
# Formatage Dataframes Présentation
def normalize_columns(df, columns):
    for column in columns:
        df[column]= df[column].apply(lambda x:x*100)
    return df

def drop_columns(df, columns):
    for column in columns:
        df = df.drop(column, axis=1) 
    return df

def column_names(df):
    results = defaultdict(list)
    metrics = ["fpr95", "auroc"]
    grouping = "drop_modality" if "drop_modality" in df.columns else "method"
    
    for name, group in df.groupby(by=grouping):
        results[grouping].append(name)
        for i, dataset in group.iterrows():
            for metric in metrics:
                results[
                    f"{dataset['dataset']}_{metric}"
                    ].append(dataset[metric])
    return pd.DataFrame(results)

def order_modalities(df):
    df['modality'] = pd.Categorical(df['modality'], categories=modalities, ordered=True)
    df = df.sort_values(['modality'])
    return df

#######################################################

def max_perturbations(id_conf_sans, id_confs, ood_confs):
    """
    id_conf_sans: scores de confiance obtenus sans react/ash
    id_conf, ood_confs: scores de confiance obtenus avec react/ash
    """
    max_perturbations = -1
    max_idx = -1
    
    for i, conf in enumerate(id_confs):
        perturbation = np.linalg.norm(id_conf_sans-conf)
        if perturbation > max_perturbations:
            max_perturbations = perturbation
            max_idx = i
    
    assert max_idx != -1 
    
    return ood_confs[max_idx]

    
        
