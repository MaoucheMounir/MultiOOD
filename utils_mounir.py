def save_results(method, dataset, layer_proc, fpr95, auroc, id_acc, exec_time):
    if "ash" in layer_proc:
        layer_proc = "ash"
    elif "react" in layer_proc:
        layer_proc = "react"
    else:
        layer_proc = "none"
        
    with open("results_postprocessings.csv", "a") as f:
        f.write("{},{},{},{:.3f},{:.3f},{:.3f},{}\n".format(
            method, dataset, layer_proc, fpr95, auroc, id_acc, exec_time))

def save_results_gen(backbone, method, dataset, layer_proc, fpr95, auroc, id_acc, exec_time, filename="results_postprocessings.csv", appen=""):
    if "ash" in layer_proc:
        layer_proc = "ash"
    elif "react" in layer_proc:
        layer_proc = "react"
    else:
        layer_proc = "none"
    
    backbone = "baseline" if "baseline" in backbone else "a2d_npmix"
    
    configuration = [backbone, method, dataset, layer_proc]
    if any(x in appen for x in ["video", "flow", "sound"]):
        configuration.append(appen.replace(f"baseline_best_{layer_proc}_", "")[:-1])
    
    metrics = [fpr95, auroc, id_acc, exec_time]
    
    with open(filename, "a") as f:
        f.write(",".join(configuration)+
                ",{:.4f},{:.4f},{:.4f},{}\n".format(*metrics))

