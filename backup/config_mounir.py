import numpy as np

"""
datasets labels: UCF,    HMDB  , EPIC,          HAC (names inside filenames)
datasets names:  UCF101, HMDB51, EPIC-KITCHENS, HAC (dataset directory names)

J'ai fait en sorte que tous les fichiers d'évaluation sauvegardés (les saved_files) aillent dans HMDB-rgb-flow
Ce fichier est censé être chargé dans un script qui va boucler sur les backbones, méthodes, layer_proc, moda_wise ou pas, les datasets etc.
"""
# VERIFIER QUE CA MARCHE ET EN MODA WISE ET EN NORMAL,
# SELON LES DIFFERENTS CAS D4APPLICATION DE layer_proc OU PAS

class Dataset():
    def __init__(self, name:str, label:str, ood_mode:str, vfa:str=''):
        """
        backbones_paths: dict[str:str]
        ood_mode: str, "near_ood" ou "far_ood"
        vfa: "vfa" ou "", pour indiquer si on utilise les 3 modalités ou pas. 
        """
        
        assert (vfa and ood_mode == "near_ood") or (not vfa) #vfa implique d'être en near_ood
        
        self.name = name
        self.label = label
        self.ood_mode = ood_mode
        self.vfa = vfa
        
    def backbone_path(self, backbone_type:str) -> str:
        if self.ood_mode == "far_ood" or self.label != "EPIC":
            prefix = "HMDB"
        else:
            prefix = "EPIC"
        #prefix = "EPIC" if self.label == "EPIC" else "HMDB"
        dataset = "HMDB" if self.ood_mode == "far_ood" else self.label
            
        return prefix+"-rgb-flow/{}_{}.pt".format(dataset, "_".join([self.ood_mode, self.vfa, backbone_type]).replace("__","_"))


def get_backbone_path(dataset, ood_mode, backbone_type):
    if dataset == "EPIC":
        return "EPIC-rgb-flow/{}_{}.pt".format(dataset, ood_mode+"_"+backbone_type)
    else:
        return "HMDB-rgb-flow/HMDB_{}.pt".format(dataset, ood_mode+"_"+backbone_type)

far_ood_datasets = [Dataset("UCF101", "UCF", "far_ood"), #Dataset("HMDB51", "HMDB", "far_ood")
                    Dataset("EPIC-KITCHENS", "EPIC", "far_ood"), Dataset("HAC", "HAC", "far_ood")]

near_ood_datasets = [Dataset("HMDB51", "HMDB", "near_ood"), Dataset("UCF101", "UCF", "near_ood"),
                     Dataset("EPIC-KITCHENS", "EPIC", "near_ood")] #

vfa_dataset = Dataset("EPIC-KITCHENS", "EPIC", "near_ood", "vfa") 

datasets = {'far_ood': far_ood_datasets, 'near_ood':near_ood_datasets, "vfa": vfa_dataset}

###################################################

modalities = ["video", "flow", "audio"]
modalities_near = modalities_far = ["video", "flow"]
ood_modes = ["near_ood", "far_ood"]
backbone_types = ["baseline", "a2d_npmix"]

test_filename = "test_video_flow"
eval_filename = "eval_video_flow"

####################################################

class Framework():
    def __init__(self, ood_mode:str, moda_wise:str=""):
        """
        ood_mode: str ("far_ood", "near_ood", "vfa")
        moda_wise: str ("" ou "moda_wise")
        """
        add_audio = "audio" if ood_mode == "vfa" else ""
        add_ood = "far_ood" if ood_mode == "far_ood" else "near_ood"
        #self.ood_mode = ood_mode
        self.datasets = datasets[ood_mode]
        self.test_filename = "_".join(["test_video_flow", add_audio, moda_wise]).strip("_").replace("__", "_")
        self.eval_filename = "_".join(["eval_video_flow", add_ood]).strip("_")
        self.modalities = [''] if not moda_wise else modalities if ood_mode == "vfa" else modalities[:-1]
        #self.results_filename = "eval_"+moda_wise+"_"+ood_mode #eval_moda_wise_ash_far_ood.csv
        self.saved_files_path = "/data/maouche/MultiOOD/HMDB-rgb-flow/saved_files/"
        
class NearOODFramework(Framework):
    def __init__(self, ood_mode:str, moda_wise:str=""):
        super().__init__(ood_mode, moda_wise)
        
        drop_modality = "" if not moda_wise else "--drop_modality {modality}"
        # --datapath: le chemin du dataset à utiliser
        # --dataset: le label du dataset à utiliser
        
        self.test_command_template = (
            "python {test_filename}.py --datapath /data/maouche/MultiOOD/{datapath}/ "
            "--bsz 16 --num_workers 2 --near_ood --dataset '{dataset}' {sparsification_method} "
            f"{drop_modality} " + "--appen '{backbone}_best_' "
            "--resumef '/data/maouche/MultiOOD/{resume_path}' "
            
            "2>error_test_{dataset}_"+ood_mode+"_{sparsification_suffix}{modality}.log | "
            "tee out_test_{dataset}_"+ood_mode+"_{sparsification_suffix}{modality}.log"
        )
        
        self.eval_command_template = (
            f"python {self.eval_filename}.py " + "--postprocessor {postprocessor} "
            "--appen '{backbone}_best_{sparsification_suffix}{modality}_' "
            "--dataset '{dataset}' --path 'HMDB-rgb-flow/' "
            
            "2>error_eval_{dataset}_"+ood_mode+"_{sparsification_suffix}{modality}.log | "
            "tee out_eval_{dataset}_"+ood_mode+"_{sparsification_suffix}{modality}.log"
        )
        

    def get_confs(self, domain, layer_proc):
        """
        retourne les scores de confiance (max MSP)
        args:
        domain: str ("id", "ood")
        """
        root_dir = self.saved_files_path
        assert domain in ["id", "ood"]
        split = "test" if domain == "id" else "eval"
                
        template_sans_layer_proc =  "id_{}_near_ood_conf_baseline_best_"+split+".npy"
        template_layer_proc_tout = "id_{}_near_ood_conf_baseline_best_"+layer_proc+"_"+split+".npy"
        template_par_modalite = "id_{}_near_ood_conf_baseline_best_"+layer_proc+"_{}_"+split+".npy"

        datasets_id = {}

        for dataset in [ds.label for ds in near_ood_datasets]:

            # Sans layer_proc
            conf_sans_layer_proc = np.load(root_dir+template_sans_layer_proc.format(dataset))
            conf_sans_layer_proc = conf_sans_layer_proc.reshape(conf_sans_layer_proc.shape[0], 1)

            # layer_proc sur tout
            conf_layer_proc_tout = np.load(root_dir+template_layer_proc_tout.format(dataset))
            conf_layer_proc_tout = conf_layer_proc_tout.reshape(conf_layer_proc_tout.shape[0], 1)


            # layer_proc par modalité
            conf_une_modalite = []
            for modality in modalities_near:
                x = np.load(root_dir+template_par_modalite.format(dataset, modality))
                x = x.reshape(x.shape[0], 1)
                conf_une_modalite.append(x)
            conf_une_modalite = np.hstack(conf_une_modalite)

            dataset_id = np.hstack([conf_sans_layer_proc, conf_layer_proc_tout, conf_une_modalite]) #(N,5)
            datasets_id[dataset] = dataset_id
        return datasets_id
    

class FarOODFramework(Framework):
    def __init__(self, moda_wise):
        super().__init__("far_ood", moda_wise)
        drop_modality = "" if not moda_wise else "--drop_modality {modality}"
        drop_modality_suffix = "" if not moda_wise else "{modality}"
        
        self.hmdb_test_command_template = (
            f"python {self.test_filename}.py --datapath /data/maouche/MultiOOD/HMDB51/ "
            "--bsz 16 --num_workers 2 --dataset 'HMDB' {sparsification_method} " +
            f"{drop_modality} " + "--appen '{backbone}_best_' "
            "--resumef '/data/maouche/MultiOOD/HMDB-rgb-flow/HMDB_far_ood_{backbone}.pt' "
            "2>error_test_HMDB_far_ood_{sparsification_suffix}"+f"{drop_modality_suffix}.log | "
            "tee out_test_HMDB_far_ood_{sparsification_suffix}"+f"{drop_modality_suffix}.log"
        )
        
        #--datapath: le chemin du dataset OOD
        #--dataset: le label du dataset ID (ici on utilise que HMDB, Kinetics n'est pas disponible)
        #La variable 'dataset' se réfère au dataset OOD
        self.test_command_template = (
            "python {test_filename}.py --datapath /data/maouche/MultiOOD/{datapath}/ "
            "--bsz 16 --num_workers 2 --far_ood --dataset 'HMDB' {sparsification_method} "
            f"{drop_modality} " + "--ood_dataset '{dataset}' --appen '{backbone}_best_' "
            "--resumef '/data/maouche/MultiOOD/{resume_path}' "
            
            "2>error_test_{dataset}_far_ood_{sparsification_suffix}"+f"{drop_modality_suffix}.log | "
            "tee out_test_{dataset}_far_ood_{sparsification_suffix}"+f"{drop_modality_suffix}.log"
        )

        self.eval_command_template = (
            f"python {self.eval_filename}.py " + "--postprocessor msp " 
            "--appen 'baseline_best_{sparsification_suffix}{modality}_' "
            "--dataset 'HMDB' --ood_dataset '{dataset}' --path 'HMDB-rgb-flow/' "
            
            "2>error_eval_{dataset}_far_ood_{sparsification_suffix}"+f"{drop_modality_suffix}.log | "
            "tee out_eval_{dataset}_far_ood_{sparsification_suffix}"+f"{drop_modality_suffix}.log"
        )
    
    def get_confs(self, domain, layer_proc):
        assert domain in ["id", "ood"]
        if domain == "id":
            return self.get_id_data(layer_proc)
        else:
            return self.get_ood_data(layer_proc)
    
    def get_id_data(self, layer_proc):
        template_id_sans_react =  "id_HMDB_conf_baseline_best_val.npy"
        template_id_react_tout = "id_HMDB_conf_baseline_best_"+layer_proc+"_val.npy"
        template_id_par_modalite = "id_HMDB_conf_baseline_best_"+layer_proc+"_{}_val.npy"

        root_dir = self.saved_files_path
        # Sans react
        conf_sans_react = np.load(root_dir+template_id_sans_react)
        conf_sans_react = conf_sans_react.reshape(conf_sans_react.shape[0], 1)

        # React sur tout
        conf_react_tout = np.load(root_dir+template_id_react_tout)
        conf_react_tout = conf_react_tout.reshape(conf_react_tout.shape[0], 1)


        # React par modalité
        conf_une_modalite = []
        for modality in modalities_far:
            x = np.load(root_dir+template_id_par_modalite.format(modality))
            x = x.reshape(x.shape[0], 1)
            conf_une_modalite.append(x)
        conf_une_modalite = np.hstack(conf_une_modalite)


        dataset_id = np.hstack([conf_sans_react, conf_react_tout, conf_une_modalite]) #(N,4)
        return dataset_id

    def get_ood_data(self, layer_proc):
        template_sans_react =  "id_HMDB_ood_{}_conf_baseline_best_eval.npy"
        template_react_tout = "id_HMDB_ood_{}_conf_baseline_best_"+layer_proc+"_eval.npy"
        template_par_modalite = "id_HMDB_ood_{}_conf_baseline_best_"+layer_proc+"_{}_eval.npy"
        
        ood_datasets = ['UCF', 'EPIC']#, 'HAC']
        root_dir = self.saved_files_path
        
        # Sans react
        conf_sans_react = []
        for dataset in ood_datasets:
            x = np.load(root_dir+template_sans_react.format(dataset))
            x = x.reshape(x.shape[0], 1)
            conf_sans_react.append(x)
        conf_sans_react = np.vstack(conf_sans_react) #(N,1), avec N = 9603, toutes les vidéos des 3 datasets (selon les filtrages far ood)

        # React sur tout 
        conf_react_tout = []
        for dataset in ood_datasets:
            x = np.load(root_dir+template_react_tout.format(dataset))
            x = x.reshape(x.shape[0], 1)
            conf_react_tout.append(x)
        conf_react_tout = np.vstack(conf_react_tout) #(N,1)


        # React par modalité
        conf_par_modalite = []
        for modality in modalities_far:
            conf_une_modalite = []
            for dataset in ood_datasets:
                x = np.load(root_dir+template_par_modalite.format(dataset, modality))
                x = x.reshape(x.shape[0], 1)
                conf_une_modalite.append(x)
            conf_une_modalite = np.vstack(conf_une_modalite)
            conf_par_modalite.append(conf_une_modalite)

        conf_par_modalite = np.reshape(conf_par_modalite, (-1, len(modalities_far)))  #(N,1)

        dataset_ood = np.hstack([conf_sans_react, conf_react_tout, conf_par_modalite]) #(N,4)
        return dataset_ood
    
        






# def get_confs(self, layer_proc):
#         root_dir = self.saved_files_path
#         #Near vfa pour le coup
#         template_sans_layer_proc =  "id_EPIC_near_ood_conf_vfa_baseline_best_test.npy"
#         template_layer_proc_tout = "id_EPIC_near_ood_conf_vfa_baseline_best_"+layer_proc+"_test.npy"
#         template_par_modalite = "id_EPIC_near_ood_conf_vfa_baseline_best_"+layer_proc+"_{}_test.npy"

#         # Sans layer_proc
#         conf_sans_layer_proc = np.load(root_dir+template_sans_layer_proc)
#         conf_sans_layer_proc = conf_sans_layer_proc.reshape(conf_sans_layer_proc.shape[0], 1)

#         # layer_proc sur tout
#         conf_layer_proc_tout = np.load(root_dir+template_layer_proc_tout)
#         conf_layer_proc_tout = conf_layer_proc_tout.reshape(conf_layer_proc_tout.shape[0], 1)


#         # layer_proc par modalité
#         conf_une_modalite = []
#         for modality in modalities:
#             x = np.load(root_dir+template_par_modalite.format(modality))
#             x = x.reshape(x.shape[0], 1)
#             conf_une_modalite.append(x)
#         conf_une_modalite = np.hstack(conf_une_modalite)

#         dataset_id = np.hstack([conf_sans_layer_proc, conf_layer_proc_tout, conf_une_modalite]) #(N,5)
#         return dataset_id
        
        