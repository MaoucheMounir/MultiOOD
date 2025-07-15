import numpy as np
from .datasets import datasets, near_ood_datasets

###################################################

ALL_MODALITIES = ["video", "flow", "audio"]
#MODALITIES_NEAR = MODALITIES_FAR = ["video", "flow"]
OOD_MODES = ["near_ood", "far_ood"]
BACKBONE_TYPES = ["baseline", "a2d_npmix"]
#MODALITIES_NEAR = MODALITIES_FAR = ["video", "flow"]
MODALITIES = {"near_ood": ["video", "flow"], "far_ood": ["video", "flow"], "vfa":["video", "flow", "audio"]}
TEST_FILENAME = "test_video_flow"
EVAL_FILENAME = "eval_video_flow"
# VERIFIER QUE CA MARCHE ET EN MODA WISE ET EN NORMAL,
# SELON LES DIFFERENTS CAS D4APPLICATION DE layer_proc OU PAS

####################################################

class Framework():
    def __init__(self, ood_mode:str, moda_wise:str=""):
        """
        ood_mode: str ("far_ood", "near_ood", "vfa")
        moda_wise: str ("" ou "moda_wise")
        """
        audio_tag = "audio" if ood_mode == "vfa" else ""
        ood_tag = "far_ood" if ood_mode == "far_ood" else "near_ood"
        self.ood_mode = ood_mode
        self.moda_wise = moda_wise
        
        self.datasets = datasets[ood_mode]
        self.test_filename = "_".join(["test_video_flow", audio_tag, moda_wise]).strip("_").replace("__", "_")
        self.eval_filename = "_".join(["eval_video_flow", ood_tag]).strip("_")
        #self.modalities = [''] if not moda_wise else modalities if ood_mode == "vfa" else modalities[:-1]
        self.modalities = MODALITIES[self.ood_mode]
        #self.results_filename = "eval_"+moda_wise+"_"+ood_mode #eval_moda_wise_ash_far_ood.csv
        self.saved_files_path = "/data/maouche/MultiOOD/HMDB-rgb-flow/saved_files/"

        
    def load_conf(self, template):
        conf = np.load(self.saved_files_path+template)
        return conf.reshape(conf.shape[0], 1)
        
    def load_conf_moda_wise(self, template, *args):
        # layer_proc par modalité
        conf_une_modalite = []
        for modality in self.modalities:
            conf_une_modalite.append(self.load_conf(template.format(*args,modality)))
        conf_une_modalite = np.hstack(conf_une_modalite)
        return conf_une_modalite
    
    def load_pred(self, template):
        pass


class NearOODFramework(Framework):
    def __init__(self, ood_mode:str="near_ood", moda_wise:str=""):
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
        if layer_proc == "none":
            return self.get_vanilla_confs(domain)
        if self.ood_mode == "vfa":
            return self.get_confs_vfa(domain, layer_proc) # np.ndarray 
        else:
            return self.get_confs_near_vf(domain, layer_proc)  # dict[str] par dataset
        
    def get_confs_near_vf(self, domain, layer_proc) -> dict:
        """
        Retourne les scores de confiance (max MSP)
        args:
        domain: str ("id", "ood")
        Le fait que ce soit ID ou OOD dépend du split
        """
        assert domain in ["id", "ood"]
        split = "test" if domain == "id" else "eval"
                
        template_sans_layer_proc =  "id_{}_near_ood_conf_baseline_best_"+split+".npy"
        template_layer_proc_tout = "id_{}_near_ood_conf_baseline_best_"+layer_proc+"_"+split+".npy"
        template_par_modalite = "id_{}_near_ood_conf_baseline_best_"+layer_proc+"_{}_"+split+".npy"

        datasets = {}
        
        for dataset_name in [ds.label for ds in near_ood_datasets]:

            # Sans layer_proc
            conf_sans_layer_proc = self.load_conf(template_sans_layer_proc.format(dataset_name))
            
            # layer_proc sur tout
            conf_layer_proc_tout = self.load_conf(template_layer_proc_tout.format(dataset_name))
            
            conf_une_modalite = self.load_conf_moda_wise(template_par_modalite, dataset_name)
            dataset = np.hstack([conf_sans_layer_proc, conf_layer_proc_tout, conf_une_modalite]) #(N,5)
            datasets[dataset_name] = dataset
        
        return datasets
    
    def get_confs_vfa(self, domain, layer_proc):
        """
        Retourne les scores de confiance (max MSP)
        args:
        domain: str ("id", "ood")
        """
        assert domain in ["id", "ood"]
        split = "test" if domain == "id" else "eval"
                
        template_sans_layer_proc =  "id_EPIC_near_ood_conf_vfa_baseline_best_"+split+".npy"
        template_layer_proc_tout = "id_EPIC_near_ood_conf_vfa_baseline_best_"+layer_proc+"_"+split+".npy"
        template_par_modalite = "id_EPIC_near_ood_conf_vfa_baseline_best_"+layer_proc+"_{}_"+split+".npy"

        # Sans layer_proc
        conf_sans_layer_proc = self.load_conf(template_sans_layer_proc)
        
        # layer_proc sur tout
        conf_layer_proc_tout = self.load_conf(template_layer_proc_tout)
        
        # React par modalité
        conf_une_modalite = self.load_conf_moda_wise(template_par_modalite)
        dataset_id = np.hstack([conf_sans_layer_proc, conf_layer_proc_tout, conf_une_modalite]) #(N,5)
        
        return dataset_id
    
    def get_vanilla_confs(self, domain):
        assert domain in ["id", "ood"]
        split = "test" if domain == "id" else "eval"
        
        if self.ood_mode == "near_ood":
            template_sans_layer_proc =  "id_{}_near_ood_conf_baseline_best_"+split+".npy"
            datasets = {}
            
            for dataset in [ds.label for ds in near_ood_datasets]:
                conf_sans_layer_proc = self.load_conf(template_sans_layer_proc.format(dataset))
                datasets[dataset] = conf_sans_layer_proc    
            return datasets
        
        else: #vfa
            template_sans_layer_proc =  "id_EPIC_near_ood_conf_vfa_baseline_best_"+split+".npy"
            conf_sans_layer_proc = self.load_conf(template_sans_layer_proc)
            
            return conf_sans_layer_proc

class FarOODFramework(Framework):
    def __init__(self, moda_wise=""):
        super().__init__("far_ood", moda_wise)
        ood_datasets = ['UCF', 'EPIC'] #HAC
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
    
    def get_confs(self, domain, layer_proc, dataset=""):
        assert domain in ["id", "ood"]
        if layer_proc == "none":
            return self.get_vanilla_confs(domain, dataset)
    
        if domain == "id":
            return self.get_id_data(layer_proc)
        else:
            return self.get_ood_data(layer_proc)
    
    def get_id_data(self, layer_proc):
        # le split était val il devient test pour être cohérent avec le papier
        template_id_sans_react =  "id_HMDB_conf_baseline_best_test.npy"
        template_id_react_tout = "id_HMDB_conf_baseline_best_"+layer_proc+"_test.npy"
        template_id_par_modalite = "id_HMDB_conf_baseline_best_"+layer_proc+"_{}_test.npy"

        # Sans react
        conf_sans_react = self.load_conf(template_id_sans_react)
        
        # React sur tout
        conf_react_tout = self.load_conf(template_id_react_tout)

        # React par modalité
        conf_une_modalite = self.load_conf_moda_wise(template_id_par_modalite)
        
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
            x = self.load_conf(template_sans_react.format(dataset))
            conf_sans_react.append(x)
        conf_sans_react = np.vstack(conf_sans_react) #(N,1), avec N = 9603, toutes les vidéos des 3 datasets (selon les filtrages far ood)

        # React sur tout 
        conf_react_tout = []
        for dataset in ood_datasets:
            x = self.load_conf(template_react_tout.format(dataset))
            conf_react_tout.append(x)
        conf_react_tout = np.vstack(conf_react_tout) #(N,1)
            
        # React par modalité
        conf_par_modalite = []
        for modality in MODALITIES["far_ood"]:
            conf_une_modalite = []
            for dataset in ood_datasets:
                x = np.load(root_dir+template_par_modalite.format(dataset, modality))
                x = x.reshape(x.shape[0], 1)
                conf_une_modalite.append(x)
            conf_une_modalite = np.vstack(conf_une_modalite)
            conf_par_modalite.append(conf_une_modalite)

        conf_par_modalite = np.reshape(conf_par_modalite, (-1, len(MODALITIES["far_ood"])))  #(N,1)

        dataset_ood = np.hstack([conf_sans_react, conf_react_tout, conf_par_modalite]) #(N,4)
        return dataset_ood
    
    def get_vanilla_confs(self, domain, dataset):
        # Certifié rend les même résultats que le code du papier
        ood_datasets = ['UCF', 'EPIC']
        if domain == "id": 
            template_id_sans_react =  "id_HMDB_conf_baseline_best_test.npy"
            conf_sans_react = self.load_conf(template_id_sans_react)
        
        if domain == "ood":
            template_sans_react =  "id_HMDB_ood_{}_conf_baseline_best_eval.npy"
            conf_sans_react = []
            if dataset:
                ood_datasets = [dataset]
            for dataset in ood_datasets:
                x = self.load_conf(template_sans_react.format(dataset))
                conf_sans_react.append(x)
            conf_sans_react = np.vstack(conf_sans_react) #(N,1), avec N = 9603, toutes les vidéos des 3 datasets (selon les filtrages far ood)
        return conf_sans_react
####################################################

FRAMEWORK_MAP = {
    "near_ood": NearOODFramework,
    "vfa": NearOODFramework,
    "far_ood": FarOODFramework
}

def FrameworkFactory(ood_mode: str, moda_wise: str = "") -> Framework:
    if ood_mode not in FRAMEWORK_MAP:
        raise ValueError(f"ood_mode inconnu : {ood_mode}")
    
    return FRAMEWORK_MAP[ood_mode](ood_mode=ood_mode, moda_wise=moda_wise) if ood_mode != "far_ood" else FRAMEWORK_MAP[ood_mode](moda_wise=moda_wise)
