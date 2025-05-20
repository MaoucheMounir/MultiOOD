"""
datasets labels: UCF,    HMDB  , EPIC,          HAC (names inside filenames)
datasets names:  UCF101, HMDB51, EPIC-KITCHENS, HAC (dataset directory names)

J'ai fait en sorte que tous les fichiers d'évaluation sauvegardés (les saved_files) aillent dans HMDB-rgb-flow
Ce fichier est censé être chargé dans un script qui va boucler sur les backbones, méthodes, layer_proc, moda_wise ou pas, les datasets etc.
"""
# VERIFIER QUE CA MARCHE ET EN MODA WISE ET EN NORMAL,
# SELON LES DIFFERENTS CAS D4APPLICATION DE REACT OU PAS

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
ood_modes = ["near_ood", "far_ood"]
backbone_types = ["baseline", "a2d_npmix"]

test_filename = "test_video_flow"
eval_filename = "eval_video_flow"

####################################################

class Framework():
    def __init__(self, ood_mode:str, moda_wise:str):
        """
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

class NearOODFramework(Framework):
    def __init__(self, ood_mode:str, moda_wise:str):
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
