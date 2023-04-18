# Introduce

## Data

```shell
#############################################
data 
├── cleaned_source_data		(cleaned  source data) 
│   ├── cphmd_csv
│   │   ├── CpHMD_pka274.csv	    (pKa results calculated by CpHMD on dataset HH274)
│   │   └── CpHMD_pka_WT69.csv	    (pKa results calculated by CpHMD on dataset HH69)
│   ├── data_pdb_CpHMD274	 (the directory containing dataset HH274's single chain pdb files)
│   ├── data_pdb_WT_accurate     (the directory containing dataset HH69's single chain pdb files)
│   └── expt_cleaned_csv
│       └── final_expt_pka.csv  (experimental pKa results on HH69) 
├── model_input		(model input data (use class PkaDataset to load))
│   ├── final_test_data
│   │   ├── data_pdb_WT_fixed_mol2.csv      (proteins' heavy atoms' features of HH69)
│   │   ├── final_expt_pka_center_coors.csv     (center coordinates of titratable residues of dataset HH69)
│   │   ├── test_chimera_f18_r4_incphmd.csv     (titratable residues' heavy atoms' 18 features of HH69)
│   │   └── test_chimera_f18_r4_incphmd_undersample.csv     (titratable residues' heavy atoms' 18 features of HH69S)
│   │   ├── test_chimera_f19_r4_incphmd.csv     (titratable residues' heavy atoms' 19 features of HH69)
│   │   └── test_chimera_f19_r4_incphmd_undersample.csv     (titratable residues' heavy atoms' 19 features of HH69S)
│   │   ├── test_chimera_f20_r4_incphmd.csv     (titratable residues' heavy atoms' 20 features of HH69)
│   │   └── test_chimera_f20_r4_incphmd_undersample.csv     (titratable residues' heavy atoms' 20 features of HH69S)
│   ├── final_train_data
│   │   ├── CpHMD_pka247_center_coors.csv   (center coordinates of titratable residues of dataset HH247)
│   │   ├── data_pdb_CpHMD247_fixed_mol2.csv    (proteins' heavy atoms' features of HH247)
│   │   └── train_n247_f18_n4.csv   (titratable residues' heavy atoms' 18 features of HH247)
│   │   └── train_n247_f19_n4.csv   (titratable residues' heavy atoms' 19 features of HH247)
│   │   └── train_n247_f20_n4.csv   (titratable residues' heavy atoms' 20 features of HH247)
│   ├── final_val_data
│   │   ├── CpHMD_pka27_center_coors.csv    (center coordinates of titratable residues of dataset HH27)
│   │   ├── data_pdb_CpHMD27_fixed_mol2.csv     (proteins' heavy atoms' features of HH27)
│   │   └── val_n27_f18_n4.csv      (titratable residues' heavy atoms' 18 features of HH27)
│   │   └── val_n27_f19_n4.csv      (titratable residues' heavy atoms' 19 features of HH27)
│   │   └── val_n27_f20_n4.csv      (titratable residues' heavy atoms' 20 features of HH27)
│   ├── old_train_data      (old data, not use anymore, temporarily reserved)
│   │   ├── CpHMD_pka279_center_coors.csv   
│   │   ├── data_pdb_CpHMD279_fixed_mol2.csv
│   │   └── train_n279_f19_n4.csv
│   └── old_val_data    (old data, not use anymore, temporarily reserved)
│       ├── data_pdb_WT_fixed_mol2.csv
│       ├── final_expt_pka_center_coors.csv
│       ├── val_chimera_f19_r4_incphmd.csv
│       └── val_chimera_f19_r4_incphmd_undersample.csv

└── predict_result 		(mpdel predicting result on dataset HH69 and HH69S)
    ├── atom_charge_result  (predicting pKa result of DeepKa using atomic charges)
    └── grid_charge_result  (predicting pKa result of DeepKa using grid charges)
└── preprocess_temp_data 		(empty directory, when run deepka/pka_process/main.py,
                                          the tempt data will created in this directory.)
    └── ...
########################################################
datasets                      (This datasets contains train dataset, validate and test dataset's residue's information, include pKa) 
├── test_n69.csv              (dataset HH69's residues information, the pKa value is experimental result)
├── test_n69_undersample.csv  (dataset HH69S's residues information, the pKa value is experimental result)
├── train_n247.csv            (dataset HH247's residues information, the pKa value is calculated by CpHMD)
└── val_n27.csv               (dataset HH27's residues information, the pKa value is calculated by CpHMD)
```
## pka_process

The main function of the program is used to prepare the input data of the Deepka.

### Installation

Install Anaconda, Charmm, Chimera, then

```
cd pka_process
conda env create -f evironment.yml
```

### Usage

Using functions preprocess_expt_data() and preprocess_features() in 'pka_process/main.py' can create validating input data(like the data in 'data/model/input/final_val_data').

Using functions preprocess_CpHMD_data() and preprocess_features() in 'pka_process/main.py' can create training input data(like the data in 'data/model/input/final_train_data').

## pka_predict

The main functions of the program is used to train models and evaluate models.

### Installation

Install Anaconda, Cuda10.1 (if use gpu), Cudnn(if use gpu), then

```
cd pka_predict
conda env create -f environment.yml
```

(If Pytorch unusable, try to install Pytroch by self.)

### Usage

Using 'pka_predict/train.py' to train models.

Using function evaluate_single_model() in 'pka_predict/evaluate.py' to evaluate models.


### Citation
Cai, Z.; Luo, F.; Wang, Y.; Li, E.; Huang, Y. Protein pKa prediction with machine learning. ACS Omega  2021, 6, 34823−34831

