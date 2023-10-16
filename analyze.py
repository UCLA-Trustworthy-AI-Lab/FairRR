import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


result_FAWOS = pd.DataFrame()
result_Direct_training = pd.DataFrame()
result_FairRR = pd.DataFrame()
result_FairSampling= pd.DataFrame()


n_seeds = np.arange(100)
dataset_list = ['Lawschool','AdultCensus','COMPAS']
Fairness_list = ['DP','EO','PE']
classifier_name = 'LR'
for seed in n_seeds:
    for dataset_name in dataset_list:
        temp_Fawos = pd.read_csv(f'Result/FAWOS/{dataset_name}/result_of_seed_{seed}')
        result_FAWOS = pd.concat([result_FAWOS,temp_Fawos])
        temp_Direct_training = pd.read_csv(f'Result/Direct_training/{dataset_name}/result_of_seed_{seed}')
        result_Direct_training = pd.concat([result_Direct_training,temp_Direct_training])
        temp_Fair_sampling = pd.read_csv(f'Result/Fiar_Sampling/{dataset_name}/result_of_seed_{seed}')
        result_FairSampling = pd.concat([result_FairSampling,temp_Fair_sampling])
        for fairness in Fairness_list:
            temp_FairRR = pd.read_csv(f'Result/FairRR/{dataset_name}/{fairness}/result_of_seed_{seed}')
            result_FairRR = pd.concat([result_FairRR, temp_FairRR])

for i in range(3):
    acc = round(result_FAWOS[::10][i::3].mean()['acc'] * 1000) / 1000
    DDP = round(result_FAWOS[::10][i::3].mean()['DDP'] * 1000) / 1000
    accs = round(result_FAWOS[::10][i::3].std()['acc'] * 1000) / 1000
    DDPs = round(result_FAWOS[::10][i::3].std()['DDP'] * 1000) / 1000
    print( f'  {acc}  &  {DDP}   \\\\ ')
    print( f' ({accs}) & ({DDPs})  \\\\ ')


for i in range(3):
    acc = round(result_FairSampling[i::3].mean()['acc'] * 1000) / 1000
    DDP = round(result_FairSampling[i::3].mean()['DDP'] * 1000) / 1000
    accs = round(result_FairSampling[i::3].std()['acc'] * 1000) / 1000
    DDPs = round(result_FairSampling[i::3].std()['DDP'] * 1000) / 1000
    print( f'  {acc}  &  {DDP}   \\\\ ')
    print( f' ({accs}) & ({DDPs})  \\\\ ')


for i in range(3):
    acc = round(result_Direct_training[i::3].mean()['acc'] * 1000) / 1000
    f1 = round(result_Direct_training[i::3].mean()['f1'] * 1000) / 1000
    DDP = round(result_Direct_training[i::3].mean()['DDP'] * 1000) / 1000
    DEO = round(result_Direct_training[i::3].mean()['DEO'] * 1000) / 1000
    DPE = round(result_Direct_training[i::3].mean()['DPE'] * 1000) / 1000
    accs = round(result_Direct_training[i::3].std()['acc'] * 1000) / 1000
    f1s = round(result_Direct_training[i::3].std()['f1'] * 1000) / 1000
    DDPs = round(result_Direct_training[i::3].std()['DDP'] * 1000) / 1000
    DEOs = round(result_Direct_training[i::3].std()['DEO'] * 1000) / 1000
    DPEs = round(result_Direct_training[i::3].std()['DPE'] * 1000) / 1000
    print( f'  {acc}  &  {f1}  &  {DDP}  &  {DEO}  &  {DPE}  \\\\ ')
    print( f' ({accs}) & ({f1s}) & ({DDPs}) & ({DEOs}) & ({DPEs}) \\\\ ')




for i in range(3):
    # acc_dp = round(result_FairRR[::10][::3][i::3].mean()['acc'] * 1000) / 1000
    # f1_dp = round(result_FairRR[::10][::3][i::3].mean()['f1'] * 1000) / 1000
    # DDP = round(result_FairRR[::10][::3][i::3].mean()['DDP'] * 1000) / 1000
    # acc_eo = round(result_FairRR[::10][1::3][i::3].mean()['acc'] * 1000) / 1000
    # f1_eo = round(result_FairRR[::10][1::3][i::3].mean()['f1'] * 1000) / 1000
    # DEO = round(result_FairRR[::10][1::3][i::3].mean()['DEO'] * 1000) / 1000
    # acc_pe = round(result_FairRR[::10][2::3][i::3].mean()['acc'] * 1000) / 1000
    # f1_pe = round(result_FairRR[::10][2::3][i::3].mean()['f1'] * 1000) / 1000
    # DPE = round(result_FairRR[::10][2::3][i::3].mean()['DPE'] * 1000) / 1000
    acc_dp = round(result_FairRR[::10][::3][i::3].std()['acc'] * 1000) / 1000
    f1_dp = round(result_FairRR[::10][::3][i::3].std()['f1'] * 1000) / 1000
    DDP = round(result_FairRR[::10][::3][i::3].std()['DDP'] * 1000) / 1000
    acc_eo = round(result_FairRR[::10][1::3][i::3].std()['acc'] * 1000) / 1000
    f1_eo = round(result_FairRR[::10][1::3][i::3].std()['f1'] * 1000) / 1000
    DEO = round(result_FairRR[::10][1::3][i::3].std()['DEO'] * 1000) / 1000
    acc_pe = round(result_FairRR[::10][2::3][i::3].std()['acc'] * 1000) / 1000
    f1_pe = round(result_FairRR[::10][2::3][i::3].std()['f1'] * 1000) / 1000
    DPE = round(result_FairRR[::10][2::3][i::3].std()['DPE'] * 1000) / 1000
    print(f' ({acc_dp}) & ({f1_dp}) & ({DDP}) & ({acc_eo}) & ({f1_dp}) & ({f1_eo}) & ({acc_pe}) & ({f1_pe}) & ({DPE}) \\\\ ')