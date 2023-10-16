
from algorithm import FAWOS,Fair_RR, Fair_Sampling, Direct_training
import warnings

warnings.filterwarnings("ignore")

from Myfunctions import *



from joblib import Parallel, delayed
import time, math

# t_list = np.arange(75)/100
n_seeds = np.arange(100)
dataset_list = ['Lawschool','AdultCensus','COMPAS']
Fairness_list = ['DP','EO','PE']
# t1 = time.time()
# s = Parallel(n_jobs=10)(delayed(Direct_training)(seed,dataset) for seed in n_seeds for dataset in dataset_list)
# t2 = time.time()
# print(t2-t1)
#



# t1 = time.time()
# s = Parallel(n_jobs=10)(delayed(Fair_Sampling)(seed,dataset) for seed in n_seeds for dataset in dataset_list)
# t2 = time.time()
# print(t2-t1)


#
# t1 = time.time()
# s = Parallel(n_jobs=10)(delayed(FAWOS)(seed,dataset) for seed in n_seeds for dataset in dataset_list)
# t2 = time.time()
# print(t2-t1)


t1 = time.time()
s = Parallel(n_jobs=10)(delayed(Fair_RR)(seed,dataset,fairness) for seed in n_seeds for dataset in dataset_list for fairness in Fairness_list)
t2 = time.time()
print(t2-t1)

print('Finished')
