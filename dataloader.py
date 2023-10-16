import os
import copy
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
#
# def adult(display=False):
#     """ Return the Adult census data in a nice package. """
#     dtypes = [
#         ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
#         ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
#         ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
#         ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
#         ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
#     ]
#     raw_train_data = pd.read_csv(
#          'Adult_data/adult.data',
#         names=[d[0] for d in dtypes],
#         na_values="?",
#         dtype=dict(dtypes)
#     )
#     raw_test_data = pd.read_csv(
#          'Adult_data/adult.test',
#         skiprows=1,
#         names=[d[0] for d in dtypes],
#         na_values="?",
#         dtype=dict(dtypes)
#     )
#
#
#     train_data = raw_train_data.drop(["Education"], axis=1)  # redundant with Education-Num
#     test_data = raw_test_data.drop(["Education"], axis=1)  # redundant with Education-Num
#     filt_dtypes = list(filter(lambda x: not (x[0] in ["Target", "Education"]), dtypes))
#
#     rcode = {
#         "Not-in-family": 0,
#         "Unmarried": 1,
#         "Other-relative": 2,
#         "Own-child": 3,
#         "Husband": 4,
#         "Wife": 5
#     }
#     for k, dtype in filt_dtypes:
#         if dtype == "category":
#             if k == "Relationship":
#                 train_data[k] = np.array([rcode[v.strip()] for v in train_data[k]])
#                 test_data[k] = np.array([rcode[v.strip()] for v in test_data[k]])
#             else:
#                 train_data[k] = train_data[k].cat.codes
#                 test_data[k] = test_data[k].cat.codes
#
#
#     # 拆分数据框
#
#
#
#     train_data["Target"] = train_data["Target"] == " >50K"
#     test_data["Target"] = test_data["Target"] == " >50K."
#     all_data = pd.concat([train_data,test_data])
#     data_size = len(all_data)
#     new_index_all = np.arange(data_size)
#     all_data.index = new_index_all
#
#
#
#
#     return all_data.drop('fnlwgt',axis = 1)
#
#
#
#
# def compas():
#     """ Downloads COMPAS data from the propublica GitHub repository.
#     :return: pandas.DataFrame with columns 'sex', 'age', 'juv_fel_count', 'juv_misd_count',
#        'juv_other_count', 'priors_count', 'two_year_recid', 'age_cat_25 - 45',
#        'age_cat_Greater than 45', 'age_cat_Less than 25', 'race_African-American',
#        'race_Caucasian', 'c_charge_degree_F', 'c_charge_degree_M'
#     """
#
#
#
#     dtypes = [
#         ("sex", "category"), ("age", "float32"), ("age_cat", "category"),
#         ("race", "category"), ("juv_fel_count", "float32"), ("juv_misd_count", "float32"),
#         ("juv_other_count", "float32"), ("priors_count", "float32"), ("c_charge_degree", "category"),
#     ]
#
#     data = pd.read_csv("data/compas_data/compas-scores-two-years.csv")  # noqa: E501
#     # filter similar to
#     # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
#     data = data[(data['days_b_screening_arrest'] <= 30) &
#                 (data['days_b_screening_arrest'] >= -30) &
#                 (data['is_recid'] != -1) &
#                 (data['c_charge_degree'] != "O") &
#                 (data['score_text'] != "N/A")]
#     # filter out all records except the ones with the most common two races
#     data = data[(data['race'] == 'African-American') | (data['race'] == 'Caucasian')]
#
#
#
#     # Select relevant columns for machine learning.
#     # We explicitly leave in age_cat to allow linear classifiers to be non-linear in age
#     data = data[["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
#                  "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid"]]
#     # map string representation of feature "sex" to 0 for Female and 1 for Male
#     # data = data.assign(sex=(data["sex"] == "Male"))
#
#
#
#
#     categories = ['Male', 'Female']
#     cat_dtype = pd.api.types.CategoricalDtype(categories=categories)
#     data['sex'] = data['sex'].astype(cat_dtype)
#     data['age'] = data['age'].astype('float32')
#     categories = ['Less than 25', '25 - 45', 'Greater than 45']
#     cat_dtype = pd.api.types.CategoricalDtype(categories=categories)
#     data['age_cat'] = data['age_cat'].astype(cat_dtype)
#
#     categories = ['African-American', 'Caucasian']
#     cat_dtype = pd.api.types.CategoricalDtype(categories=categories)
#     data['race'] = data['race'].astype(cat_dtype)
#     data['juv_fel_count'] = data['juv_fel_count'].astype('float32')
#     data['juv_misd_count'] = data['juv_misd_count'].astype('float32')
#     data['juv_other_count'] = data['juv_other_count'].astype('float32')
#     data['priors_count'] = data['priors_count'].astype('float32')
#     categories = ['M', 'F']
#     cat_dtype = pd.api.types.CategoricalDtype(categories=categories)
#     data['c_charge_degree'] = data['c_charge_degree'].astype(cat_dtype)
#
#
#     filt_dtypes = list(filter(lambda x: not (x[0] in ["Target", "Education"]), dtypes))
#
#     rcode = {
#         'Less than 25' : 0,
#         '25 - 45': 1,
#         'Greater than 45': 2,
#     }
#
#
#     rcode1 = {
#         'M' : 0,
#         'F': 1,
#     }
#
#
#     rcode2 = {
#         'African-American' : 0,
#         'Caucasian': 1,
#     }
#     for k, dtype in filt_dtypes:
#         if dtype == "category":
#             if k == "age_cat":
#                 data[k] = np.array([rcode[v.strip()] for v in data[k]])
#             elif k == 'c_charge_degree':
#                 data[k] = np.array([rcode1[v.strip()] for v in data[k]])
#             elif k == 'race':
#                 data[k] = np.array([rcode2[v.strip()] for v in data[k]])
#             else:
#                 data[k] = data[k].cat.codes
#
#
#
#
#     data_size = len(data)
#     new_index_all = np.arange(data_size)
#     data.index = new_index_all
#
#     # 拆分数据框
#
#     return data
#         # train_data.drop(["two_year_recid"], axis=1), train_data["two_year_recid"].values, test_data.drop(
#         # ["two_year_recid"], axis=1), test_data["two_year_recid"].values



class FairnessDataset():
    def __init__(self, dataset, seed, device=torch.device('cpu')):
        self.dataset = dataset
        self.seed = seed

        self.device = device

        if self.dataset == 'AdultCensus':
            self.get_adult_data(seed)

        elif self.dataset == 'COMPAS':
            self.get_compas_data(seed)

        elif self.dataset == 'Lawschool':
            self.get_lawschool_data(seed)

        else:
            raise ValueError('Your argument {} for dataset name is invalid.'.format(self.dataset))

    def get_adult_data(self,seed):

        data = pd.read_csv('Datasets/AdultCensus')
        data = data.drop('Unnamed: 0',axis = 1)

        self.training_set, self.test_set = train_test_split(data, test_size=0.2, random_state=seed)


    def get_compas_data(self,seed):

        data = pd.read_csv('Datasets/COMPAS')
        data = data.drop('Unnamed: 0',axis = 1)

        self.training_set, self.test_set = train_test_split(data, test_size=0.2, random_state=seed)



    def get_lawschool_data(self,seed):

        data = pd.read_csv('Datasets/Lawschool')
        data = data.drop('Unnamed: 0',axis = 1)

        self.training_set, self.test_set = train_test_split(data, test_size=0.2, random_state=seed)




    def get_dataset(self):
            return (self.training_set,  self.test_set)
