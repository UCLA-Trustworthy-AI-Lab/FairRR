
import warnings

warnings.filterwarnings("ignore")



from sklearn.utils import shuffle


from dataloader import FairnessDataset

from Myfunctions import *



###Fair RR method: adjust t for specific disparity level###

def Fair_RR(seed,dataset_name,fairness):

    classifier_list = [ LogisticRegression()]
    classifier_name_list = ['LR']
    dataset = FairnessDataset(dataset=dataset_name, seed = seed, device='cpu')
    training_set,  test_set = dataset.get_dataset()
    training_set = training_set.drop(['weight', 'neighbour'], axis=1)
    test_set = test_set.drop(['weight', 'neighbour'], axis=1)

    Z_train = training_set['Sensitive']
    Y_train = training_set['Label']
    X_train = training_set.drop('Label', axis=1)




    Z_test = test_set['Sensitive']
    Y_test = test_set['Label']
    X_test = test_set.drop('Label', axis=1)

    training_set11 = training_set[(training_set['Sensitive'] == 1) & (training_set['Label'] == 1)]
    training_set10 = training_set[(training_set['Sensitive'] == 1) & (training_set['Label'] == 0)]
    training_set01 = training_set[(training_set['Sensitive'] == 0) & (training_set['Label'] == 1)]
    training_set00 = training_set[(training_set['Sensitive'] == 0) & (training_set['Label'] == 0)]

    n11 = len(training_set11)
    n10 = len(training_set10)
    n01 = len(training_set01)
    n00 = len(training_set00)
    n = len(training_set)
    p11 = n11 / n
    p10 = n10 / n
    p01 = n01 / n
    p00 = n00 / n

    for ic, classifier in enumerate(classifier_list):
        classifier_name = classifier_name_list[ic]
        print(f'we are running seed: {seed}, dataset: {dataset_name}, fairness: {fairness}, classifier: {classifier_name}', )

        random.seed(seed)
        np.random.seed(seed)
        training_set11_s = shuffle(training_set11)
        training_set10_s = shuffle(training_set10)
        training_set01_s = shuffle(training_set01)
        training_set00_s = shuffle(training_set00)

        Result_test = pd.DataFrame()

        if fairness == 'DP':
            if dataset_name == 'AdultCensus':
                level_list = [0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18]
            if dataset_name == 'COMPAS':
                level_list = [0,0.03,0.06,0.09,0.12,0.15,0.18,0.21,0.24,0.27]
            if dataset_name == 'Lawschool':
                level_list = [0,0.006,0.012,0.018,0.024,0.03,0.036,0.042,0.048,0.054]
        if fairness == 'EO':
            if dataset_name == 'AdultCensus':
                level_list = [0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18]
            if dataset_name == 'COMPAS':
                level_list = [0,0.03,0.06,0.09,0.12,0.15,0.18,0.21,0.24,0.27]
            if dataset_name == 'Lawschool':
                level_list = [0,0.009,0.018,0.027,0.036,0.045,0.054,0.063,0.072,0.081]

        if fairness == 'PE':
            if dataset_name == 'AdultCensus':
                level_list = [0,0.009,0.018,0.027,0.036,0.045,0.054,0.063,0.072,0.081]
            if dataset_name == 'COMPAS':
                level_list = [0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18]
            if dataset_name == 'Lawschool':
                level_list = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225]

        for level in level_list:
            acc_val, f1_val,disparity_val = train_test_acc_parity(X_train, Y_train, X_train, Y_train, Z_train, fairness, classifier)
            if abs(disparity_val) < level:
                acc_test,f1_test, disparity_test = train_test_acc_parity(X_train, Y_train, X_test, Y_test, Z_test, fairness, classifier)

                data_result = [seed, 0, level, classifier_name, acc_test, f1_test, abs(disparity_test)]

                columns = ['seed', 't', 'level','classifier', 'acc','f1',f'D{fairness}']

                temp = pd.DataFrame([data_result], columns=columns)
                Result_test = pd.concat([Result_test, temp])

                continue
            elif disparity_val > level:
                t_max = p11 + p10
                t_min = 0
                level0 = level
            else:
                t_max = 0
                t_min = -p01 - p00
                level0 = -1 * level
            t_mid = (t_max + t_min) / 2


            while abs(t_max - t_min) > 1e-4:

                X_train11 = training_set11_s.copy()
                X_train10 = training_set10_s.copy()
                X_train01 = training_set01_s.copy()
                X_train00 = training_set00_s.copy()


                theta = Find_Theta(t_mid,p11,p10,p01,p00 , fairness)
                num_change11 =  int(n11 * theta[0])
                num_change10 =  int(n10 * theta[1])
                num_change01 =  int(n01 * theta[2])
                num_change00 =  int(n00 * theta[3])

                X_train11['Label'].values[:num_change11] = 0
                X_train10['Label'].values[:num_change10] = 1
                X_train01['Label'].values[:num_change01] = 0
                X_train00['Label'].values[:num_change00] = 1


                X_syn = pd.concat([X_train11,X_train10,X_train01,X_train00])
                Y_syn = X_syn['Label']
                Z_syn = X_syn['Sensitive']
                X_syn = X_syn.drop('Label',axis =1)


                acc_val, f1_val, disparity_val = train_test_acc_parity(X_syn, Y_syn, X_train, Y_train, Z_train, fairness, classifier)
                if disparity_val > level0:
                    t_min = t_min / 3 + 2 * t_mid / 3
                else:
                    t_max = t_max / 3 + 2 * t_mid / 3

                t_mid = (t_max + t_min) / 2


            acc_test, f1_test,disparity_test = train_test_acc_parity(X_syn, Y_syn, X_test, Y_test, Z_test,fairness, classifier)

            data_result = [seed, t_mid, level, classifier_name, acc_test,f1_test, abs(disparity_test)]
            columns = ['seed', 't', 'level', 'classifier', 'acc', 'f1', f'D{fairness}']

            temp = pd.DataFrame([data_result], columns=columns)
            Result_test = pd.concat([Result_test,temp])
        Result_test.to_csv(f'Result/FairRR/{dataset_name}/{fairness}/result_of_seed_{seed}')







###Training on original data###

def Direct_training(seed,dataset_name):

    classifier_list = [ LogisticRegression()]
    classifier_name_list = ['LR']
    dataset = FairnessDataset(dataset=dataset_name, seed = seed, device='cpu')
    training_set,  test_set = dataset.get_dataset()
    training_set = training_set.drop(['weight', 'neighbour'], axis=1)
    test_set = test_set.drop(['weight', 'neighbour'], axis=1)


    Z_train = training_set['Sensitive']
    Y_train = training_set['Label']
    X_train = training_set.drop('Label', axis=1)




    Z_test = test_set['Sensitive']
    Y_test = test_set['Label']
    X_test = test_set.drop('Label', axis=1)

    for ic, classifier in enumerate(classifier_list):
        classifier_name = classifier_name_list[ic]
        print(f'we are running seed: {seed}, dataset: {dataset_name},  classifier: {classifier_name}', )

        random.seed(seed)
        np.random.seed(seed)

        acc_test,f1_test, DDP_test,DEO_test,DPE_test = train_test_acc_all_parity(X_train, Y_train, X_test, Y_test, Z_test, classifier)
        data_result = [seed, classifier_name, acc_test, f1_test, abs(DDP_test),abs(DEO_test),abs(DPE_test)]
        columns = ['seed', 'classifier', 'acc', 'f1', 'DDP','DEO','DPE']

        temp = pd.DataFrame([data_result], columns=columns)

        temp.to_csv(f'Result/Direct_training/{dataset_name}/result_of_seed_{seed}')






###Training on dataset generated by fair sampling method###
def Fair_Sampling(seed, dataset_name):
    classifier_list = [ LogisticRegression()]
    classifier_name_list = ['LR']
    dataset = FairnessDataset(dataset=dataset_name, seed = seed, device='cpu')
    training_set,  test_set = dataset.get_dataset()
    training_set = training_set.drop(['weight', 'neighbour'], axis=1)
    test_set = test_set.drop(['weight', 'neighbour'], axis=1)

    X_train = training_set.copy()

    X_train11 = X_train[(X_train['Sensitive'] == 1) & (X_train['Label'] == 1)]
    X_train10 = X_train[(X_train['Sensitive'] == 1) & (X_train['Label'] == 0)]
    X_train01 = X_train[(X_train['Sensitive'] == 0) & (X_train['Label'] == 1)]
    X_train00 = X_train[(X_train['Sensitive'] == 0) & (X_train['Label'] == 0)]
    n_ori = len(X_train)
    n11_ori = len(X_train11)
    n10_ori = len(X_train10)
    n01_ori = len(X_train01)
    n00_ori = len(X_train00)

    Z_test = test_set['Sensitive']
    Y_test = test_set['Label']
    X_test = test_set.drop('Label', axis=1)

    n11 = round((n11_ori + n10_ori) * (n11_ori + n01_ori) / n_ori)
    n10 = round((n11_ori + n10_ori) * (n10_ori + n00_ori) / n_ori)
    n01 = round((n01_ori + n00_ori) * (n11_ori + n01_ori) / n_ori)
    n00 = round((n01_ori + n00_ori) * (n10_ori + n00_ori) / n_ori)

    np.random.seed(seed)
    random.seed(seed)

    if n11 > n11_ori:
        index11 = np.random.choice(X_train11.index, n11, replace=True)
    else:
        index11 = np.random.choice(X_train11.index, n11, replace=False)
    if n10 > n10_ori:
        index10 = np.random.choice(X_train10.index, n10, replace=True)
    else:
        index10 = np.random.choice(X_train10.index, n10, replace=False)
    if n01 > n01_ori:
        index01 = np.random.choice(X_train01.index, n01, replace=True)
    else:
        index01 = np.random.choice(X_train01.index, n01, replace=False)
    if n00 > n00_ori:
        index00 = np.random.choice(X_train00.index, n00, replace=True)
    else:
        index00 = np.random.choice(X_train00.index, n00, replace=False)

    X_syn11 = X_train11.loc[index11, :]
    X_syn10 = X_train10.loc[index10, :]
    X_syn01 = X_train01.loc[index01, :]
    X_syn00 = X_train00.loc[index00, :]

    X_syn = pd.concat([X_syn11, X_syn10, X_syn01, X_syn00])
    Y_syn = X_syn['Label']
    X_syn = X_syn.drop('Label', axis=1)


    for ic, classifier in enumerate(classifier_list):
        classifier_name = classifier_name_list[ic]
        print(f'we are running seed: {seed}, dataset: {dataset_name},  classifier: {classifier_name}', )

        random.seed(seed)
        np.random.seed(seed)

        acc_test,f1_test, DDP_test,DEO_test,DPE_test = train_test_acc_all_parity(X_syn, Y_syn, X_test, Y_test, Z_test, classifier)
        data_result = [seed, classifier_name, acc_test, f1_test, abs(DDP_test),abs(DEO_test),abs(DPE_test)]
        columns = ['seed', 'classifier', 'acc', 'f1', 'DDP','DEO','DPE']

        temp = pd.DataFrame([data_result], columns=columns)


        temp.to_csv(f'Result/Fiar_Sampling/{dataset_name}/result_of_seed_{seed}')


###Training on dataset generated by SMOTE: adjust alpha for specific disparity level###

def FAWOS(seed, dataset_name):
    classifier_list = [ LogisticRegression()]
    classifier_name_list = ['LR']

    fairness = 'DP'

    dataset = FairnessDataset(dataset=dataset_name, seed = seed,device='cpu')

    training_set,  test_set = dataset.get_dataset()

    X_train = training_set.copy()
    train_Y = X_train['Label']
    train_Z = X_train['Sensitive']
    train_X = X_train.drop(['weight','neighbour','Label'], axis=1)

    X_train11 = X_train[(X_train['Sensitive'] == 1) & (X_train['Label'] == 1)]
    X_train10 = X_train[(X_train['Sensitive'] == 1) & (X_train['Label'] == 0)]
    X_train01 = X_train[(X_train['Sensitive'] == 0) & (X_train['Label'] == 1)]
    X_train00 = X_train[(X_train['Sensitive'] == 0) & (X_train['Label'] == 0)]


    n_ori = len(X_train)
    n11_ori = len(X_train11)
    n10_ori = len(X_train10)
    n01_ori = len(X_train01)
    n00_ori = len(X_train00)

    train_set11 = X_train11.copy().drop(['neighbour', 'weight'], axis=1)
    train_set10 = X_train10.copy().drop(['neighbour', 'weight'], axis=1)
    train_set01 = X_train01.copy().drop(['neighbour', 'weight'], axis=1)
    train_set00 = X_train00.copy().drop(['neighbour', 'weight'], axis=1)

    Z_test = test_set['Sensitive']
    Y_test = test_set['Label']
    X_test = test_set.drop(['neighbour', 'weight','Label'], axis=1)

    for ic, classifier in enumerate(classifier_list):
        classifier_name = classifier_name_list[ic]
        print(f'we are running seed: {seed}, dataset: {dataset_name}, fairness: {fairness}, classifier: {classifier_name}', )


        random.seed(seed)
        np.random.seed(seed)

        Result_test = pd.DataFrame()
        if dataset_name == 'AdultCensus':
            level_list = [0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18]
        if dataset_name == 'COMPAS':
            level_list = list(-1 * np.array([0,0.03,0.06,0.09,0.12,0.15,0.18,0.21,0.24,0.27]))
        if dataset_name == 'Lawschool':
            level_list = [0,0.006,0.012,0.018,0.024,0.03,0.036,0.042,0.048,0.054]

        for level in level_list:
            acc_val, f1_val,disparity_val = train_test_acc_parity(train_X, train_Y, train_X, train_Y, train_Z, fairness, classifier)
            if abs(disparity_val) < level:
                acc_test,f1_test, disparity_test = train_test_acc_parity(train_X, train_Y, X_test, Y_test, Z_test, fairness, classifier)

                data_result = [seed, 0, level, classifier_name, acc_test, f1_test, disparity_test]

                columns = ['seed', 't', 'level','classifier', 'acc','f1',f'D{fairness}']

                temp = pd.DataFrame([data_result], columns=columns)
                Result_test = pd.concat([Result_test, temp])

                continue

            t_max = 4
            t_min = 0
            t_mid = (t_max + t_min) / 2

            while abs(t_max - t_min) > 1e-4:
                if dataset_name == 'Lawschool':
                    n01 = round(t_mid * (n11_ori * n00_ori / n10_ori - n01_ori))
                    X_syn11 = train_set11.copy()
                    X_syn10 = train_set10.copy()
                    X_syn01 = sample_points(X_train01, X_train, n01)
                    X_syn01 = pd.concat([X_syn01,train_set01])
                    X_syn00 = train_set00.copy()
                if dataset_name == 'AdultCensus':
                    n01 = round(t_mid * (n11_ori * n00_ori / n10_ori - n01_ori))
                    X_syn11 = train_set11.copy()
                    X_syn10 = train_set10.copy()
                    X_syn01 = sample_points(X_train01, X_train, n01)
                    X_syn01 = pd.concat([X_syn01,train_set01])
                    X_syn00 = train_set00.copy()
                if dataset_name == 'COMPAS':
                    n00 = round(t_mid * (n10_ori * n01_ori / n11_ori - n00_ori))
                    X_syn11 = train_set11.copy()
                    X_syn10 = train_set10.copy()
                    X_syn01 = train_set01.copy()
                    X_syn00 = sample_points(X_train00, X_train, n00)
                    X_syn00 = pd. concat([X_syn00,train_set00])


                X_syn = pd.concat([X_syn11, X_syn10, X_syn01, X_syn00])
                Y_syn = X_syn['Label']
                X_syn = X_syn.drop('Label', axis=1)

                acc_val, f1_val,disparity_val = train_test_acc_parity(X_syn, Y_syn, train_X, train_Y, train_Z, fairness, classifier)
                if dataset_name == 'AdultCensus':
                    if disparity_val > level:
                        t_min = t_min / 3 + 2 * t_mid / 3
                    else:
                        t_max = t_max / 3 + 2 * t_mid / 3
                    t_mid = (t_max + t_min) / 2
                if dataset_name == 'Lawschool':
                    if disparity_val > level:
                        t_min = t_min / 3 + 2 * t_mid / 3
                    else:
                        t_max = t_max / 3 + 2 * t_mid / 3
                    t_mid = (t_max + t_min) / 2
                if dataset_name == 'COMPAS':
                    if disparity_val > level:
                        t_max = t_max / 3 + 2 * t_mid / 3
                    else:
                        t_min = t_min / 3 + 2 * t_mid / 3
                    t_mid = (t_max + t_min) / 2

            acc_test, f1_test,disparity_test = train_test_acc_parity(X_syn, Y_syn, X_test, Y_test, Z_test,fairness, classifier)
            data_result = [seed, t_mid, level, classifier_name, acc_test,f1_test, disparity_test]
            columns = ['seed', 't', 'level', 'classifier', 'acc', 'f1', f'D{fairness}']
            temp = pd.DataFrame([data_result], columns=columns)
            Result_test = pd.concat([Result_test,temp])


        Result_test.to_csv(f'Result/FAWOS/{dataset_name}/result_of_seed_{seed}')

