import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc,precision_recall_curve,accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning

df = pd.read_csv('../data/NGS_counts_info.csv')
target_list = ['hsa-let-7a/hsa-miR-21', 'hsa-miR-200a/hsa-miR-150', 'hsa-miR-125a/hsa-miR-21']
gene_list = df.columns.tolist()[:-5]
fea1=[]
for i in target_list:
    a,b = i.split('/')
    if {a, b} < set(gene_list) :
        print(i)
        
        fea1.append(i)
        df[i] =  (df[a]+1)/(df[b]+1)
X = df[gene_list]
y = df['condition']
random_state = 42
lr = LogisticRegression(random_state=random_state,C = 10,max_iter=1000,class_weight='balanced') 
lr = lr.fit(X,y)
train_s = lr.predict_proba(X)[:,1]

def decimal_up_reservations(decimal, n=2):
        decimal_1 = decimal * math.pow(10, n)
        decimal_1 = math.ceil(decimal_1)
        return decimal_1 / math.pow(10, n)
def find_optimal_cutoff(fpr, tpr, thresholds):
        y = tpr - fpr
        youden_index = np.argmax(y)  
        optimal_threshold = thresholds[youden_index]
        point = [fpr[youden_index], tpr[youden_index]]
        return optimal_threshold, point
def cutoff( y_test, pred_prob):

        fpr, tpr, thresholds = roc_curve(y_test, pred_prob, pos_label=1)
        optimal_threshold, point = find_optimal_cutoff(fpr, tpr, thresholds)
        threshold = optimal_threshold
        b = -np.log(10 / 6 - 1) - threshold
        b = decimal_up_reservations(b, 4)  

        def prob_z(_z):
            return 100 / (1 + np.exp(-(b + _z)))

        z = list(map(prob_z, pred_prob))
        return np.array(list(map(lambda x: 1 if x >= 60 else 0, z))), b, threshold

_,_,cutoff = cutoff(y, train_s)


# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
def train_and_evaluate_model_prc(df, feature, target, test_size=0.4, random_seed=400, num_iterations=99):
    X = df[feature]
    y = df[target]
    np.random.seed(random_seed)
    pr_values = []

    for i in range(num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        model = LogisticRegression(C=10, max_iter=1000, class_weight='balanced')
        model.fit(X, y)

        y_score = model.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_score)
        pr_auc = auc(recall, precision)
        pr_values.append((precision, recall, pr_auc))

    mean_recall = np.unique(np.concatenate([recall for _, recall, _ in pr_values]))

    mean_precision = np.zeros_like(mean_recall)
    for precision, recall, _ in pr_values:
        mean_precision += np.interp(mean_recall, recall[::-1], precision[::-1])
    mean_precision /= len(pr_values)

    precisions = np.array([np.interp(mean_recall, recall[::-1], precision[::-1]) for precision, recall, _ in pr_values])
    mean_pr_auc = auc(mean_recall, mean_precision)
    std_pr_auc = np.std([auc(recall, precision) for precision, recall, _ in pr_values])
    precisions_upper = np.percentile(precisions, 97.5, axis=0)
    precisions_lower = np.percentile(precisions, 2.5, axis=0)



    return  mean_recall, mean_precision, mean_pr_auc, std_pr_auc, precisions_lower, precisions_upper


mean_recall, mean_precision, mean_pr_auc, std_pr_auc, precisions_lower, precisions_upper = train_and_evaluate_model_prc(df, gene_list, 'condition')

import matplotlib.pyplot as plt
plt.rcParams['pdf.use14corefonts'] = True
plt.figure(dpi=300,figsize=(5, 5))

plt.plot(mean_recall, mean_precision, color='red', label=f'Mean PRC curve (AUC = {mean_pr_auc:.3f} ± {std_pr_auc:.3f})')
plt.fill_between(mean_recall, precisions_lower, precisions_upper, color='darkgrey', alpha=0.3)

plt.xlim((-0.05, 1.05))
plt.ylim((-0.05, 1.05))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')


plt.grid(True)
plt.savefig("../result/NGS_PRC_bootstrap.pdf", format="pdf")
plt.savefig("../result/NGS_PRC_bootstrap.tiff", format="tiff")
plt.show()


def train_and_evaluate_model(df, feature, target, test_size=0.4, random_seed=400, num_iterations=99):
    X = df[feature]
    y = df[target]
    np.random.seed(random_seed)
    roc_values = []


    for i in range(num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        model = LogisticRegression(C=10,max_iter=1000,class_weight='balanced')
        model.fit(X, y)

        y_score = model.predict_proba(X_test)[:, 1]

       
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        roc_values.append((fpr, tpr, roc_auc))

    mean_fpr = np.unique(np.concatenate([fpr for fpr, _, _ in roc_values]))

    mean_tpr = np.zeros_like(mean_fpr)
    for fpr, tpr, _ in roc_values:
        mean_tpr += np.interp(mean_fpr, fpr, tpr)

    mean_tpr /= len(roc_values)


    tprs = np.array([np.interp(mean_fpr, fpr, tpr) for fpr, tpr, _ in roc_values])
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std([auc(mean_fpr, tpr) for tpr in tprs])
    tprs_upper = np.percentile(tprs, 97.5, axis=0)
    tprs_lower = np.percentile(tprs, 2.5, axis=0)

    
    return mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper


mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper = train_and_evaluate_model(df, gene_list,'condition')


plt.rcParams['pdf.use14corefonts'] = True
plt.figure(dpi=300,figsize=(5, 5))
plt.plot(mean_fpr, mean_tpr, color='red', label=f'Mean PRC curve (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='darkgrey', alpha=0.3)
plt.xlim((-0.05, 1.05))
plt.ylim((-0.05, 1.05))


plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

plt.grid(True)
plt.savefig("../result/NGS_ROC_bootstrap.pdf", format="pdf")
plt.savefig("../result/NGS_ROC_bootstrap.tiff", format="tiff")
plt.show()
