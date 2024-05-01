import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.metrics import precision_recall_curve

df = pd.read_csv('../data/PCR_info.csv')
gene_list = ['hsa-let-7a/hsa-miR-21','hsa-miR-125a/hsa-miR-21','hsa-miR-200a/hsa-miR-150']
gene_list2 = ['hsa-let-7a/hsa-miR-21','hsa-miR-125a/hsa-miR-21','hsa-miR-200a/hsa-miR-150','AFP']
gene_list3 = ['hsa-let-7a/hsa-miR-21','hsa-miR-125a/hsa-miR-21','hsa-miR-200a/hsa-miR-150','DCP']
gene_list4 = ['hsa-let-7a/hsa-miR-21','hsa-miR-125a/hsa-miR-21','hsa-miR-200a/hsa-miR-150','AFP','DCP']
fea = ['hsa-let-7a','hsa-miR-21',
        'hsa-miR-125a','hsa-miR-150','hsa-miR-200a']

fea1=[]
for i in gene_list:
    a,b = i.split('/')
    if {a, b} < set(fea) :
        print(i)
        fea1.append(i)
        df[i] =  df[b]-df[a]
X = df[gene_list]
y = df['condition']
def find_optimal_cutoff(fpr, tpr, thresholds):

        y = tpr - fpr
        youden_index = np.argmax(y)  
        optimal_threshold = thresholds[youden_index]
        point = [fpr[youden_index], tpr[youden_index]]
        return optimal_threshold, point
def cutoff_pred(y_test, pred_prob):

        fpr, tpr, thresholds = roc_curve(y_test, pred_prob, pos_label=1)
        optimal_threshold, point = find_optimal_cutoff(fpr, tpr, thresholds)
        threshold = optimal_threshold
        
        return np.array(list(map(lambda x: 1 if x >= threshold else 0, pred_prob))), threshold

def calculate_roc_values2(df, feature, target, test_size=0.4, random_seed=400, num_iterations=99):
    np.random.seed(random_seed)
    roc_values = []
    
    for _ in range(num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(df[[feature]], df[target], test_size=test_size)

        y_score = X_test

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

    return mean_fpr, mean_tpr, tprs_lower, tprs_upper, mean_auc, std_auc

mean_fpr5, mean_tpr5, tprs_lower5, tprs_upper5, mean_auc5, std_auc5 = calculate_roc_values2(df, 'AFP公司整理', 'condition')
mean_fpr6, mean_tpr6, tprs_lower6, tprs_upper6, mean_auc6, std_auc6 = calculate_roc_values2(df, 'DCP公司整理', 'condition')


def train_and_evaluate_model(df, feature, target, test_size=0.4, random_seed=400, num_iterations=99):
    X = df[feature]
    y = df[target]
    np.random.seed(random_seed)
    roc_values = []
    y_pred_list = []

    for i in range(num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        model = RandomForestClassifier(random_state=0,max_depth = 2,n_estimators=100,class_weight='balanced')
        model.fit(X_train, y_train)

        y_score = model.predict_proba(X_test)[:, 1]

        y_score_all= model.predict_proba(X)[:, 1]
        y_pred,cutoff = cutoff_pred(y,y_score_all)
        y_pred_list.append(y_pred)

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

    y_pred_combined = np.vstack(y_pred_list)
    df_predictions = pd.DataFrame(y_pred_combined.T, columns=[f'Iteration_{i+1}' for i in range(num_iterations)])
    df_with_predictions = pd.concat([df[['ID', 'condition']], df_predictions], axis=1)

    return df_with_predictions, mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper

df_with_predictions, mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper = train_and_evaluate_model(df, gene_list,'condition')
df_with_predictions2, mean_fpr2, mean_tpr2, mean_auc2, std_auc2, tprs_lower2, tprs_upper2 = train_and_evaluate_model(df, gene_list2,'condition')
df_with_predictions3, mean_fpr3, mean_tpr3, mean_auc3, std_auc3, tprs_lower3, tprs_upper3 = train_and_evaluate_model(df, gene_list3,'condition')
df_with_predictions4, mean_fpr4, mean_tpr4, mean_auc4, std_auc4, tprs_lower4, tprs_upper4 = train_and_evaluate_model(df, gene_list4,'condition')

import matplotlib.pyplot as plt
plt.rcParams['pdf.use14corefonts'] = True
plt.figure(dpi=300,figsize=(5, 5))
plt.plot(mean_fpr, mean_tpr, color='royalblue', label=f'miRNA (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='darkgrey', alpha=0.3)

plt.plot(mean_fpr2, mean_tpr2, color='darkorange', label=f'AFP+miRNA (AUC = {mean_auc2:.2f} ± {std_auc2:.2f})')
plt.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color='darkgrey', alpha=0.3)

plt.plot(mean_fpr3, mean_tpr3, color='seagreen', label=f'DCP+miRNA (AUC = {mean_auc3:.2f} ± {std_auc3:.2f})')
plt.fill_between(mean_fpr3, tprs_lower3, tprs_upper3, color='darkgrey', alpha=0.3)


plt.plot(mean_fpr4, mean_tpr4, color='crimson', label=f'DCP+AFP+miRNA (AUC = {mean_auc4:.2f} ± {std_auc4:.2f})')
plt.fill_between(mean_fpr4, tprs_lower4, tprs_upper4, color='darkgrey', alpha=0.3)

plt.plot(mean_fpr5, mean_tpr5, color='darkorchid', label=f'AFP (AUC = {mean_auc5:.2f} ± {std_auc5:.2f})')
plt.fill_between(mean_fpr5, tprs_lower5, tprs_upper5, color='darkgrey', alpha=0.3)

plt.plot(mean_fpr6, mean_tpr6, color='sienna', label=f'DCP (AUC = {mean_auc6:.2f} ± {std_auc6:.2f})')
plt.fill_between(mean_fpr6, tprs_lower6, tprs_upper6, color='darkgrey', alpha=0.3)

plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curve with RandomForest ')
plt.legend(loc='lower right')
plt.savefig("../result/RF_ROC_bootstrap.pdf", format="pdf")
plt.savefig("../result/RF_ROC_bootstrap.tiff", format="tiff")
plt.show()



def majority_vote(row):

    counts = row.iloc[2:].value_counts()

    if counts.get(1, 0) >= counts.get(0, 0):
        return 1
    else:
        return 0

result = df_with_predictions[['ID','condition']]
df_s = df[['ID','AFP-label','DCP-label']]
result = pd.merge(result,df_s)
result['pred_miR'] = df_with_predictions.apply(lambda row: majority_vote(row), axis=1)
result['pred_miR+AFP'] = df_with_predictions2.apply(lambda row: majority_vote(row), axis=1)

result['pred_miR+DCP'] = df_with_predictions3.apply(lambda row: majority_vote(row), axis=1)

result['pred_miR+DCP+AFP'] = df_with_predictions4.apply(lambda row: majority_vote(row), axis=1)

new_columns = {'condition': 'condition+',
               'AFP-label': 'AFP+',
               'DCP-label': 'DCP+',
               'pred_miR': 'miRNA+',
               'pred_miR+AFP': 'miR&AFP+',
               'pred_miR+DCP': 'miR&DCP+',
               'pred_miR+DCP+AFP': 'miR&DCP&AFP+'}

result = result.rename(columns=new_columns)

a = result.groupby(['condition+', 'AFP+', 'DCP+', 'miRNA+', 'miR&AFP+', 'miR&DCP+', 'miR&DCP&AFP+']).size()

from upsetplot import generate_counts,plot
fig = plt.figure(dpi=300,figsize=(12, 5))
plot(a.unstack().query('`condition+`==1').stack(),sort_by='cardinality',show_counts='{}',fig =fig,element_size=None)
plt.savefig("../result/RF_upset.pdf", format="pdf")
plt.savefig("../result/RF_upset.tiff", format="tiff")

def calculate_pr_values(df, feature, target, test_size=0.4, random_seed=400, num_iterations=99):
    np.random.seed(random_seed)
    pr_values = []
    
    for _ in range(num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(df[[feature]], df[target], test_size=test_size)

        y_score = X_test

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

    return mean_recall, mean_precision, precisions_lower, precisions_upper, mean_pr_auc, std_pr_auc

mean_recall5, mean_precision5, precisions_lower5, precisions_upper5, mean_pr_auc5, std_pr_auc5 = calculate_pr_values(df, 'AFP公司整理', 'condition')
mean_recall6, mean_precision6, precisions_lower6, precisions_upper6, mean_pr_auc6, std_pr_auc6 = calculate_pr_values(df, 'DCP公司整理', 'condition')


def train_and_evaluate_model_pr(df, feature, target, test_size=0.4, random_seed=400, num_iterations=99):
    X = df[feature]
    y = df[target]
    np.random.seed(random_seed)
    pr_values = []
    y_pred_list = []

    for i in range(num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        model = RandomForestClassifier(random_state=0,max_depth = 2,n_estimators=100,class_weight='balanced')
        model.fit(X_train, y_train)

        y_score = model.predict_proba(X_test)[:, 1]
        y_score_all = model.predict_proba(X)[:, 1]
        y_pred, cutoff = cutoff_pred(y, y_score_all)
        y_pred_list.append(y_pred)

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

    y_pred_combined = np.vstack(y_pred_list)
    df_predictions = pd.DataFrame(y_pred_combined.T, columns=[f'Iteration_{i+1}' for i in range(num_iterations)])
    df_with_predictions = pd.concat([df[['ID', 'condition']], df_predictions], axis=1)

    return df_with_predictions, mean_recall, mean_precision, mean_pr_auc, std_pr_auc, precisions_lower, precisions_upper

df_with_predictions, mean_recall, mean_precision, mean_pr_auc, std_pr_auc, precisions_lower, precisions_upper = train_and_evaluate_model_pr(df, gene_list, 'condition')
df_with_predictions2, mean_recall2, mean_precision2, mean_pr_auc2, std_pr_auc2, precisions_lower2, precisions_upper2 = train_and_evaluate_model_pr(df, gene_list2, 'condition')
df_with_predictions3, mean_recall3, mean_precision3, mean_pr_auc3, std_pr_auc3, precisions_lower3, precisions_upper3 = train_and_evaluate_model_pr(df, gene_list3, 'condition')
df_with_predictions4, mean_recall4, mean_precision4, mean_pr_auc4, std_pr_auc4, precisions_lower4, precisions_upper4 = train_and_evaluate_model_pr(df, gene_list4, 'condition')


plt.rcParams['pdf.use14corefonts'] = True
plt.figure(dpi=300,figsize=(5, 5))
plt.plot(mean_recall, mean_precision, color='royalblue', label=f'miRNA (AUC = {mean_pr_auc:.2f} ± {std_pr_auc:.2f})')
plt.fill_between(mean_recall, precisions_lower, precisions_upper, color='darkgrey', alpha=0.3)

plt.plot(mean_recall2, mean_precision2, color='darkorange', label=f'AFP+miRNA (AUC = {mean_pr_auc2:.2f} ± {std_pr_auc2:.2f})')
plt.fill_between(mean_recall2, precisions_lower2, precisions_upper2, color='darkgrey', alpha=0.3)

plt.plot(mean_recall3, mean_precision3, color='seagreen', label=f'DCP+miRNA (AUC = {mean_pr_auc3:.2f} ± {std_pr_auc3:.2f})')
plt.fill_between(mean_recall3, precisions_lower3, precisions_upper3, color='darkgrey', alpha=0.3)

plt.plot(mean_recall4, mean_precision4, color='crimson', label=f'DCP+AFP+miRNA (AUC = {mean_pr_auc4:.2f} ± {std_pr_auc4:.2f})')
plt.fill_between(mean_recall4, precisions_lower4, precisions_upper4, color='darkgrey', alpha=0.3)

plt.plot(mean_recall5, mean_precision5, color='darkorchid', label=f'AFP (AUC = {mean_pr_auc5:.2f} ± {std_pr_auc5:.2f})')
plt.fill_between(mean_recall5, precisions_lower5, precisions_upper5, color='darkgrey', alpha=0.3)

plt.plot(mean_recall6, mean_precision6, color='sienna', label=f'DCP (AUC = {mean_pr_auc6:.2f} ± {std_pr_auc6:.2f})')
plt.fill_between(mean_recall6, precisions_lower6, precisions_upper6, color='darkgrey', alpha=0.3)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim((-0.05, 1.05))
plt.ylim((-0.05, 1.05))
plt.title('Mean PRC Curve with RandomForest')
plt.legend(loc='lower left')
plt.savefig("../result/RF_PRC_bootstrap.pdf", format="pdf")
plt.savefig("../result/RF_PRC_bootstrap.tiff", format="tiff")
plt.show()
