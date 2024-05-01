import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve,auc,accuracy_score
from sklearn.metrics import precision_recall_curve
import numpy as np

df = pd.read_csv('../data/PCR_info.csv')
fea = ['hsa-let-7a','hsa-miR-21',
        'hsa-miR-125a','hsa-miR-150','hsa-miR-200a']
gene_list = ['hsa-let-7a/hsa-miR-21','hsa-miR-125a/hsa-miR-21','hsa-miR-200a/hsa-miR-150']
fea1=[]
for i in gene_list:
    a,b = i.split('/')
    if {a, b} < set(fea) :
        print(i)
        fea1.append(i)
        df[i] =  df[b]-df[a]

df_t = df.query("队列 =='训练集'")
df_c = df.query("队列 =='中心内'")
df_m = df.query("队列 =='多中心'")
X_t = df_t[gene_list]
y_t = df_t['condition']
X_vc = df_c[gene_list]
y_vc = df_c['condition']
X_vm = df_m[gene_list]
y_vm = df_m['condition']
X = df[gene_list]
y = df['condition']

random_state = 42
lr = LogisticRegression(random_state=random_state,C = 10,max_iter=1000,class_weight='balanced') 
lr = lr.fit(X_t,y_t)
train_s = lr.predict_proba(X_t)[:,1]
val1_s = lr.predict_proba(X_vc)[:,1]
val2_s = lr.predict_proba(X_vm)[:,1]
all_s = lr.predict_proba(X)[:,1]

joblib.dump(lr, filename='../model/model_3m_0901.pkl')
def plot_roc(nested_list):
    plt.rcParams['pdf.use14corefonts'] = True
    _n = len(nested_list)
    category_colors = ['red', 'orange', 'forestgreen']  
    plt.figure(figsize=(5,5),dpi=300)
    for i in range(_n):
        true_label, pred_prob, name = nested_list[i][0], nested_list[i][1], nested_list[i][2]
        fpr, tpr, thresholds = roc_curve(true_label, pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'b-',
                     label='%s = %0.3f' % (name, roc_auc),
                     color = category_colors[i],
                     alpha=1)

    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.plot([0, 1], [0, 1], linestyle="--",color = 'grey')
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05,1.05))
    plt.legend(loc='lower right')
    plt.savefig("../result/qp-ROC.pdf", format="pdf")
    plt.savefig("../result/qp-ROC.tiff", format="tiff")
    plt.show()
parm = [
    [y_t, train_s, 'Train'],
    [y_vc, val1_s, 'Within-center'],
    [y_vm,val2_s,'Multi-center']]
    

plt.rcParams['pdf.use14corefonts'] = True
plot_roc(parm)



def plot_pr_curve(nested_list):
    plt.rcParams['pdf.use14corefonts'] = True
    _n = len(nested_list)
    category_colors = ['red', 'orange', 'forestgreen']  
    plt.figure(figsize=(5,5),dpi=300)
    
    for i in range(_n):
        true_label, pred_prob, name = nested_list[i][0], nested_list[i][1], nested_list[i][2]
        precision, recall, thresholds = precision_recall_curve(true_label, pred_prob)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, 'b-',
                 label='%s = %0.3f' % (name, pr_auc),
                 color=category_colors[i],
                 alpha=1)
        
    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.legend(loc='lower left')
    plt.savefig("../result/qp-PRC.pdf", format="pdf")
    plt.savefig("../result/qp-PRC.tiff", format="tiff")
    plt.show()
parm = [
    [y_t, train_s, 'Train'],
    [y_vc, val1_s, 'Within-center'],
    [y_vm,val2_s,'Multi-center']] 
from sklearn.metrics import roc_curve,auc,accuracy_score
plt.rcParams['pdf.use14corefonts'] = True
plot_pr_curve(parm)    


class_names = np.array([0,1]) 

y_pred = []
for i in all_s:
    if i >= 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

result = pd.DataFrame({'True label':y,
                      'Predicted label':y_pred})

a = result.groupby('True label')['Predicted label'].value_counts().unstack()

from matplotlib.colors import Normalize

plt.rcParams['pdf.use14corefonts'] = True
plt.figure(dpi=300, figsize=(5, 5))

trans_mat = a.values


row_sums = trans_mat.sum(axis=1, keepdims=True)


normalized_trans_mat = trans_mat / row_sums

plt.imshow(normalized_trans_mat, cmap='Blues', norm=Normalize(vmin=0, vmax=1))


for i in range(normalized_trans_mat.shape[0]):
    for j in range(normalized_trans_mat.shape[1]):
        text = f'{trans_mat[i, j]}\n({normalized_trans_mat[i, j]:.2f})'  
        if normalized_trans_mat[i, j] < 0.5:  
            text_color = 'black'
        else:
            text_color = 'white'
        plt.text(j, i, text, ha='center', va='center', color=text_color)

plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(np.arange(len(trans_mat[0])), labels=['0', '1'])
plt.yticks(np.arange(len(trans_mat)), labels=['0', '1'])
plt.savefig("../result/qp_Confusion matrix.pdf", format="pdf")
plt.savefig("../result/qp_Confusion matrix.tiff", format="tiff")
plt.show()

    
