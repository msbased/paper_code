import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import joblib

df = pd.read_csv('../data/other_cancers.csv')
df.dropna(subset=['分组'], inplace=True)
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

condition = dict(zip(['肝癌','肝硬化','肝炎','非肝癌良性','健康','胃癌','肺癌','乳腺癌','肠癌'],[1,0,0,0,0,0,0,0,0]))
df.loc[:,'condition'] = df['分组'].map(condition)


model = joblib.load(filename='../model/model_3m_0901.pkl')

X = df[gene_list]
y = df['condition']
all_s = model.predict_proba(X)[:,1]
result = pd.DataFrame({'score':all_s})
result.loc[:,'pre'] = \
        result["score"].apply(lambda x: 1 if x>=0.5 else 0)

result = pd.concat([df,result],axis=1)

index_order = ['胃癌','肺癌', '乳腺癌','肠癌']
result_other = result.query("队列=='其它癌症'")

result_other['分组'] = pd.Categorical(result_other['分组'], categories=index_order, ordered=True)
result_other.groupby('分组')['pre'].value_counts().unstack()

a = result_other.groupby('分组')['pre'].value_counts().unstack()



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
plt.xticks(np.arange(len(trans_mat[0])), labels=['non-HCC', 'HCC'])
plt.yticks(np.arange(len(trans_mat)), labels=['Gastric Cancer', 'Lung Cancer','Breast Cancer','Colorectal Cancer'])
plt.savefig("../result/other_cancer_Confusion matrix.pdf", format="pdf")
plt.savefig("../result/other_cancer_Confusion matrix.tiff", format="tiff")
plt.show()
result.to_csv('../result/other.cancers.model.score.csv',index=0)