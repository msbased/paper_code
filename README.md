# Program description

This is a python code that includes NGS feature screening, NGS modeling, and PCR data modeling. The performance of the PCR model was also verified by other cancers, and the robustness of the four algorithms was verified by random partitioning method.

# Requirements

- Language:     Python 3.11.3
- Python     Libraries:
  - numpy==1.24.1
  - pandas==1.5.0
  - sklearn==1.2.2
  - Matplotlib==3.7.1
  - upsetplot==0.9.0

**01-Feature screening**

The tpm data of NGS is used for recursive feature elimination, and the feature combination with the highest average accuracy of cross-validation is selected. The feature combination builds a random forest model and selects the top10 features according to the importance of the features.

**02-NGSmodel**

Three paired miRNAs were used to establish logistic regression models on NGS data, and ROC and PRC curves and confusion matrices with confidence intervals were plotted.

**03-RNAmodel**

Three paired miRNAs were used to model on the training set of PCR data, and the model performance was verified on two validation sets. ROC and PRC curves and confusion matrix were plotted.

**04-other cancer**

Analyze the performance of the model built by the training set on other cancer data and plot the confusion matrix.

**05-1-LR**

All PCR data were combined, then divided, and logistic regression models were established, PRC and ROC with confidence intervals were plotted, and the performance of miRNA was compared with AFP and DCP.

**05-2-RF**

All PCR data were combined, then divided, and randomforest models were established, PRC and ROC with confidence intervals were plotted, and the performance of miRNA was compared with AFP and DCP.

**05-3-SVC**

All PCR data were combined, then divided, and SVC models were established, PRC and ROC with confidence intervals were plotted, and the performance of miRNA was compared with AFP and DCP.

**05-4-MLP**

All PCR data were combined, then divided, and MPL models were established, PRC and ROC with confidence intervals were plotted, and the performance of miRNA was compared with AFP and DCP.
