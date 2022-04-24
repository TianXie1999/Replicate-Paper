### Introduction

In this notebook, we are going to replicate the experiments in paper *Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification without Disparate Mistreatment* (Zafar, Muhammad Bilal, et al.)

This paper is mainly about fairness of machine learning algorithms. There are several fairness metrics, and this paper focuses on one of them: **Disparate Mistreatment**. According to the paper, **disparate mistreatment means with respect to a sensitive attribute, the model would have misclassification rates differ for groups of people having different values of that sensitive attribute (e.g., blacks and whites).** There are also multiple ways to measure the so-called misclassification rate, where the paper mainly discusses two of them: false positive rate and false negative rate. Here are the definitions:

$$FPR = \frac{FP}{FP + TN}$$

$$FNR = \frac{FN}{FN + TP}$$

Therefore, the paper discusses how accuracy would suffer when we add constraints about disparate mistreatment using FPR or FNR. The experiments contain three parts: one synthesized experiment, one real data experiments, and finally the comparison for different mistreatment constraints. We are going to replicate each of them.

### Previous Code and New Contributions

The paper has its github but the codes are out-dated since it was written in 2018. Also, certain functions are erased in Python packages now. We modifed the code from Python 2 to Python 3 to replicate the results. More importantly, we integrate all experiments in one Jupyter notebook to provide better visualizations instead of the original .py files

### Part 1. Synthesized Data for Accuracy-Fairness Tradeoffs

#### 1.1 Data Generating Process

This part generates a dataset with a multivariate normal distribution. The dataset consists of two non-sensitive features and one sensitive feature. We use *generate_synthetic_data.py* to finish the task, where the function can generate certain kinds of synthetic data suffering from different disparate mistreatment effect. The way to achieve it is relatively simple that the function generates different multivariate normal distributions for non-sensitive features when sensitive features are different (i.e. differet covariance matrices and means).

With this technique, we generate a dataset suffering from different disparate mistreatment effect both on FPR and FNR, which means for different values of z (the sensitive attribute), the classifer aiming to optimize accuracy would have different FPR and FNR. Codes below generate the dataset:

(**Note: the results are slightly different from the original ones in the paper, this happens because data is a bit randomized**)

### Part 2. Actual Data

We then apply the constraints on actual data: ProPublica COMPAS dataset. Basically, this dataset is about the crime offenders for different race groups. According to the paper, the data includes information on the offenders’ demographic features (gender, race, age), criminal history (charge for which the person was arrested, number of prior offenses) and the risk score assigned to the offender by COMPAS.The data also contains whether each individual recidivated after two years. 

Therefore, **race** is a sensitive attribute and **y** is whether the individual recidivated. Maybe sex is also a sensitive attribute, but the experiment is focusing on **race**.  Thus, it would be a perfect demonstration of disparate treatment effect. Let's see the results.  
