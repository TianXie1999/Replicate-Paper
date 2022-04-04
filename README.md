# Replicate-Paper

In this notebook, we are going to replicate the experiments in paper *Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification without Disparate Mistreatment* (Zafar, Muhammad Bilal, et al.)

This paper is mainly about fairness of machine learning algorithms. There are several fairness metrics, and this paper focuses on one of them: **Disparate Mistreatment**. According to the paper, **disparate mistreatment means with respect to a sensitive attribute, the model would have misclassification rates differ for groups of people having different values of that sensitive attribute (e.g., blacks and whites).** There are also multiple ways to measure the so-called misclassification rate, where the paper mainly discusses two of them: false positive rate and false negative rate. Here are the definitions:

$$FPR = \frac{FP}{FP + TN}$$

$$FNR = \frac{FN}{FN + TP}$$

Therefore, the paper discusses how accuracy would suffer when we add constraints about disparate mistreatment using FPR or FNR. The experiments contain three parts: one synthesized experiment, one real data experiments, and finally the comparison for different mistreatment constraints. We are going to replicate each of them.

**Previous Code and New Contributions**

The paper has its github but the codes are out-dated since it was written in 2018. Also, certain functions are erased in Python packages now. We modifed the code from Python 2 to Python 3 to replicate the results. More importantly, we integrate all experiments in one Jupyter notebook to provide better visualizations instead of the original .py files
