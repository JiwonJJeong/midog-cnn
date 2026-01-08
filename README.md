# midog-cnn
Demonstration of covariate shift and solutions for classification of mitotic figures across domains given varying access to cross-domain data.

[Scientific report](.final_report.pdf)

**Abstract**

*Incremental investigation of the MIDOG++ dataset provided insights about covariate shifts in the classification of mitotic figures of cancer histology images. For single domain classification, AlexNet based models performed well. As the task grew to require cross domain generaliz ability, basic training methods were still performant when the target domain label data is accessible, but single domain trained models failed under covariate shifts. When target labels are removed or target distributions are entirely unavailable, empirical weighted risk minimization and meta-learning domain generalization algorithms allow models to generalize across covariate shifts. A final AlexNet-based model was trained with various covariate shift corrections to achieve a 0.6854 F1 score while naive algorithms achieved 0.6093.*

# Experiments
[Scientific report](.final_report.pdf)
1. Single-domain classification
    1. Baseline CNNs
    2. Factorizing improvements to AlexNet
    3. Data augmentations
    4. Padding methods
    5. RegNets
    6. Designing channel magnitude and ratio network space
2. Data-abundant covariate shifts
    1. Cross domain generalization on single-domain and naive combined models
    2. Empirical weighted risk minimization
3. Data deficient covariate shifts with single domain training
    1. Augmentations and generalization
4. Data deficient covariate shifts with multi domain training
    1. Meta-learning domain generalization
    2. Scaling to entire dataset

# Achieved Learning Goals
* **Applied deep learning**: Trained various CNN models and created data insights through controlled experiments on model architecture, hyperparameters, and data augmentations.
* **Implementing novel algorithms**: Applied training algorithms to resolve covariate shifts and demonstrated effectiveness to cross-domain evaluation.
* **Network design**: Applied AnyNet network design protocol on RegNet to create simple design constraints on deep learning architecture.

