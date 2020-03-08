# WhiteBox-Part2

![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FTooTouch%2FWhiteBox-Part2)

The White Box Project is a project that introduces many ways to solve the part of the black box of machine learning. This project is based on [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) by Christoph Molnar [1]. I recommend you to read the book first and practice this project. If you are R user, you can see R code used in examples [here](https://github.com/christophM/interpretable-ml-book). 

한글로 번역된 내용은 [여기](https://tootouch.github.io/IML/start/)서 확인하실 수 있습니다. 번역은 저자와 협의 후 진행되었음을 알립니다.

만약 번역본에 잘못된 해석이 있다면 wogur379@gmail.com 또는 issue에 남겨주세요. 감사합니다.

# Purpose

The goal is to analysis various data into black box models and to build **a pipeline of analysis reports** using interpretable methods. 

# Requirements

```
numpy == 1.17.3
scikit-learn == 0.21.2
xgboost == 0.90
tensorflow == 1.14.0
```

# Dataset
1. Titanic: Machine Learning from Disaster (Classification) [2]
2. Cervical Cancer (Classification) [3]
3. House Prices: Advanced Regression Techniques (Regression) [4]
4. Bike Sharing (Regression) [5]
5. Youtube Spam (Classification & NLP) [6]

# Black Box Models
The parameters used to learn the model can be found [here](https://github.com/bllfpc/WhiteBox-Part2/tree/master/params).
1. Random Forest (RF)
2. XGboost (XGB)
3. LigthGBM (LGB)
4. Deep Neural Network (DNN) 

# Interpretable Methods

**Model-specific methods** [ [English](https://christophm.github.io/interpretable-ml-book/simple.html) , [Korean](https://tootouch.github.io/IML/interpretable_models/) ]
- Linear Regression [ [English](https://christophm.github.io/interpretable-ml-book/limo.html) , [Korean](https://tootouch.github.io/IML/linear_regression/) ]
- Logistic Regression [ [English](https://christophm.github.io/interpretable-ml-book/logistic.html) , [Korean](https://tootouch.github.io/IML/logistic_regression/) ]
- GLM, GAM and more [ [English](https://christophm.github.io/interpretable-ml-book/extend-lm.html) , [Korean](https://tootouch.github.io/IML/glm_gam_and_more/) ]
- Decision Tree [ [English](https://christophm.github.io/interpretable-ml-book/tree.html) , [Korean](https://tootouch.github.io/IML/decision_tree/) ]
- Decision Rules [ [English](https://christophm.github.io/interpretable-ml-book/rules.html) , [Korean](https://tootouch.github.io/IML/decision_rules/) ]
- RuleFit [ [English](https://christophm.github.io/interpretable-ml-book/rulefit.html) , [Korean](https://tootouch.github.io/IML/rulefit/) ]
- Other Interpretable Models [ [English](https://christophm.github.io/interpretable-ml-book/other-interpretable.html) , [Korean](https://tootouch.github.io/IML/other_interpretable_models/) ]

**Model-agnostic methods** [ [English](https://christophm.github.io/interpretable-ml-book/agnostic.html) , [Korean](https://tootouch.github.io/IML/model_agnostic_methods/) ]
- Partial Dependence Plot (PDP) [ [English](https://christophm.github.io/interpretable-ml-book/pdp.html) , [Korean](https://tootouch.github.io/IML/partial_dependence_plot/) ]
- Individual Conditional Expectation (ICE) [ [English](https://christophm.github.io/interpretable-ml-book/ice.html) , [Korean](https://tootouch.github.io/IML/individual_conditional_expectation/) ]
- Accumulated Local Effects (ALE) Plot [ [English](https://christophm.github.io/interpretable-ml-book/ale.html) , [Korean](https://tootouch.github.io/IML/accumulated_local_effects/) ] 
- Feature Interaction [ [English](https://christophm.github.io/interpretable-ml-book/interaction.html) , [Korean](https://tootouch.github.io/IML/feature_interaction/) ]
- Permutation Feature Importance [ [English](https://christophm.github.io/interpretable-ml-book/feature-importance.html) , [Korean](https://tootouch.github.io/IML/permutation_feature_importance/) ]
- Global Surrogate [ [English](https://christophm.github.io/interpretable-ml-book/global.html) , [Korean](https://tootouch.github.io/IML/global_surrogate/) ]
- Local Surrogate (LIME) [ [English](https://christophm.github.io/interpretable-ml-book/lime.html) , [Korean](https://tootouch.github.io/IML/local_surrogate/) ]
- Scoped Rules (Anchors) [ [English](https://christophm.github.io/interpretable-ml-book/anchors.html) , [Korean](https://tootouch.github.io/IML/scoped_rules/) ]
- Shapley Values [ [English](https://christophm.github.io/interpretable-ml-book/shapley.html) , [Korean](https://tootouch.github.io/IML/shapley_values/) ]
- SHAP (SHapley Additive exPlanations) [ [English](https://christophm.github.io/interpretable-ml-book/shap.html) , [Korean](https://tootouch.github.io/IML/shap/) ]
  

# Python Implementation

**Interpretable Models**

Name | Packages 
---|---
Linear Regression | [`scikit-learn`](#scikit-learn) [`statsmodels`](#statsmodels)
Logistic Regression | [`scikit-learn`](#scikit-learn) [`statsmodels`](#statsmodels)
Ridge Regression | [`scikit-learn`](#scikit-learn) [`statsmodels`](#statsmodels)
Lasso Regression | [`scikit-learn`](#scikit-learn) [`statsmodels`](#statsmodels)
Generalized Linear Model (GLM) | [`statsmodels`](#statsmodels)
Generalized Additive Model (GAM) | [`statsmodels`](#statsmodels) [`pyGAM`](#pyGAM)
Decision Tree | [`scikit-learn`](#scikit-learn)
Baysian Rule Lists | [`skater`](#skater)
RuleFit | [`rulefit`](#rulefit)
Skope-rules | [`skope-rules`](#skope-rules)


**Model-Agnostic Methods**

Name | Packages
---|---
Partial Dependence Plot (PDP) | [`skater`](#skater) [`scikit-learn`](#scikit-learn) 
Individual Conditional Expectation (ICE) Plot | [`PyCEbox`](#PyCEbox)
Feature Importance | [`skater`](#skater)
Local Surrogate | [`skater`](#skater) [`lime`](#lime)
Global Surrogate | [`skater`](#skater)
Scoped Rules (Anchors) | [`alibi`](#alibi)
SHapley Additive exPlanation (SHAP) | [`shap`](#shap)

**Example-Based Explanations**

Name | Packages
---|---
Contrastive Explanations Method (CEM) | [`alibi`](#alibi)
Counterfactual Instances | [`alibi`](#alibi)
Prototype Counterfactuals | [`MMD-critic`](#MMD-critic)
Influence Instances | [`influence-release`](#influence-release)


## Install packages

### scikit-learn

scikit-learn is a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license.

The project was started in 2007 by David Cournapeau as a Google Summer of Code project, and since then many volunteers have contributed. See the About us page for a list of core contributors.

It is currently maintained by a team of volunteers.

Scikit-learn is available in through conda provided by Anaconda.

- **Documentation** : [https://scikit-learn.org/stable/  ](https://scikit-learn.org/stable/  )
- **Github Repository** : [https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)

**Installation** 

```bash
# Pip
pip install -U scikit-learn
# Conda
onda install scikit-learn
```

```python
import sklearn
```

### statsmodels

statsmodels is a Python module that provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests, and statistical data exploration. An extensive list of result statistics are available for each estimator. The results are tested against existing statistical packages to ensure that they are correct. The package is released under the open source Modified BSD (3-clause) license. The online documentation is hosted at statsmodels.org.

Statsmodels is available in through conda provided by Anaconda.

- **Documentation** : [https://www.statsmodels.org/stable/index.html  ](https://www.statsmodels.org/stable/index.html  )
- **Github Repository** : [https://github.com/statsmodels/statsmodels](https://github.com/statsmodels/statsmodels)

**Installation**  

```bash
# Pip
pip install statsmodels 
# Conda
conda install -c conda-forge statsmodels
```

```python
import statsmodels
```

### pyGAM 

pyGAM is a package for building Generalized Additive Models in Python, with an emphasis on modularity and performance. The API will be immediately familiar to anyone with experience of scikit-learn or scipy.

- **Documentation** : [https://pygam.readthedocs.io/en/latest/notebooks/quick_start.html  ](https://pygam.readthedocs.io/en/latest/notebooks/quick_start.html  )
- **Github Repository** : [https://github.com/dswah/pyGAM](https://github.com/dswah/pyGAM)

**Installation**  

```bash
# Pip
pip install pygam
# Conda
conda install -c conda-forge pygam
```

```python
import pygam
```

### skater    

Skater is a open source unified framework to enable Model Interpretation for all forms of model to help one build an Interpretable machine learning system often needed for real world use-cases. Skater supports algorithms to demystify the learned structures of a black box model both globally(inference on the basis of a complete data set) and locally(inference about an individual prediction).

- **Documentation** : [https://oracle.github.io/Skater/index.html  ](https://oracle.github.io/Skater/index.html  )
- **Github Repository** : [https://github.com/oracle/Skater](https://github.com/oracle/Skater)

**Installation** 

```bash
# Option 1: without rule lists and without deepinterpreter
pip install -U skater

# Option 2: without rule lists and with deepinterpreter:
pip3 install --upgrade tensorflow 
sudo pip install keras
pip install -U skater

# Option 3: For everything included
conda install gxx_linux-64
pip3 install --upgrade tensorflow 
sudo pip install keras
sudo pip install -U --no-deps --force-reinstall --install-option="--rl=True" skater==1.1.1b1

# Conda
conda install -c conda-forge Skater
```

```python
import skater
```


### PDPbox

python partial dependence plot toolbox

This repository is inspired by ICEbox. The goal is to visualize the impact of certain features towards model prediction for any supervised learning algorithm using partial dependence plots [R1](https://pdpbox.readthedocs.io/en/latest/papers.html#r1) [R2](https://pdpbox.readthedocs.io/en/latest/papers.html#r2). PDPbox now supports all scikit-learn algorithms.

- **Documentation** : [https://pdpbox.readthedocs.io/en/latest/index.html#](https://pdpbox.readthedocs.io/en/latest/index.html#)
- **Github Repository** : [https://github.com/SauceCat/PDPbox](https://github.com/SauceCat/PDPbox)

**Installation**  

```bash
# Pip
pip install pdpbox
```

```python
import pdpbox
```


### LIME

This project is about explaining what machine learning classifiers (or models) are doing. At the moment, we support explaining individual predictions for text classifiers or classifiers that act on tables (numpy arrays of numerical or categorical data) or images, with a package called lime (short for local interpretable model-agnostic explanations). Lime is based on the work presented in [this paper](https://arxiv.org/abs/1602.04938) ([bibtex here for citation](https://github.com/marcotcr/lime/blob/master/citation.bib)).

- **Documentation** : [https://lime-ml.readthedocs.io/en/latest/index.html](https://lime-ml.readthedocs.io/en/latest/index.html)
- **Github Repository** : [https://github.com/marcotcr/lime](https://github.com/marcotcr/lime)

**Installation**  

```bash
# Pip
pip install lime
```

```python
import lime
```


### PyCEbox

A Python implementation of individual conditional expecation plots inspired by R's [ICEbox](https://cran.r-project.org/web/packages/ICEbox/index.html). Individual conditional expectation plots were introduced in Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation ([arXiv:1309.6392](https://arxiv.org/abs/1309.6392)).

- **Documentation** : [http://austinrochford.github.io/PyCEbox/docs/](http://austinrochford.github.io/PyCEbox/docs/)
- **Github Repository** : [https://github.com/AustinRochford/PyCEbox](https://github.com/AustinRochford/PyCEbox)

**Installation** 

```bash
# Pip 
pip install pycebox
```

```python
import pycebox
```

### rulefit

Implementation of a rule based prediction algorithm based on [the rulefit algorithm from Friedman and Popescu (PDF)(http://statweb.stanford.edu/~jhf/ftp/RuleFit.pdf)]

- **Github Repository** : [https://github.com/christophM/rulefit](https://github.com/christophM/rulefit)

**Installation** 

```bash
# Pip
pip install git+git://github.com/christophM/rulefit.git
```

```python
import rulefit
```

### skope-rules

Skope-rules is a Python machine learning module built on top of scikit-learn and distributed under the 3-Clause BSD license.

Skope-rules aims at learning logical, interpretable rules for "scoping" a target class, i.e. detecting with high precision instances of this class.

Skope-rules is a trade off between the interpretability of a Decision Tree and the modelization power of a Random Forest.

- **Documentation** : [https://skope-rules.readthedocs.io/en/latest/index.html](https://skope-rules.readthedocs.io/en/latest/index.html)
- **Github Repository** : [https://github.com/scikit-learn-contrib/skope-rules](https://github.com/scikit-learn-contrib/skope-rules)

**Installation**  

```bash
# Pip
pip install skope-rules
```

```python
import skrules
```

### alibi

Alibi is an open source Python library aimed at machine learning model inspection and interpretation. The initial focus on the library is on black-box, instance based model explanations.


- **Documentation** : [https://docs.seldon.io/projects/alibi/en/latest/#](https://docs.seldon.io/projects/alibi/en/latest/#)
- **Github Repository** : [https://github.com/SeldonIO/alibi](https://github.com/SeldonIO/alibi)

**Installation** 

```bash
# Pip 
pip install alibi
```

```python
import alibi
```

### influence-instance

This code replicates the experiments from the following paper:

> Pang Wei Koh and Percy Liang
[Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)
International Conference on Machine Learning (ICML), 2017.

We have a reproducible, executable, and Dockerized version of these scripts on [Codalab](https://worksheets.codalab.org/worksheets/0x2b314dc3536b482dbba02783a24719fd/).

The datasets for the experiments can also be found at the Codalab link.

- **Github Repository** : [https://github.com/kohpangwei/influence-release](https://github.com/kohpangwei/influence-release)


### shap

SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions (see [papers](https://github.com/slundberg/shap#citations) for details and citations).


- **Documentation** : [https://shap.readthedocs.io/en/latest/#](https://shap.readthedocs.io/en/latest/#)
- **Github Repository** : [https://github.com/slundberg/shap](https://github.com/slundberg/shap)

**Installation** 

```bash
# Pip
pip install shap
# Conda
conda install -c conda-forge shap
```

```python
import shap
```

### MMD-critic

This method is proposed in [this papaer](http://people.csail.mit.edu/beenkim/papers/KIM2016NIPS_MMD.pdf). 

**Abstract**  
Example-based explanations are widely used in the effort to improve the interpretability of highly complex distributions. However, prototypes alone are rarely
sufficient to represent the gist of the complexity. In order for users to construct
better mental models and understand complex data distributions, we also need
criticism to explain what are not captured by prototypes. Motivated by the Bayesian
model criticism framework, we develop MMD-critic which efficiently learns prototypes and criticism, designed to aid human interpretability. A human subject pilot
study shows that the MMD-critic selects prototypes and criticism that are useful
to facilitate human understanding and reasoning. We also evaluate the prototypes
selected by MMD-critic via a nearest prototype classifier, showing competitive
performance compared to baselines.

- **Github Repository** : [https://github.com/BeenKim/MMD-critic](https://github.com/BeenKim/MMD-critic)


# Reference
[1] Molnar, Christoph. "Interpretable machine learning. A Guide for Making Black Box Models Explainable", 2019. https://christophm.github.io/interpretable-ml-book/.

[2] Kaggle Competiton : [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

[3] Kelwin Fernandes, Jaime S. Cardoso, and Jessica Fernandes. 'Transfer Learning with Partial Observability Applied to Cervical Cancer Screening.' Iberian Conference on Pattern Recognition and Image Analysis. Springer International Publishing, 2017. [[Link]](https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29)

[4] Kaggle Competition : [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/description)

[5] Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg. [[Link]](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

[6] Alberto, T.C., Lochter J.V., Almeida, T.A. TubeSpam: Comment Spam Filtering on YouTube. Proceedings of the 14th IEEE International Conference on Machine Learning and Applications (ICMLA'15), 1-6, Miami, FL, USA, December, 2015. [[Link]](http://www.dt.fee.unicamp.br/~tiago//youtubespamcollection/)

[7] Lundberg, Scott M., and Su-In Lee. “[A unified approach to interpreting model predictions.](https://arxiv.org/pdf/1705.07874.pdf)” Advances in Neural Information Processing Systems. 2017. ([Korean Version](https://www.notion.so/tootouch/A-Unified-Approach-to-Interpreting-Model-Predictions-96de8a9e08b149c48cdd802cd62ad59f))