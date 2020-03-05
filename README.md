# WhiteBox-Part2

![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FTooTouch%2FWhiteBox-Part2)

The White Box Project is a project that introduces many ways to solve the part of the black box of machine learning. This project is based on [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) by Christoph Molnar [1]. I recommend you to read the book first and practice this project. If you are R user, you can see R code used in examples [here](https://github.com/christophM/interpretable-ml-book). 

한글로 변역된 내용은 [여기](https://tootouch.github.io/IML/start/)서 확인하실 수 있습니다. 변역은 저자와 협의 후 진행되었음을 알립니다.

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
The parameters used to learn the model can be found [here](https://github.com/bllfpc/WhiteBox-Part2/tree/master/params)
1. Random Forest (RF)
2. XGboost (XGB)
3. LigthGBM (LGB)
4. Deep Neural Network (DNN) 

# Interpretable Methods

**Model-specific methods** [[English](https://christophm.github.io/interpretable-ml-book/simple.html)|[Korean](https://tootouch.github.io/IML/interpretable_models/)]
- Linear Regression [[English](https://christophm.github.io/interpretable-ml-book/limo.html)|[Korean](https://tootouch.github.io/IML/linear_regression/)]
- Logistic Regression [[English](https://christophm.github.io/interpretable-ml-book/logistic.html)|[Korean](https://tootouch.github.io/IML/logistic_regression/)]
- GLM, GAM and more [[English](https://christophm.github.io/interpretable-ml-book/extend-lm.html)|[Korean](https://tootouch.github.io/IML/glm_gam_and_more/)]
- Decision Tree [[English](https://christophm.github.io/interpretable-ml-book/tree.html)|[Korean](https://tootouch.github.io/IML/decision_tree/)]
- Decision Rules [[English](https://christophm.github.io/interpretable-ml-book/rules.html)|[Korean](https://tootouch.github.io/IML/decision_rules/)]
- RuleFit [[English](https://christophm.github.io/interpretable-ml-book/rulefit.html)|[Korean](https://tootouch.github.io/IML/rulefit/)]
- Other Interpretable Models [[English](https://christophm.github.io/interpretable-ml-book/other-interpretable.html)|[Korean](https://tootouch.github.io/IML/other_interpretable_models/)]

**Model-agnostic methods** [[English](https://christophm.github.io/interpretable-ml-book/agnostic.html)|[Korean](https://tootouch.github.io/IML/model_agnostic_methods/)]
- Partial Dependence Plot (PDP) [[English](https://christophm.github.io/interpretable-ml-book/pdp.html)|[Korean](https://tootouch.github.io/IML/partial_dependence_plot/)]
- Individual Conditional Expectation (ICE) [[English](https://christophm.github.io/interpretable-ml-book/ice.html)|[Korean](https://tootouch.github.io/IML/individual_conditional_expectation/)]
- Accumulated Local Effects (ALE) Plot [[English](https://christophm.github.io/interpretable-ml-book/ale.html)|[Korean](https://tootouch.github.io/IML/accumulated_local_effects/)]
- Feature Interaction [[English](https://christophm.github.io/interpretable-ml-book/interaction.html)|[Korean](https://tootouch.github.io/IML/feature_interaction/)]
- Permutation Feature Importance [[English](https://christophm.github.io/interpretable-ml-book/feature-importance.html)|[Korean](https://tootouch.github.io/IML/permutation_feature_importance/)]
- Global Surrogate [[English](https://christophm.github.io/interpretable-ml-book/global.html)|[Korean](https://tootouch.github.io/IML/global_surrogate/)]
- Local Surrogate (LIME) [[English](https://christophm.github.io/interpretable-ml-book/lime.html)|[Korean](https://tootouch.github.io/IML/local_surrogate/)]
- Scoped Rules (Anchors) [[English](https://christophm.github.io/interpretable-ml-book/anchors.html)|[Korean](https://tootouch.github.io/IML/scoped_rules/)]
- Shapley Values [[English](https://christophm.github.io/interpretable-ml-book/shapley.html)|[Korean](https://tootouch.github.io/IML/shapley_values/)]
- SHAP (SHapley Additive exPlanations) [[English](https://christophm.github.io/interpretable-ml-book/shap.html)|[Korean](https://tootouch.github.io/IML/shap/)]
  

# Reference
[1] Molnar, Christoph. "Interpretable machine learning. A Guide for Making Black Box Models Explainable", 2019. https://christophm.github.io/interpretable-ml-book/.

[2] Kaggle Competiton : [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

[3] Kelwin Fernandes, Jaime S. Cardoso, and Jessica Fernandes. 'Transfer Learning with Partial Observability Applied to Cervical Cancer Screening.' Iberian Conference on Pattern Recognition and Image Analysis. Springer International Publishing, 2017. [[Link]](https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29)

[4] Kaggle Competition : [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/description)

[5] Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg. [[Link]](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

[6] Alberto, T.C., Lochter J.V., Almeida, T.A. TubeSpam: Comment Spam Filtering on YouTube. Proceedings of the 14th IEEE International Conference on Machine Learning and Applications (ICMLA'15), 1-6, Miami, FL, USA, December, 2015. [[Link]](http://www.dt.fee.unicamp.br/~tiago//youtubespamcollection/)

[7] Lundberg, Scott M., and Su-In Lee. “[A unified approach to interpreting model predictions.](https://arxiv.org/pdf/1705.07874.pdf)” Advances in Neural Information Processing Systems. 2017. ([Korean Version](https://www.notion.so/tootouch/A-Unified-Approach-to-Interpreting-Model-Predictions-96de8a9e08b149c48cdd802cd62ad59f))

