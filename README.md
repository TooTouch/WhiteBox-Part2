# WhiteBox-Part2
The White Box Project is a project that introduces many ways to solve the part of the black box of machine learning. In this part, i've introduced and experimented with ways to interpret and evaluate models in the field of **tabular data**.

# Requirements
```
numpy == 1.17.3
scikit-learn == 0.21.2
xgboost == 0.90
tensorflow == 1.14.0
```

# Dataset
1. Titanic: Machine Learning from Disaster [1] (Classification)
2. Cervical Cancer [2] (Classification)
3. House Prices: Advanced Regression Techniques [3] (Regression)
4. Bike Sharing [4] (Regression)
5. Youtube Spam [5] (Classification & NLP)

# Model 
The parameters used to learn the model can be found [here]()
1. Linear Regression or Logistic Regression (LR)
2. Decision Tree (DT)
3. Random Forest (RF)
4. XGboost (XGB)
5. Deep Neural Network (DNN) 

# Method 
## SHapley Additive exPlanations (SHAP) [5]
- **SHAP에 대한 모든 것***
  - Part 1 : [Shapley Values 알아보기](https://datanetworkanalysis.github.io/2019/12/23/shap1)
  - Part 2 : [SHAP 소개](https://datanetworkanalysis.github.io/2019/12/24/shap2)
  - Part 3 : [SHAP을 통한 시각화해석](https://datanetworkanalysis.github.io/2019/12/24/shap3)
  
- **Practice**
  
# How-to
## Model Training
```python
> python main.py --model=['LR','DT','RF','XGB','DNN'] \
                 --dataset=['titanic','house_price','bike_sharing','cervical_cancer','youtube_spam'] \
                 --file_path=dataset_directory \
                 --save_path=save_directory \
                 --params_path=parameter_directory
```

## Load Model
```python
# LR, DT, RF, XGB
import pickle
model = pickle.load(open(saved_model_path, 'rb'))

# DNN
import tensorflow as tf
model = tf.keras.models.load_model(saved_model_path)
```

# Results


# Reference
[1] Kaggle Competiton : [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

[2] Kelwin Fernandes, Jaime S. Cardoso, and Jessica Fernandes. 'Transfer Learning with Partial Observability Applied to Cervical Cancer Screening.' Iberian Conference on Pattern Recognition and Image Analysis. Springer International Publishing, 2017.

[3] Kaggle Competition : [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/description)

[4] Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.

[5] Alberto, T.C., Lochter J.V., Almeida, T.A. TubeSpam: Comment Spam Filtering on YouTube. Proceedings of the 14th IEEE International Conference on Machine Learning and Applications (ICMLA'15), 1-6, Miami, FL, USA, December, 2015.

[6] Lundberg, Scott M., and Su-In Lee. “[A unified approach to interpreting model predictions.](https://arxiv.org/pdf/1705.07874.pdf)” Advances in Neural Information Processing Systems. 2017. ([Korean Version](https://www.notion.so/tootouch/A-Unified-Approach-to-Interpreting-Model-Predictions-96de8a9e08b149c48cdd802cd62ad59f))
