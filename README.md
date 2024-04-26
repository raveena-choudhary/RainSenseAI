# RainSenseAI

RainSenseAI is a machine learning project that focuses on rain prediction using Decision Trees and Deep Neural Networks. The project aims to address the challenge of accurate rain prediction, which is crucial for various applications such as agriculture, water resource management, urban planning, and transportation.

## Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Related Works](#related-works)
- [Methodologies](#methodologies)
- [Datasets](#datasets)
- [Decision Tree Model](#decision-tree-model)
- [Deep Neural Network Model](#deep-neural-network-model)
- [Optimization](#optimization)
- [Results](#results)
- [Usage](#usage)

## Abstract
This comprehensive study focuses on rain prediction using machine learning techniques, specifically Decision Trees and Deep Neural Networks. The goal is to address a machine learning problem by employing supervised and semi-supervised learning methods with Decision Trees, as well as supervised learning using a deep learning model (RNN). The study evaluates the performance of these models using various evaluation metrics and provides valuable insights into their effectiveness for rain prediction.

## Introduction
The problem statement addressed in the project revolves around accurate rain prediction and its importance in various application fields. The report discusses the complexity and uncertainty of atmospheric processes and the challenges associated with rain prediction. It highlights the advantages and limitations of existing solutions and proposes the use of machine learning techniques to capture the nonlinear relationships between meteorological variables and rainfall occurrence.

## Related Works
This section provides an overview of various studies that have explored the use of machine learning algorithms for rainfall prediction. It discusses the effectiveness of different techniques such as linear regression, SVM, RF, DT, and deep learning models. The impact of environmental features, data size, and algorithm choice on prediction accuracy is also analyzed.

## Methodologies
The project employs a combination of machine learning techniques, including Decision Trees (supervised and semi-supervised) and Recurrent Neural Networks (RNN). It describes the data preprocessing steps, feature selection and engineering techniques, and the evaluation metrics used to assess the models' performance. The importance of hyper-parameter optimization and the effectiveness of semi-supervised learning are discussed.

## Datasets
The project utilizes the "Rain in Australia" dataset sourced from Kaggle, which contains a large number of records and features. The data preprocessing steps include handling missing values, normalization, and encoding categorical variables. Feature extraction techniques are applied to transform raw data into a more meaningful representation.

## Decision Tree Model
The project compares the performance of two Decision Tree classifiers: one without hyper-parameter tuning and the other with hyper-parameter tuning using GridSearchCV. The models are trained on the dataset and evaluated using various evaluation metrics such as accuracy, F1 score, precision, recall, and ROC. The importance of selecting appropriate hyper-parameters is emphasized, and t-SNE visualizations are used for further analysis.

## Deep Neural Network Model
The project employs a Recurrent Neural Network (RNN) model for rain prediction. The model consists of multiple hidden layers with ReLU activation functions and batch normalization. The Binary Cross Entropy Loss function is used for optimization. The model's performance is evaluated using accuracy metrics, and t-SNE visualization is used to analyze weather patterns associated with rain and non-rain conditions.

## Optimization
This section discusses the optimization techniques used in the models, such as hyper-parameter tuning using GridSearchCV for Decision Trees and Adam optimization algorithm for the RNN model. The rationale behind the choices of optimization algorithms and learning rates is explained.

## Results
The results section presents the evaluation metrics and visualizations obtained from the Decision Tree and RNN models. It highlights the performance of the models in accurately predicting rain events and demonstrates the effectiveness of the chosen machine learning techniques. Comparative analysis of the models' performance and discussions on the limitations and future improvements are also provided.

| Model Name                             | Accuracy | Precision | Recall | F1-Score | ROC  |
| -------------------------------------- | -------- | --------- | ------ | -------- | ---- |
| Supervised W/O parameter                | 97.4%    | 0.983     | 1.0    | 0.991    | 1.0  |
| Supervised With parameter               | 99.50%   | 1.0       | 1.0    | 1.0      | 1.0  |
| Semi-Supervised                         | 97.4%    | 1.0       | 1.0    | 1.0      | 1.0  |
| Recurrent Neural Network                | 100%     | 1.0       | 1.0    | 1.0      | 1.0  |

Table: Comparison of 4 models.
## Usage
To use RainSenseAI, follow these steps:
1. Clone the repository: `git clone https://github.com/your-username/RainSenseAI.git`
2. Install the required dependencies such as numpy, matplotlib, tabulate, seaborn, sklearn, scipy, torch, thop -> local execution
3. For google colab, only install thop
4. Run the file sequentially

Make sure to have Python 3.7 or higher installed on your system.

## Documentation Folder:
1) Final report
2) Presentation
3) Contribution
4) Guidelines to run Project

## Data folder:
1) weather.csv -> dataset for this project
2) sample_test_data.csv

## Code folder:
1) AI_project_final.ipynb -> file with code for all three models.
