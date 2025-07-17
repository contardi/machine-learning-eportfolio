# 🤖 Machine Learning Module - e-Portfolio

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Module-blue)
![MSc AI](https://img.shields.io/badge/MSc-Artificial%20Intelligence-green)
![Python](https://img.shields.io/badge/Python-3.12%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-red)

Welcome to my e-Portfolio for the **Machine Learning** module of the MSc in Artificial Intelligence programme. This repository contains practical implementations, exercises, and projects covering fundamental to advanced machine learning concepts.

---

## 📚 Module Overview

This comprehensive module covers the theoretical foundations and practical applications of machine learning, progressing from basic statistical concepts through deep learning architectures. Each unit builds upon previous knowledge, culminating in a thorough understanding of modern ML techniques.

---

## 🗂️ Repository Structure

### Unit 1: Introduction to Machine Learning

**Objective**: Establish foundational understanding of machine learning paradigms, terminology, and core concepts.

**Learning Outcomes**: 
- Distinguish between supervised, unsupervised, and reinforcement learning
- Understand the machine learning pipeline and workflow
- Identify appropriate ML approaches for different problem types

**Collaborative Discussion**  
[**The 4th Industrial Revolution**](unit-01/collaborative-discussion.md) 


---

### Unit 2: Exploratory Data Analysis (EDA)

**Objective**: Develop proficiency in data exploration, visualization, and preprocessing techniques essential for machine learning.

**Learning Outcomes**:
- Conduct comprehensive exploratory data analysis on complex datasets
- Apply appropriate visualization techniques to uncover patterns and relationships
- Handle missing data and outliers effectively
- Select relevant features based on statistical analysis

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [**EDA-Tutorial.ipynb**](unit-02/EDA-Tutorial.ipynb) | Comprehensive EDA tutorial using house price data (79 features) - missing value analysis, correlation heatmaps, advanced visualizations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-02/EDA-Tutorial.ipynb) |
| [**eda-auto-mpg.ipynb**](unit-02/eda-auto-mpg.ipynb) | EDA on automotive MPG dataset with multiple regression models (Linear, Decision Tree, Random Forest, Gradient Boosting, SVR, k-NN, MLP) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-02/eda-auto-mpg.ipynb) |

**Datasets**: `auto-mpg.csv`, `unit2-train.csv`, `unit2-test.csv`

---

### Unit 3: Statistical Analysis and Regression Fundamentals

**Objective**: Learn fundamental statistical concepts and regression techniques that form the basis of predictive modeling.

**Learning Outcomes**:
- Calculate and interpret correlation coefficients and covariance
- Implement linear, multiple, and polynomial regression models
- Understand the mathematical foundations of regression analysis
- Apply regression techniques to real-world prediction problems

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [**01-covariance_pearson_correlation.ipynb**](unit-03/01-covariance_pearson_correlation.ipynb) | Understanding covariance and Pearson correlation coefficient | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-03/01-covariance_pearson_correlation.ipynb) |
| [**02-linear_regression.ipynb**](unit-03/02-linear_regression.ipynb) | Simple linear regression implementation and visualization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-03/02-linear_regression.ipynb) |
| [**03-multiple_linear_regression.ipynb**](unit-03/03-multiple_linear_regression.ipynb) | Multiple linear regression for CO2 emissions prediction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-03/03-multiple_linear_regression.ipynb) |
| [**04-polynomial_regression.ipynb**](unit-03/04-polynomial_regression.ipynb) | Polynomial regression for non-linear relationships | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-03/04-polynomial_regression.ipynb) |

---

### Unit 4: Advanced Regression Analysis

**Objective**: Apply regression techniques to complex real-world datasets and evaluate model performance.

**Learning Outcomes**:
- Perform correlation analysis on multivariate datasets
- Build and validate regression models for practical applications
- Interpret model coefficients and statistical significance
- Compare different regression approaches for optimal results

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [**01-demo-correlation-regression-fuel-consumption.ipynb**](unit-04/01-demo-correlation-regression-fuel-consumption.ipynb) | Fuel consumption prediction using correlation analysis and linear regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-04/01-demo-correlation-regression-fuel-consumption.ipynb) |
| [**02-population-gdp-analysis.ipynb**](unit-04/02-population-gdp-analysis.ipynb) | Global population and GDP correlation analysis (2001-2020) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-04/02-population-gdp-analysis.ipynb) |

**Datasets**: `FuelConsumption.csv`, `Global_GDP.csv`, `Global_Population.csv`

---

### Unit 5: Similarity and Distance Metrics

**Objective**: Understand and implement various similarity and distance measures used in machine learning algorithms.

**Learning Outcomes**:
- Calculate and interpret similarity coefficients (Jaccard, cosine, etc.)
- Apply distance metrics for classification and clustering tasks
- Choose appropriate similarity measures for different data types
- Implement similarity-based recommendation systems

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [**jaccard-sklearn-solution.ipynb**](unit-05/jaccard-sklearn-solution.ipynb) | Jaccard coefficient implementation for pathological test similarity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-05/jaccard-sklearn-solution.ipynb) |

---

### Unit 6: Unsupervised Learning - Clustering

**Objective**: Understand and use unsupervised learning techniques with focus on K-Means clustering for pattern discovery.

**Learning Outcomes**:
- Implement K-Means clustering algorithm and understand its mathematical foundations
- Determine optimal number of clusters using elbow method and silhouette analysis
- Apply clustering to diverse datasets for segmentation and pattern recognition
- Evaluate clustering quality using appropriate metrics

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [**01-k-means-demo-notebook.ipynb**](unit-06/01-k-means-demo-notebook.ipynb) | Customer segmentation using K-Means with elbow method | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-06/01-k-means-demo-notebook.ipynb) |
| [**02-task-a-iris-kmeans.ipynb**](unit-06/02-task-a-iris-kmeans.ipynb) | K-Means clustering on Iris dataset (K=3) with species comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-06/02-task-a-iris-kmeans.ipynb) |
| [**03-task-b-wine-kmeans.ipynb**](unit-06/03-task-b-wine-kmeans.ipynb) | Wine quality clustering based on chemical composition | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-06/03-task-b-wine-kmeans.ipynb) |
| [**04-task-c-weather-aus-kmeans.ipynb**](unit-06/04-task-c-weather-aus-kmeans.ipynb) | Australian weather pattern clustering (145K records) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-06/04-task-c-weather-aus-kmeans.ipynb) |

**Datasets**: `iris.csv`, `wine.csv`, `weatherAUS.csv`

The group assignment repository from this unit can be found [here]([https://github.com/tm3-machine-learning/airbnb])

---

### Unit 7: Neural Networks - Perceptron

**Objective**: Understand the fundamentals of neural networks through perceptron implementation and analysis.

**Learning Outcomes**:
- Implement single and multi-layer perceptrons from scratch
- Understand activation functions and weight update mechanisms
- Apply perceptrons to binary classification problems
- Recognize the limitations and capabilities of perceptron models

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [**01-simple_perceptron.ipynb**](unit-07/01-simple_perceptron.ipynb) | Simple perceptron implementation from scratch using NumPy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-07/01-simple_perceptron.ipynb) |
| [**02-perceptron_AND_operator.ipynb**](unit-07/02-perceptron_AND_operator.ipynb) | Perceptron learning the AND logic gate | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-07/02-perceptron_AND_operator.ipynb) |
| [**03-multi-layer-perceptron.ipynb**](unit-07/03-multi-layer-perceptron.ipynb) | Multi-layer perceptron (MLP) implementation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-07/03-multi-layer-perceptron.ipynb) |

---

### Unit 8: Optimization Algorithms

**Objective**: Learn optimization techniques essential for training machine learning models.

**Learning Outcomes**:
- Implement gradient descent algorithm with various learning rates
- Understand cost functions and their role in optimization
- Analyze convergence behavior and optimization challenges
- Apply optimization techniques to neural network training

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [**01-gradient_descent_cost_function.ipynb**](unit-08/01-gradient_descent_cost_function.ipynb) | Gradient descent implementation with learning rate analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-08/01-gradient_descent_cost_function.ipynb) |

---

### Unit 9: Deep Learning - Convolutional Neural Networks

**Objective**: Design and implement convolutional neural networks for computer vision applications.

**Learning Outcomes**:
- Understand CNN architecture components (convolutional layers, pooling, etc.)
- Implement CNNs for image classification tasks
- Apply data augmentation and regularization techniques
- Evaluate and optimize CNN performance on benchmark datasets

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [**01-convolutional-neural-networks-CNN-object-recognition.ipynb**](unit-09/01-convolutional-neural-networks-CNN-object-recognition.ipynb) | CNN for CIFAR-10 image classification (10 classes, 60K images) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-09/01-convolutional-neural-networks-CNN-object-recognition.ipynb) |

---

### Unit 10: Advanced Deep Learning

**Objective**: Explore deep learning architectures and techniques for complex AI applications.

**Learning Outcomes**:
- Understand advanced architectures (RNNs, Transformers, GANs)
- Apply transfer learning and fine-tuning strategies
- Implement deep learning solutions for diverse domains
- Optimize models for production deployment

In this module we have used a CNN explainer to help visualize the CNN model's decision-making process:
[CNN Explainer/](https://poloclub.github.io/cnn-explainer/)


---

### Unit 11: Model Evaluation and Performance Metrics

**Objective**: Understand model evaluation techniques to ensure reliable and robust ML systems.

**Learning Outcomes**:
- Apply appropriate metrics for classification and regression tasks
- Implement cross-validation and model selection strategies
- Interpret confusion matrices, ROC curves, and performance reports
- Identify and address overfitting, underfitting, and bias issues

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [**01-model_performance_measurement.ipynb**](unit-11/01-model_performance_measurement.ipynb) | Complete guide to ML metrics - confusion matrix, F1, ROC-AUC, RMSE, MAE, R² | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contardi/machine-learning-portfolio/blob/main/unit-11/01-model_performance_measurement.ipynb) |

**Assignment**:  
The repository for the unit's assignment can be found [here]([https://github.com/contardi/cnn-product-recognition])

---

### Unit 12: Machine Learning in Production

**Objective**: Learn to deploy, monitor, and maintain machine learning models in production environments. Lean how to use AI ethically and responsibly.

**Learning Outcomes**:
- Understand Industry 4.0 and Industry 5.0
- Understand MLOps principles and best practices
- Implement model versioning and experiment tracking
- Deploy models using cloud services and APIs
- Monitor model performance and implement continuous improvement

---

## 🛠️ Technologies & Libraries

- **Core Libraries**: NumPy, Pandas, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, TensorFlow/Keras
- **Data Analysis**: SciPy, Missingno
- **Visualization**: Plotly (selected notebooks)

---

## 📋 Prerequisites

- Python 3.12
- Jupyter Notebook/Lab or Google Colab
- Basic understanding of:
  - Linear algebra
  - Statistics and probability
  - Python programming

---

## 🚀 Getting Started

### Option 1: Google Colab (Recommended)
Click any "Open in Colab" button above to run notebooks directly in your browser with free GPU access.

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/contardi/machine-learning-portfolio.git
cd e-portfolio

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install jupyter pandas numpy matplotlib seaborn scikit-learn tensorflow

# Start Jupyter
jupyter lab
```

---

## 👤 Author

**Thiago Contardi**
- MSc in Artificial Intelligence Student
- [GitHub Profile](https://github.com/contardi)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- University of Essex and course instructors
- Open-source community for datasets and libraries
- Fellow students for collaborative learning

---
