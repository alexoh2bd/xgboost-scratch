# GradientBoostedTrees

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Building multiple types of gradient-boosted trees from scratch on simple datasets for classification and regression

## Project Overview

This project provides a comprehensive implementation of gradient-boosted trees from scratch, including:

- Gradient Boosted Decision Trees (GBDT) from scratch
- XGBoost regularization techniques (L1, L2, and Gamma)
- Handling of missing values
- Feature importance calculation
- Tree visualization capabilities
- Proper handling of categorical variables
- Advanced splitting criteria optimization

## Why Implement Gradient Boosting from Scratch?

My ultimate *goal* is to become a data scientist and push the frontier of AI research. Tree-based models like XGBoost are bread and butter for a typical data science role with applications in any industry. By researching the statistics behind the model and translating the numbers to code, I deepen my understanding of the model, and enhance my ability to wield it in future situations.

1. **Deep Understanding**: Gain intimate knowledge of how gradient boosting works
2. **Learning Benefits**:
   - Understand tree-based algorithms and their optimizations
   - Learn about gradient boosting and its mathematical foundations
   - Explore regularization techniques and their impact
   - Gain insights into handling missing values
   - Understand feature importance calculations
   - Learn about efficient tree construction algorithms
3. **Customization**: Ability to modify and experiment with different aspects of the algorithm

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - tqdm
  - typer

### Setup

1. Clone the repository:
```bash
git clone https://github.com/alexoh2bd/xgboost-scratch.git
cd xgboost-scratch
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source ./venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Model

1. Download the Spaceship Titanic dataset from Kaggle (https://www.kaggle.com/competitions/spaceship-titanic/data)
2. Place the `train.csv` and `test.csv` files in the `data/raw/` directory
3. Run the preprocessing script:
```bash
python TreesFromScratch/dataset.py
```
4. Train and test the model:
```bash
python TreesFromScratch/modeling/xgb.py
```


## Model Parameters

Key parameters that can be tuned:
- `num_estimators`: Number of trees in the ensemble
- `max_depth`: Maximum depth of each tree
- `learning_rate`: Step size shrinkage used to prevent overfitting
- `min_child_weight`: Minimum sum of instance weight needed in a child
- `reg_alpha`: L1 regularization term
- `reg_lambda`: L2 regularization term
- `reg_gamma`: Minimum loss reduction required to make a further partition


## Learning Outcomes

By implementing this:
1. Understand the mathematical foundations of gradient boosting
2. Learn about tree construction algorithms and optimizations
3. Gain insights into regularization techniques
4. Understand how to handle missing values in tree-based models
5. Learn about feature importance calculation
6. Gain experience with model evaluation and validation
7. Understand the trade-offs
