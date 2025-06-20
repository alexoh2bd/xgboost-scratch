# GradientBoostedTrees

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**XGBoost** is a foundational tool for data scientists, driving some of the most _accurate_ and _scalable_ machine learning solutions across diverse industries. Its tree-based approach delivers efficient, interpretable, and high-performing models, especially suited for _structured_, _tabular_ datasets.

By building XGBoost from scratch, I deepen my understanding of the model and similar algorithms, enabling me to fine-tune performance, interpret results more clearly, and develop robust solutions for real-world challenges.

I built all tree construction components manually, including the calculation of residuals and second-order gradients (hessians), gain scoring, and the identification of optimal feature splits. I designed the model to support binary classification tasks, implementing techniques such as regularized loss minimization and greedy tree growth with depth control. This implementation also considers missing values in the datasets and draws splits accordingly.

I trained and evaluated the custom model on Kaggle’s Spaceship Titanic dataset to validate its performance against a real-world classification problem. This project deepened my understanding of gradient boosting mechanics, optimization strategies, and the inner workings of scalable ML systems.

## Setup

Set up Python

python -m venv venv
source ./venv/bin/activate
pip install -r requirements



Download mock [Spaceship Titanic Data](https://www.kaggle.com/competitions/spaceship-titanic/data), and move train.csv and test.csv files to "Trees/data/raw/" directory:

run dataset.py in terminal to process data
then run xgboost.py to build xgboost model


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         TreesFromScratch and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── TreesFromScratch   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes TreesFromScratch a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

