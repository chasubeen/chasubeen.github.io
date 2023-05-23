---
layout: single
title:  "[ECC DS 10주차] 회귀 3_Pycaret Introduction"
categories: ML
tags: [ECC, DS, house price] 
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# **0. Pycaret**


## **0-1. AutoML**

- ```AutoML(Automated Machine Learning)```

  - 현재의 머신러닝 모델링은 Machine Learning Process 동안 많은 시간과 노력이 요구됨

  - AutoML은 기계 학습 파이프라인에서 수작업과 반복되는 작업을 자동화하는 프로세스

  - 머신러닝을 자동화하는 AI 기술



cf) Machine Learning Process: 문제 정의 과정, 데이터 수집, 전처리, 모델 학습 및 평가, 서비스 적용  

cf) 파이프라인: 한 데이터 처리 단계의 출력이 다음 단계의 입력으로 이어지는 형태로 연결된 구조


## **0-2. Pycaret**

- Low-code machine learning

- AutoML을 가능하게 해주는 파이썬 라이브러리

- scikit-learn 패키지 기반

- 분류, 회귀, 군집화 등 다양한 모델 지원



cf) low-code: 어플리케이션과 시스템을 빌딩할 때 거의 코딩이 필요 없는 방식의 소프트웨어



---

```Iteration마다```   

- setup

- compare models

- create and store models in variable

- blend models

- stack models


# **1. 데이터 준비**

- 데이터는 이전 예제에서 다룬 house price 데이터를 활용



```python
!pip install pycaret
```

<pre>
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: pycaret in /usr/local/lib/python3.10/dist-packages (3.0.2)
Requirement already satisfied: ipython>=5.5.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (7.34.0)
Requirement already satisfied: ipywidgets>=7.6.5 in /usr/local/lib/python3.10/dist-packages (from pycaret) (7.7.1)
Requirement already satisfied: tqdm>=4.62.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (4.64.1)
Requirement already satisfied: numpy<1.24,>=1.21 in /usr/local/lib/python3.10/dist-packages (from pycaret) (1.22.4)
Requirement already satisfied: pandas<2.0.0,>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (1.5.3)
Requirement already satisfied: jinja2>=1.2 in /usr/local/lib/python3.10/dist-packages (from pycaret) (3.1.2)
Requirement already satisfied: scipy<2.0.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (1.9.3)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (1.2.0)
Requirement already satisfied: scikit-learn>=1.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (1.2.2)
Requirement already satisfied: pyod>=1.0.8 in /usr/local/lib/python3.10/dist-packages (from pycaret) (1.0.9)
Requirement already satisfied: imbalanced-learn>=0.8.1 in /usr/local/lib/python3.10/dist-packages (from pycaret) (0.10.1)
Requirement already satisfied: category-encoders>=2.4.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (2.6.1)
Requirement already satisfied: lightgbm>=3.0.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (3.3.5)
Requirement already satisfied: numba>=0.55.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (0.56.4)
Requirement already satisfied: requests>=2.27.1 in /usr/local/lib/python3.10/dist-packages (from pycaret) (2.27.1)
Requirement already satisfied: psutil>=5.9.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (5.9.5)
Requirement already satisfied: markupsafe>=2.0.1 in /usr/local/lib/python3.10/dist-packages (from pycaret) (2.1.2)
Requirement already satisfied: importlib-metadata>=4.12.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (6.6.0)
Requirement already satisfied: nbformat>=4.2.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (5.8.0)
Requirement already satisfied: cloudpickle in /usr/local/lib/python3.10/dist-packages (from pycaret) (2.2.1)
Requirement already satisfied: deprecation>=2.1.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (2.1.0)
Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from pycaret) (3.2.0)
Requirement already satisfied: matplotlib>=3.3.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (3.6.3)
Requirement already satisfied: scikit-plot>=0.3.7 in /usr/local/lib/python3.10/dist-packages (from pycaret) (0.3.7)
Requirement already satisfied: yellowbrick>=1.4 in /usr/local/lib/python3.10/dist-packages (from pycaret) (1.5)
Requirement already satisfied: plotly>=5.0.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (5.13.1)
Requirement already satisfied: kaleido>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from pycaret) (0.2.1)
Requirement already satisfied: schemdraw==0.15 in /usr/local/lib/python3.10/dist-packages (from pycaret) (0.15)
Requirement already satisfied: plotly-resampler>=0.8.3.1 in /usr/local/lib/python3.10/dist-packages (from pycaret) (0.8.3.2)
Requirement already satisfied: statsmodels>=0.12.1 in /usr/local/lib/python3.10/dist-packages (from pycaret) (0.13.5)
Requirement already satisfied: sktime!=0.17.1,<0.17.2,>=0.16.1 in /usr/local/lib/python3.10/dist-packages (from pycaret) (0.17.0)
Requirement already satisfied: tbats>=1.1.3 in /usr/local/lib/python3.10/dist-packages (from pycaret) (1.1.3)
Requirement already satisfied: pmdarima!=1.8.1,<3.0.0,>=1.8.0 in /usr/local/lib/python3.10/dist-packages (from pycaret) (2.0.3)
Requirement already satisfied: wurlitzer in /usr/local/lib/python3.10/dist-packages (from pycaret) (3.0.3)
Requirement already satisfied: patsy>=0.5.1 in /usr/local/lib/python3.10/dist-packages (from category-encoders>=2.4.0->pycaret) (0.5.3)
Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from deprecation>=2.1.0->pycaret) (23.1)
Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from imbalanced-learn>=0.8.1->pycaret) (3.1.0)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata>=4.12.0->pycaret) (3.15.0)
Requirement already satisfied: setuptools>=18.5 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret) (67.7.2)
Requirement already satisfied: jedi>=0.16 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret) (0.18.2)
Requirement already satisfied: decorator in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret) (4.4.2)
Requirement already satisfied: pickleshare in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret) (0.7.5)
Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret) (5.7.1)
Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret) (3.0.38)
Requirement already satisfied: pygments in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret) (2.14.0)
Requirement already satisfied: backcall in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret) (0.2.0)
Requirement already satisfied: matplotlib-inline in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret) (0.1.6)
Requirement already satisfied: pexpect>4.3 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret) (4.8.0)
Requirement already satisfied: ipykernel>=4.5.1 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.6.5->pycaret) (5.5.6)
Requirement already satisfied: ipython-genutils~=0.2.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.6.5->pycaret) (0.2.0)
Requirement already satisfied: widgetsnbextension~=3.6.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.6.5->pycaret) (3.6.4)
Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.6.5->pycaret) (3.0.7)
Requirement already satisfied: wheel in /usr/local/lib/python3.10/dist-packages (from lightgbm>=3.0.0->pycaret) (0.40.0)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->pycaret) (1.0.7)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->pycaret) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->pycaret) (4.39.3)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->pycaret) (1.4.4)
Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->pycaret) (8.4.0)
Requirement already satisfied: pyparsing>=2.2.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->pycaret) (3.0.9)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->pycaret) (2.8.2)
Requirement already satisfied: fastjsonschema in /usr/local/lib/python3.10/dist-packages (from nbformat>=4.2.0->pycaret) (2.16.3)
Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.10/dist-packages (from nbformat>=4.2.0->pycaret) (4.3.3)
Requirement already satisfied: jupyter-core in /usr/local/lib/python3.10/dist-packages (from nbformat>=4.2.0->pycaret) (5.3.0)
Requirement already satisfied: llvmlite<0.40,>=0.39.0dev0 in /usr/local/lib/python3.10/dist-packages (from numba>=0.55.0->pycaret) (0.39.1)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<2.0.0,>=1.3.0->pycaret) (2022.7.1)
Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from plotly>=5.0.0->pycaret) (8.2.2)
Requirement already satisfied: dash<3.0.0,>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from plotly-resampler>=0.8.3.1->pycaret) (2.9.3)
Requirement already satisfied: jupyter-dash>=0.4.2 in /usr/local/lib/python3.10/dist-packages (from plotly-resampler>=0.8.3.1->pycaret) (0.4.2)
Requirement already satisfied: orjson<4.0.0,>=3.8.0 in /usr/local/lib/python3.10/dist-packages (from plotly-resampler>=0.8.3.1->pycaret) (3.8.12)
Requirement already satisfied: trace-updater>=0.0.8 in /usr/local/lib/python3.10/dist-packages (from plotly-resampler>=0.8.3.1->pycaret) (0.0.9.1)
Requirement already satisfied: Cython!=0.29.18,!=0.29.31,>=0.29 in /usr/local/lib/python3.10/dist-packages (from pmdarima!=1.8.1,<3.0.0,>=1.8.0->pycaret) (0.29.34)
Requirement already satisfied: urllib3 in /usr/local/lib/python3.10/dist-packages (from pmdarima!=1.8.1,<3.0.0,>=1.8.0->pycaret) (1.26.15)
Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from pyod>=1.0.8->pycaret) (1.16.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->pycaret) (2022.12.7)
Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->pycaret) (2.0.12)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->pycaret) (3.4)
Requirement already satisfied: deprecated>=1.2.13 in /usr/local/lib/python3.10/dist-packages (from sktime!=0.17.1,<0.17.2,>=0.16.1->pycaret) (1.2.13)
Requirement already satisfied: Flask>=1.0.4 in /usr/local/lib/python3.10/dist-packages (from dash<3.0.0,>=2.2.0->plotly-resampler>=0.8.3.1->pycaret) (2.2.3)
Requirement already satisfied: dash-html-components==2.0.0 in /usr/local/lib/python3.10/dist-packages (from dash<3.0.0,>=2.2.0->plotly-resampler>=0.8.3.1->pycaret) (2.0.0)
Requirement already satisfied: dash-core-components==2.0.0 in /usr/local/lib/python3.10/dist-packages (from dash<3.0.0,>=2.2.0->plotly-resampler>=0.8.3.1->pycaret) (2.0.0)
Requirement already satisfied: dash-table==5.0.0 in /usr/local/lib/python3.10/dist-packages (from dash<3.0.0,>=2.2.0->plotly-resampler>=0.8.3.1->pycaret) (5.0.0)
Requirement already satisfied: wrapt<2,>=1.10 in /usr/local/lib/python3.10/dist-packages (from deprecated>=1.2.13->sktime!=0.17.1,<0.17.2,>=0.16.1->pycaret) (1.14.1)
Requirement already satisfied: jupyter-client in /usr/local/lib/python3.10/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.6.5->pycaret) (6.1.12)
Requirement already satisfied: tornado>=4.2 in /usr/local/lib/python3.10/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.6.5->pycaret) (6.3.1)
Requirement already satisfied: parso<0.9.0,>=0.8.0 in /usr/local/lib/python3.10/dist-packages (from jedi>=0.16->ipython>=5.5.0->pycaret) (0.8.3)
Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=4.2.0->pycaret) (23.1.0)
Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=4.2.0->pycaret) (0.19.3)
Requirement already satisfied: retrying in /usr/local/lib/python3.10/dist-packages (from jupyter-dash>=0.4.2->plotly-resampler>=0.8.3.1->pycaret) (1.3.4)
Requirement already satisfied: ansi2html in /usr/local/lib/python3.10/dist-packages (from jupyter-dash>=0.4.2->plotly-resampler>=0.8.3.1->pycaret) (1.8.0)
Requirement already satisfied: nest-asyncio in /usr/local/lib/python3.10/dist-packages (from jupyter-dash>=0.4.2->plotly-resampler>=0.8.3.1->pycaret) (1.5.6)
Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.10/dist-packages (from pexpect>4.3->ipython>=5.5.0->pycaret) (0.7.0)
Requirement already satisfied: wcwidth in /usr/local/lib/python3.10/dist-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=5.5.0->pycaret) (0.2.6)
Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.10/dist-packages (from widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (6.4.8)
Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.10/dist-packages (from jupyter-core->nbformat>=4.2.0->pycaret) (3.3.0)
Requirement already satisfied: Werkzeug>=2.2.2 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.0.4->dash<3.0.0,>=2.2.0->plotly-resampler>=0.8.3.1->pycaret) (2.3.0)
Requirement already satisfied: itsdangerous>=2.0 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.0.4->dash<3.0.0,>=2.2.0->plotly-resampler>=0.8.3.1->pycaret) (2.1.2)
Requirement already satisfied: click>=8.0 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.0.4->dash<3.0.0,>=2.2.0->plotly-resampler>=0.8.3.1->pycaret) (8.1.3)
Requirement already satisfied: pyzmq>=17 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (23.2.1)
Requirement already satisfied: argon2-cffi in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (21.3.0)
Requirement already satisfied: nbconvert in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (6.5.4)
Requirement already satisfied: Send2Trash>=1.8.0 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (1.8.0)
Requirement already satisfied: terminado>=0.8.3 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (0.17.1)
Requirement already satisfied: prometheus-client in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (0.16.0)
Requirement already satisfied: argon2-cffi-bindings in /usr/local/lib/python3.10/dist-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (21.2.0)
Requirement already satisfied: lxml in /usr/local/lib/python3.10/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (4.9.2)
Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (4.11.2)
Requirement already satisfied: bleach in /usr/local/lib/python3.10/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (6.0.0)
Requirement already satisfied: defusedxml in /usr/local/lib/python3.10/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (0.7.1)
Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.10/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (0.4)
Requirement already satisfied: jupyterlab-pygments in /usr/local/lib/python3.10/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (0.2.2)
Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (0.8.4)
Requirement already satisfied: nbclient>=0.5.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (0.7.4)
Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (1.5.0)
Requirement already satisfied: tinycss2 in /usr/local/lib/python3.10/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (1.2.1)
Requirement already satisfied: cffi>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (1.15.1)
Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (2.4.1)
Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (0.5.1)
Requirement already satisfied: pycparser in /usr/local/lib/python3.10/dist-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret) (2.21)
</pre>

```python
### Import libraries

import numpy as np 
import pandas as pd 

from pycaret.regression import * # pycaret에서 회귀 관련 모듈 import
```


```python
### Import dataset

train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/10주차/data/house_price_train.csv')
test  = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/10주차/data/house_price_test.csv')
sample= pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/10주차/data/house_sample_submission.csv')
```


```python
train.head()
```

<pre>
   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
0   1          60       RL         65.0     8450   Pave   NaN      Reg   
1   2          20       RL         80.0     9600   Pave   NaN      Reg   
2   3          60       RL         68.0    11250   Pave   NaN      IR1   
3   4          70       RL         60.0     9550   Pave   NaN      IR1   
4   5          60       RL         84.0    14260   Pave   NaN      IR1   

  LandContour Utilities  ... PoolArea PoolQC Fence MiscFeature MiscVal MoSold  \
0         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   
1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      5   
2         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      9   
3         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   
4         Lvl    AllPub  ...        0    NaN   NaN         NaN       0     12   

  YrSold  SaleType  SaleCondition  SalePrice  
0   2008        WD         Normal     208500  
1   2007        WD         Normal     181500  
2   2008        WD         Normal     223500  
3   2006        WD        Abnorml     140000  
4   2008        WD         Normal     250000  

[5 rows x 81 columns]
</pre>

```python
### 데이터 정보 확인

train.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 81 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1460 non-null   int64  
 1   MSSubClass     1460 non-null   int64  
 2   MSZoning       1460 non-null   object 
 3   LotFrontage    1201 non-null   float64
 4   LotArea        1460 non-null   int64  
 5   Street         1460 non-null   object 
 6   Alley          91 non-null     object 
 7   LotShape       1460 non-null   object 
 8   LandContour    1460 non-null   object 
 9   Utilities      1460 non-null   object 
 10  LotConfig      1460 non-null   object 
 11  LandSlope      1460 non-null   object 
 12  Neighborhood   1460 non-null   object 
 13  Condition1     1460 non-null   object 
 14  Condition2     1460 non-null   object 
 15  BldgType       1460 non-null   object 
 16  HouseStyle     1460 non-null   object 
 17  OverallQual    1460 non-null   int64  
 18  OverallCond    1460 non-null   int64  
 19  YearBuilt      1460 non-null   int64  
 20  YearRemodAdd   1460 non-null   int64  
 21  RoofStyle      1460 non-null   object 
 22  RoofMatl       1460 non-null   object 
 23  Exterior1st    1460 non-null   object 
 24  Exterior2nd    1460 non-null   object 
 25  MasVnrType     1452 non-null   object 
 26  MasVnrArea     1452 non-null   float64
 27  ExterQual      1460 non-null   object 
 28  ExterCond      1460 non-null   object 
 29  Foundation     1460 non-null   object 
 30  BsmtQual       1423 non-null   object 
 31  BsmtCond       1423 non-null   object 
 32  BsmtExposure   1422 non-null   object 
 33  BsmtFinType1   1423 non-null   object 
 34  BsmtFinSF1     1460 non-null   int64  
 35  BsmtFinType2   1422 non-null   object 
 36  BsmtFinSF2     1460 non-null   int64  
 37  BsmtUnfSF      1460 non-null   int64  
 38  TotalBsmtSF    1460 non-null   int64  
 39  Heating        1460 non-null   object 
 40  HeatingQC      1460 non-null   object 
 41  CentralAir     1460 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1460 non-null   int64  
 44  2ndFlrSF       1460 non-null   int64  
 45  LowQualFinSF   1460 non-null   int64  
 46  GrLivArea      1460 non-null   int64  
 47  BsmtFullBath   1460 non-null   int64  
 48  BsmtHalfBath   1460 non-null   int64  
 49  FullBath       1460 non-null   int64  
 50  HalfBath       1460 non-null   int64  
 51  BedroomAbvGr   1460 non-null   int64  
 52  KitchenAbvGr   1460 non-null   int64  
 53  KitchenQual    1460 non-null   object 
 54  TotRmsAbvGrd   1460 non-null   int64  
 55  Functional     1460 non-null   object 
 56  Fireplaces     1460 non-null   int64  
 57  FireplaceQu    770 non-null    object 
 58  GarageType     1379 non-null   object 
 59  GarageYrBlt    1379 non-null   float64
 60  GarageFinish   1379 non-null   object 
 61  GarageCars     1460 non-null   int64  
 62  GarageArea     1460 non-null   int64  
 63  GarageQual     1379 non-null   object 
 64  GarageCond     1379 non-null   object 
 65  PavedDrive     1460 non-null   object 
 66  WoodDeckSF     1460 non-null   int64  
 67  OpenPorchSF    1460 non-null   int64  
 68  EnclosedPorch  1460 non-null   int64  
 69  3SsnPorch      1460 non-null   int64  
 70  ScreenPorch    1460 non-null   int64  
 71  PoolArea       1460 non-null   int64  
 72  PoolQC         7 non-null      object 
 73  Fence          281 non-null    object 
 74  MiscFeature    54 non-null     object 
 75  MiscVal        1460 non-null   int64  
 76  MoSold         1460 non-null   int64  
 77  YrSold         1460 non-null   int64  
 78  SaleType       1460 non-null   object 
 79  SaleCondition  1460 non-null   object 
 80  SalePrice      1460 non-null   int64  
dtypes: float64(3), int64(35), object(43)
memory usage: 924.0+ KB
</pre>
- 일부 결측치가 존재함을 확인할 수 있다.


# **2. 데이터 구성(데이터 전처리)**

- ```setup()``` 함수를 통해 지정



```python
reg = setup(data = train, 
             target = 'SalePrice',
             numeric_imputation = 'mean',
             categorical_features = ['MSZoning','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType',
                                     'Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood',   
                                     'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',    
                                     'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',   
                                     'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',   
                                     'Electrical','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',
                                     'SaleCondition']  , 
             ignore_features = ['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'],
             normalize = True)
```

<pre>
<pandas.io.formats.style.Styler at 0x7f4390d17f70>
</pre>
# **3. 모델 성능 비교**

- ```models()```를 통해 모델을 확인할 수 있음

  - setup을 한 후에만 가능

- ```compare_models()```

  - ```models()```에서 제공하는 모델들이나 scikit-learn에서 제공하는 모델을 별도로 선언한 이후에 입력한 모델들의 성능(MAE, MSE, RMSE, R^2, train time)등을 데이터프레임 형태로 제공



```python
compare_models()
```


  <div id="df-68a22232-955b-4f01-a63e-727779391219">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Initiated</th>
      <td>. . . . . . . . . . . . . . . . . .</td>
      <td>05:49:16</td>
    </tr>
    <tr>
      <th>Status</th>
      <td>. . . . . . . . . . . . . . . . . .</td>
      <td>Fitting 10 Folds</td>
    </tr>
    <tr>
      <th>Estimator</th>
      <td>. . . . . . . . . . . . . . . . . .</td>
      <td>Random Forest Regressor</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-68a22232-955b-4f01-a63e-727779391219')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-68a22232-955b-4f01-a63e-727779391219 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-68a22232-955b-4f01-a63e-727779391219');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  


<style type="text/css">
#T_b873d th {
  text-align: left;
}
#T_b873d_row0_col0, #T_b873d_row0_col1, #T_b873d_row0_col2, #T_b873d_row0_col3, #T_b873d_row0_col4, #T_b873d_row0_col5, #T_b873d_row0_col6, #T_b873d_row0_col7, #T_b873d_row1_col0, #T_b873d_row1_col1, #T_b873d_row1_col2, #T_b873d_row1_col3, #T_b873d_row1_col4, #T_b873d_row1_col5, #T_b873d_row1_col6, #T_b873d_row1_col7, #T_b873d_row2_col0, #T_b873d_row2_col1, #T_b873d_row2_col2, #T_b873d_row2_col3, #T_b873d_row2_col4, #T_b873d_row2_col5, #T_b873d_row2_col6, #T_b873d_row2_col7, #T_b873d_row3_col0, #T_b873d_row3_col1, #T_b873d_row3_col2, #T_b873d_row3_col3, #T_b873d_row3_col4, #T_b873d_row3_col5, #T_b873d_row3_col6, #T_b873d_row3_col7, #T_b873d_row4_col0, #T_b873d_row4_col1, #T_b873d_row4_col2, #T_b873d_row4_col3, #T_b873d_row4_col4, #T_b873d_row4_col5, #T_b873d_row4_col6, #T_b873d_row4_col7, #T_b873d_row5_col0, #T_b873d_row5_col1, #T_b873d_row5_col2, #T_b873d_row5_col3, #T_b873d_row5_col4, #T_b873d_row5_col5, #T_b873d_row5_col6, #T_b873d_row5_col7, #T_b873d_row6_col0, #T_b873d_row6_col1, #T_b873d_row6_col2, #T_b873d_row6_col3, #T_b873d_row6_col4, #T_b873d_row6_col5, #T_b873d_row6_col6, #T_b873d_row6_col7, #T_b873d_row7_col0, #T_b873d_row7_col1, #T_b873d_row7_col2, #T_b873d_row7_col3, #T_b873d_row7_col4, #T_b873d_row7_col5, #T_b873d_row7_col6, #T_b873d_row7_col7, #T_b873d_row8_col0, #T_b873d_row8_col1, #T_b873d_row8_col2, #T_b873d_row8_col3, #T_b873d_row8_col4, #T_b873d_row8_col5, #T_b873d_row8_col6, #T_b873d_row8_col7, #T_b873d_row9_col0, #T_b873d_row9_col1, #T_b873d_row9_col2, #T_b873d_row9_col3, #T_b873d_row9_col4, #T_b873d_row9_col5, #T_b873d_row9_col6, #T_b873d_row9_col7, #T_b873d_row10_col0, #T_b873d_row10_col1, #T_b873d_row10_col2, #T_b873d_row10_col3, #T_b873d_row10_col4, #T_b873d_row10_col5, #T_b873d_row10_col6, #T_b873d_row10_col7, #T_b873d_row11_col0, #T_b873d_row11_col1, #T_b873d_row11_col2, #T_b873d_row11_col3, #T_b873d_row11_col4, #T_b873d_row11_col5, #T_b873d_row11_col6, #T_b873d_row11_col7 {
  text-align: left;
}
</style>
<table id="T_b873d" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b873d_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_b873d_level0_col1" class="col_heading level0 col1" >MAE</th>
      <th id="T_b873d_level0_col2" class="col_heading level0 col2" >MSE</th>
      <th id="T_b873d_level0_col3" class="col_heading level0 col3" >RMSE</th>
      <th id="T_b873d_level0_col4" class="col_heading level0 col4" >R2</th>
      <th id="T_b873d_level0_col5" class="col_heading level0 col5" >RMSLE</th>
      <th id="T_b873d_level0_col6" class="col_heading level0 col6" >MAPE</th>
      <th id="T_b873d_level0_col7" class="col_heading level0 col7" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b873d_level0_row0" class="row_heading level0 row0" >en</th>
      <td id="T_b873d_row0_col0" class="data row0 col0" >Elastic Net</td>
      <td id="T_b873d_row0_col1" class="data row0 col1" >18364.3952</td>
      <td id="T_b873d_row0_col2" class="data row0 col2" >1344589957.1260</td>
      <td id="T_b873d_row0_col3" class="data row0 col3" >35059.9474</td>
      <td id="T_b873d_row0_col4" class="data row0 col4" >0.7834</td>
      <td id="T_b873d_row0_col5" class="data row0 col5" >0.1449</td>
      <td id="T_b873d_row0_col6" class="data row0 col6" >0.1024</td>
      <td id="T_b873d_row0_col7" class="data row0 col7" >0.8530</td>
    </tr>
    <tr>
      <th id="T_b873d_level0_row1" class="row_heading level0 row1" >par</th>
      <td id="T_b873d_row1_col0" class="data row1 col0" >Passive Aggressive Regressor</td>
      <td id="T_b873d_row1_col1" class="data row1 col1" >18134.1562</td>
      <td id="T_b873d_row1_col2" class="data row1 col2" >1374167341.7881</td>
      <td id="T_b873d_row1_col3" class="data row1 col3" >35373.6928</td>
      <td id="T_b873d_row1_col4" class="data row1 col4" >0.7773</td>
      <td id="T_b873d_row1_col5" class="data row1 col5" >0.1699</td>
      <td id="T_b873d_row1_col6" class="data row1 col6" >0.1033</td>
      <td id="T_b873d_row1_col7" class="data row1 col7" >2.4800</td>
    </tr>
    <tr>
      <th id="T_b873d_level0_row2" class="row_heading level0 row2" >br</th>
      <td id="T_b873d_row2_col0" class="data row2 col0" >Bayesian Ridge</td>
      <td id="T_b873d_row2_col1" class="data row2 col1" >19102.7516</td>
      <td id="T_b873d_row2_col2" class="data row2 col2" >1509230751.8566</td>
      <td id="T_b873d_row2_col3" class="data row2 col3" >36789.4023</td>
      <td id="T_b873d_row2_col4" class="data row2 col4" >0.7551</td>
      <td id="T_b873d_row2_col5" class="data row2 col5" >0.1706</td>
      <td id="T_b873d_row2_col6" class="data row2 col6" >0.1097</td>
      <td id="T_b873d_row2_col7" class="data row2 col7" >0.9030</td>
    </tr>
    <tr>
      <th id="T_b873d_level0_row3" class="row_heading level0 row3" >omp</th>
      <td id="T_b873d_row3_col0" class="data row3 col0" >Orthogonal Matching Pursuit</td>
      <td id="T_b873d_row3_col1" class="data row3 col1" >19565.0295</td>
      <td id="T_b873d_row3_col2" class="data row3 col2" >1604341893.6740</td>
      <td id="T_b873d_row3_col3" class="data row3 col3" >37530.4782</td>
      <td id="T_b873d_row3_col4" class="data row3 col4" >0.7370</td>
      <td id="T_b873d_row3_col5" class="data row3 col5" >0.1815</td>
      <td id="T_b873d_row3_col6" class="data row3 col6" >0.1121</td>
      <td id="T_b873d_row3_col7" class="data row3 col7" >1.2170</td>
    </tr>
    <tr>
      <th id="T_b873d_level0_row4" class="row_heading level0 row4" >llar</th>
      <td id="T_b873d_row4_col0" class="data row4 col0" >Lasso Least Angle Regression</td>
      <td id="T_b873d_row4_col1" class="data row4 col1" >19787.4115</td>
      <td id="T_b873d_row4_col2" class="data row4 col2" >1620639261.1369</td>
      <td id="T_b873d_row4_col3" class="data row4 col3" >38028.6722</td>
      <td id="T_b873d_row4_col4" class="data row4 col4" >0.7366</td>
      <td id="T_b873d_row4_col5" class="data row4 col5" >0.1867</td>
      <td id="T_b873d_row4_col6" class="data row4 col6" >0.1157</td>
      <td id="T_b873d_row4_col7" class="data row4 col7" >0.9700</td>
    </tr>
    <tr>
      <th id="T_b873d_level0_row5" class="row_heading level0 row5" >dt</th>
      <td id="T_b873d_row5_col0" class="data row5 col0" >Decision Tree Regressor</td>
      <td id="T_b873d_row5_col1" class="data row5 col1" >26873.8594</td>
      <td id="T_b873d_row5_col2" class="data row5 col2" >1656701282.2708</td>
      <td id="T_b873d_row5_col3" class="data row5 col3" >40601.1973</td>
      <td id="T_b873d_row5_col4" class="data row5 col4" >0.7338</td>
      <td id="T_b873d_row5_col5" class="data row5 col5" >0.2134</td>
      <td id="T_b873d_row5_col6" class="data row5 col6" >0.1554</td>
      <td id="T_b873d_row5_col7" class="data row5 col7" >0.9180</td>
    </tr>
    <tr>
      <th id="T_b873d_level0_row6" class="row_heading level0 row6" >ridge</th>
      <td id="T_b873d_row6_col0" class="data row6 col0" >Ridge Regression</td>
      <td id="T_b873d_row6_col1" class="data row6 col1" >20390.6023</td>
      <td id="T_b873d_row6_col2" class="data row6 col2" >1675546296.8694</td>
      <td id="T_b873d_row6_col3" class="data row6 col3" >38612.7187</td>
      <td id="T_b873d_row6_col4" class="data row6 col4" >0.7280</td>
      <td id="T_b873d_row6_col5" class="data row6 col5" >0.2029</td>
      <td id="T_b873d_row6_col6" class="data row6 col6" >0.1197</td>
      <td id="T_b873d_row6_col7" class="data row6 col7" >1.1390</td>
    </tr>
    <tr>
      <th id="T_b873d_level0_row7" class="row_heading level0 row7" >lasso</th>
      <td id="T_b873d_row7_col0" class="data row7 col0" >Lasso Regression</td>
      <td id="T_b873d_row7_col1" class="data row7 col1" >20348.5520</td>
      <td id="T_b873d_row7_col2" class="data row7 col2" >1679664973.5803</td>
      <td id="T_b873d_row7_col3" class="data row7 col3" >38597.0093</td>
      <td id="T_b873d_row7_col4" class="data row7 col4" >0.7271</td>
      <td id="T_b873d_row7_col5" class="data row7 col5" >0.2045</td>
      <td id="T_b873d_row7_col6" class="data row7 col6" >0.1196</td>
      <td id="T_b873d_row7_col7" class="data row7 col7" >3.0070</td>
    </tr>
    <tr>
      <th id="T_b873d_level0_row8" class="row_heading level0 row8" >huber</th>
      <td id="T_b873d_row8_col0" class="data row8 col0" >Huber Regressor</td>
      <td id="T_b873d_row8_col1" class="data row8 col1" >19236.5019</td>
      <td id="T_b873d_row8_col2" class="data row8 col2" >1766235862.2651</td>
      <td id="T_b873d_row8_col3" class="data row8 col3" >38848.9916</td>
      <td id="T_b873d_row8_col4" class="data row8 col4" >0.7128</td>
      <td id="T_b873d_row8_col5" class="data row8 col5" >0.2052</td>
      <td id="T_b873d_row8_col6" class="data row8 col6" >0.1123</td>
      <td id="T_b873d_row8_col7" class="data row8 col7" >1.2150</td>
    </tr>
    <tr>
      <th id="T_b873d_level0_row9" class="row_heading level0 row9" >knn</th>
      <td id="T_b873d_row9_col0" class="data row9 col0" >K Neighbors Regressor</td>
      <td id="T_b873d_row9_col1" class="data row9 col1" >27602.8084</td>
      <td id="T_b873d_row9_col2" class="data row9 col2" >2038986154.7901</td>
      <td id="T_b873d_row9_col3" class="data row9 col3" >44370.3270</td>
      <td id="T_b873d_row9_col4" class="data row9 col4" >0.6870</td>
      <td id="T_b873d_row9_col5" class="data row9 col5" >0.2092</td>
      <td id="T_b873d_row9_col6" class="data row9 col6" >0.1559</td>
      <td id="T_b873d_row9_col7" class="data row9 col7" >1.1220</td>
    </tr>
    <tr>
      <th id="T_b873d_level0_row10" class="row_heading level0 row10" >lr</th>
      <td id="T_b873d_row10_col0" class="data row10 col0" >Linear Regression</td>
      <td id="T_b873d_row10_col1" class="data row10 col1" >489462244852579.7500</td>
      <td id="T_b873d_row10_col2" class="data row10 col2" >25087786038259630322198507421696.0000</td>
      <td id="T_b873d_row10_col3" class="data row10 col3" >3296273264514296.0000</td>
      <td id="T_b873d_row10_col4" class="data row10 col4" >-3909394621073952604160.0000</td>
      <td id="T_b873d_row10_col5" class="data row10 col5" >3.5936</td>
      <td id="T_b873d_row10_col6" class="data row10 col6" >4067501731.2203</td>
      <td id="T_b873d_row10_col7" class="data row10 col7" >4.4590</td>
    </tr>
    <tr>
      <th id="T_b873d_level0_row11" class="row_heading level0 row11" >lar</th>
      <td id="T_b873d_row11_col0" class="data row11 col0" >Least Angle Regression</td>
      <td id="T_b873d_row11_col1" class="data row11 col1" >42521860416708132924698326635424220409273253888.0000</td>
      <td id="T_b873d_row11_col2" class="data row11 col2" >1844180973733323350448562275845971804997761547450196885667534503848224189678324096601303512252416.0000</td>
      <td id="T_b873d_row11_col3" class="data row11 col3" >429448190934811211097724065035493310127918809088.0000</td>
      <td id="T_b873d_row11_col4" class="data row11 col4" >-356872355765253379694331818216569495633078728307084395869623944858328491348458530144256.0000</td>
      <td id="T_b873d_row11_col5" class="data row11 col5" >49.8038</td>
      <td id="T_b873d_row11_col6" class="data row11 col6" >310381041719748206784084963210196848476160.0000</td>
      <td id="T_b873d_row11_col7" class="data row11 col7" >1.4570</td>
    </tr>
  </tbody>
</table>


<pre>
Processing:   0%|          | 0/81 [00:00<?, ?it/s]
</pre>
<pre>
<pandas.io.formats.style.Styler at 0x7f43a28770a0>
</pre>
<pre>
<IPython.core.display.HTML object>
</pre>
<pre>
<IPython.core.display.HTML object>
</pre>
<pre>
GradientBoostingRegressor(random_state=679)
</pre>
- LGBM의 성능이 가장 좋다.


# **4. 단일 모델 생성**

- ```create_model()```

  - 여러 모델이 아닌 하나의 모델에 대해서 ```setup()```의 설정대로 학습을 진행하고 학습 결과 확인

  - 세부적으로 각 fold에 대한 성능을 제시



```python
lgb = create_model('lightgbm')
```


  <div id="df-de60f496-593d-455e-a595-24f90184c52b">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Initiated</th>
      <td>. . . . . . . . . . . . . . . . . .</td>
      <td>05:55:18</td>
    </tr>
    <tr>
      <th>Status</th>
      <td>. . . . . . . . . . . . . . . . . .</td>
      <td>Loading Dependencies</td>
    </tr>
    <tr>
      <th>Estimator</th>
      <td>. . . . . . . . . . . . . . . . . .</td>
      <td>Compiling Library</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-de60f496-593d-455e-a595-24f90184c52b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-de60f496-593d-455e-a595-24f90184c52b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-de60f496-593d-455e-a595-24f90184c52b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  




<pre>
Processing:   0%|          | 0/4 [00:00<?, ?it/s]
</pre>
<pre>
<IPython.core.display.HTML object>
</pre>
<pre>
<IPython.core.display.HTML object>
</pre>
<pre>
<pandas.io.formats.style.Styler at 0x7f4390d16e60>
</pre>
- ```verbose``` 옵션

  - 함수 수행 시 발생하는 상세한 정보들을 표준 출력으로 자세히 내보낼 것인가를 결정

  - True의 경우 자세히 출력함


# **5. 모델 튜닝하기**

- ```tune_model()```

  - 입력한 모델에 대해서 hyper parameter tuning을 수행



```python
tuned_lgb = tune_model(lgb)
```

<pre>
<IPython.core.display.HTML object>
</pre>
<pre>
<pandas.io.formats.style.Styler at 0x7f43a27db880>
</pre>
<pre>
Processing:   0%|          | 0/7 [00:00<?, ?it/s]
</pre>
<pre>
Fitting 10 folds for each of 10 candidates, totalling 100 fits
</pre>
<pre>
<IPython.core.display.HTML object>
</pre>
# **6. 모델링 결과 해석**

- ```interpret_model()```

  - 모델을 해석

  - 훈련된 모델 객체와 플롯 유형을 문자열로 받음

  - 해석은 SHAP를 기반으로 구현되며 트리 기반 모델에서만 사용 가능함



cf) pycaret.classification, pycaret.regression 모듈에서만 사용 가능  

cf) SHAP: SHapley Addictive exPlanations



```python
!pip install shap
```


```python
interpret_model(tuned_lgb)
```

<pre>
<Figure size 800x950 with 2 Axes>
</pre>

```python
predictions = predict_model(tuned_lgb, data = test)
sample['SalePrice'] = predictions['Label']
sample.to_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/10주차/data/house_sample_submission.csv',index = False)
sample.head()
```
