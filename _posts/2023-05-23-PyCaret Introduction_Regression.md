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


  <div id="df-1e853581-3d3b-4819-b864-58d30eb0c6ba">
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1e853581-3d3b-4819-b864-58d30eb0c6ba')"
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
          document.querySelector('#df-1e853581-3d3b-4819-b864-58d30eb0c6ba button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1e853581-3d3b-4819-b864-58d30eb0c6ba');
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

<style type="text/css">
#T_f5efe_row12_col1, #T_f5efe_row18_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_f5efe" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f5efe_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_f5efe_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f5efe_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_f5efe_row0_col0" class="data row0 col0" >Session id</td>
      <td id="T_f5efe_row0_col1" class="data row0 col1" >679</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_f5efe_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_f5efe_row1_col1" class="data row1 col1" >SalePrice</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_f5efe_row2_col0" class="data row2 col0" >Target type</td>
      <td id="T_f5efe_row2_col1" class="data row2 col1" >Regression</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_f5efe_row3_col0" class="data row3 col0" >Original data shape</td>
      <td id="T_f5efe_row3_col1" class="data row3 col1" >(1460, 81)</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_f5efe_row4_col0" class="data row4 col0" >Transformed data shape</td>
      <td id="T_f5efe_row4_col1" class="data row4 col1" >(1460, 262)</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_f5efe_row5_col0" class="data row5 col0" >Transformed train set shape</td>
      <td id="T_f5efe_row5_col1" class="data row5 col1" >(1021, 262)</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_f5efe_row6_col0" class="data row6 col0" >Transformed test set shape</td>
      <td id="T_f5efe_row6_col1" class="data row6 col1" >(439, 262)</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_f5efe_row7_col0" class="data row7 col0" >Ignore features</td>
      <td id="T_f5efe_row7_col1" class="data row7 col1" >6</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_f5efe_row8_col0" class="data row8 col0" >Ordinal features</td>
      <td id="T_f5efe_row8_col1" class="data row8 col1" >2</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_f5efe_row9_col0" class="data row9 col0" >Numeric features</td>
      <td id="T_f5efe_row9_col1" class="data row9 col1" >37</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_f5efe_row10_col0" class="data row10 col0" >Categorical features</td>
      <td id="T_f5efe_row10_col1" class="data row10 col1" >37</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_f5efe_row11_col0" class="data row11 col0" >Rows with missing values</td>
      <td id="T_f5efe_row11_col1" class="data row11 col1" >100.0%</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_f5efe_row12_col0" class="data row12 col0" >Preprocess</td>
      <td id="T_f5efe_row12_col1" class="data row12 col1" >True</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_f5efe_row13_col0" class="data row13 col0" >Imputation type</td>
      <td id="T_f5efe_row13_col1" class="data row13 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_f5efe_row14_col0" class="data row14 col0" >Numeric imputation</td>
      <td id="T_f5efe_row14_col1" class="data row14 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_f5efe_row15_col0" class="data row15 col0" >Categorical imputation</td>
      <td id="T_f5efe_row15_col1" class="data row15 col1" >mode</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_f5efe_row16_col0" class="data row16 col0" >Maximum one-hot encoding</td>
      <td id="T_f5efe_row16_col1" class="data row16 col1" >25</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_f5efe_row17_col0" class="data row17 col0" >Encoding method</td>
      <td id="T_f5efe_row17_col1" class="data row17 col1" >None</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_f5efe_row18_col0" class="data row18 col0" >Normalize</td>
      <td id="T_f5efe_row18_col1" class="data row18 col1" >True</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_f5efe_row19_col0" class="data row19 col0" >Normalize method</td>
      <td id="T_f5efe_row19_col1" class="data row19 col1" >zscore</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_f5efe_row20_col0" class="data row20 col0" >Fold Generator</td>
      <td id="T_f5efe_row20_col1" class="data row20 col1" >KFold</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_f5efe_row21_col0" class="data row21 col0" >Fold Number</td>
      <td id="T_f5efe_row21_col1" class="data row21 col1" >10</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_f5efe_row22_col0" class="data row22 col0" >CPU Jobs</td>
      <td id="T_f5efe_row22_col1" class="data row22 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_f5efe_row23_col0" class="data row23 col0" >Use GPU</td>
      <td id="T_f5efe_row23_col1" class="data row23 col1" >False</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_f5efe_row24_col0" class="data row24 col0" >Log Experiment</td>
      <td id="T_f5efe_row24_col1" class="data row24 col1" >False</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_f5efe_row25_col0" class="data row25 col0" >Experiment Name</td>
      <td id="T_f5efe_row25_col1" class="data row25 col1" >reg-default-name</td>
    </tr>
    <tr>
      <th id="T_f5efe_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_f5efe_row26_col0" class="data row26 col0" >USI</td>
      <td id="T_f5efe_row26_col1" class="data row26 col1" >6ade</td>
    </tr>
  </tbody>
</table>


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
<style type="text/css">
#T_f8c93 th {
  text-align: left;
}
#T_f8c93_row0_col0, #T_f8c93_row1_col0, #T_f8c93_row1_col1, #T_f8c93_row1_col2, #T_f8c93_row1_col3, #T_f8c93_row1_col4, #T_f8c93_row1_col5, #T_f8c93_row1_col6, #T_f8c93_row2_col0, #T_f8c93_row2_col1, #T_f8c93_row2_col2, #T_f8c93_row2_col3, #T_f8c93_row2_col4, #T_f8c93_row2_col5, #T_f8c93_row2_col6, #T_f8c93_row3_col0, #T_f8c93_row3_col1, #T_f8c93_row3_col2, #T_f8c93_row3_col3, #T_f8c93_row3_col4, #T_f8c93_row3_col5, #T_f8c93_row3_col6, #T_f8c93_row4_col0, #T_f8c93_row4_col1, #T_f8c93_row4_col2, #T_f8c93_row4_col3, #T_f8c93_row4_col4, #T_f8c93_row4_col5, #T_f8c93_row4_col6, #T_f8c93_row5_col0, #T_f8c93_row5_col1, #T_f8c93_row5_col2, #T_f8c93_row5_col3, #T_f8c93_row5_col4, #T_f8c93_row5_col5, #T_f8c93_row5_col6, #T_f8c93_row6_col0, #T_f8c93_row6_col1, #T_f8c93_row6_col2, #T_f8c93_row6_col3, #T_f8c93_row6_col4, #T_f8c93_row6_col5, #T_f8c93_row6_col6, #T_f8c93_row7_col0, #T_f8c93_row7_col1, #T_f8c93_row7_col2, #T_f8c93_row7_col3, #T_f8c93_row7_col4, #T_f8c93_row7_col5, #T_f8c93_row7_col6, #T_f8c93_row8_col0, #T_f8c93_row8_col1, #T_f8c93_row8_col2, #T_f8c93_row8_col3, #T_f8c93_row8_col4, #T_f8c93_row8_col5, #T_f8c93_row8_col6, #T_f8c93_row9_col0, #T_f8c93_row9_col1, #T_f8c93_row9_col2, #T_f8c93_row9_col3, #T_f8c93_row9_col4, #T_f8c93_row9_col5, #T_f8c93_row9_col6, #T_f8c93_row10_col0, #T_f8c93_row10_col1, #T_f8c93_row10_col2, #T_f8c93_row10_col3, #T_f8c93_row10_col4, #T_f8c93_row10_col5, #T_f8c93_row10_col6, #T_f8c93_row11_col0, #T_f8c93_row11_col1, #T_f8c93_row11_col2, #T_f8c93_row11_col3, #T_f8c93_row11_col4, #T_f8c93_row11_col5, #T_f8c93_row11_col6, #T_f8c93_row12_col0, #T_f8c93_row12_col1, #T_f8c93_row12_col2, #T_f8c93_row12_col3, #T_f8c93_row12_col4, #T_f8c93_row12_col5, #T_f8c93_row12_col6, #T_f8c93_row13_col0, #T_f8c93_row13_col1, #T_f8c93_row13_col2, #T_f8c93_row13_col3, #T_f8c93_row13_col4, #T_f8c93_row13_col5, #T_f8c93_row13_col6, #T_f8c93_row14_col0, #T_f8c93_row14_col1, #T_f8c93_row14_col2, #T_f8c93_row14_col3, #T_f8c93_row14_col4, #T_f8c93_row14_col5, #T_f8c93_row14_col6, #T_f8c93_row15_col0, #T_f8c93_row15_col1, #T_f8c93_row15_col2, #T_f8c93_row15_col3, #T_f8c93_row15_col4, #T_f8c93_row15_col5, #T_f8c93_row15_col6, #T_f8c93_row16_col0, #T_f8c93_row16_col1, #T_f8c93_row16_col2, #T_f8c93_row16_col3, #T_f8c93_row16_col4, #T_f8c93_row16_col5, #T_f8c93_row16_col6, #T_f8c93_row17_col0, #T_f8c93_row17_col1, #T_f8c93_row17_col2, #T_f8c93_row17_col3, #T_f8c93_row17_col4, #T_f8c93_row17_col5, #T_f8c93_row17_col6, #T_f8c93_row18_col0, #T_f8c93_row18_col1, #T_f8c93_row18_col2, #T_f8c93_row18_col3, #T_f8c93_row18_col4, #T_f8c93_row18_col5, #T_f8c93_row18_col6 {
  text-align: left;
}
#T_f8c93_row0_col1, #T_f8c93_row0_col2, #T_f8c93_row0_col3, #T_f8c93_row0_col4, #T_f8c93_row0_col5, #T_f8c93_row0_col6 {
  text-align: left;
  background-color: yellow;
}
#T_f8c93_row0_col7, #T_f8c93_row1_col7, #T_f8c93_row2_col7, #T_f8c93_row3_col7, #T_f8c93_row4_col7, #T_f8c93_row6_col7, #T_f8c93_row7_col7, #T_f8c93_row8_col7, #T_f8c93_row9_col7, #T_f8c93_row10_col7, #T_f8c93_row11_col7, #T_f8c93_row12_col7, #T_f8c93_row13_col7, #T_f8c93_row14_col7, #T_f8c93_row15_col7, #T_f8c93_row16_col7, #T_f8c93_row17_col7, #T_f8c93_row18_col7 {
  text-align: left;
  background-color: lightgrey;
}
#T_f8c93_row5_col7 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_f8c93" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f8c93_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_f8c93_level0_col1" class="col_heading level0 col1" >MAE</th>
      <th id="T_f8c93_level0_col2" class="col_heading level0 col2" >MSE</th>
      <th id="T_f8c93_level0_col3" class="col_heading level0 col3" >RMSE</th>
      <th id="T_f8c93_level0_col4" class="col_heading level0 col4" >R2</th>
      <th id="T_f8c93_level0_col5" class="col_heading level0 col5" >RMSLE</th>
      <th id="T_f8c93_level0_col6" class="col_heading level0 col6" >MAPE</th>
      <th id="T_f8c93_level0_col7" class="col_heading level0 col7" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f8c93_level0_row0" class="row_heading level0 row0" >gbr</th>
      <td id="T_f8c93_row0_col0" class="data row0 col0" >Gradient Boosting Regressor</td>
      <td id="T_f8c93_row0_col1" class="data row0 col1" >16668.8275</td>
      <td id="T_f8c93_row0_col2" class="data row0 col2" >813316846.6913</td>
      <td id="T_f8c93_row0_col3" class="data row0 col3" >27912.6374</td>
      <td id="T_f8c93_row0_col4" class="data row0 col4" >0.8666</td>
      <td id="T_f8c93_row0_col5" class="data row0 col5" >0.1348</td>
      <td id="T_f8c93_row0_col6" class="data row0 col6" >0.0963</td>
      <td id="T_f8c93_row0_col7" class="data row0 col7" >1.9690</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row1" class="row_heading level0 row1" >rf</th>
      <td id="T_f8c93_row1_col0" class="data row1 col0" >Random Forest Regressor</td>
      <td id="T_f8c93_row1_col1" class="data row1 col1" >18261.1246</td>
      <td id="T_f8c93_row1_col2" class="data row1 col2" >998294529.0476</td>
      <td id="T_f8c93_row1_col3" class="data row1 col3" >31130.8689</td>
      <td id="T_f8c93_row1_col4" class="data row1 col4" >0.8383</td>
      <td id="T_f8c93_row1_col5" class="data row1 col5" >0.1493</td>
      <td id="T_f8c93_row1_col6" class="data row1 col6" >0.1068</td>
      <td id="T_f8c93_row1_col7" class="data row1 col7" >3.5390</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row2" class="row_heading level0 row2" >lightgbm</th>
      <td id="T_f8c93_row2_col0" class="data row2 col0" >Light Gradient Boosting Machine</td>
      <td id="T_f8c93_row2_col1" class="data row2 col1" >17672.3315</td>
      <td id="T_f8c93_row2_col2" class="data row2 col2" >1012800681.3536</td>
      <td id="T_f8c93_row2_col3" class="data row2 col3" >31313.5997</td>
      <td id="T_f8c93_row2_col4" class="data row2 col4" >0.8345</td>
      <td id="T_f8c93_row2_col5" class="data row2 col5" >0.1418</td>
      <td id="T_f8c93_row2_col6" class="data row2 col6" >0.0999</td>
      <td id="T_f8c93_row2_col7" class="data row2 col7" >1.3330</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row3" class="row_heading level0 row3" >xgboost</th>
      <td id="T_f8c93_row3_col0" class="data row3 col0" >Extreme Gradient Boosting</td>
      <td id="T_f8c93_row3_col1" class="data row3 col1" >19010.3780</td>
      <td id="T_f8c93_row3_col2" class="data row3 col2" >1029452959.8392</td>
      <td id="T_f8c93_row3_col3" class="data row3 col3" >31447.4011</td>
      <td id="T_f8c93_row3_col4" class="data row3 col4" >0.8309</td>
      <td id="T_f8c93_row3_col5" class="data row3 col5" >0.1482</td>
      <td id="T_f8c93_row3_col6" class="data row3 col6" >0.1083</td>
      <td id="T_f8c93_row3_col7" class="data row3 col7" >2.6500</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row4" class="row_heading level0 row4" >et</th>
      <td id="T_f8c93_row4_col0" class="data row4 col0" >Extra Trees Regressor</td>
      <td id="T_f8c93_row4_col1" class="data row4 col1" >18237.5227</td>
      <td id="T_f8c93_row4_col2" class="data row4 col2" >1037767632.2362</td>
      <td id="T_f8c93_row4_col3" class="data row4 col3" >31830.1485</td>
      <td id="T_f8c93_row4_col4" class="data row4 col4" >0.8297</td>
      <td id="T_f8c93_row4_col5" class="data row4 col5" >0.1515</td>
      <td id="T_f8c93_row4_col6" class="data row4 col6" >0.1073</td>
      <td id="T_f8c93_row4_col7" class="data row4 col7" >3.0070</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row5" class="row_heading level0 row5" >en</th>
      <td id="T_f8c93_row5_col0" class="data row5 col0" >Elastic Net</td>
      <td id="T_f8c93_row5_col1" class="data row5 col1" >18364.3952</td>
      <td id="T_f8c93_row5_col2" class="data row5 col2" >1344589957.1260</td>
      <td id="T_f8c93_row5_col3" class="data row5 col3" >35059.9474</td>
      <td id="T_f8c93_row5_col4" class="data row5 col4" >0.7834</td>
      <td id="T_f8c93_row5_col5" class="data row5 col5" >0.1449</td>
      <td id="T_f8c93_row5_col6" class="data row5 col6" >0.1024</td>
      <td id="T_f8c93_row5_col7" class="data row5 col7" >0.8530</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row6" class="row_heading level0 row6" >par</th>
      <td id="T_f8c93_row6_col0" class="data row6 col0" >Passive Aggressive Regressor</td>
      <td id="T_f8c93_row6_col1" class="data row6 col1" >18134.1562</td>
      <td id="T_f8c93_row6_col2" class="data row6 col2" >1374167341.7881</td>
      <td id="T_f8c93_row6_col3" class="data row6 col3" >35373.6928</td>
      <td id="T_f8c93_row6_col4" class="data row6 col4" >0.7773</td>
      <td id="T_f8c93_row6_col5" class="data row6 col5" >0.1699</td>
      <td id="T_f8c93_row6_col6" class="data row6 col6" >0.1033</td>
      <td id="T_f8c93_row6_col7" class="data row6 col7" >2.4800</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row7" class="row_heading level0 row7" >ada</th>
      <td id="T_f8c93_row7_col0" class="data row7 col0" >AdaBoost Regressor</td>
      <td id="T_f8c93_row7_col1" class="data row7 col1" >25297.1088</td>
      <td id="T_f8c93_row7_col2" class="data row7 col2" >1483109549.4956</td>
      <td id="T_f8c93_row7_col3" class="data row7 col3" >38170.5192</td>
      <td id="T_f8c93_row7_col4" class="data row7 col4" >0.7603</td>
      <td id="T_f8c93_row7_col5" class="data row7 col5" >0.2061</td>
      <td id="T_f8c93_row7_col6" class="data row7 col6" >0.1631</td>
      <td id="T_f8c93_row7_col7" class="data row7 col7" >1.6810</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row8" class="row_heading level0 row8" >br</th>
      <td id="T_f8c93_row8_col0" class="data row8 col0" >Bayesian Ridge</td>
      <td id="T_f8c93_row8_col1" class="data row8 col1" >19102.7516</td>
      <td id="T_f8c93_row8_col2" class="data row8 col2" >1509230751.8566</td>
      <td id="T_f8c93_row8_col3" class="data row8 col3" >36789.4023</td>
      <td id="T_f8c93_row8_col4" class="data row8 col4" >0.7551</td>
      <td id="T_f8c93_row8_col5" class="data row8 col5" >0.1706</td>
      <td id="T_f8c93_row8_col6" class="data row8 col6" >0.1097</td>
      <td id="T_f8c93_row8_col7" class="data row8 col7" >0.9030</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row9" class="row_heading level0 row9" >omp</th>
      <td id="T_f8c93_row9_col0" class="data row9 col0" >Orthogonal Matching Pursuit</td>
      <td id="T_f8c93_row9_col1" class="data row9 col1" >19565.0295</td>
      <td id="T_f8c93_row9_col2" class="data row9 col2" >1604341893.6740</td>
      <td id="T_f8c93_row9_col3" class="data row9 col3" >37530.4782</td>
      <td id="T_f8c93_row9_col4" class="data row9 col4" >0.7370</td>
      <td id="T_f8c93_row9_col5" class="data row9 col5" >0.1815</td>
      <td id="T_f8c93_row9_col6" class="data row9 col6" >0.1121</td>
      <td id="T_f8c93_row9_col7" class="data row9 col7" >1.2170</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row10" class="row_heading level0 row10" >llar</th>
      <td id="T_f8c93_row10_col0" class="data row10 col0" >Lasso Least Angle Regression</td>
      <td id="T_f8c93_row10_col1" class="data row10 col1" >19787.4115</td>
      <td id="T_f8c93_row10_col2" class="data row10 col2" >1620639261.1369</td>
      <td id="T_f8c93_row10_col3" class="data row10 col3" >38028.6722</td>
      <td id="T_f8c93_row10_col4" class="data row10 col4" >0.7366</td>
      <td id="T_f8c93_row10_col5" class="data row10 col5" >0.1867</td>
      <td id="T_f8c93_row10_col6" class="data row10 col6" >0.1157</td>
      <td id="T_f8c93_row10_col7" class="data row10 col7" >0.9700</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row11" class="row_heading level0 row11" >dt</th>
      <td id="T_f8c93_row11_col0" class="data row11 col0" >Decision Tree Regressor</td>
      <td id="T_f8c93_row11_col1" class="data row11 col1" >26873.8594</td>
      <td id="T_f8c93_row11_col2" class="data row11 col2" >1656701282.2708</td>
      <td id="T_f8c93_row11_col3" class="data row11 col3" >40601.1973</td>
      <td id="T_f8c93_row11_col4" class="data row11 col4" >0.7338</td>
      <td id="T_f8c93_row11_col5" class="data row11 col5" >0.2134</td>
      <td id="T_f8c93_row11_col6" class="data row11 col6" >0.1554</td>
      <td id="T_f8c93_row11_col7" class="data row11 col7" >0.9180</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row12" class="row_heading level0 row12" >ridge</th>
      <td id="T_f8c93_row12_col0" class="data row12 col0" >Ridge Regression</td>
      <td id="T_f8c93_row12_col1" class="data row12 col1" >20390.6023</td>
      <td id="T_f8c93_row12_col2" class="data row12 col2" >1675546296.8694</td>
      <td id="T_f8c93_row12_col3" class="data row12 col3" >38612.7187</td>
      <td id="T_f8c93_row12_col4" class="data row12 col4" >0.7280</td>
      <td id="T_f8c93_row12_col5" class="data row12 col5" >0.2029</td>
      <td id="T_f8c93_row12_col6" class="data row12 col6" >0.1197</td>
      <td id="T_f8c93_row12_col7" class="data row12 col7" >1.1390</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row13" class="row_heading level0 row13" >lasso</th>
      <td id="T_f8c93_row13_col0" class="data row13 col0" >Lasso Regression</td>
      <td id="T_f8c93_row13_col1" class="data row13 col1" >20348.5520</td>
      <td id="T_f8c93_row13_col2" class="data row13 col2" >1679664973.5803</td>
      <td id="T_f8c93_row13_col3" class="data row13 col3" >38597.0093</td>
      <td id="T_f8c93_row13_col4" class="data row13 col4" >0.7271</td>
      <td id="T_f8c93_row13_col5" class="data row13 col5" >0.2045</td>
      <td id="T_f8c93_row13_col6" class="data row13 col6" >0.1196</td>
      <td id="T_f8c93_row13_col7" class="data row13 col7" >3.0070</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row14" class="row_heading level0 row14" >huber</th>
      <td id="T_f8c93_row14_col0" class="data row14 col0" >Huber Regressor</td>
      <td id="T_f8c93_row14_col1" class="data row14 col1" >19236.5019</td>
      <td id="T_f8c93_row14_col2" class="data row14 col2" >1766235862.2651</td>
      <td id="T_f8c93_row14_col3" class="data row14 col3" >38848.9916</td>
      <td id="T_f8c93_row14_col4" class="data row14 col4" >0.7128</td>
      <td id="T_f8c93_row14_col5" class="data row14 col5" >0.2052</td>
      <td id="T_f8c93_row14_col6" class="data row14 col6" >0.1123</td>
      <td id="T_f8c93_row14_col7" class="data row14 col7" >1.2150</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row15" class="row_heading level0 row15" >knn</th>
      <td id="T_f8c93_row15_col0" class="data row15 col0" >K Neighbors Regressor</td>
      <td id="T_f8c93_row15_col1" class="data row15 col1" >27602.8084</td>
      <td id="T_f8c93_row15_col2" class="data row15 col2" >2038986154.7901</td>
      <td id="T_f8c93_row15_col3" class="data row15 col3" >44370.3270</td>
      <td id="T_f8c93_row15_col4" class="data row15 col4" >0.6870</td>
      <td id="T_f8c93_row15_col5" class="data row15 col5" >0.2092</td>
      <td id="T_f8c93_row15_col6" class="data row15 col6" >0.1559</td>
      <td id="T_f8c93_row15_col7" class="data row15 col7" >1.1220</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row16" class="row_heading level0 row16" >dummy</th>
      <td id="T_f8c93_row16_col0" class="data row16 col0" >Dummy Regressor</td>
      <td id="T_f8c93_row16_col1" class="data row16 col1" >57764.4783</td>
      <td id="T_f8c93_row16_col2" class="data row16 col2" >6442602524.5583</td>
      <td id="T_f8c93_row16_col3" class="data row16 col3" >79947.6263</td>
      <td id="T_f8c93_row16_col4" class="data row16 col4" >-0.0024</td>
      <td id="T_f8c93_row16_col5" class="data row16 col5" >0.4042</td>
      <td id="T_f8c93_row16_col6" class="data row16 col6" >0.3610</td>
      <td id="T_f8c93_row16_col7" class="data row16 col7" >1.1870</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row17" class="row_heading level0 row17" >lr</th>
      <td id="T_f8c93_row17_col0" class="data row17 col0" >Linear Regression</td>
      <td id="T_f8c93_row17_col1" class="data row17 col1" >489462244852579.7500</td>
      <td id="T_f8c93_row17_col2" class="data row17 col2" >25087786038259630322198507421696.0000</td>
      <td id="T_f8c93_row17_col3" class="data row17 col3" >3296273264514296.0000</td>
      <td id="T_f8c93_row17_col4" class="data row17 col4" >-3909394621073952604160.0000</td>
      <td id="T_f8c93_row17_col5" class="data row17 col5" >3.5936</td>
      <td id="T_f8c93_row17_col6" class="data row17 col6" >4067501731.2203</td>
      <td id="T_f8c93_row17_col7" class="data row17 col7" >4.4590</td>
    </tr>
    <tr>
      <th id="T_f8c93_level0_row18" class="row_heading level0 row18" >lar</th>
      <td id="T_f8c93_row18_col0" class="data row18 col0" >Least Angle Regression</td>
      <td id="T_f8c93_row18_col1" class="data row18 col1" >42521860416708132924698326635424220409273253888.0000</td>
      <td id="T_f8c93_row18_col2" class="data row18 col2" >1844180973733323350448562275845971804997761547450196885667534503848224189678324096601303512252416.0000</td>
      <td id="T_f8c93_row18_col3" class="data row18 col3" >429448190934811211097724065035493310127918809088.0000</td>
      <td id="T_f8c93_row18_col4" class="data row18 col4" >-356872355765253379694331818216569495633078728307084395869623944858328491348458530144256.0000</td>
      <td id="T_f8c93_row18_col5" class="data row18 col5" >49.8038</td>
      <td id="T_f8c93_row18_col6" class="data row18 col6" >310381041719748206784084963210196848476160.0000</td>
      <td id="T_f8c93_row18_col7" class="data row18 col7" >1.4570</td>
    </tr>
  </tbody>
</table>






<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GradientBoostingRegressor(random_state=679)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">GradientBoostingRegressor</label><div class="sk-toggleable__content"><pre>GradientBoostingRegressor(random_state=679)</pre></div></div></div></div></div>


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




<style type="text/css">
#T_9c673_row10_col0, #T_9c673_row10_col1, #T_9c673_row10_col2, #T_9c673_row10_col3, #T_9c673_row10_col4, #T_9c673_row10_col5 {
  background: yellow;
}
</style>
<table id="T_9c673" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_9c673_level0_col0" class="col_heading level0 col0" >MAE</th>
      <th id="T_9c673_level0_col1" class="col_heading level0 col1" >MSE</th>
      <th id="T_9c673_level0_col2" class="col_heading level0 col2" >RMSE</th>
      <th id="T_9c673_level0_col3" class="col_heading level0 col3" >R2</th>
      <th id="T_9c673_level0_col4" class="col_heading level0 col4" >RMSLE</th>
      <th id="T_9c673_level0_col5" class="col_heading level0 col5" >MAPE</th>
    </tr>
    <tr>
      <th class="index_name level0" >Fold</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_9c673_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_9c673_row0_col0" class="data row0 col0" >15442.1645</td>
      <td id="T_9c673_row0_col1" class="data row0 col1" >500820041.6930</td>
      <td id="T_9c673_row0_col2" class="data row0 col2" >22379.0090</td>
      <td id="T_9c673_row0_col3" class="data row0 col3" >0.9257</td>
      <td id="T_9c673_row0_col4" class="data row0 col4" >0.1161</td>
      <td id="T_9c673_row0_col5" class="data row0 col5" >0.0871</td>
    </tr>
    <tr>
      <th id="T_9c673_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_9c673_row1_col0" class="data row1 col0" >18244.9876</td>
      <td id="T_9c673_row1_col1" class="data row1 col1" >855920787.4039</td>
      <td id="T_9c673_row1_col2" class="data row1 col2" >29256.1239</td>
      <td id="T_9c673_row1_col3" class="data row1 col3" >0.8332</td>
      <td id="T_9c673_row1_col4" class="data row1 col4" >0.1369</td>
      <td id="T_9c673_row1_col5" class="data row1 col5" >0.1017</td>
    </tr>
    <tr>
      <th id="T_9c673_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_9c673_row2_col0" class="data row2 col0" >20166.5540</td>
      <td id="T_9c673_row2_col1" class="data row2 col1" >1435737692.1195</td>
      <td id="T_9c673_row2_col2" class="data row2 col2" >37891.1295</td>
      <td id="T_9c673_row2_col3" class="data row2 col3" >0.7530</td>
      <td id="T_9c673_row2_col4" class="data row2 col4" >0.1675</td>
      <td id="T_9c673_row2_col5" class="data row2 col5" >0.1193</td>
    </tr>
    <tr>
      <th id="T_9c673_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_9c673_row3_col0" class="data row3 col0" >17853.1263</td>
      <td id="T_9c673_row3_col1" class="data row3 col1" >1729976226.4035</td>
      <td id="T_9c673_row3_col2" class="data row3 col2" >41592.9829</td>
      <td id="T_9c673_row3_col3" class="data row3 col3" >0.6375</td>
      <td id="T_9c673_row3_col4" class="data row3 col4" >0.1523</td>
      <td id="T_9c673_row3_col5" class="data row3 col5" >0.1004</td>
    </tr>
    <tr>
      <th id="T_9c673_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_9c673_row4_col0" class="data row4 col0" >20165.9642</td>
      <td id="T_9c673_row4_col1" class="data row4 col1" >1283772957.2518</td>
      <td id="T_9c673_row4_col2" class="data row4 col2" >35829.7775</td>
      <td id="T_9c673_row4_col3" class="data row4 col3" >0.8314</td>
      <td id="T_9c673_row4_col4" class="data row4 col4" >0.1748</td>
      <td id="T_9c673_row4_col5" class="data row4 col5" >0.1221</td>
    </tr>
    <tr>
      <th id="T_9c673_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_9c673_row5_col0" class="data row5 col0" >16520.0521</td>
      <td id="T_9c673_row5_col1" class="data row5 col1" >686462662.0092</td>
      <td id="T_9c673_row5_col2" class="data row5 col2" >26200.4325</td>
      <td id="T_9c673_row5_col3" class="data row5 col3" >0.9053</td>
      <td id="T_9c673_row5_col4" class="data row5 col4" >0.1295</td>
      <td id="T_9c673_row5_col5" class="data row5 col5" >0.0917</td>
    </tr>
    <tr>
      <th id="T_9c673_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_9c673_row6_col0" class="data row6 col0" >17028.9203</td>
      <td id="T_9c673_row6_col1" class="data row6 col1" >786230258.6556</td>
      <td id="T_9c673_row6_col2" class="data row6 col2" >28039.7978</td>
      <td id="T_9c673_row6_col3" class="data row6 col3" >0.8479</td>
      <td id="T_9c673_row6_col4" class="data row6 col4" >0.1132</td>
      <td id="T_9c673_row6_col5" class="data row6 col5" >0.0887</td>
    </tr>
    <tr>
      <th id="T_9c673_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_9c673_row7_col0" class="data row7 col0" >16193.2443</td>
      <td id="T_9c673_row7_col1" class="data row7 col1" >1033335960.9131</td>
      <td id="T_9c673_row7_col2" class="data row7 col2" >32145.5434</td>
      <td id="T_9c673_row7_col3" class="data row7 col3" >0.8559</td>
      <td id="T_9c673_row7_col4" class="data row7 col4" >0.1365</td>
      <td id="T_9c673_row7_col5" class="data row7 col5" >0.0908</td>
    </tr>
    <tr>
      <th id="T_9c673_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_9c673_row8_col0" class="data row8 col0" >16631.2041</td>
      <td id="T_9c673_row8_col1" class="data row8 col1" >685488507.1789</td>
      <td id="T_9c673_row8_col2" class="data row8 col2" >26181.8354</td>
      <td id="T_9c673_row8_col3" class="data row8 col3" >0.8899</td>
      <td id="T_9c673_row8_col4" class="data row8 col4" >0.1640</td>
      <td id="T_9c673_row8_col5" class="data row8 col5" >0.1019</td>
    </tr>
    <tr>
      <th id="T_9c673_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_9c673_row9_col0" class="data row9 col0" >18477.0974</td>
      <td id="T_9c673_row9_col1" class="data row9 col1" >1130261719.9075</td>
      <td id="T_9c673_row9_col2" class="data row9 col2" >33619.3653</td>
      <td id="T_9c673_row9_col3" class="data row9 col3" >0.8654</td>
      <td id="T_9c673_row9_col4" class="data row9 col4" >0.1274</td>
      <td id="T_9c673_row9_col5" class="data row9 col5" >0.0954</td>
    </tr>
    <tr>
      <th id="T_9c673_level0_row10" class="row_heading level0 row10" >Mean</th>
      <td id="T_9c673_row10_col0" class="data row10 col0" >17672.3315</td>
      <td id="T_9c673_row10_col1" class="data row10 col1" >1012800681.3536</td>
      <td id="T_9c673_row10_col2" class="data row10 col2" >31313.5997</td>
      <td id="T_9c673_row10_col3" class="data row10 col3" >0.8345</td>
      <td id="T_9c673_row10_col4" class="data row10 col4" >0.1418</td>
      <td id="T_9c673_row10_col5" class="data row10 col5" >0.0999</td>
    </tr>
    <tr>
      <th id="T_9c673_level0_row11" class="row_heading level0 row11" >Std</th>
      <td id="T_9c673_row11_col0" class="data row11 col0" >1530.8375</td>
      <td id="T_9c673_row11_col1" class="data row11 col1" >365042550.9815</td>
      <td id="T_9c673_row11_col2" class="data row11 col2" >5679.7143</td>
      <td id="T_9c673_row11_col3" class="data row11 col3" >0.0797</td>
      <td id="T_9c673_row11_col4" class="data row11 col4" >0.0206</td>
      <td id="T_9c673_row11_col5" class="data row11 col5" >0.0116</td>
    </tr>
  </tbody>
</table>


- ```verbose``` 옵션

  - 함수 수행 시 발생하는 상세한 정보들을 표준 출력으로 자세히 내보낼 것인가를 결정

  - True의 경우 자세히 출력함


# **5. 모델 튜닝하기**

- ```tune_model()```

  - 입력한 모델에 대해서 hyper parameter tuning을 수행



```python
tuned_lgb = tune_model(lgb)
```



<style type="text/css">
#T_230b8_row10_col0, #T_230b8_row10_col1, #T_230b8_row10_col2, #T_230b8_row10_col3, #T_230b8_row10_col4, #T_230b8_row10_col5 {
  background: yellow;
}
</style>
<table id="T_230b8" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_230b8_level0_col0" class="col_heading level0 col0" >MAE</th>
      <th id="T_230b8_level0_col1" class="col_heading level0 col1" >MSE</th>
      <th id="T_230b8_level0_col2" class="col_heading level0 col2" >RMSE</th>
      <th id="T_230b8_level0_col3" class="col_heading level0 col3" >R2</th>
      <th id="T_230b8_level0_col4" class="col_heading level0 col4" >RMSLE</th>
      <th id="T_230b8_level0_col5" class="col_heading level0 col5" >MAPE</th>
    </tr>
    <tr>
      <th class="index_name level0" >Fold</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_230b8_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_230b8_row0_col0" class="data row0 col0" >16220.2312</td>
      <td id="T_230b8_row0_col1" class="data row0 col1" >672187894.3403</td>
      <td id="T_230b8_row0_col2" class="data row0 col2" >25926.5866</td>
      <td id="T_230b8_row0_col3" class="data row0 col3" >0.9003</td>
      <td id="T_230b8_row0_col4" class="data row0 col4" >0.1230</td>
      <td id="T_230b8_row0_col5" class="data row0 col5" >0.0895</td>
    </tr>
    <tr>
      <th id="T_230b8_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_230b8_row1_col0" class="data row1 col0" >18389.4504</td>
      <td id="T_230b8_row1_col1" class="data row1 col1" >957053892.6023</td>
      <td id="T_230b8_row1_col2" class="data row1 col2" >30936.2876</td>
      <td id="T_230b8_row1_col3" class="data row1 col3" >0.8135</td>
      <td id="T_230b8_row1_col4" class="data row1 col4" >0.1396</td>
      <td id="T_230b8_row1_col5" class="data row1 col5" >0.1032</td>
    </tr>
    <tr>
      <th id="T_230b8_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_230b8_row2_col0" class="data row2 col0" >18155.1805</td>
      <td id="T_230b8_row2_col1" class="data row2 col1" >1124713680.5994</td>
      <td id="T_230b8_row2_col2" class="data row2 col2" >33536.7512</td>
      <td id="T_230b8_row2_col3" class="data row2 col3" >0.8065</td>
      <td id="T_230b8_row2_col4" class="data row2 col4" >0.1665</td>
      <td id="T_230b8_row2_col5" class="data row2 col5" >0.1146</td>
    </tr>
    <tr>
      <th id="T_230b8_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_230b8_row3_col0" class="data row3 col0" >17656.8498</td>
      <td id="T_230b8_row3_col1" class="data row3 col1" >1042311405.3105</td>
      <td id="T_230b8_row3_col2" class="data row3 col2" >32284.8479</td>
      <td id="T_230b8_row3_col3" class="data row3 col3" >0.7816</td>
      <td id="T_230b8_row3_col4" class="data row3 col4" >0.1439</td>
      <td id="T_230b8_row3_col5" class="data row3 col5" >0.1019</td>
    </tr>
    <tr>
      <th id="T_230b8_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_230b8_row4_col0" class="data row4 col0" >18224.1143</td>
      <td id="T_230b8_row4_col1" class="data row4 col1" >1133540184.1908</td>
      <td id="T_230b8_row4_col2" class="data row4 col2" >33668.0885</td>
      <td id="T_230b8_row4_col3" class="data row4 col3" >0.8512</td>
      <td id="T_230b8_row4_col4" class="data row4 col4" >0.1765</td>
      <td id="T_230b8_row4_col5" class="data row4 col5" >0.1181</td>
    </tr>
    <tr>
      <th id="T_230b8_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_230b8_row5_col0" class="data row5 col0" >16493.6110</td>
      <td id="T_230b8_row5_col1" class="data row5 col1" >687283374.5380</td>
      <td id="T_230b8_row5_col2" class="data row5 col2" >26216.0900</td>
      <td id="T_230b8_row5_col3" class="data row5 col3" >0.9052</td>
      <td id="T_230b8_row5_col4" class="data row5 col4" >0.1354</td>
      <td id="T_230b8_row5_col5" class="data row5 col5" >0.0950</td>
    </tr>
    <tr>
      <th id="T_230b8_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_230b8_row6_col0" class="data row6 col0" >14976.2889</td>
      <td id="T_230b8_row6_col1" class="data row6 col1" >584667369.0686</td>
      <td id="T_230b8_row6_col2" class="data row6 col2" >24179.8960</td>
      <td id="T_230b8_row6_col3" class="data row6 col3" >0.8869</td>
      <td id="T_230b8_row6_col4" class="data row6 col4" >0.1024</td>
      <td id="T_230b8_row6_col5" class="data row6 col5" >0.0797</td>
    </tr>
    <tr>
      <th id="T_230b8_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_230b8_row7_col0" class="data row7 col0" >19611.3610</td>
      <td id="T_230b8_row7_col1" class="data row7 col1" >1654645129.8896</td>
      <td id="T_230b8_row7_col2" class="data row7 col2" >40677.3294</td>
      <td id="T_230b8_row7_col3" class="data row7 col3" >0.7693</td>
      <td id="T_230b8_row7_col4" class="data row7 col4" >0.1606</td>
      <td id="T_230b8_row7_col5" class="data row7 col5" >0.1084</td>
    </tr>
    <tr>
      <th id="T_230b8_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_230b8_row8_col0" class="data row8 col0" >16973.8506</td>
      <td id="T_230b8_row8_col1" class="data row8 col1" >744827085.8664</td>
      <td id="T_230b8_row8_col2" class="data row8 col2" >27291.5204</td>
      <td id="T_230b8_row8_col3" class="data row8 col3" >0.8803</td>
      <td id="T_230b8_row8_col4" class="data row8 col4" >0.1582</td>
      <td id="T_230b8_row8_col5" class="data row8 col5" >0.1068</td>
    </tr>
    <tr>
      <th id="T_230b8_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_230b8_row9_col0" class="data row9 col0" >20922.3215</td>
      <td id="T_230b8_row9_col1" class="data row9 col1" >1753138613.2070</td>
      <td id="T_230b8_row9_col2" class="data row9 col2" >41870.4981</td>
      <td id="T_230b8_row9_col3" class="data row9 col3" >0.7912</td>
      <td id="T_230b8_row9_col4" class="data row9 col4" >0.1580</td>
      <td id="T_230b8_row9_col5" class="data row9 col5" >0.1162</td>
    </tr>
    <tr>
      <th id="T_230b8_level0_row10" class="row_heading level0 row10" >Mean</th>
      <td id="T_230b8_row10_col0" class="data row10 col0" >17762.3259</td>
      <td id="T_230b8_row10_col1" class="data row10 col1" >1035436862.9613</td>
      <td id="T_230b8_row10_col2" class="data row10 col2" >31658.7896</td>
      <td id="T_230b8_row10_col3" class="data row10 col3" >0.8386</td>
      <td id="T_230b8_row10_col4" class="data row10 col4" >0.1464</td>
      <td id="T_230b8_row10_col5" class="data row10 col5" >0.1033</td>
    </tr>
    <tr>
      <th id="T_230b8_level0_row11" class="row_heading level0 row11" >Std</th>
      <td id="T_230b8_row11_col0" class="data row11 col0" >1629.3522</td>
      <td id="T_230b8_row11_col1" class="data row11 col1" >382505984.9453</td>
      <td id="T_230b8_row11_col2" class="data row11 col2" >5758.2901</td>
      <td id="T_230b8_row11_col3" class="data row11 col3" >0.0494</td>
      <td id="T_230b8_row11_col4" class="data row11 col4" >0.0211</td>
      <td id="T_230b8_row11_col5" class="data row11 col5" >0.0117</td>
    </tr>
  </tbody>
</table>


<pre>
Processing:   0%|          | 0/7 [00:00<?, ?it/s]
</pre>
<pre>
Fitting 10 folds for each of 10 candidates, totalling 100 fits
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

<img src="data:image/png;base64,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XbPvap5++mlMJhNLly4lJiaGvn378uyzzxb46544cYIePXrkqO+4cePo3r17AUckkntlS5loWBb+OJf19nLpiVl/r4++H958uCBDExERKbKU7BciDw8Phg0bxrBhwwr1dcuVK2d3M6zrqVChQgFHI5J36/u5Efhu1newvqtWAoYJTNdO73t5FnxgIiIiRZSS/RLA09OTunVzsFKJSBFX2tuNVhWtbDyZeVttn2Qyztlek+2XK525s4iIuC6T6nNyQzX7IuJS6pfNut2jtCfJdzW0b/T3gftaFHxQIiIiRZRm9kXEpTQs5wZcW8pjUKNUGhfe7k+pWpXgh51QMxTG94WKwU6IUkREpGhQsi8iLuWpW02M3whnr1pmuY7HOaqWSsPwLwPvPum84ERERIoYJfsi4lLc3Uz8PdiN0esNtp826BSaSI1ja4B6zg5NRESkyFGyLyIux8fDjbc7Zvz9zJlUVh5zbjwiIiJFlS7QFREREREppjSzLyIiIiKuQytv5opm9kVEREREiikl+yIiIiIixZTKeERERETEhaiOJzc0sy8iIiIiUkwp2RcRERERKaZUxiMiLun7v6zM/t0gLc2f+qkVdEstEZGSQlU8uaJkX0RczvLDVnostf7/My9W0oFq549wX6hTwxIRESlyVMYjIi7n9d+sds8NTMz/u5yTohERESm6lOyLiMs5EJe57e9k74y/7D8OS36D4+cLNygREZEiSGU8IuJyksyZ29IsJgInfgsf/pjR4O4GswbDE10KNzgREZEiRDP7IuJS4lMMLEbm9pQ0CLiS6ANYrPDsR3A5ufCCExERKWKU7IuIS1mwx5Jlu3dqaubGpDT461QBRyQiIlJ0qYxHRFzKpM1Zt9967EjWG9w0pyEiUqxo6c1c0b+CIuJSTiVm3d739w1Zbxg2t+CCERERKeKU7IuIS1h60Mo931jIolwfgAWNb896wy9/wm/7CiwuERGRoqxYlfHMnj2b6OhouzY/Pz9CQ0Np0aIFjz/+OEFBQc4Jjqzju9bMmTNp3bp1jo8ZERHByZMnWb58OQCRkZF89913bNu2DYBt27YxZMgQu328vLwoX7484eHhPP7449SuXTuX7yRrV97fDz/8QNmyZfPlmCIAE36zMPY3R2l+ht/C6mLg4Ozu+yug9c0FEZqIiEiRVqyS/SuioqKoWLEiAImJifzv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Di2pmSua2RcRERERKaaU7IuIiIiIFFNK9kVEREREiikl+yIiIiIixZSSfRERERGRYkqr8YiIiIiI69BqPLmimX0RERERkWJKyb6IiIiISDGlZF9EREREpJhSsi8iIiIiUkwp2RcRERERKaaU7IuIiIiIFFNaelNEREREXIdJa2/mhmb2RURERESKKSX7IiIiIlLiDRgwgOeff97ZYeQ7lfGIiIiIiOtQFU+uaGZfRERERCQbq1ev5v7776dx48a0aNGCl156ibi4OFJSUmjYsCE///yzrW90dDR169Zl27Zttra3336bBx98sNDjVrIvIiIiInIdW7ZsYdiwYTzyyCNs2rSJr7/+mr/++ovnnnsOHx8fmjVrxpYtW2z9N27cSJ06ddi0aZNdW/v27Qs9dpXxSIm3f/9+0tLSnB1GtgzDAODQoUOYtBKBjcVioXbt2gCcO3eOCxcuODmiokWfG8c0No5pbBzT2Njz8vKibt26zg6jwH322We0atWKe++9F4AqVarw9NNPM3ToUE6cOEG7du2IiYkBIC0tjR07dvDaa6+xbNkynnnmGS5fvsyff/7J2LFjCz12JfsiLsJkMuHl5eXsMIocd3d3/P39nR1GkaXPjWMaG8c0No5pbJzPeKnw09ejR4/SsmVLu7YrE01///037dq1IyoqikuXLvHnn39StWpVOnfuzIQJE0hNTWXz5s0EBQVRv379Qo9dyb6UeCVhRkJERETyLjU1NVOb1WoFMn4A1qpVi4oVK7Jlyxb++OMPWrZsSWBgIGFhYezYsYNNmzbRrl07p5wNUs2+iIiIiMh1hIWFsX//fru2gwcP2rYBtG3bli1btrBx40ZatWoFQPPmzdm0aRObNm3i9ttvL9SYr1CyLyIiIiJyHf369WPTpk0sW7aM9PR0jh49yvvvv0/Hjh0JDQ0FoH379vz666/s3buX5s2bAxnJ/k8//URsbCxt2rRxSuwm48qVJiIiIiIiJdSAAQPYtm0bHh72Ve6lSpVi8+bNLF26lE8++YS///6bMmXK0KlTJ5577jn8/PwASEhIoGXLloSHh7N48WIA4uPjadmyJbfddhsLFy4s9PcESvZFRERERIotlfGIiIiIiBRTSvZFRERERIopJfsiIiIiIsWUkn0RERERkWJKN9USKUQnTpzg3XffZceOHaSkpFCjRg2GDBlid1e+H374gQULFvDPP/8QHBxMp06dePrpp3F3dwcyruyfOnUqO3bsIDk5mZtuuonhw4dzyy235OsxXEVKSgrTp0/nt99+4+LFi9SoUYOIiAhat27t7NBuSHx8PO+99x4bN27k0qVLVKlShUcffZSuXbsCsHnzZmbPns1ff/2Fv78/rVq14oUXXsDX1xfI2bjkxzGc6ejRo/Tv35/OnTsTGRkJaFy+++47PvnkE06cOEHZsmXp3bs3AwYMAEr22MTGxvL+++/zxx9/kJKSQuXKlenfvz933303ULLHRkoAQ0QKRWpqqtGzZ0/jlVdeMS5cuGAkJycb7777rtG6dWvjxIkThmEYxrZt24zmzZsbK1asMFJSUoyDBw8ad999t/HBBx/YjjN48GDjySefNE6ePGkkJiYaH3zwgdGxY0cjLi4u347hSiIjI43evXsbf/31l5GSkmIsWbLEaNmypfHXX385O7QbMmjQINt/o/T0dOOrr74ymjVrZvzxxx/G0aNHjVatWhkLFiwwkpKSjGPHjhn9+/c3xowZY9s/u3HJj2M4k9lsNh599FHj9ttvN8aNG2cYRv68J1cel9WrVxt33HGHsXHjRiM1NdXYunWr8cADDxi7du0q0WNjNpuNbt262b5709LSjBUrVhhNmzY1Nm7cWKLHRkoGJfsiheT48eNGZGSkcfbsWVtbQkKC0aRJE2PFihWGYRjGyy+/bAwbNsxuvy+++MLo0KGDkZ6ebhw8eNBo0qSJsWvXLtv29PR0o1OnTsaCBQvy7Riu4uLFi0aLFi2MVatW2bX369fPiIqKclJUN+7y5cvG+PHjjQMHDti133HHHcbHH39svPPOO0avXr3stv38889G8+bNjfPnz+doXPLjGM40d+5cY8CAAcaYMWNsyX5JH5devXoZH330UZbbSvLYnD592mjSpInx66+/2rV36dLFmDt3bokeGykZVMYjUkgqVarEuHHj7NqOHTsGQIUKFQDYvXs39913n12fW265hcuXL3P06FF2796Nu7s74eHhtu0eHh7cfPPN7Nq1K9+O4Sr27t2L2WymQYMGdu233HILu3fvdlJUN87f35/XXnvNru3ChQskJiYSGhrKr7/+Sv369e22169fH4vFwp49e/D09Mx2XHbt2nXDx3CW/fv3M3/+fObNm8f8+fNt7fnxnlx1XM6dO8eRI0cICAggIiKC/fv3ExoaysCBA7n77rtL9NiUK1eOhg0bsnTpUurWrUuZMmX46aefSEpKon379kyePLnEjo2UDLpAVySfmM1mLl++7PBhtVrt+ickJDBu3Djatm3LrbfeCmQkdKVLl7brFxQUZNt24cIF/P39bbX3V/e5cOFCvh3DVVyJN6v3GxcX54yQCkRaWhqvvvoqNWvW5M477+TChQsEBgba9bny3zguLi5H45Ifx3CG9PR0xo0bx2OPPUbNmjXttpXkcTl16hQAS5Ys4aWXXmLlypXcf//9jBs3ji1btpTosTGZTEydOpXTp09z11130bJlSyZMmMDYsWOpU6dOiR4bKRk0sy+ST3bu3MmQIUMcbl+yZAlhYWFAxoWFL7zwAqGhoUyaNKmQIhRXdO7cOUaMGEFKSgrvvvtuptu4lzQffvghPj4+totOxV7v3r256aabAOjbty/ff/89y5cvd3JUzpWens6wYcOoUqUKU6dOpXTp0mzYsIHIyMhMybdIcVSy/9UQyUdNmzZl27Zt2fbbunUrI0aM4O677+a5556zS96Cg4O5ePGiXf/4+HgAQkJCCA4OJiEhAYvFYjczHx8fT0hISL4dw1VciffixYuUKlXK1u6K7yUrBw8e5LnnnqNBgwaMHTvW9h5DQkKu+9/Yy8sLuP645McxCtvOnTtZvHgxCxYsyHRmCkruuACULVsWgICAALv2ypUrc+7cuRI9Nlu3bmX//v288847lC9fHoDOnTvz3//+lyVLlpTosZGSQWU8IoVo+/btvPjiizz33HO89NJLmWZpGzZsmKl+c+fOnQQGBlKtWjUaNWqExWJh7969tu3p6ens2bOHhg0b5tsxXEV4eDheXl6ZrjX4/fffXe69XOvIkSM89dRT9OzZk8mTJ9slCA0bNsz0nnfu3Im7uzu33HJLjsYlP45R2L799lvS09N5/PHH6dSpE506dWLVqlWsWrWKTp06ldhxgYy69MDAQLv/ryHjuqBKlSqV6LG5UkJ5bSmlxWLBarWW6LGRkkHJvkghSUpKYuzYsQwePJiePXtm2eehhx5i8+bN/PDDD6SlpbFnzx4+//xz+vXrh7u7O2FhYbRu3ZoZM2Zw6tQpEhISmDlzJu7u7nTr1i3fjuEq/P396dGjB9HR0cTGxpKSksKCBQs4fvw4ffr0cXZ4eWaxWBg7dix33XUXERERmbY/8MADnDp1igULFpCSkkJsbCyzZ8/mnnvuISgoKEfjkh/HKGzPP/883377LV988YXt0b59e9q3b88XX3xRYscFwN3dnf79+/PNN9+wadMm0tLSWLx4Mfv27aNXr14lemwaNWpESEgIM2bM4Pz586Snp/Pzzz+zefNm7rzzzhI9NlIymAzDMJwdhEhJ8P333zN27Fjb6dyrdevWjTFjxgDw008/MWvWLNsNse677z4GDRqEm1vGb/PLly/z1ltvsX79etLT02nYsCEvvPACtWvXth0vP47hKtLS0pg5cyarVq0iISHBdoOwxo0bOzu0PNu5cydPPPEEnp6emEwmu2233XYb77//Pjt27GDGjBkcPHgQf39/unbtyrBhw2yfr5yMS34cw9mu3Ezryp8leVwMw+Cjjz5i6dKlxMXFUbVqVZ577jnbTZtK8tgcPnz4/9q7+7Cezz2A4+9kZUXTDipPW8Mp60GRkIg8bIUjzPPKhnU8nzomYS0srgsjKj1wPGVhq36EOro0m6GwnDWMnHk4rEIecrFyej5/dP2+x6/frweW2drndV2uXX2/9/e+7+/9q12f+/5+vvePiIgI/vWvf1FSUkLbtm3x9vbGy8sL+GOPjWj8JNgXQgghhBCikZI0HiGEEEIIIRopCfaFEEIIIYRopCTYF0IIIYQQopGSYF8IIYQQQohGSoJ9IYQQQgghGikJ9oUQQgghhGikJNgXQgghhBCikZJgXwghhBBCiEZKgn0hRIPLz88nNDSUYcOG4eTkhI2NDa6ursyaNYuzZ89qlA0PD8fKyoorV67orCsnJwcrKys+/fRTnedv3bpF165dsbKy4uLFizrLqNuo/s/Z2RlfX1/OnDnzy274KajvJzw8/Fdr80kZGRnY2tpy8OBBANzd3fH29n4hfRENx9vbm759+z71dYGBgVhZWT2HHv22bNu2DUdHRy5cuPCiuyLEr06CfSFEg7p79y5jxoxBpVIxZswYoqOj2bp1K3/961/Jzs7Gx8eH77//vsHaS0hIQF9fn5dffpnExMRay4aFhZGQkEBCQgKff/45S5Ys4datW3h7e5Oent5gffqtunPnDv7+/nh5eTF8+HAAoqKiWLZs2Qvu2bNLTU3F3d39RXdDPIWKigqcnJw4derUc2vj/v37WFtbk5OTA8B7771Hz549mTdvHj///PNza1eI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"/>


```python
predictions = predict_model(tuned_lgb, data = test)
sample['SalePrice'] = predictions['Label']
sample.to_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/10주차/data/house_sample_submission.csv',index = False)
sample.head()
```
