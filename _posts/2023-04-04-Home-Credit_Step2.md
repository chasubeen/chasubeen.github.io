---
layout: single
title:  "[ECC DS 3주차] 2. Introduction: Manual Feature Engineering"
categories: ML
tags: [ECC, DS, Home Credit Default Risk] 
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


# **📚 Reference**

- [한글화 작업 된 커널](https://github.com/Aete/Kaggle-Kernels-translations-korean/blob/master/Introduction%20Manual%20Feature%20Engineering%20(%ED%95%9C%EA%B8%80%20%EB%B2%88%EC%97%AD%EB%B3%B8).ipynb)


# **1. Introduction: Manual Feature Engineering**

- 이 커널에서는 ```The Home Credit Defalut Risk Competition``` 데이터를 기반으로 Feature를 생성하는 방법을 다루고 있음

- 커널 초반 부분에는 모델을 만들기 위해 ```application``` 데이터만 사용함

  - 해당 데이터를 기반으로 만들어진 가장 성능이 좋은 모델은 리더보드에서 약 **0.74**를 기록

- 다른 데이터프레임들을 활용하여 정보를 좀 더 모으려 함

  - 여기서는 ```bureau``` 및 ```bureau_balance``` 데이터를 활용하였음

  - ```bureau```

    -  'Home Credit'에 제출된 고객(Client)의 다른 금융 기관에서의 과거의 대출 기록

    - 각각의 대출 기록은 각각의 열로 정리되어 있음

  - ```bureau_balance```

    - 과거 대출들의 월별 데이터

    - 각각의 월별 데이터는 각각의 열로 정리되어 있음



- **Manual Feature Engineering**은 어떻게 보면 지루한 과정일 수 있음 + 해당 작업은 도메인 지식(domain expertise)을 필요로 하기도 함

  - 대출 및 채무 불이행의 주된 원인에 대한 지식을 갖추는데는 한계가 있기 때문에, 최종 학습용 데이터프레임에서 가능한 많은 정보들을 얻는데 주안점을 두었음

  - 즉, 이 커널은 어떤 feature들이 중요한지를 결정하는 것에 있어서 사람보다 모델이 고르도록 하는 접근방식을 택함

  - 최대한 많은 feature들을 만들고, 모델은 이러한 feature들을 전부 활용 -> 추후 모델에서 얻어진 feature importance나 PCA를 통해 feautre reduction을 할 수 있음



- **Manual Feature Engineering**의 각 과정은 많은 양의 Pandas 코드와 약간의 인내심, 특히 데이터 처리에 있어서 많은 인내심을 필요로 함

  - 비록 자동화된 Feature Engineering 도구들이 활용되기 시작했지만, 당분간 Feature Engineering은 여전히 전처리 작업을 필요로 함



```python
# 데이터 처리을 위한 Pandas 및 Numpy
import pandas as pd
import numpy as np

# 시각화를 위한 matplotlib 및 seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# pandas에서 나오는 경고문 무시
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
```

## **1-1. Example: 고객의 이전 대출 수량 파악(Counts of a client's previous loans)**

- **Manual Feature Engineering**의 보편적인 방법을 설명하기 위해, 먼저 고객의 과거 타 금융기관에서의 대출 수량을 간단히 파악하고자 함



### **📌 자주 사용되는 Pandas 명령어**


- [groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html)

  - column값에 따라 데이터 프레임을 그룹화

  - 이 과정에서는 ```SK_ID_CURR``` 컬럼의 값에 따라 고객별로 데이터 프레임을 그룹화

- [agg](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html)

  - 그룹화된 데이터의 평균 등을 계산

  - ```grouped_df.mean()```을 통해 직접 평균을 계산하거나, agg 명령어와 리스트를 활용하여 평균, 최대값, 최소값, 합계 등을 계산(```grouped_df.agg([mean, max, min, sum])```)

- [merge](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html#pandas.DataFrame.merge)

  - 집계된(aggregated) 값을 해당 고객와 매칭

  - ```SK_ID_CURR``` 컬럼을 활용하여 집계된 값을 원본 training 데이터로 병합하고, 해당 값이 없을 경우에는 NaN 값을 입력.

- 또한 ```rename``` 명령어를 통해 컬럼을 dict을 활용하여 변경

  - 이러한 방식은 생성된 변수들을 계속해서 추적하는데 유용



```python
from google.colab import drive
drive.mount('/content/drive')
```

<pre>
Mounted at /content/drive
</pre>

```python
### bureau 데이터 불러오기

bureau = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/3주차/data/bureau.csv')
bureau.head()
```


  <div id="df-5ba33850-4bc1-4a67-8582-cf1deceb75d2">
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
      <th>SK_ID_CURR</th>
      <th>SK_ID_BUREAU</th>
      <th>CREDIT_ACTIVE</th>
      <th>CREDIT_CURRENCY</th>
      <th>DAYS_CREDIT</th>
      <th>CREDIT_DAY_OVERDUE</th>
      <th>DAYS_CREDIT_ENDDATE</th>
      <th>DAYS_ENDDATE_FACT</th>
      <th>AMT_CREDIT_MAX_OVERDUE</th>
      <th>CNT_CREDIT_PROLONG</th>
      <th>AMT_CREDIT_SUM</th>
      <th>AMT_CREDIT_SUM_DEBT</th>
      <th>AMT_CREDIT_SUM_LIMIT</th>
      <th>AMT_CREDIT_SUM_OVERDUE</th>
      <th>CREDIT_TYPE</th>
      <th>DAYS_CREDIT_UPDATE</th>
      <th>AMT_ANNUITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>215354</td>
      <td>5714462</td>
      <td>Closed</td>
      <td>currency 1</td>
      <td>-497</td>
      <td>0</td>
      <td>-153.0</td>
      <td>-153.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>91323.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-131</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>215354</td>
      <td>5714463</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-208</td>
      <td>0</td>
      <td>1075.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>225000.0</td>
      <td>171342.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Credit card</td>
      <td>-20</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>215354</td>
      <td>5714464</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-203</td>
      <td>0</td>
      <td>528.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>464323.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-16</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>215354</td>
      <td>5714465</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-203</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>90000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Credit card</td>
      <td>-16</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>215354</td>
      <td>5714466</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-629</td>
      <td>0</td>
      <td>1197.0</td>
      <td>NaN</td>
      <td>77674.5</td>
      <td>0</td>
      <td>2700000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-21</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5ba33850-4bc1-4a67-8582-cf1deceb75d2')"
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
          document.querySelector('#df-5ba33850-4bc1-4a67-8582-cf1deceb75d2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5ba33850-4bc1-4a67-8582-cf1deceb75d2');
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
### 고객 id (SK_ID_CURR)를 기준으로 groupby
# 이전 대출 갯수를 파악하고, 컬럼 이름을 변경

previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index = False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()
```


  <div id="df-8b629fcd-3777-4ca3-a677-629e988d7091">
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
      <th>SK_ID_CURR</th>
      <th>previous_loan_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8b629fcd-3777-4ca3-a677-629e988d7091')"
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
          document.querySelector('#df-8b629fcd-3777-4ca3-a677-629e988d7091 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8b629fcd-3777-4ca3-a677-629e988d7091');
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
### 학습용 데이터프레임과 병합(join)
train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/3주차/data/application_train.csv')
train = train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')

### 결측치 -> 0으로 채우기
train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)
train.head()
```


  <div id="df-e8dc9862-86e8-4939-9533-676bd8cd1d65">
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
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
      <th>previous_loan_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 123 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e8dc9862-86e8-4939-9533-676bd8cd1d65')"
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
          document.querySelector('#df-e8dc9862-86e8-4939-9533-676bd8cd1d65 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e8dc9862-86e8-4939-9533-676bd8cd1d65');
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
  


- 스크롤을 우측 끝까지 움직여 새롭게 만들어진 column(```previous_loan_counts```)을 확인해 보세요!


## **1-2. R Value를 활용한 변수 유용성 평가**

- 새롭게 생성된 변수들이 유용한지 판단하기 위해, 우선 target 변수와 해당 변수 간의 ```피어슨 상관계수(Pearson Correlation Coefficient, r-value)```를 계산



### **📌 피어슨 상관계수(Pearson Correlation Coefficient)**

- [피어슨 상관계수](https://ko.wikipedia.org/wiki/%ED%94%BC%EC%96%B4%EC%8A%A8_%EC%83%81%EA%B4%80_%EA%B3%84%EC%88%98)

- 두 변수 사이의 **선형** 관계는 -1(완벽한 음의 선형관계)에서부터 +1(완벽한 양의 선형관계) 사이의 값으로 표현됨



-  ```r-value```가 변수의 유용성을 평가하기 위한 최선의 방식은 아니지만, 머신러닝 모델을 발전시키는 데 효과가 있을 지에 대한 대략적인 정보를 줄 수는 있음

  - 목표값에 대한 ```r-value```가 **커질수록**, 해당 변수가 목표값에 영향을 끼칠 가능성이 높아짐

  - 목표값에 대해 가장 큰 ```r-value```의 절댓값을 가지는 변수를 찾고자 함


### **📌 커널 밀도 추정 그래프(Kernel Density Estimate Plots)**

- 목표값(target)과의 상관관계를 시각화

- 단일 변수의 분포를 보여줌(히스토그램을 부드럽게 한(smoothed) 것으로 생각하면 될 듯..)

- 변수들이 머신러닝 모델과 관련성을 가지는지를 보여줄 수 있는 지표로 활용될 수 있음

---

- **범주형 변수**의 값 차이에 따른 분포 차이를 보기 위해, 카테고리에 따라 색을 다르게 칠하였음

  - ```TARGET``` 값이 0인지 1인지에 따라 색을 다르게 칠한 ```previous_loan_count```의 커널밀도추정그래프를 그릴 수 있음




```python
### 커널밀도그래프 시각화를 위한 함수

def kde_target(var_name, df):
    '''
    input:
      var_name = str, 변수가 되는 Column
      df : DataFrame, 대상 데이터프레임
        
    return: None
    '''
   
   ### 통계값 얻기(상관계수, 중간값)

    # 새롭게 생성된 변수와 target 변수 간의 상관계수 계산
    corr = df['TARGET'].corr(df[var_name])
    # 각 그룹의 중앙값 계산
    avg_repaid = df.loc[df['TARGET'] == 0, var_name].median() # 대출 상환
    avg_not_repaid = df.loc[df['TARGET'] == 1, var_name].median() # 대출 상환x
    

    ### 시각화

    # 시각화 map 설정
    plt.figure(figsize = (12, 6))
    # target값에 따라 색을 달리하여 그래프 그리기
    sns.kdeplot(df.loc[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.loc[df['TARGET'] == 1, var_name], label = 'TARGET == 1')
    # labeling
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend()
    

    ### 결과 출력

    # 상관계수 출력
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # 중간값 출력
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)
    print()
```

**```EXT_SOURCE_3``` 변수 시각화**

- [이전 노트북에 의해](https://chasubeen.github.io/ml/Home-Credit_Step1/) RandomForest 및 GradientBoostingMachine에 의해 가장 중요한 변수로 판명된 ```EXT_SOURCE_3```를 활용하여 테스트



```python
kde_target('EXT_SOURCE_3', train)
```

<pre>
The correlation between EXT_SOURCE_3 and the TARGET is -0.1789
Median value for loan that was not repaid = 0.3791
Median value for loan that was repaid =     0.5460

</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABHQAAAJMCAYAAACb9fFMAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAEAAElEQVR4nOzdd3xT1fsH8M/NaroXbaHQMspG9iwbBIpsEJAhICgKIoggIuJC8MsQvw5kKiJlqywZInsXlL2EsimrlLbpSrPv74/+mm9Lkg7apk36eb9evCj3nHvzJD0tuU/OeY6gUqlEEBERERERERGRw5AUdwBERERERERERJQ/TOgQERERERERETkYJnSIiIiIiIiIiBwMEzpERERERERERA6GCR0iIiIiIiIiIgfDhA4RERERERERkYNhQoeIiIiIiIiIyMEwoUNERERERERE5GCY0CEiIiIiIiIicjBM6BARERFRoVmzZg18fHzg4+ODu3fvFnc4eTJ27Fj4+Pigbt26Vtszn8/s2bPtHFnBzJ492xw7ERE5H1lxB0BERKXbkSNH0LNnz3yd061bN6xdu9b87z179mDAgAEAgB49emD16tU5nm80GhEREYFTp05BJpNh3759ePXVVxETE5P/J5DFwoULMXTo0AJdAwBEUcTu3buxceNGnD59GrGxsUhPT4e7uzvKli2LqlWrolGjRmjXrh0aN24MiSTnz2d0Oh02bdqEv/76C+fOncPTp09hMBgQEBCAWrVqISIiAgMGDICnp2eO15k9ezbmzp0LANi2bRvatGmT63MZO3Ys1q1bBwA4f/48KlasmK29e/fuOHbsmMV5EokEnp6eqFChApo1a4Zhw4ahUaNGuT5epkePHmHjxo04ePAgoqOjkZCQAL1eDx8fH1SvXh3NmzdH//79UbNmTYtz7969i/r16+f5sQDghRdewNGjR/N1ji0PHz7Evn37cPbsWVy4cAGxsbHm+P38/FCnTh10794dgwYNgpubW4Eey9br7+rqCi8vL3h7e6NWrVpo0KABunTpgjp16hTo8YiIiKjwcIYOERE5vM6dO2PgwIEAgO3bt2Pr1q059l+0aBFOnToFAHj33XfzffNelOLi4tCjRw+88sor+PXXX3Hz5k2kpqbCaDQiOTkZ0dHR2LlzJ2bNmoXOnTtj//79OV5vz549aNasGcaMGYPN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+Phg8+bNqFatmkNfl8qhZyNyA23atMHGjRsxdepUrFixAiaTCbVr18bkyZPvT9N/GIVCgYULF+Lrr7/G4sWL8d133yEoKAjDhg3De++9x2QOlbjiXscZGRn3nwJv2bIFW7ZskYyJjIxkQodKlD3ej4mczV7X8SeffILatWvj559/xsKFCyEIAurVq4evvvqKywapRNnjGu7UqRP+/vtvfPPNN9i7dy/i4+Oh0+lQvXp1vP322xgxYkQJvwoq7fbs2YNFixbladu7d2+e5VP3Ejr5mTFjBmrXro3ffvsN33//PXx8fNCzZ0+8//7795NDjsQZOkREREREREREboY1dIiIiIiIiIiI3AwTOkRERERERET0f+3deXDN1//H8WciEpIItUyJEEnUUrVkqIZIKTW0obRUJbbqlEnSUSrUFkNb8yVtmSAVQ6c00UlVUFU0RiOoLLWEatOq1i7UEnplE8n1+yNz77iSyCKL29/rMWNG7j3n3PdHPjPk5XzeR6yMAh0RERERERERESujQEdERERERERExMoo0BERERERERERsTIKdERERERERERErIwCHRERERERERERK6NAR0RERERERETEyijQERERERERERGxMgp0RERERERERESsjF1tFyAiIiKPL39/fw4ePFiusbdu3QIgMjKSsLAwGjZsSEpKCi1atChxfHp6Ov369SM/P59169axe/duYmNjK1yjr68vO3bsqPA8gHv37rF161Y2btzI8ePHuXHjBvXq1aNp06a4ubnh4+ND37596dOnT6lrnDp1ivXr15OYmMjFixfJzc2lSZMmdOnShaFDhzJ69Gjs7Er+J9fixYsJDw+nVatWnDhx4qG1mr4XAQEBREVFmV8/d+4cXbt2LTbe3t7eXEdAQADDhw8v88/j4MGDxMXFkZyczOXLl8nOzsbFxYV27drRp08fAgIC8PLyKrGu8jDdI5Vx9uxZYmNjSUtL49SpU9y4cYPc3FwaN25Mt27dCAwMZNiwYZVeX0RExNoo0BEREZEyubm54ebmVq6xISEhfP/996SkpDBt2jQ2btxYbExBQQFBQUHk5+czfPhwXn31VU6fPo2Pj0+xsX///TfXrl3DxcWFp59+utj7Jb1WHjk5OYwdO5aEhAQAnJyc8PT0xMnJiStXrnDgwAEOHDhAXFwcaWlpxebfu3ePRYsWsXz5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cXBwYPDgwRiNRrZu3coLL7xQ6nij0Yi/vz/JyckMHjyYr7/+usRx1RHoQNEunISEBBITE/n111+5cuUKd+7cwcnJidatW9O7d2/Gjx/PM88889B1/vzzT9atW8e+ffu4ePEieXl5NG7cmC5dujBkyBACAgKoW7dumfXs27ePmJgYfv75Z65du4bRaKRp06Z069aN0aNHM2TIEGxsbIrNK2+gA0W7XNauXYu7uztHjhwptmPowIEDxMXFkZKSwuXLl8nJycHFxYV27drh5+dHYGAgHh4eFnMqeo/4+fmVa+yDdu3aRUJCAocOHeLy5ctkZmZib2+Pq6sr3bt3JyAggL59+1ZqbREREWukQEdERERERERExMqoh46IiIiIiIiIiJVRoCMiIiIiIiIiYmXsyh4iIiIi8viKiYmp0ElZAwcOLLHJsdQMfb9ERESqhgIdERERsWoXL14kJSWl3OMfbOorNUvfLxERkaqhpsgiIiIiIiIiIlZGPXRERERERERERKyMAh0RERERERERESujQEdERERERERExMoo0BERERERERERsTIKdERERERERERErIwCHRERERERERERK6NAR0RERERERETEyijQERERERERERGxMv8Hcjr3kM0s3SIAAAAASUVORK5CYII="/>

**```previous_loan_counts``` 시각화**  

- 새로 생성한 변수에 대해 시각화



```python
kde_target('previous_loan_counts', train)
```

<pre>
The correlation between previous_loan_counts and the TARGET is -0.0100
Median value for loan that was not repaid = 3.0000
Median value for loan that was repaid =     4.0000

</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABHQAAAJMCAYAAACb9fFMAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAADTx0lEQVR4nOzdd3iUZdbH8d+UhAQSCAaIlIRIF5EqoiCgoIuKqFQpi4oFZS0ouuqqq6uwIvuqa0OsoIiAiqAoSkeQIiLNggJLKKEFCKSXyZT3j5gxwzyTOslMku/nurhM7qfdM9zJ7hzOfY4pJSXFJQAAAAAAAFQZ5kBPAAAAAAAAAKVDQAcAAAAAAKCKIaADAAAAAABQxRDQAQAAAAAAqGII6AAAAAAAAFQxBHQAAAAAAACqGAI6AAAAAAAAVQwBHQAAAAAAgCqGgA4AAAAAAEAVQ0AHAFCtREVFKSoqSlOnTg30VILOhRdeqKioKE2YMCHQUwGCwoQJExQVFaULL7ww0FMpsaJ+jr/77jv378DvvvsuALMru4EDByoqKkoDBw4M9FQAoMqwBnoCAAAAQHGioqK8xkwmk+rUqaO6desqOjpaF154obp27apBgwYpJiam8icJAEAlIkMHAAAAhqZOnerO+AhGLpdLGRkZOnr0qH7++WfNnTtXDz/8sDp06KBx48bpyJEjAZlXVc6U8aeqmAEFAFUJGToAgGolJSUl0FMAUIG6dOmi6dOnu7+32WxKTU3VwYMH9f333+vLL79Uenq6Fi1apDVr1ujdd9/VlVdeaXivGTNmaMaMGZU1db/4+eefAz2FCrFkyZJATwEAqhwCOgAAAKgyateurfbt2xseu/nmm/X888/r3//+t9566y2lpKTolltu0TfffKOOHTtW8kwBAKhYbLkCAABAtVG3bl1NmzZNTz75pCQpMzNT999/f4BnBQCA/xHQAYBq6Oy6F2lpaXr++efVs2dPNWvWTHFxcRowYIBmz54tp9Pp8z5ndx1JSEjQI488oosuukhNmzZVVFSUfvrpJ49rbDabZs6cqSFDhqht27Zq2LChWrRooWuvvVZvvvmmcnJyvJ6zYcMG93zfeuutYl/fp59+6j7/66+/9jhWki5XLpdLCxcu1MiRI9WuXTs1bNhQ5513nv7yl7/olVdeUWZmps9rS1pT5ODBg+7zPvroI8NzEhIS9Nhjj7n/Xho2bKg2bdqoZ8+euvPOOzVv3jylp6cX+ZyKsHLlSo0bN04XXHCBYmJi1Lx5c/Xt21f//ve/lZycXOS1x48f17vvvqubb75ZXbt2VZMmTdSoUSOdf/75GjVqlBYuXFjkmjOqPfLFF1/oxhtvVKtWrRQTE6POnTvrkUceUVJSkl9f99m2bdumBx98UD169FBcXJwaNmyotm3bavDgwXr11VeLfP6WLVs0YcIEderUSY0bN1ZsbKwuvfRSPf7440pMTPR5XWlqrxTV7eijjz5y3+fgwYNyOp2aPXu2rr76ap133nlq3LixevToocmTJys1NdXn9dOmTXOPFdyv8J+DBw96XLdz507df//96t69u8ff/WWXXab77rtPixYtUm5ubpGvy18eeughde3aVZK0Y8cOrVq1yuucktR4WbJkif7617+qQ4cOiomJUZMmTXThhRfqqquu0j//+U+tW7fOfW7Bz/2gQYPcY4MGDfJ63wr/TjD6ff3CCy+ob9++io+PV1RUlN544w33+aXpVlfw937NNdeoRYsWaty4sS655BL9+9//LvJ3S0lr3/harwWvad68eZKkxMREw/VTWEm7XJX1Z8vod/LatWs1evRotWvXTo0aNdIFF1ygv/3tb0pISChyDgAQLNhyBQDV3MGDBzV48GCv/4O6efNmbd68WYsWLdLcuXMVHh5e5H2++eYb3XnnncrIyPB5zq5duzRmzBjt37/fY/z06dPauHGjNm7cqPfee08ff/yxWrRo4T7es2dPxcbGKjExUZ9++qnuuuuuIufy6aefSpLOOeccXXXVVUWee7aUlBSNGTNGGzZs8Bg/c+aMfvjhB/3www966623NH/+/ArdovHFF19o/PjxXh9uT5w4oRMnTmjXrl369NNP1bBhQ5/1P/wtNzdXd999txYtWuQ1vnPnTu3cuVNvvfWWZs+ercsvv9zreofDofbt2xsGbI4dO6Zjx47pm2++0YcffqgPP/xQERERRc7H6XTqrrvu0scff+wxfuDAAb399ttavHixlixZopYtW5b+xRYhNzdXDz74oObOnet1LCkpSUlJSVqzZo1+++03r/orLpdLjz/+uGFdlt9++02//fabZs6cqVdeeUU33XSTX+ftS3Z2toYOHao1a9Z4jO/evVu7d+/WV199pSVLlqhBgwbles6bb76pxx9/3Ovvv+Dv/pdfftGHH36oH374QW3atCnXs0rCZDLp7rvv1vjx4yVJX375pfr371/i6x0Oh+68804tXLjQ61hWVpYSExO1ZcsWffzxx9qzZ49f5pyQkKAhQ4bowIED5b5XXl6ebrrpJq1YscJj/Pfff9fvv/+u+fPna/HixTrvvPPK/azK4O+frWeffVYvvfSSx9iRI0c0d+5cffnll/rss8908cUX+23+AFARCOgAQDU3btw4HThwQDfffLMGDx6s+vXr6/fff9frr7+uX375RWvWrNE999yjmTNn+rzH4cOHdeeddyo0NFRPPfWULr30UoWGhuqnn35S/fr1JeV/yL7mmmuUmpqqOnXqaNy4cbr44osVGxurtLQ0rV69Wm+//bb27t2rYcOGac2aNapXr56k/A9ew4cP10svvaQff/xR+/fv9/kh49SpU1q9erUkafDgwQoJCSnxe+FwODRq1Cht2rRJknTxxRfrrrvuUsuWLXXq1Cl9+umn+vjjj3X06FFdf/312rBhg5o2bVri+5fUiRMn9Le//U25ublq0KCBbr/9dvXo0UPR0dHKzc3VgQMHtHnz5kovEnrPPfe4gznt2rXTvffeqwsuuEBpaWlasmSJZs6cqbS0NI0YMUIrVqxQp06dPK53uVySpD59+uiqq65S+/btFR0drYyMDB04cECzZ8/WDz/8oDVr1ujhhx/Wm2++WeR8nnvuOW3evFkDBgzQ6NGj1bx5c505c0Zz587Vp59+quPHj+vee+/VN99847f3wOVy6eabb9ayZcskSXFxcbrzzjvVtWtXRURE6NSpU9q6dau++OILw+snT57s/sDZtGlTPfDAA+ratatyc3O1evVqTZ8+XdnZ2br77rsVFRWlAQMG+G3uvkycOFE//PCDRowYocGDB6tJkyY6fvy43n77ba1atUq7d+/W448/rrffftt9zcCBA9WlSxe99957eu+99yRJGzdu9Lp3kyZNJEm//PKLO5hT8J517NhR9evXV1ZWlvbt26cNGzZ4ZdRVtH79+rm/Lvi5L6mZM2e6gzk9evTQ2LFjdd555ykyMlJnzpzRb7/9pm+//Va//PKL+5omTZpo48aN2rZtm+69915J0uuvv+7OFCp8npGxY8fqyJEjuuOOO3TttdfqnHPO0cGDB92/Z0tjypQp2rZtm/r06aPbb79dzZs317FjxzRnzhwtWbJEiYmJGjx4sDZs2KA6deqU+v5FueOOO3TDDTdoypQp+vrrr9W4cWN99tln5bqnP3+2Zs+erc2bN+uSSy7RbbfdptatWyszM1NffPGF3n33XaWnp2v8+PHasmVLqf43BgAqGwEdAKjmtm3bphkzZmjUqFHusc6dO2vIkCEaMmSI1q9fr4ULF2rs2LG64oorDO9x8OBBxcTEaPny5WrevLl7vFu3bu6vJ0yYoNTUVJ1//vn6/PPPFRMT43GPvn376sYbb9TAgQOVkJCg1157zV3jQpJGjBjh/tfSTz75RI8++qjhXBYuXCi73e6+pjTef/9994e666+/Xu+//77M5j93H1955ZXq3r27Hn74YaWkpOjRRx/VnDlzSvWMkli2bJl7W9cXX3yhCy64wOP4xRdfrBEjRuj555833KJWEVasWKEFCxZIyv/w+vnnn3tkbfXt21f9+vXT6NGjZbPZdP/992vt2rUe97BYLPrxxx89sq8KXHbZZfrrX/+q5557Tv/5z3/08ccf6+9//3uR2TWbN2/Wo48+qn/84x8e41dccYVCQ0P10UcfadOmTfr555/91hb5vffecwdz/vKXv+iDDz7wyl7r37+/HnnkER0+fNhj/LffftPLL78sSWrZsqWWL1+u6Oho9/GePXvq2muv1XXXXaesrCxNnDhRO3fuVK1atfwyd182b96s6dOna8yYMe6xTp066aqrrtLgwYO1du1aLVq0SFOnTnXPt2BrSuGsHV+FiKX8dex0OlWnTh2tWLHC6+f/kksu0ZgxY5SVleXxM1fRGjRooKZNm+rIkSPat29fqa4tCOZ069ZNS5YskdXq+X+b+/btq7vvvlunT592j4WEhKh9+/YeWxObN29e5HtX2G+//aaPP/7YI/Owc+fOpZp3gW3btumvf/2rXn/9dY97XXPNNe7slAMHDuill17SP//5zzI9w5eGDRuqYcOG7qC91Wot8XtgxN8/W5s3b9aYMWP02muveazHyy67TA0aNNDUqVN14MABLV++vNgtYAAQSNTQAYBq7i9/+YtHMKdAaGioXn/9dVksFkny+Nd5I08//bRHMKewTZs2uQMl06dP9/owV6BLly664447JMmrrky7du3cW5wKtlQZ+eSTTyTlf0jq0aNHkXM+2zvvvCMpv2jqq6++avjB8o477lCfPn0kSV9/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XayKU9Ay+ujRozp48GCZ7lEgIiLC/cFoyZIlys7OlsPhcLcQvvzyy9WoUaNS3/f88893f11QWNmXwu/Z2QGqwgVWiwpM7Nmzp5QzzP/geOutt2r16tXu5y5durTYrUnlFR8fr9q1a0sq/r356aef3B1xCr83Z86ccX+QuvHGG30GczIyMrR3715/TNvvCreH3rhxY6murVWrlrtrUHE/cydOnHDXYSnr+kpOTq6UQr3lzYho2bKl7rnnHn377bfun9vPP//cDzMrnsvl0ltvveX+ftCgQeW+p8ViUbdu3fSvf/1LCxYscI+f/ZqCIZOkpL/7Jd/rMDU1tch7FPd7zh/vg79+tgCgOiKgAwDV3Ny5c30eO3jwoL777jtJUu/evcuc7j9w4ED312+88UaZ7lFYQVee9PR0ff3111qzZo37X2aL6thTlMaNG6tdu3aSpC+//LLIDyoffPCBpPx/pT67o0p8fLz7a6NOXAWKar1enNDQUHc2QWlbLZeF1Wp1dztbv359kZ1dCt4bSbriiivcXxcu5lzUv9rPnj07IIWfS6JDhw7ubLaPPvqo2A+zZyt4PxISEopsdzx79myvawoUDqoWFBs3Up71VRqFOwHl5uaW+T5RUVHq1KmTJFVax6gXX3zR/R526dLF670ur4svvtgdCD37NRV+32w2m1+fW1JffPGFz59Fu92u+fPnS8rPzunYsaPH8YLfc+np6T6DNi6XyyOoZaTgfSjve+CPny0AqI4I6ABANbds2TLDjlE2m03333+/u/7NnXfeWeZnXH755e7uI2+//bY++uijIs8/cOBAkR9I+/Xr5y7Y+cknn7jnX7t2bY/gUWkVvMaUlBQ99NBDhtsJZs2apW+//VZSfjeq2NhYj+M9evSQ1Zq/Y/n111833BYzf/58ffXVVz7nsWrVKnddDyM5OTnatGmTpPwtLNHR0UW/MD8oeG8cDofuueceww/vy5cvd2//6tSpky655BL3sQYNGrjbfC9YsMDw+m3btlXaVpuyMJvNmjhxoqT8f+kfP358kdlRR44c8fj+9ttvd2cmTZo0yTDDZseOHfrvf/8rKb8I+A033OBxPCoqyt3p6KOPPjIMfuzatavS3sfCxaH379/v87wvv/yyyIyiM2fOaMeOHZKMa2f5U1pamh599FFNmTJFUn724auvvlrq+8yfP9+djWZk06ZN7oDJ2a+ppO9bRTpx4oTP2j7PP/+8/ve//0nKb/UdGhrqcbxXr17ur1955RXDe/zf//2fR2c4IwXvw8mTJz26jZWWP362AKA6ooYOAFRzXbt21d13362NGzdq8ODBqlevnvbs2aPXXnvN3T76hhtuUP/+/cv1nHfeeUf9+/dXcnKy7rnnHn322WcaNmyY2rRpI6vVqtOnT+uXX37RihUrtGHDBl133XU+ixtbrVYNHjxYb7/9tlavXu3+sDFw4ECPLSmldeutt2rBggXatGmTFixYoCNHjmj8+PE677zzlJycrAULFrj/1ToqKkrTpk3zukeDBg00ZMgQffLJJ/r22281YsQIjR8/XjExMTp27JgWLVqkTz75RJdccom+//57w3ksWLBACxYsUN++fdWvXz+1b99e9evXV1ZWlvbu3av33nvP3T3n5ptvdgeQKtJVV12lYcOGacGCBdqwYYOuuOIK3XvvvWrfvr3S0tL09ddf691335XT6VRoaKjXB2Sz2awRI0bonXfe0a+//qqrr75a99xzj1q2bKm0tDQtX75c7733nurUqaNzzz3X/WEy2Nx+++1atmyZVq5cqWXLlumSSy7RHXfcoW7duikiIkLJycnavn27Fi1apA4dOmjGjBnua88//3w98MADeumll7R792717t1bDzzwgLp06aLc3FytXr1a06dPV1ZWlkwmk1555RXVqlXLaw7jx4/X/fffr5MnT+rqq6/W3//+d7Vt21ZpaWlas2aN3n77bcXExCg0NLTMHeZKqkePHu6vH3/8cT300EM699xz3Vtp4uLiZLVa9eabb2r8+PG66qqr1KdPH7Vp00ZRUVFKS0vTL7/8onfeecedZXf77beXa05ZWVke3aVsNptSU1N18OBBbd68WV9++aU7q61evXp677333NtCS+Puu+/WP//5Tw0cOFA9evRQixYtFBYWpuTkZG3cuFHvvPOOpPzfV7fccovHtbGxsWratKmOHDmi1157TU2aNFHr1q3dWZANGzYsVcetsujatas++OADHTx4ULfffrtiY2OVlJSkOXPm6Msvv5SU//f30EMPeV3bsWNH9++wjz76SHl5eRozZozq1aungwcPat68eVq6dGmRv+ekP9eP0+nUpEmTNH78eI8AdYsWLUr0Wvz1swUA1Q0BHQCo5mbOnKkbb7xR77//vmEHqj59+nh8KC2r+Ph4rVixQjfffLN++eUXrV692qMg6dmK+zBz00036e2331ZeXp77X8nLut2qgMVi0bx58zRmzBht2LBBmzZtcmfCFNakSRPNnz9fTZs2NbzPc889px07dmjPnj1auXKlVzvmvn37atq0aR4ZLGfLy8szvLawG2+8UU899VQJX135TZ8+XQ6HQ4sWLdKuXbv0t7/9zeucunXravbs2e7tM4U9+eST+v777/Xzzz9r+/btuuOOOzyO169fX7Nnz9Zzzz0XtAEds9msDz/8UPfdd58WLFiggwcP6p///KfhuQWZNIX985//VFZWlt58800lJiYaflgOCwvTK6+8ogEDBhjed+zYsVq1apW++OIL7d27V+PHj/c4HhcXp/nz57uLaVekFi1aaPDgwVq0aJHhz/TOnTvd2SnZ2dlavHixFi9e7PN+d911l9frKa3t27erZ8+eRZ4TEhKigQMHasqUKUUWhS/OyZMnff7ulPLbX7/yyiuGPw+TJk3SQw89pIMHD2r06NEex6ZPn64xY8aUeV4l8eSTT2r69OlatWqVO+uwsGbNmmnRokU+g+TTp0/Xtddeq6SkJI9MyQIjRozQmDFjisyE6dOnj7p3764tW7bo008/9crMLE2BdH/8bAFAdUNABwCqufj4eH377beaPn26vvrqKx06dEhms1nnn3++Ro8erZtvvrnYtuQl1aJFC61bt06LFy/WF198oa1bt+rkyZOy2+2KiopSy5Yt1b17d11zzTXFfiDr1q2bWrVq5f7g37BhQ7/URIiKitJXX32lhQsX6pNPPtGOHTt0+vRp1alTR23atNHAgQN1++23F5kJ1KBBA61YsUKvvfaaFi9erEOHDqlWrVpq27atRo0apVtuuUWJiYk+r586daquuOIKfffdd/r111+VlJSkkydPymKx6Nxzz9VFF12kkSNHljtrqrRq1aqlWbNmacyYMZozZ462bNmikydPKiwsTPHx8frLX/6iCRMm+NwCVq9ePS1btkzTp0/XokWLlJCQIKvVqqZNm+ovf/mL7r77bp9BsmASHh6ud999V7fffrvmzJmjjRs3KikpSXl5eYqOjtYFF1yg/v3766abbvK61mQy6fnnn9fQoUP13nvvaePGjTpx4oSsVqtiY2N1xRVX6O6771ZcXJzP55tMJs2cOVMffvihPvroI/3+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7d27F0OHDoVSqcSOHTtKza4ZPHgwZDIZvvjiC6O/Q2NjY/HTTz+hXr16aNmypUXmJqSnp+sscid6bsXExKBv376wt7dHeHg4nJ2dsX37diQmJmLWrFkYN26ctadIldycOXMwd+5cODs7Y/To0aLBnLCwMISEhAAo+klNt27dcP78eXTq1AkvvfQSzp49i+joaDRp0gQ7d+6Eg4NDRT8GPWPGjBmDDRs2YO/evWjWrJmhneuPLOn69evo2rUr0tLS0K1bN8NWl5iYGPj4+GDfvn2GdG6uPbIUjUaDnj174siRI3B3d0ePHj3g5uaGs2fPIiYmBg4ODvj111/RtGlTAFx79PgiIyNx9OhRAMCFCxdw9uxZtGjRArVr1wZQdKra0KFDATzeOhs/fjwiIyPRoEEDdO3aFSkpKdi6dSucnJywd+9eBAYGVuwDk80o69q7cuUK2rZti4KCAvTt21d0zfj6+mLw4MFGbQsWLMDs2bPh4+ODXr16ITs7Gz///DNUKhW2bdtmCJw/KQZ0yCJOnTqFOXPm4Pjx41Cr1QgODsbYsWMRHh5u7anRM0D/xrkky5YtM/qLNCMjA59//jl27Nhh2L/ap08fTJw40XBcINGTMBfQAbj+yLKSkpLw2WefISoqCvfv34enpyd69OiB//znPya1nbj2yFIKCgqwfPlybN26FXFxcVCpVPDw8ECbNm3w4Ycfol69ekbjufbocZT2f7yBAwdixYoVhs/Lu860Wi1WrVqFtWvXGrapdujQAdOmTTO8cafnU1nX3sGDB9GzZ88S79W6dWvs3LnTpH3Tpk1YsWIFLl26BLlcjhYtWmDy5MkmtWefBAM6RERERERERESVDGvoEBERERERERFVMgzoEBERERERERFVMgzoEBERERERERFVMgzoEBERERERERFVMgzoEBERERERERFVMgzoEBERERERERFVMgzoEBERERERERFVMgzoEBERERERERFVMgzoEBERERERERFVMgzoEBERkcWtW7cOSqUSYWFh1p5KhQgLC4NSqcS6deusPRUiIiJ6TsisPQEiIiIiIlsUGxuLnTt3wtfXF4MHD7b2dIiIiIwwQ4eIiIgsztXVFUFBQahVq5a1p0L02M6dO4e5c+di/fr11p4KERGRCWboEBERkcX17NkTPXv2tPY0iIiIiJ5ZzNAhIiIiIiIiIqpk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"/>

- 새롭게 생성된 변수(```previous_loan_counts```)가 중요하지 않음을 알 수 있음

  - 상관계수가 너무 작음

  - target 값에 따른 분포의 차이도 거의 없음


## **1-3. 새로운 변수 생성하기**


### **a) 수치형 변수들의 대표값 계산**

- ```agg()```를 활용하여 데이터 프레임의 평균, 최대값, 최소값, 합계 등을 구할 수 있음

  - 별도의 함수를 작성한 후 이를 불러오는 것도 가능함

- **bureau** 데이터 프레임 안의 수치형 변수들을 활용하기 위해, 모든 수치형 변수들의 대표값들을 계산

  - 이를 위해 고객 ID(```SK_ID_CURR```)별로 ```groupby()```하고, 그룹화된 데이터 프레임의 대표값들을 ```agg()```를 통해 구한 뒤, 결과를 train 데이터 셋과 병합(```merge()```)



```python
### 대표값 계산
# 고객id에 따라 데이터 프레임을 그룹화하여 계산

bureau_agg = bureau.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
bureau_agg.head()
```


  <div id="df-17c4449c-246d-4d4e-af2c-7a5981ea598e">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>SK_ID_CURR</th>
      <th colspan="5" halign="left">DAYS_CREDIT</th>
      <th colspan="4" halign="left">CREDIT_DAY_OVERDUE</th>
      <th>...</th>
      <th colspan="5" halign="left">DAYS_CREDIT_UPDATE</th>
      <th colspan="5" halign="left">AMT_ANNUITY</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>max</th>
      <th>min</th>
      <th>sum</th>
      <th>count</th>
      <th>mean</th>
      <th>max</th>
      <th>min</th>
      <th>...</th>
      <th>count</th>
      <th>mean</th>
      <th>max</th>
      <th>min</th>
      <th>sum</th>
      <th>count</th>
      <th>mean</th>
      <th>max</th>
      <th>min</th>
      <th>sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>7</td>
      <td>-735.000000</td>
      <td>-49</td>
      <td>-1572</td>
      <td>-5145</td>
      <td>7</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>-93.142857</td>
      <td>-6</td>
      <td>-155</td>
      <td>-652</td>
      <td>7</td>
      <td>3545.357143</td>
      <td>10822.5</td>
      <td>0.0</td>
      <td>24817.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>8</td>
      <td>-874.000000</td>
      <td>-103</td>
      <td>-1437</td>
      <td>-6992</td>
      <td>8</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>-499.875000</td>
      <td>-7</td>
      <td>-1185</td>
      <td>-3999</td>
      <td>7</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>4</td>
      <td>-1400.750000</td>
      <td>-606</td>
      <td>-2586</td>
      <td>-5603</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>-816.000000</td>
      <td>-43</td>
      <td>-2131</td>
      <td>-3264</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>2</td>
      <td>-867.000000</td>
      <td>-408</td>
      <td>-1326</td>
      <td>-1734</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>-532.000000</td>
      <td>-382</td>
      <td>-682</td>
      <td>-1064</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>3</td>
      <td>-190.666667</td>
      <td>-62</td>
      <td>-373</td>
      <td>-572</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>-54.333333</td>
      <td>-11</td>
      <td>-121</td>
      <td>-163</td>
      <td>3</td>
      <td>1420.500000</td>
      <td>4261.5</td>
      <td>0.0</td>
      <td>4261.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-17c4449c-246d-4d4e-af2c-7a5981ea598e')"
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
          document.querySelector('#df-17c4449c-246d-4d4e-af2c-7a5981ea598e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-17c4449c-246d-4d4e-af2c-7a5981ea598e');
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
  


- 새로 생성된 column들에 대해 새롭게 이름을 적어주는 것이 좋을 것 같음

  - 밑의 코드들은 원본데이터의 Column에 대표값들의 종류들을 추가적으로 기입해주는 역할을 수행

- 여기서는 Multi-Level index 데이터프레임을 작업의 대상으로 함

  - 이러한 부분은 혼동을 줄 수 있고, 작업하기도 어렵기 때문에, single-level index로 최대한 빠르게 변환하고자 하였음



```python
### 컬럼명 재정의

# 컬럼명을 저장한 리스트
columns = ['SK_ID_CURR']

for var in bureau_agg.columns.levels[0]: # 원본 컬럼(변수)들만 가져옴
                                         # min, max 이런것들 없앰
    # id 컬럼은 생략
    if var != 'SK_ID_CURR':
        # 대표값의 종류에 따라 반복문을 생성
        for stat in bureau_agg.columns.levels[1][:-1]:
            # 변수 및 대표값의 종류에 따라 새로운 column name을 생성
            columns.append('bureau_%s_%s' % (var, stat)) # bureau_변수_대표값
```


```python
### 생성된 list를 데이터 프레임의 column name으로 지정

bureau_agg.columns = columns
bureau_agg.head()
```


  <div id="df-a6ec96e9-bf79-49cc-90e8-2f800bc8fb5f">
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
      <th>SK_ID_CURR</th>
      <th>bureau_DAYS_CREDIT_count</th>
      <th>bureau_DAYS_CREDIT_mean</th>
      <th>bureau_DAYS_CREDIT_max</th>
      <th>bureau_DAYS_CREDIT_min</th>
      <th>bureau_DAYS_CREDIT_sum</th>
      <th>bureau_CREDIT_DAY_OVERDUE_count</th>
      <th>bureau_CREDIT_DAY_OVERDUE_mean</th>
      <th>bureau_CREDIT_DAY_OVERDUE_max</th>
      <th>bureau_CREDIT_DAY_OVERDUE_min</th>
      <th>...</th>
      <th>bureau_DAYS_CREDIT_UPDATE_count</th>
      <th>bureau_DAYS_CREDIT_UPDATE_mean</th>
      <th>bureau_DAYS_CREDIT_UPDATE_max</th>
      <th>bureau_DAYS_CREDIT_UPDATE_min</th>
      <th>bureau_DAYS_CREDIT_UPDATE_sum</th>
      <th>bureau_AMT_ANNUITY_count</th>
      <th>bureau_AMT_ANNUITY_mean</th>
      <th>bureau_AMT_ANNUITY_max</th>
      <th>bureau_AMT_ANNUITY_min</th>
      <th>bureau_AMT_ANNUITY_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>7</td>
      <td>-735.000000</td>
      <td>-49</td>
      <td>-1572</td>
      <td>-5145</td>
      <td>7</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>-93.142857</td>
      <td>-6</td>
      <td>-155</td>
      <td>-652</td>
      <td>7</td>
      <td>3545.357143</td>
      <td>10822.5</td>
      <td>0.0</td>
      <td>24817.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>8</td>
      <td>-874.000000</td>
      <td>-103</td>
      <td>-1437</td>
      <td>-6992</td>
      <td>8</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>-499.875000</td>
      <td>-7</td>
      <td>-1185</td>
      <td>-3999</td>
      <td>7</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>4</td>
      <td>-1400.750000</td>
      <td>-606</td>
      <td>-2586</td>
      <td>-5603</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>-816.000000</td>
      <td>-43</td>
      <td>-2131</td>
      <td>-3264</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>2</td>
      <td>-867.000000</td>
      <td>-408</td>
      <td>-1326</td>
      <td>-1734</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>-532.000000</td>
      <td>-382</td>
      <td>-682</td>
      <td>-1064</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>3</td>
      <td>-190.666667</td>
      <td>-62</td>
      <td>-373</td>
      <td>-572</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>-54.333333</td>
      <td>-11</td>
      <td>-121</td>
      <td>-163</td>
      <td>3</td>
      <td>1420.500000</td>
      <td>4261.5</td>
      <td>0.0</td>
      <td>4261.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a6ec96e9-bf79-49cc-90e8-2f800bc8fb5f')"
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
          document.querySelector('#df-a6ec96e9-bf79-49cc-90e8-2f800bc8fb5f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a6ec96e9-bf79-49cc-90e8-2f800bc8fb5f');
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
### 훈련 데이터와 병합

train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
train.head()
```


  <div id="df-a88bad0c-f6c9-46ee-8d3f-10d5ed949b75">
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
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>bureau_DAYS_CREDIT_UPDATE_count</th>
      <th>bureau_DAYS_CREDIT_UPDATE_mean</th>
      <th>bureau_DAYS_CREDIT_UPDATE_max</th>
      <th>bureau_DAYS_CREDIT_UPDATE_min</th>
      <th>bureau_DAYS_CREDIT_UPDATE_sum</th>
      <th>bureau_AMT_ANNUITY_count</th>
      <th>bureau_AMT_ANNUITY_mean</th>
      <th>bureau_AMT_ANNUITY_max</th>
      <th>bureau_AMT_ANNUITY_min</th>
      <th>bureau_AMT_ANNUITY_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>...</td>
      <td>8.0</td>
      <td>-499.875</td>
      <td>-7.0</td>
      <td>-1185.0</td>
      <td>-3999.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>...</td>
      <td>4.0</td>
      <td>-816.000</td>
      <td>-43.0</td>
      <td>-2131.0</td>
      <td>-3264.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>-532.000</td>
      <td>-382.0</td>
      <td>-682.0</td>
      <td>-1064.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>...</td>
      <td>1.0</td>
      <td>-783.000</td>
      <td>-783.0</td>
      <td>-783.0</td>
      <td>-783.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 183 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a88bad0c-f6c9-46ee-8d3f-10d5ed949b75')"
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
          document.querySelector('#df-a88bad0c-f6c9-46ee-8d3f-10d5ed949b75 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a88bad0c-f6c9-46ee-8d3f-10d5ed949b75');
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
  


### **b) target과 대표값들의 상관계수 분석**

- 새롭게 생성된 값들과 목표값과의 **상관 계수**를 분석



```python
# 새로 생성된 변수들에 대한 상관계수를 저장할 리스트
new_corrs = []

# 변수별로 반복문을 수행하며..
for col in columns:
    # target과의 상관계수 계산
    corr = train['TARGET'].corr(train[col])
    # 튜플(tuple)로 리스트에 추가
    new_corrs.append((col, corr))
```

- 상관계수들을 **절댓값**에 따라서 정렬

  - ```sorted()``` 함수 활용




```python
### 상관계수들을 절대값에 따라 정렬

new_corrs = sorted(new_corrs, key = lambda x: abs(x[1]), reverse = True) # 내림차순 정렬
new_corrs[:15] # 상관도가 높은 상위 15개 변수만 추출
```

<pre>
[('bureau_DAYS_CREDIT_mean', 0.08972896721998114),
 ('bureau_DAYS_CREDIT_min', 0.0752482510301036),
 ('bureau_DAYS_CREDIT_UPDATE_mean', 0.06892735266968673),
 ('bureau_DAYS_ENDDATE_FACT_min', 0.05588737984392077),
 ('bureau_DAYS_CREDIT_ENDDATE_sum', 0.0537348956010205),
 ('bureau_DAYS_ENDDATE_FACT_mean', 0.05319962585758616),
 ('bureau_DAYS_CREDIT_max', 0.04978205463997299),
 ('bureau_DAYS_ENDDATE_FACT_sum', 0.048853502611115894),
 ('bureau_DAYS_CREDIT_ENDDATE_mean', 0.046982754334835494),
 ('bureau_DAYS_CREDIT_UPDATE_min', 0.042863922470730155),
 ('bureau_DAYS_CREDIT_sum', 0.041999824814846716),
 ('bureau_DAYS_CREDIT_UPDATE_sum', 0.04140363535306002),
 ('bureau_DAYS_CREDIT_ENDDATE_max', 0.036589634696329094),
 ('bureau_DAYS_CREDIT_ENDDATE_min', 0.034281109921616024),
 ('bureau_DAYS_ENDDATE_FACT_count', -0.030492306653325495)]
</pre>
- 새로 생성된 변수들 중 대상과 유의한 상관 관계가 있는 변수가 없음




```python
### 가장 상관도가 높은 bureau_DAYS_CREDIT_mean 변수 시각화

kde_target('bureau_DAYS_CREDIT_mean', train)
```

<pre>
The correlation between bureau_DAYS_CREDIT_mean and the TARGET is 0.0897
Median value for loan that was not repaid = -835.3333
Median value for loan that was repaid =     -1067.0000

</pre>
<img src="data:image/png;base64,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pk2blmWtIrlcji5duqBSpUrZPp+cSk1NxbBhw2A0GiGTybBmzRqrCaMM/v7+mDVrFn777TcolcpcP2aVKlXQvXt3AOl1pA4fPvxCsec1QRDQqVMn7Ny50zxiaebMmbh8+bKdI6OC6kWvR0REVDQxaURE5CA0Gg2+//57NGrUCKVKlUJwcDDatm2LlStXZlmjJauVYp6V3WptGX3Tpk0DkD7SY8iQIXj11Vfh7++P4OBgi31iYmIwbdo0tGrVCqVLl4afnx8qVqyIfv36YevWrVnGk5ycjA0bNmD06NFo0qQJgoOD4evri7Jly6J9+/aYN28ekpKSrO4fERFhjnn16tVZPpajrDozfvx48/Sf8+fPY+/evVlunzE1rVatWihfvrx5Cpler8f69esl93F2dsa8efMgCAISExPxwQcfZOo/e/asebpJixYtMHDgQItjZCSavL297bK636pVq/D48WMA6VPAsitO/awuXbrA1dX1hR43o4g0ALvVerHmlVdeweeffw4gvU7NrFmz7BxR7kj9vm7evBldu3ZFuXLlEBgYiMaNG2PJkiWZanaJoog///wTHTp0QLly5VCiRAk0a9YMy5YtgyiK2T5uYmIi5s6di3bt2qFcuXLw8/ND+fLl0b17d4SFhcFoNFrdV6fT4Z9//sFHH32Eli1bIiQkBL6+vihdujRatWqFadOmWU3eZnj++nzr1i188MEHqFatGgICAlC2bFn06tULBw4cyMnLmKdycj3KSc2dbdu24a233sKrr76KgIAABAYGomrVqmjTpg0+//xzHDx40LxtxnnwbDH5Tp06mc8NqWv68+9dGo0GM2bMQPPmzREaGgovLy8sXLjQvH1O3xOB9OmgK1euRLt27VCmTBmUKFECDRo0wLfffovExESr++W0FpG12k0ZzynjGh8ZGWnxGjx/7c3p+9ipU6cwcuRIVK9eHSVKlEBQUBAaNmyITz/9FJGRkVb3k/odPXDgAPr164eKFSvC398fVapUwXvvvZejmnlEVHhwehoRkQOIiIhA165dLT6InThxAidOnMDGjRsRFhYmuTJUfvj2228xY8aMTDdlTk5OmbbZuHEjxowZY/HB+vHjx9i+fTu2b9+Odu3a4ddff5W8ie/Vq5fkik6xsbE4evQojh49il9//RV//vknXnnllTx6ZvYlCAJGjBiB4cOHAwC2bNmCVq1aSW5769YtnDp1CgDMI4zKlCmDunXr4tSpU1izZg3effddyX0bNWqEd955B7/88gt2796NtWvXok+fPtDr9Rg1ahSMRiPc3NwwZ84cyf1VKhWA9GLYcXFxKFas2Es979zKuGERBCFHN355RS6Xm/+tUDjeR6S33noLU6dOhUajwY4dO2AwGBwyzpwYP368uZZShsuXL2PixIk4fPgwVqxYAYPBgOHDh2PTpk2Ztrt48SI+/PBDXLhwweo5DABHjhzB22+/jejo6Ezt0dHR2Lt3L/bu3YsVK1YgLCws0zS9DGPHjpVc4S4uLg5nzpzBmTNn8MsvvyAsLAwNGjTI9jlv3boV7777LpKTk81taWlp2LVrF3bt2oUffvjBfG2whdxcj6QYjUYMGzYMGzZssOhLSUlBZGQkTp06hT/++AM3btzIk5hv376Nbt264e7duy99LL1ej969e2P37t2Z2q9du4Zr165h7dq12Lx5M0qXLv3Sj2ULoiji008/laxBdfXqVVy9ehXLli3DnDlzMtWws+abb76xSE4/ePAAYWFh2LJlC/76668crepIRAVfwfykQURUyAwePBh3797FwIED0bVrVxQrVgzXrl3D/Pnz8d9//2H//v14//33s63Xkhe2bt2Ky5cvo1KlShg5ciSqVKmCtLQ0nDlzxrzNpk2bMGTIEIiiiFKlSmH48OHmbyIfPXqE9evX46+//sI///yD999/HytWrLB4HKPRiMqVK6N9+/aoUaMGSpQoAVEUERkZia1bt2Ljxo2IiIhA//79cejQIYukVUH12muvmf997Ngxq9tl3KwqFArztCkgfZraqVOncP78eVy9etXqtLAvv/wSO3fuxL179zBp0iS89tprWLZsGa5cuQIA+OKLLyTr6ADpI24uX74MURTx/vvvY8mSJdkWc84rGo3GXMi2XLlyKFOmjE0eF0i/WcwgNbLO3lxcXNCgQQPs2rULycnJuHDhAmrXrm3vsHJt+fLlOH36NF5//XUMGDAAQUFBePDgAWbPno3Tp09jy5YtWL16Nf777z9s2rQJPXv2RI8ePRAQEIDbt2/j+++/x40bN/Dbb7+hU6dOkkWcT506ha5du0Kn08Hb2xvDhg1D9erVERgYiNjYWGzbtg2//fYbTp48if79+2Pr1q0W0xqNRiNCQ0PRsWNH1K5dG6VKlYJC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"/>

- 해당 column은 '고객이 신용관리국에 신용등급을 신청한 날로부터 현 대출 신청까지 걸린 기간은 몇 일인가?'에 관한 데이터임

- 이전 대출을 받고나서 'Home Credit'에서 대출을 받기전까지 걸린 일수로 해석

  - 마이너스 수치가 크다는 것은 이전 대출이 이루어진 시점이 더 오래됐음을 의미

  - 더 오래된 과거에 대출을 신청했던 고객들은 'Home Credit'에서 대출을 상환할 가능성이 높다는 것을 파악할 수 있음(약한 양의 상관관계)

    - 하지만 너무 낮은 상관관계 -> 그저 노이즈일 수도 있다.



#### **✔ 다중 비교 문제(Multiple Comparisons Problem)**

- 변수가 매우 많을 때는, **다중 비교 문제**로 알려진 우연에 의해 변수들이 목표값(target)에 대해 연관성을 가질 것으로 기대할 수 있음

- 수백개의 변수(feature)들을 만들 수 있지만, 몇개의 경우 그저 데이터 안에 랜덤하게 있는 노이즈값들에 의해 목표값(target)과 연관성을 가지는 것처럼 보일 수 있음

  - 이러한 변수들은 train 데이터 상에서는 target 값과 관련성이 있는 것처럼 보이지만, 관련성들이 테스트 데이터까지 일반화 될 수 없기 때문에 모델에 있어 overfitting을 야기할 수 있음

  - feature들을 만드는 것에 대한 많은 고려 필요

- [다중 비교 문제](https://ko.wikipedia.org/wiki/%EB%8B%A4%EC%A4%91_%EB%B9%84%EA%B5%90)


### **c) 수치형 변수들의 대표값 연산을 위한 함수 생성**

- 이전의 작업들을 요약하여, 수치형 변수들의 대표값 계산을 위한 함수를 생성

- 데이터 프레임 전체에 걸쳐 수치형 변수들의 대표값들을 계산하는 역할을 수행

---

- **파라미터(Parameters)>** 

  - ```df(dataframe)```: 연산의 대상이 되는 데이터 프레임

  - ```group_var(string)```: 그룹화(groupby)의 기준이 되는 column 

  - ```df_name(string)```: column명을 재정의하는데 쓰이는 변수



- **출력값(Returns)>**

  - ```agg (dataframe)```

    - 모든 수치형 변수들의 대표값들이 연산된 데이터프레임

    - 각각의 그룹화된 인스턴스들은 대표값(평균, 최소값, 최대값, 합계)들을 가짐

    - 새롭게 생성된 feature들을 구분하기위해 column들의 이름을 재정의    



```python
def agg_numeric(df, group_var, df_name):
    # 그룹화 대상이 아닌 id들을 제거
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids
    
    # 특정 변수들을 그룹화하고 대표값들을 계산
    agg = numeric_df.groupby(group_var).agg(['count','mean','max','min','sum']).reset_index()
    
    # 새로운 column명 생성
    columns = [group_var]
    
    # 변수들 별로..
    for var in agg.columns.levels[0]:
        # id column은 생략
        if var != group_var:
            # 종류별로 대표값 구하기
            for stat in agg.columns.levels[1][:-1]:
                # 변수 및 대표값의 종류에 따라 새로운 column name을 생성
                columns.append('%s_%s_%s' % (df_name, var, stat))
    agg.columns = columns
    
    return agg
```


```python
### 수치형 변수들의 대표값 계산

bureau_agg_new = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), 
                             group_var = 'SK_ID_CURR', 
                             df_name = 'bureau')
bureau_agg_new.head()
```


  <div id="df-ffefce43-8758-42fd-8c30-1ae41e6a6bf1">
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
      <th>SK_ID_CURR</th>
      <th>bureau_DAYS_CREDIT_count</th>
      <th>bureau_DAYS_CREDIT_mean</th>
      <th>bureau_DAYS_CREDIT_max</th>
      <th>bureau_DAYS_CREDIT_min</th>
      <th>bureau_DAYS_CREDIT_sum</th>
      <th>bureau_CREDIT_DAY_OVERDUE_count</th>
      <th>bureau_CREDIT_DAY_OVERDUE_mean</th>
      <th>bureau_CREDIT_DAY_OVERDUE_max</th>
      <th>bureau_CREDIT_DAY_OVERDUE_min</th>
      <th>...</th>
      <th>bureau_DAYS_CREDIT_UPDATE_count</th>
      <th>bureau_DAYS_CREDIT_UPDATE_mean</th>
      <th>bureau_DAYS_CREDIT_UPDATE_max</th>
      <th>bureau_DAYS_CREDIT_UPDATE_min</th>
      <th>bureau_DAYS_CREDIT_UPDATE_sum</th>
      <th>bureau_AMT_ANNUITY_count</th>
      <th>bureau_AMT_ANNUITY_mean</th>
      <th>bureau_AMT_ANNUITY_max</th>
      <th>bureau_AMT_ANNUITY_min</th>
      <th>bureau_AMT_ANNUITY_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>7</td>
      <td>-735.000000</td>
      <td>-49</td>
      <td>-1572</td>
      <td>-5145</td>
      <td>7</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>-93.142857</td>
      <td>-6</td>
      <td>-155</td>
      <td>-652</td>
      <td>7</td>
      <td>3545.357143</td>
      <td>10822.5</td>
      <td>0.0</td>
      <td>24817.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>8</td>
      <td>-874.000000</td>
      <td>-103</td>
      <td>-1437</td>
      <td>-6992</td>
      <td>8</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>-499.875000</td>
      <td>-7</td>
      <td>-1185</td>
      <td>-3999</td>
      <td>7</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>4</td>
      <td>-1400.750000</td>
      <td>-606</td>
      <td>-2586</td>
      <td>-5603</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>-816.000000</td>
      <td>-43</td>
      <td>-2131</td>
      <td>-3264</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>2</td>
      <td>-867.000000</td>
      <td>-408</td>
      <td>-1326</td>
      <td>-1734</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>-532.000000</td>
      <td>-382</td>
      <td>-682</td>
      <td>-1064</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>3</td>
      <td>-190.666667</td>
      <td>-62</td>
      <td>-373</td>
      <td>-572</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>-54.333333</td>
      <td>-11</td>
      <td>-121</td>
      <td>-163</td>
      <td>3</td>
      <td>1420.500000</td>
      <td>4261.5</td>
      <td>0.0</td>
      <td>4261.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ffefce43-8758-42fd-8c30-1ae41e6a6bf1')"
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
          document.querySelector('#df-ffefce43-8758-42fd-8c30-1ae41e6a6bf1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ffefce43-8758-42fd-8c30-1ae41e6a6bf1');
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
### 직접 만든 df

bureau_agg.head()
```


  <div id="df-e839bc58-7467-4949-a4b8-ff132b2ba90b">
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
      <th>SK_ID_CURR</th>
      <th>bureau_DAYS_CREDIT_count</th>
      <th>bureau_DAYS_CREDIT_mean</th>
      <th>bureau_DAYS_CREDIT_max</th>
      <th>bureau_DAYS_CREDIT_min</th>
      <th>bureau_DAYS_CREDIT_sum</th>
      <th>bureau_CREDIT_DAY_OVERDUE_count</th>
      <th>bureau_CREDIT_DAY_OVERDUE_mean</th>
      <th>bureau_CREDIT_DAY_OVERDUE_max</th>
      <th>bureau_CREDIT_DAY_OVERDUE_min</th>
      <th>...</th>
      <th>bureau_DAYS_CREDIT_UPDATE_count</th>
      <th>bureau_DAYS_CREDIT_UPDATE_mean</th>
      <th>bureau_DAYS_CREDIT_UPDATE_max</th>
      <th>bureau_DAYS_CREDIT_UPDATE_min</th>
      <th>bureau_DAYS_CREDIT_UPDATE_sum</th>
      <th>bureau_AMT_ANNUITY_count</th>
      <th>bureau_AMT_ANNUITY_mean</th>
      <th>bureau_AMT_ANNUITY_max</th>
      <th>bureau_AMT_ANNUITY_min</th>
      <th>bureau_AMT_ANNUITY_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>7</td>
      <td>-735.000000</td>
      <td>-49</td>
      <td>-1572</td>
      <td>-5145</td>
      <td>7</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>-93.142857</td>
      <td>-6</td>
      <td>-155</td>
      <td>-652</td>
      <td>7</td>
      <td>3545.357143</td>
      <td>10822.5</td>
      <td>0.0</td>
      <td>24817.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>8</td>
      <td>-874.000000</td>
      <td>-103</td>
      <td>-1437</td>
      <td>-6992</td>
      <td>8</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>-499.875000</td>
      <td>-7</td>
      <td>-1185</td>
      <td>-3999</td>
      <td>7</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>4</td>
      <td>-1400.750000</td>
      <td>-606</td>
      <td>-2586</td>
      <td>-5603</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>-816.000000</td>
      <td>-43</td>
      <td>-2131</td>
      <td>-3264</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>2</td>
      <td>-867.000000</td>
      <td>-408</td>
      <td>-1326</td>
      <td>-1734</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>-532.000000</td>
      <td>-382</td>
      <td>-682</td>
      <td>-1064</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>3</td>
      <td>-190.666667</td>
      <td>-62</td>
      <td>-373</td>
      <td>-572</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>-54.333333</td>
      <td>-11</td>
      <td>-121</td>
      <td>-163</td>
      <td>3</td>
      <td>1420.500000</td>
      <td>4261.5</td>
      <td>0.0</td>
      <td>4261.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e839bc58-7467-4949-a4b8-ff132b2ba90b')"
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
          document.querySelector('#df-e839bc58-7467-4949-a4b8-ff132b2ba90b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e839bc58-7467-4949-a4b8-ff132b2ba90b');
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
  


- 두 데이터프레임이 동등하게 생성되었다는 것을 확인할 수 있음


### **d) 상관계수 계산을 위한 함수**

- target과 변수 간 상관계수를 계산



```python
### 데이터프레임 상에서 target과의 상관계수를 계산하기 위한 함수

def target_corrs(df):
    # 상관관계를 저장하기 위한 리스트 생성
    corrs = []
    # 변수별로..
    for col in df.columns:
        print(col)

        # target 컬럼은 생략(자기 자신과 상관계수 구하면 1이니까..)
        if col != 'TARGET':
            # 상관계수 계산
            corr = df['TARGET'].corr(df[col])
            # 튜플(tuple)로 리스트에 추가
            corrs.append((col, corr))
            
    # 상관계수들을 절대값 크기에 따라 내림차순 정렬
    corrs = sorted(corrs, key = lambda x: abs(x[1]), reverse = True)
    
    return corrs
```

## **1-4. 범주형 변수(Categorical Variables)**

1️⃣ 주로 문자열 데이터로, 이러한 데이터들에 대해서는 평균이나 최대치 등 통계값을 활용하기가 어려움

  - 그 대신, 각 범주별로 값들의 개수를 count



👉 **Example**  

- 이러한 데이터를..



| SK_ID_CURR | Loan type |

|------------|-----------|

| 1          | home      |

| 1          | home      |

| 1          | home      |

| 1          | credit    |

| 2          | credit    |

| 3          | credit    |

| 3          | cash      |

| 3          | cash      |

| 4          | credit    |

| 4          | home      |

| 4          | home      |



-

  - 각 고객의 범주별 대출 갯수를 활용하여 다음과 같이 변경할 수 있음



| SK_ID_CURR | credit count | cash count | home count | total count |

|------------|--------------|------------|------------|-------------|

| 1          | 1            | 0          | 3          | 4           |

| 2          | 1            | 0          | 0          | 1           |

| 3          | 1            | 2          | 0          | 3           |

| 4          | 1            | 0          | 2          | 3           |



2️⃣ 그 다음, 고객별 대출 횟수를 활용하여 값들을 ```정규화(normalize)```

  - 고객별 대출 횟수의 합계가 1이 되도록 스케일 조정



| SK_ID_CURR | credit count | cash count | home count | total count | credit count norm | cash count norm | home count norm |

|------------|--------------|------------|------------|-------------|-------------------|-----------------|-----------------|

| 1          | 1            | 0          | 3          | 4           | 0.25              | 0               | 0.75            |

| 2          | 1            | 0          | 0          | 1           | 1.00              | 0               | 0               |

| 3          | 1            | 2          | 0          | 3           | 0.33              | 0.66            | 0               |

| 4          | 1            | 0          | 2          | 3           | 0.33              | 0               | 0.66            |



### **a) 범주형 변수 인코딩(Encoding)**

- 범주형 변수들을 인코딩(encoding)함으로써 데이터들이 담고 있는 정보를 확인할 수 있음

- 원-핫 인코딩(One-hot Encoding) 적용

  - 범주형 변수들(```dtype=object```)에 한하여 적용

- 



```python
### One-hot Encoding

categorical = pd.get_dummies(bureau.select_dtypes('object'))
categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
categorical.head()
```


  <div id="df-7881dbfe-3897-4870-9135-705c93647d68">
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
      <th>CREDIT_ACTIVE_Active</th>
      <th>CREDIT_ACTIVE_Bad debt</th>
      <th>CREDIT_ACTIVE_Closed</th>
      <th>CREDIT_ACTIVE_Sold</th>
      <th>CREDIT_CURRENCY_currency 1</th>
      <th>CREDIT_CURRENCY_currency 2</th>
      <th>CREDIT_CURRENCY_currency 3</th>
      <th>CREDIT_CURRENCY_currency 4</th>
      <th>CREDIT_TYPE_Another type of loan</th>
      <th>CREDIT_TYPE_Car loan</th>
      <th>...</th>
      <th>CREDIT_TYPE_Loan for business development</th>
      <th>CREDIT_TYPE_Loan for purchase of shares (margin lending)</th>
      <th>CREDIT_TYPE_Loan for the purchase of equipment</th>
      <th>CREDIT_TYPE_Loan for working capital replenishment</th>
      <th>CREDIT_TYPE_Microloan</th>
      <th>CREDIT_TYPE_Mobile operator loan</th>
      <th>CREDIT_TYPE_Mortgage</th>
      <th>CREDIT_TYPE_Real estate loan</th>
      <th>CREDIT_TYPE_Unknown type of loan</th>
      <th>SK_ID_CURR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>215354</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>215354</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>215354</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>215354</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>215354</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7881dbfe-3897-4870-9135-705c93647d68')"
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
          document.querySelector('#df-7881dbfe-3897-4870-9135-705c93647d68 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7881dbfe-3897-4870-9135-705c93647d68');
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
### 고객 id를 기준으로 그룹화

categorical_grouped = categorical.groupby('SK_ID_CURR').agg(['sum', 'mean'])
categorical_grouped.head()
```


  <div id="df-ce5c7b16-c0b0-4fa1-9ca3-5b103be0c668">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">CREDIT_ACTIVE_Active</th>
      <th colspan="2" halign="left">CREDIT_ACTIVE_Bad debt</th>
      <th colspan="2" halign="left">CREDIT_ACTIVE_Closed</th>
      <th colspan="2" halign="left">CREDIT_ACTIVE_Sold</th>
      <th colspan="2" halign="left">CREDIT_CURRENCY_currency 1</th>
      <th>...</th>
      <th colspan="2" halign="left">CREDIT_TYPE_Microloan</th>
      <th colspan="2" halign="left">CREDIT_TYPE_Mobile operator loan</th>
      <th colspan="2" halign="left">CREDIT_TYPE_Mortgage</th>
      <th colspan="2" halign="left">CREDIT_TYPE_Real estate loan</th>
      <th colspan="2" halign="left">CREDIT_TYPE_Unknown type of loan</th>
    </tr>
    <tr>
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>sum</th>
      <th>mean</th>
      <th>sum</th>
      <th>mean</th>
      <th>sum</th>
      <th>mean</th>
      <th>sum</th>
      <th>mean</th>
      <th>...</th>
      <th>sum</th>
      <th>mean</th>
      <th>sum</th>
      <th>mean</th>
      <th>sum</th>
      <th>mean</th>
      <th>sum</th>
      <th>mean</th>
      <th>sum</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100001</th>
      <td>3</td>
      <td>0.428571</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>0.571429</td>
      <td>0</td>
      <td>0.0</td>
      <td>7</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>2</td>
      <td>0.250000</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.750000</td>
      <td>0</td>
      <td>0.0</td>
      <td>8</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>1</td>
      <td>0.250000</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0.750000</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>2</td>
      <td>0.666667</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.333333</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ce5c7b16-c0b0-4fa1-9ca3-5b103be0c668')"
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
          document.querySelector('#df-ce5c7b16-c0b0-4fa1-9ca3-5b103be0c668 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ce5c7b16-c0b0-4fa1-9ca3-5b103be0c668');
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
  


- ```sum```: 고객별 해당 범주에 속한 대출의 총 횟수

- ```mean```: 횟수를 정규화시킨 것



> **원-핫 인코딩**을 통해 이러한 수치들을 쉽게 계산할 수 있음


- 여기서, 지난번 column 제목들을 재정의할 때 사용하였던 것과 비슷한 함수를 활용하겠습니다.

  - multi-level index로 작성되어 있는 column들을 다룰 것임

- 범주형 데이터가 속한 column의 이름을 차용한 first-level(level 0)을 따라 반복문을 먼저 실행하고, 그 다음 계산된 통계치들을 따라 반복문을 한번 더 실행

- 이후 level 0의 이름에 통계치의 종류를 합쳐 column 제목들을 재정의

  - 예를 들면, ```CREDIT_ACTIVE_Active```가 level 0이고, ```sum```이 level 1인 column은 ```CREDIT_ACTIVE_Active_count```로 정의됨


**함수화**



```python
### Level 0: 범주형 데이터가 속한 column명

categorical_grouped.columns.levels[0][:10]
```

<pre>
Index(['CREDIT_ACTIVE_Active', 'CREDIT_ACTIVE_Bad debt',
       'CREDIT_ACTIVE_Closed', 'CREDIT_ACTIVE_Sold',
       'CREDIT_CURRENCY_currency 1', 'CREDIT_CURRENCY_currency 2',
       'CREDIT_CURRENCY_currency 3', 'CREDIT_CURRENCY_currency 4',
       'CREDIT_TYPE_Another type of loan', 'CREDIT_TYPE_Car loan'],
      dtype='object')
</pre>

```python
### Level 1: 계산될 통계치(sum, mean)

categorical_grouped.columns.levels[1]
```

<pre>
Index(['sum', 'mean'], dtype='object')
</pre>

```python
group_var = 'SK_ID_CURR'

# 새로운 column 제목들을 저장하기 위한 리스트
columns = []

# 변수들을 순환하며...
for var in categorical_grouped.columns.levels[0]:
    # 고객 id column(그룹화 변수)은 생략
    if var != group_var:
        # 통계치의 종류에 따라 반복문 실행
        for stat in ['count', 'count_norm']:
            # 새로운 column 제목 정의
            columns.append('%s_%s' % (var, stat))
```


```python
### 변수 이름 재정의

categorical_grouped.columns = columns
categorical_grouped.head()
```


  <div id="df-1c81c224-b910-403e-995a-56caf69d2bd1">
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
      <th>CREDIT_ACTIVE_Active_count</th>
      <th>CREDIT_ACTIVE_Active_count_norm</th>
      <th>CREDIT_ACTIVE_Bad debt_count</th>
      <th>CREDIT_ACTIVE_Bad debt_count_norm</th>
      <th>CREDIT_ACTIVE_Closed_count</th>
      <th>CREDIT_ACTIVE_Closed_count_norm</th>
      <th>CREDIT_ACTIVE_Sold_count</th>
      <th>CREDIT_ACTIVE_Sold_count_norm</th>
      <th>CREDIT_CURRENCY_currency 1_count</th>
      <th>CREDIT_CURRENCY_currency 1_count_norm</th>
      <th>...</th>
      <th>CREDIT_TYPE_Microloan_count</th>
      <th>CREDIT_TYPE_Microloan_count_norm</th>
      <th>CREDIT_TYPE_Mobile operator loan_count</th>
      <th>CREDIT_TYPE_Mobile operator loan_count_norm</th>
      <th>CREDIT_TYPE_Mortgage_count</th>
      <th>CREDIT_TYPE_Mortgage_count_norm</th>
      <th>CREDIT_TYPE_Real estate loan_count</th>
      <th>CREDIT_TYPE_Real estate loan_count_norm</th>
      <th>CREDIT_TYPE_Unknown type of loan_count</th>
      <th>CREDIT_TYPE_Unknown type of loan_count_norm</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100001</th>
      <td>3</td>
      <td>0.428571</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>0.571429</td>
      <td>0</td>
      <td>0.0</td>
      <td>7</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>2</td>
      <td>0.250000</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.750000</td>
      <td>0</td>
      <td>0.0</td>
      <td>8</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>1</td>
      <td>0.250000</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0.750000</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>2</td>
      <td>0.666667</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.333333</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1c81c224-b910-403e-995a-56caf69d2bd1')"
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
          document.querySelector('#df-1c81c224-b910-403e-995a-56caf69d2bd1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1c81c224-b910-403e-995a-56caf69d2bd1');
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
  


- ```sum```: 총 횟수

- ```count_norm```: 횟수를 정규화시킨 것



```python
### 훈련 데이터와 결합

train = train.merge(categorical_grouped, left_on = 'SK_ID_CURR', 
                    right_index = True, how = 'left')
train.head()
```


  <div id="df-1019fb62-5129-45cf-91f4-6ddcaf1e9b56">
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
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>CREDIT_TYPE_Microloan_count</th>
      <th>CREDIT_TYPE_Microloan_count_norm</th>
      <th>CREDIT_TYPE_Mobile operator loan_count</th>
      <th>CREDIT_TYPE_Mobile operator loan_count_norm</th>
      <th>CREDIT_TYPE_Mortgage_count</th>
      <th>CREDIT_TYPE_Mortgage_count_norm</th>
      <th>CREDIT_TYPE_Real estate loan_count</th>
      <th>CREDIT_TYPE_Real estate loan_count_norm</th>
      <th>CREDIT_TYPE_Unknown type of loan_count</th>
      <th>CREDIT_TYPE_Unknown type of loan_count_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 229 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1019fb62-5129-45cf-91f4-6ddcaf1e9b56')"
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
          document.querySelector('#df-1019fb62-5129-45cf-91f4-6ddcaf1e9b56 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1019fb62-5129-45cf-91f4-6ddcaf1e9b56');
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
train.shape
```

<pre>
(307511, 229)
</pre>

```python
train.iloc[:10, 123:]
```


  <div id="df-d9c825f1-671b-4355-81d9-cc5c236a480e">
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
      <th>bureau_DAYS_CREDIT_count</th>
      <th>bureau_DAYS_CREDIT_mean</th>
      <th>bureau_DAYS_CREDIT_max</th>
      <th>bureau_DAYS_CREDIT_min</th>
      <th>bureau_DAYS_CREDIT_sum</th>
      <th>bureau_CREDIT_DAY_OVERDUE_count</th>
      <th>bureau_CREDIT_DAY_OVERDUE_mean</th>
      <th>bureau_CREDIT_DAY_OVERDUE_max</th>
      <th>bureau_CREDIT_DAY_OVERDUE_min</th>
      <th>bureau_CREDIT_DAY_OVERDUE_sum</th>
      <th>...</th>
      <th>CREDIT_TYPE_Microloan_count</th>
      <th>CREDIT_TYPE_Microloan_count_norm</th>
      <th>CREDIT_TYPE_Mobile operator loan_count</th>
      <th>CREDIT_TYPE_Mobile operator loan_count_norm</th>
      <th>CREDIT_TYPE_Mortgage_count</th>
      <th>CREDIT_TYPE_Mortgage_count_norm</th>
      <th>CREDIT_TYPE_Real estate loan_count</th>
      <th>CREDIT_TYPE_Real estate loan_count_norm</th>
      <th>CREDIT_TYPE_Unknown type of loan_count</th>
      <th>CREDIT_TYPE_Unknown type of loan_count_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.0</td>
      <td>-874.000000</td>
      <td>-103.0</td>
      <td>-1437.0</td>
      <td>-6992.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>-1400.750000</td>
      <td>-606.0</td>
      <td>-2586.0</td>
      <td>-5603.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>-867.000000</td>
      <td>-408.0</td>
      <td>-1326.0</td>
      <td>-1734.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>-1149.000000</td>
      <td>-1149.0</td>
      <td>-1149.0</td>
      <td>-1149.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.0</td>
      <td>-757.333333</td>
      <td>-78.0</td>
      <td>-1097.0</td>
      <td>-2272.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>18.0</td>
      <td>-1271.500000</td>
      <td>-239.0</td>
      <td>-2882.0</td>
      <td>-22887.0</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.0</td>
      <td>-1939.500000</td>
      <td>-1138.0</td>
      <td>-2741.0</td>
      <td>-3879.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.0</td>
      <td>-1773.000000</td>
      <td>-1309.0</td>
      <td>-2508.0</td>
      <td>-7092.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 106 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d9c825f1-671b-4355-81d9-cc5c236a480e')"
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
          document.querySelector('#df-d9c825f1-671b-4355-81d9-cc5c236a480e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d9c825f1-671b-4355-81d9-cc5c236a480e');
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
  


### **b) 범주형 데이터들을 처리하기 위한 함수**

- 데이터 프레임을 그룹화

- 이후 각각의 범주에 따라 ```counts```와 ```normalized_counts```를 계산

---

- **파라미터(Parameters)>**

  - ```df (dataframe)```: 연산의 대상이 되는 데이터프레임

  - ```group_var(string)```: **groupby**의 기준이되는 column

  - ```df_name(string)```: column 명을 재정의하는데 쓰이는 변수



- **출력값(Returns)>**  

  - ```categorical```: 데이터프레임 **group_var**에 대해 각 범주들의 **counts** 및 **normalized_counts** 값이 포함된 데이터 프레임   



```python
### 범주형 데이터 처리를 위한 함수

def count_categorical(df, group_var, df_name):    
    # 범주형 column들을 선택
    categorical = pd.get_dummies(df.select_dtypes('object'))
    # 확실히 id가 column에 있도록 지정하기
    categorical[group_var] = df[group_var]

    # group_var를 기준으로 그룹화하고 sum과 mean을 계산
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    
    column_names = []
    
    # level = 0의 column들에 따라 반복문을 실행
    for var in categorical.columns.levels[0]:
      # level = 1의 통계값들에 대해 반복문을 실행
        for stat in ['count', 'count_norm']:
            # 컬럼명 재정의
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical
```


```python
bureau_counts = count_categorical(bureau, 
                                  group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_counts.head()
```


  <div id="df-a9ab38c7-6a60-4694-8f2c-4df6224b5819">
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
      <th>bureau_CREDIT_ACTIVE_Active_count</th>
      <th>bureau_CREDIT_ACTIVE_Active_count_norm</th>
      <th>bureau_CREDIT_ACTIVE_Bad debt_count</th>
      <th>bureau_CREDIT_ACTIVE_Bad debt_count_norm</th>
      <th>bureau_CREDIT_ACTIVE_Closed_count</th>
      <th>bureau_CREDIT_ACTIVE_Closed_count_norm</th>
      <th>bureau_CREDIT_ACTIVE_Sold_count</th>
      <th>bureau_CREDIT_ACTIVE_Sold_count_norm</th>
      <th>bureau_CREDIT_CURRENCY_currency 1_count</th>
      <th>bureau_CREDIT_CURRENCY_currency 1_count_norm</th>
      <th>...</th>
      <th>bureau_CREDIT_TYPE_Microloan_count</th>
      <th>bureau_CREDIT_TYPE_Microloan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count_norm</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count_norm</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100001</th>
      <td>3</td>
      <td>0.428571</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>0.571429</td>
      <td>0</td>
      <td>0.0</td>
      <td>7</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>2</td>
      <td>0.250000</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.750000</td>
      <td>0</td>
      <td>0.0</td>
      <td>8</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>1</td>
      <td>0.250000</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0.750000</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>2</td>
      <td>0.666667</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.333333</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a9ab38c7-6a60-4694-8f2c-4df6224b5819')"
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
          document.querySelector('#df-a9ab38c7-6a60-4694-8f2c-4df6224b5819 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a9ab38c7-6a60-4694-8f2c-4df6224b5819');
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
  


### **c) 다른 데이터프레임에 연산 적용하기**

- **bureau_balance** 데이터프레임을 활용

  - 이 데이터프레임은 월별 각 고객의 과거 타 금융기관 대출 데이터를 포함하고 있음

- 고객들의 ID인 ```SK_ID_CURR```에 따라 그룹화하기보다, 이전 대출의 ID인 ```SK_ID_BUREAU```를 활용하여 1차 그룹화를 진행할 것임

  - 그룹화한 데이터프레임은 각각의 대출에 대한 정보를 행별로 포함할 것임

- 그 다음, ```SK_ID_CURR```을 활용하여 그룹화한 뒤, 각 고객별 대출의 대표값들을 계산

- 최종 산출물은 각각의 행에 고객별로 대출에 대한 대표값들을 포함



```python
### 데이터 불러오기

bureau_balance = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/3주차/data/bureau_balance.csv')
bureau_balance.head()
```


  <div id="df-baf4ad1e-3d22-4e97-8f42-6e8b07178311">
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
      <th>SK_ID_BUREAU</th>
      <th>MONTHS_BALANCE</th>
      <th>STATUS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5715448</td>
      <td>0</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5715448</td>
      <td>-1</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5715448</td>
      <td>-2</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5715448</td>
      <td>-3</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5715448</td>
      <td>-4</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-baf4ad1e-3d22-4e97-8f42-6e8b07178311')"
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
          document.querySelector('#df-baf4ad1e-3d22-4e97-8f42-6e8b07178311 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-baf4ad1e-3d22-4e97-8f42-6e8b07178311');
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
### 1. 각각의 이전 대출에 대한 상태 개수 파악

bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', 
                                          df_name = 'bureau_balance')
bureau_balance_counts.head()
```


  <div id="df-b4dda05f-75d7-479d-8f41-4a425042d748">
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
      <th>bureau_balance_STATUS_0_count</th>
      <th>bureau_balance_STATUS_0_count_norm</th>
      <th>bureau_balance_STATUS_1_count</th>
      <th>bureau_balance_STATUS_1_count_norm</th>
      <th>bureau_balance_STATUS_2_count</th>
      <th>bureau_balance_STATUS_2_count_norm</th>
      <th>bureau_balance_STATUS_3_count</th>
      <th>bureau_balance_STATUS_3_count_norm</th>
      <th>bureau_balance_STATUS_4_count</th>
      <th>bureau_balance_STATUS_4_count_norm</th>
      <th>bureau_balance_STATUS_5_count</th>
      <th>bureau_balance_STATUS_5_count_norm</th>
      <th>bureau_balance_STATUS_C_count</th>
      <th>bureau_balance_STATUS_C_count_norm</th>
      <th>bureau_balance_STATUS_X_count</th>
      <th>bureau_balance_STATUS_X_count_norm</th>
    </tr>
    <tr>
      <th>SK_ID_BUREAU</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5001709</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>86</td>
      <td>0.886598</td>
      <td>11</td>
      <td>0.113402</td>
    </tr>
    <tr>
      <th>5001710</th>
      <td>5</td>
      <td>0.060241</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>48</td>
      <td>0.578313</td>
      <td>30</td>
      <td>0.361446</td>
    </tr>
    <tr>
      <th>5001711</th>
      <td>3</td>
      <td>0.750000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>5001712</th>
      <td>10</td>
      <td>0.526316</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>9</td>
      <td>0.473684</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5001713</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>22</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b4dda05f-75d7-479d-8f41-4a425042d748')"
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
          document.querySelector('#df-b4dda05f-75d7-479d-8f41-4a425042d748 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b4dda05f-75d7-479d-8f41-4a425042d748');
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
### 2. 수치형 변수 처리
# MONTHS_BALACE: 신청일을 기준으로 한 남은 개월 수

## 2-1. 각각의 'SK_ID_CURR'별 대표값 계산
bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', 
                                 df_name = 'bureau_balance')
bureau_balance_agg.head()
```


  <div id="df-210ebaad-d45b-4c8f-9b60-34d4b7d104af">
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
      <th>SK_ID_BUREAU</th>
      <th>bureau_balance_MONTHS_BALANCE_count</th>
      <th>bureau_balance_MONTHS_BALANCE_mean</th>
      <th>bureau_balance_MONTHS_BALANCE_max</th>
      <th>bureau_balance_MONTHS_BALANCE_min</th>
      <th>bureau_balance_MONTHS_BALANCE_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5001709</td>
      <td>97</td>
      <td>-48.0</td>
      <td>0</td>
      <td>-96</td>
      <td>-4656</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5001710</td>
      <td>83</td>
      <td>-41.0</td>
      <td>0</td>
      <td>-82</td>
      <td>-3403</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5001711</td>
      <td>4</td>
      <td>-1.5</td>
      <td>0</td>
      <td>-3</td>
      <td>-6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5001712</td>
      <td>19</td>
      <td>-9.0</td>
      <td>0</td>
      <td>-18</td>
      <td>-171</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5001713</td>
      <td>22</td>
      <td>-10.5</td>
      <td>0</td>
      <td>-21</td>
      <td>-231</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-210ebaad-d45b-4c8f-9b60-34d4b7d104af')"
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
          document.querySelector('#df-210ebaad-d45b-4c8f-9b60-34d4b7d104af button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-210ebaad-d45b-4c8f-9b60-34d4b7d104af');
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
## 2-2. 각 고객별 계산

# 대출을 기준으로 데이터프레임을 그룹화
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, 
                                          left_on = 'SK_ID_BUREAU', how = 'outer')

# SK_ID_CURR을 포함하여 병합
bureau_by_loan = bureau_by_loan.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], 
                                      on = 'SK_ID_BUREAU', how = 'left')

bureau_by_loan.head()
```


  <div id="df-cd8ab225-7161-4bb8-a558-14dc7c94b068">
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
      <th>SK_ID_BUREAU</th>
      <th>bureau_balance_MONTHS_BALANCE_count</th>
      <th>bureau_balance_MONTHS_BALANCE_mean</th>
      <th>bureau_balance_MONTHS_BALANCE_max</th>
      <th>bureau_balance_MONTHS_BALANCE_min</th>
      <th>bureau_balance_MONTHS_BALANCE_sum</th>
      <th>bureau_balance_STATUS_0_count</th>
      <th>bureau_balance_STATUS_0_count_norm</th>
      <th>bureau_balance_STATUS_1_count</th>
      <th>bureau_balance_STATUS_1_count_norm</th>
      <th>...</th>
      <th>bureau_balance_STATUS_3_count_norm</th>
      <th>bureau_balance_STATUS_4_count</th>
      <th>bureau_balance_STATUS_4_count_norm</th>
      <th>bureau_balance_STATUS_5_count</th>
      <th>bureau_balance_STATUS_5_count_norm</th>
      <th>bureau_balance_STATUS_C_count</th>
      <th>bureau_balance_STATUS_C_count_norm</th>
      <th>bureau_balance_STATUS_X_count</th>
      <th>bureau_balance_STATUS_X_count_norm</th>
      <th>SK_ID_CURR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5001709</td>
      <td>97</td>
      <td>-48.0</td>
      <td>0</td>
      <td>-96</td>
      <td>-4656</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>86</td>
      <td>0.886598</td>
      <td>11</td>
      <td>0.113402</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5001710</td>
      <td>83</td>
      <td>-41.0</td>
      <td>0</td>
      <td>-82</td>
      <td>-3403</td>
      <td>5</td>
      <td>0.060241</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>48</td>
      <td>0.578313</td>
      <td>30</td>
      <td>0.361446</td>
      <td>162368.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5001711</td>
      <td>4</td>
      <td>-1.5</td>
      <td>0</td>
      <td>-3</td>
      <td>-6</td>
      <td>3</td>
      <td>0.750000</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.250000</td>
      <td>162368.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5001712</td>
      <td>19</td>
      <td>-9.0</td>
      <td>0</td>
      <td>-18</td>
      <td>-171</td>
      <td>10</td>
      <td>0.526316</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>9</td>
      <td>0.473684</td>
      <td>0</td>
      <td>0.000000</td>
      <td>162368.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5001713</td>
      <td>22</td>
      <td>-10.5</td>
      <td>0</td>
      <td>-21</td>
      <td>-231</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>22</td>
      <td>1.000000</td>
      <td>150635.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-cd8ab225-7161-4bb8-a558-14dc7c94b068')"
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
          document.querySelector('#df-cd8ab225-7161-4bb8-a558-14dc7c94b068 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-cd8ab225-7161-4bb8-a558-14dc7c94b068');
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
bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), 
                                       group_var = 'SK_ID_CURR', df_name = 'client')
bureau_balance_by_client.head()
```


  <div id="df-be6b9d46-a7ad-4004-b225-922b02ffd13e">
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
      <th>SK_ID_CURR</th>
      <th>client_bureau_balance_MONTHS_BALANCE_count_count</th>
      <th>client_bureau_balance_MONTHS_BALANCE_count_mean</th>
      <th>client_bureau_balance_MONTHS_BALANCE_count_max</th>
      <th>client_bureau_balance_MONTHS_BALANCE_count_min</th>
      <th>client_bureau_balance_MONTHS_BALANCE_count_sum</th>
      <th>client_bureau_balance_MONTHS_BALANCE_mean_count</th>
      <th>client_bureau_balance_MONTHS_BALANCE_mean_mean</th>
      <th>client_bureau_balance_MONTHS_BALANCE_mean_max</th>
      <th>client_bureau_balance_MONTHS_BALANCE_mean_min</th>
      <th>...</th>
      <th>client_bureau_balance_STATUS_X_count_count</th>
      <th>client_bureau_balance_STATUS_X_count_mean</th>
      <th>client_bureau_balance_STATUS_X_count_max</th>
      <th>client_bureau_balance_STATUS_X_count_min</th>
      <th>client_bureau_balance_STATUS_X_count_sum</th>
      <th>client_bureau_balance_STATUS_X_count_norm_count</th>
      <th>client_bureau_balance_STATUS_X_count_norm_mean</th>
      <th>client_bureau_balance_STATUS_X_count_norm_max</th>
      <th>client_bureau_balance_STATUS_X_count_norm_min</th>
      <th>client_bureau_balance_STATUS_X_count_norm_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001.0</td>
      <td>7</td>
      <td>24.571429</td>
      <td>52</td>
      <td>2</td>
      <td>172</td>
      <td>7</td>
      <td>-11.785714</td>
      <td>-0.5</td>
      <td>-25.5</td>
      <td>...</td>
      <td>7</td>
      <td>4.285714</td>
      <td>9</td>
      <td>0</td>
      <td>30.0</td>
      <td>7</td>
      <td>0.214590</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>1.502129</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002.0</td>
      <td>8</td>
      <td>13.750000</td>
      <td>22</td>
      <td>4</td>
      <td>110</td>
      <td>8</td>
      <td>-21.875000</td>
      <td>-1.5</td>
      <td>-39.5</td>
      <td>...</td>
      <td>8</td>
      <td>1.875000</td>
      <td>3</td>
      <td>0</td>
      <td>15.0</td>
      <td>8</td>
      <td>0.161932</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>1.295455</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100005.0</td>
      <td>3</td>
      <td>7.000000</td>
      <td>13</td>
      <td>3</td>
      <td>21</td>
      <td>3</td>
      <td>-3.000000</td>
      <td>-1.0</td>
      <td>-6.0</td>
      <td>...</td>
      <td>3</td>
      <td>0.666667</td>
      <td>1</td>
      <td>0</td>
      <td>2.0</td>
      <td>3</td>
      <td>0.136752</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.410256</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100010.0</td>
      <td>2</td>
      <td>36.000000</td>
      <td>36</td>
      <td>36</td>
      <td>72</td>
      <td>2</td>
      <td>-46.000000</td>
      <td>-19.5</td>
      <td>-72.5</td>
      <td>...</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100013.0</td>
      <td>4</td>
      <td>57.500000</td>
      <td>69</td>
      <td>40</td>
      <td>230</td>
      <td>4</td>
      <td>-28.250000</td>
      <td>-19.5</td>
      <td>-34.0</td>
      <td>...</td>
      <td>4</td>
      <td>10.250000</td>
      <td>40</td>
      <td>0</td>
      <td>41.0</td>
      <td>4</td>
      <td>0.254545</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.018182</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 106 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-be6b9d46-a7ad-4004-b225-922b02ffd13e')"
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
          document.querySelector('#df-be6b9d46-a7ad-4004-b225-922b02ffd13e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-be6b9d46-a7ad-4004-b225-922b02ffd13e');
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
  


### **📌 정리**

- **bureau_balance** 데이터 프레임은 다음과 같은 과정으로 가공됨



  1. 각각의 대출에 대해 수치형(numeric) 대표값 계산

  2. 각각의 대출에 대해 범주형(categorical) 데이터들의 개수를 파악

  3. 각각의 대출에 대한 대표값들과 갯수를 병합

  4. 각 고객별로 3의 결과에 대한 수치형 대표값 계산



- 최종 데이터프레임은 각 고객에 대한 개별 행으로 구성되며, 각 행은 이전 모든 대출들의 월별 정보들의 통계치들로 구성되어 있음

---

- ```client_bureau_balance_MONTHS_BALANCE_mean_mean```: 각각의 대출에 대한 ```MONTHS_BALANCE```의 평균값을 계산 -> 클라이언트별 대출의 평균값을 계산

- ```client_bureau_balance_STATUS_X_count_norm_sum```: 각각의 대출에 대해 ```STATUS``` == x 인것의 빈도를 총 ```STATUS``` 수로 나눈 다음, 개별 클라이언트별로 그 수를 합산


We will hold off on calculating the correlations until we have all the variables together in one dataframe. 


# **2. 이전까지 생성한 함수 활용하기**

- 모든 변수들을 초기화 한 뒤, 생성된 함수들을 사용하여 처음부터 만들어나가자.



```python
# 오래된 객체들(objects)을 제거함으로써 메모리를 확보

import gc
gc.enable()

del train, bureau, bureau_balance, bureau_agg, bureau_agg_new, bureau_balance_agg, bureau_balance_counts, bureau_by_loan, bureau_balance_by_client, bureau_counts
gc.collect()
```

<pre>
0
</pre>

```python
# 원본 데이터 다시 불러오기(초기화)

train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/3주차/data/application_train.csv')
bureau = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/3주차/data/bureau.csv')
bureau_balance = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/3주차/data/bureau_balance.csv')
```

## **📌 Bureau 데이터프레임 내 범주형 데이터의 갯수 세기**



```python
bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', 
                                  df_name = 'bureau')
bureau_counts.head()
```


  <div id="df-a4e29a4d-6344-4a41-88fa-1ea670973fcc">
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
      <th>bureau_CREDIT_ACTIVE_Active_count</th>
      <th>bureau_CREDIT_ACTIVE_Active_count_norm</th>
      <th>bureau_CREDIT_ACTIVE_Bad debt_count</th>
      <th>bureau_CREDIT_ACTIVE_Bad debt_count_norm</th>
      <th>bureau_CREDIT_ACTIVE_Closed_count</th>
      <th>bureau_CREDIT_ACTIVE_Closed_count_norm</th>
      <th>bureau_CREDIT_ACTIVE_Sold_count</th>
      <th>bureau_CREDIT_ACTIVE_Sold_count_norm</th>
      <th>bureau_CREDIT_CURRENCY_currency 1_count</th>
      <th>bureau_CREDIT_CURRENCY_currency 1_count_norm</th>
      <th>...</th>
      <th>bureau_CREDIT_TYPE_Microloan_count</th>
      <th>bureau_CREDIT_TYPE_Microloan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count_norm</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count_norm</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100001</th>
      <td>3</td>
      <td>0.428571</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>0.571429</td>
      <td>0</td>
      <td>0.0</td>
      <td>7</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>2</td>
      <td>0.250000</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.750000</td>
      <td>0</td>
      <td>0.0</td>
      <td>8</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>1</td>
      <td>0.250000</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0.750000</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>2</td>
      <td>0.666667</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.333333</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a4e29a4d-6344-4a41-88fa-1ea670973fcc')"
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
          document.querySelector('#df-a4e29a4d-6344-4a41-88fa-1ea670973fcc button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a4e29a4d-6344-4a41-88fa-1ea670973fcc');
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
  


## **📌Bureau 데이터프레임의 대표값 계산**



```python
bureau_agg = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), 
                         group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_agg.head()
```


  <div id="df-bc51607c-e873-44de-826a-ef5b414f08b6">
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
      <th>SK_ID_CURR</th>
      <th>bureau_DAYS_CREDIT_count</th>
      <th>bureau_DAYS_CREDIT_mean</th>
      <th>bureau_DAYS_CREDIT_max</th>
      <th>bureau_DAYS_CREDIT_min</th>
      <th>bureau_DAYS_CREDIT_sum</th>
      <th>bureau_CREDIT_DAY_OVERDUE_count</th>
      <th>bureau_CREDIT_DAY_OVERDUE_mean</th>
      <th>bureau_CREDIT_DAY_OVERDUE_max</th>
      <th>bureau_CREDIT_DAY_OVERDUE_min</th>
      <th>...</th>
      <th>bureau_DAYS_CREDIT_UPDATE_count</th>
      <th>bureau_DAYS_CREDIT_UPDATE_mean</th>
      <th>bureau_DAYS_CREDIT_UPDATE_max</th>
      <th>bureau_DAYS_CREDIT_UPDATE_min</th>
      <th>bureau_DAYS_CREDIT_UPDATE_sum</th>
      <th>bureau_AMT_ANNUITY_count</th>
      <th>bureau_AMT_ANNUITY_mean</th>
      <th>bureau_AMT_ANNUITY_max</th>
      <th>bureau_AMT_ANNUITY_min</th>
      <th>bureau_AMT_ANNUITY_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>7</td>
      <td>-735.000000</td>
      <td>-49</td>
      <td>-1572</td>
      <td>-5145</td>
      <td>7</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>-93.142857</td>
      <td>-6</td>
      <td>-155</td>
      <td>-652</td>
      <td>7</td>
      <td>3545.357143</td>
      <td>10822.5</td>
      <td>0.0</td>
      <td>24817.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>8</td>
      <td>-874.000000</td>
      <td>-103</td>
      <td>-1437</td>
      <td>-6992</td>
      <td>8</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>-499.875000</td>
      <td>-7</td>
      <td>-1185</td>
      <td>-3999</td>
      <td>7</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>4</td>
      <td>-1400.750000</td>
      <td>-606</td>
      <td>-2586</td>
      <td>-5603</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>-816.000000</td>
      <td>-43</td>
      <td>-2131</td>
      <td>-3264</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>2</td>
      <td>-867.000000</td>
      <td>-408</td>
      <td>-1326</td>
      <td>-1734</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>-532.000000</td>
      <td>-382</td>
      <td>-682</td>
      <td>-1064</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>3</td>
      <td>-190.666667</td>
      <td>-62</td>
      <td>-373</td>
      <td>-572</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>-54.333333</td>
      <td>-11</td>
      <td>-121</td>
      <td>-163</td>
      <td>3</td>
      <td>1420.500000</td>
      <td>4261.5</td>
      <td>0.0</td>
      <td>4261.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bc51607c-e873-44de-826a-ef5b414f08b6')"
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
          document.querySelector('#df-bc51607c-e873-44de-826a-ef5b414f08b6 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bc51607c-e873-44de-826a-ef5b414f08b6');
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
  


## **📌 Bureau Balance 데이터 프레임의 각 대출 별 범주형 데이터 개수**



```python
bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', 
                                          df_name = 'bureau_balance')
bureau_balance_counts.head()
```


  <div id="df-1b93c4e0-9563-4b33-a35a-8a49113dfb52">
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
      <th>bureau_balance_STATUS_0_count</th>
      <th>bureau_balance_STATUS_0_count_norm</th>
      <th>bureau_balance_STATUS_1_count</th>
      <th>bureau_balance_STATUS_1_count_norm</th>
      <th>bureau_balance_STATUS_2_count</th>
      <th>bureau_balance_STATUS_2_count_norm</th>
      <th>bureau_balance_STATUS_3_count</th>
      <th>bureau_balance_STATUS_3_count_norm</th>
      <th>bureau_balance_STATUS_4_count</th>
      <th>bureau_balance_STATUS_4_count_norm</th>
      <th>bureau_balance_STATUS_5_count</th>
      <th>bureau_balance_STATUS_5_count_norm</th>
      <th>bureau_balance_STATUS_C_count</th>
      <th>bureau_balance_STATUS_C_count_norm</th>
      <th>bureau_balance_STATUS_X_count</th>
      <th>bureau_balance_STATUS_X_count_norm</th>
    </tr>
    <tr>
      <th>SK_ID_BUREAU</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5001709</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>86</td>
      <td>0.886598</td>
      <td>11</td>
      <td>0.113402</td>
    </tr>
    <tr>
      <th>5001710</th>
      <td>5</td>
      <td>0.060241</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>48</td>
      <td>0.578313</td>
      <td>30</td>
      <td>0.361446</td>
    </tr>
    <tr>
      <th>5001711</th>
      <td>3</td>
      <td>0.750000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>5001712</th>
      <td>10</td>
      <td>0.526316</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>9</td>
      <td>0.473684</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5001713</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>22</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1b93c4e0-9563-4b33-a35a-8a49113dfb52')"
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
          document.querySelector('#df-1b93c4e0-9563-4b33-a35a-8a49113dfb52 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1b93c4e0-9563-4b33-a35a-8a49113dfb52');
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
  


## **📌 Bureau Balance 데이터프레임의 각 대출별 대표값**



```python
bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', 
                                 df_name = 'bureau_balance')
bureau_balance_agg.head()
```


  <div id="df-65470b4b-a355-4948-a8a1-886cef195cfa">
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
      <th>SK_ID_BUREAU</th>
      <th>bureau_balance_MONTHS_BALANCE_count</th>
      <th>bureau_balance_MONTHS_BALANCE_mean</th>
      <th>bureau_balance_MONTHS_BALANCE_max</th>
      <th>bureau_balance_MONTHS_BALANCE_min</th>
      <th>bureau_balance_MONTHS_BALANCE_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5001709</td>
      <td>97</td>
      <td>-48.0</td>
      <td>0</td>
      <td>-96</td>
      <td>-4656</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5001710</td>
      <td>83</td>
      <td>-41.0</td>
      <td>0</td>
      <td>-82</td>
      <td>-3403</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5001711</td>
      <td>4</td>
      <td>-1.5</td>
      <td>0</td>
      <td>-3</td>
      <td>-6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5001712</td>
      <td>19</td>
      <td>-9.0</td>
      <td>0</td>
      <td>-18</td>
      <td>-171</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5001713</td>
      <td>22</td>
      <td>-10.5</td>
      <td>0</td>
      <td>-21</td>
      <td>-231</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-65470b4b-a355-4948-a8a1-886cef195cfa')"
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
          document.querySelector('#df-65470b4b-a355-4948-a8a1-886cef195cfa button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-65470b4b-a355-4948-a8a1-886cef195cfa');
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
  


## **📌 Bureau Balance 데이터프레임의 고객별 대표값**



```python
# 각 대출별 데이터프레임 병합
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, 
                                          left_on = 'SK_ID_BUREAU', how = 'outer')

# 데이터 프레임에 SK_ID_CURR을 포함
bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, 
                                                              on = 'SK_ID_BUREAU', 
                                                              how = 'left')

# 고객별 대표값 계산
bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), 
                                       group_var = 'SK_ID_CURR', df_name = 'client')
```

## **📌 계산된 특성(Feature)들을 훈련용 데이터와 병합**



```python
original_features = list(train.columns) # 원래 변수들
print('Original Number of Features: ', len(original_features))
```

<pre>
Original Number of Features:  122
</pre>

```python
# bureau_count 병합
train = train.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')

# bureau 대표값 병합
train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')

# 월별, 고객별 정보 병합
train = train.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
```


```python
new_features = list(train.columns) # 변수 추가 후 변수 목록
print('Number of features using previous loans from other institutions data: ', len(new_features))
```

<pre>
Number of features using previous loans from other institutions data:  333
</pre>
- 많은 변수들이 새로 생성되었다.


# **3. Feature Engineering 결과물**

- 결측치의 비율, target과의 상관계수, 변수 간 상관계수들을 파악

  - 각 변수 간 높은 상관관계는 변수 간 **collinear** 관계를 가지는 지 여부를 보여줄 수 있으며, 이는 변수들이 서로 강한 연관관계를 가짐을 의미

  - 종종 collinear한 두 변수를 모두 갖는 것은 중복이기 때문에 하나를 제거해야할 필요가 있기도 함(**다중공선성** 문제 등)



- **feature seletion**은 모델 학습 및 일반화를 위해 변수들을 제거하는 과정

  - 필요없고, 중복인 변수들을 제거하고, 중요한 변수들을 보존하는 것이 목적

   - ```Curse of dimensionality(차원의 저주)```

    - 너무 많은 feature를 가질 때 생기는 문제(Feature 갯수가 많은 것 -> 지나치게 고차원)

    - 변수의 수가 증가함에 따라 변수와 목표값 사이의 상관관계를 학습하는데 필요한 데이터의 수가 기하급수적으로 증가

  - Feature의 수를 줄이는 것은 모델 학습과 더불어 일반화를 도울 수 있음

    - 결측치들의 백분율을 활용하여 대부분의 값이 존재하지 않는 feature들을 제거할 수 있음

    - Gradient Boosting Machine과 RandomForest 모델로부터 반환된 **feature importance**를 활용할 수 있음


## **3-1. 결측치 처리**



```python
### column별 결측치의 개수를 계산하기 위한 함수

def missing_values_table(df):
        # 결측치의 총 개수
        mis_val = df.isnull().sum()
        
        # 결측치의 비율
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # 결과 테이블
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)
        
        # 컬럼명 재정의
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # 결과를 내림차순으로 정렬
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending = False).round(1)
        
        # 요약통계량 출력
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        return mis_val_table_ren_columns
```


```python
missing_train = missing_values_table(train)
missing_train.head(10)
```

<pre>
Your selected dataframe has 333 columns.
There are 278 columns that have missing values.
</pre>

  <div id="df-82287cfb-e181-4bfc-96a4-4c51ee6ace07">
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
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bureau_AMT_ANNUITY_min</th>
      <td>227502</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>bureau_AMT_ANNUITY_max</th>
      <td>227502</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>bureau_AMT_ANNUITY_mean</th>
      <td>227502</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>client_bureau_balance_STATUS_4_count_min</th>
      <td>215280</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>client_bureau_balance_STATUS_3_count_norm_mean</th>
      <td>215280</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>client_bureau_balance_MONTHS_BALANCE_count_min</th>
      <td>215280</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>client_bureau_balance_STATUS_4_count_max</th>
      <td>215280</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>client_bureau_balance_STATUS_4_count_mean</th>
      <td>215280</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>client_bureau_balance_STATUS_3_count_norm_min</th>
      <td>215280</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>client_bureau_balance_STATUS_3_count_norm_max</th>
      <td>215280</td>
      <td>70.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-82287cfb-e181-4bfc-96a4-4c51ee6ace07')"
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
          document.querySelector('#df-82287cfb-e181-4bfc-96a4-4c51ee6ace07 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-82287cfb-e181-4bfc-96a4-4c51ee6ace07');
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
  


- 누락된 값의 비율이 높은 column이 여러 개 있음을 알 수 있음

  - 훈련 데이터 및 테스트 데이터에 있어 **90% 이상**의 누락값을 가진 column들을 제거 



```python
missing_train_vars = list(missing_train.index[missing_train['% of Total Values'] > 90])
len(missing_train_vars)
```

<pre>
0
</pre>
- 테스트 데이터에 대해서도 동일한 작업 수행


## **3-2. 테스트 데이터 처리**



```python
# 테스트 데이터 불러오기
test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/3주차/data/application_test.csv')

# bureau 데이터의 개수들을 계산한 데이터프레임과 병합
test = test.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')

# bureau 데이터의 대표값들을 계산한 데이터프레임과 병합
test = test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')

# bureau balance 데이터의 개수들을 계산한 데이터프레임과 병합
test = test.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
```


```python
print('Shape of Testing Data: ', test.shape)
```

<pre>
Shape of Testing Data:  (48744, 332)
</pre>
## **3-3. 데이터 개수 맞춰주기**

- 훈련용 데이터와 테스트 데이터프레임이 같은 column들을 가지도록 맞춰보자.

  - 여기서는 문제가 되지 않지만, 변수들을 원-핫 인코딩할 때에는 데이터프레임들이 동일한 column을 가지도록 맞춰야 함



```python
train_labels = train['TARGET']

# 데이터프레임을 align
# 'target' column은 일단 제거하고 맞춰주기
train, test = train.align(test, join = 'inner', axis = 1)

train['TARGET'] = train_labels # train data에 대해서는 다시 target값 결합
```


```python
print('Training Data Shape: ', train.shape)
print('Testing Data Shape: ', test.shape)
```

<pre>
Training Data Shape:  (307511, 333)
Testing Data Shape:  (48744, 332)
</pre>
- 데이터 형태가 통일되었음을 확인할 수 있다.


## **3-4. 결측치 처리**



```python
### 결측치 상태(?) 확인

missing_test = missing_values_table(test)
missing_test.head(10)
```

<pre>
Your selected dataframe has 332 columns.
There are 275 columns that have missing values.
</pre>

  <div id="df-f3df87af-4805-4fb3-a12c-ceeb98a7369d">
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
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>COMMONAREA_MEDI</th>
      <td>33495</td>
      <td>68.7</td>
    </tr>
    <tr>
      <th>COMMONAREA_MODE</th>
      <td>33495</td>
      <td>68.7</td>
    </tr>
    <tr>
      <th>COMMONAREA_AVG</th>
      <td>33495</td>
      <td>68.7</td>
    </tr>
    <tr>
      <th>NONLIVINGAPARTMENTS_MEDI</th>
      <td>33347</td>
      <td>68.4</td>
    </tr>
    <tr>
      <th>NONLIVINGAPARTMENTS_AVG</th>
      <td>33347</td>
      <td>68.4</td>
    </tr>
    <tr>
      <th>NONLIVINGAPARTMENTS_MODE</th>
      <td>33347</td>
      <td>68.4</td>
    </tr>
    <tr>
      <th>FONDKAPREMONT_MODE</th>
      <td>32797</td>
      <td>67.3</td>
    </tr>
    <tr>
      <th>LIVINGAPARTMENTS_MEDI</th>
      <td>32780</td>
      <td>67.2</td>
    </tr>
    <tr>
      <th>LIVINGAPARTMENTS_MODE</th>
      <td>32780</td>
      <td>67.2</td>
    </tr>
    <tr>
      <th>LIVINGAPARTMENTS_AVG</th>
      <td>32780</td>
      <td>67.2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f3df87af-4805-4fb3-a12c-ceeb98a7369d')"
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
          document.querySelector('#df-f3df87af-4805-4fb3-a12c-ceeb98a7369d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f3df87af-4805-4fb3-a12c-ceeb98a7369d');
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
missing_test_vars = list(missing_test.index[missing_test['% of Total Values'] > 90])
len(missing_test_vars)
```

<pre>
0
</pre>

```python
missing_columns = list(set(missing_test_vars + missing_train_vars))
print('There are %d columns with more than 90%% missing in either the training or testing data.' % len(missing_columns))
```

<pre>
There are 0 columns with more than 90% missing in either the training or testing data.
</pre>

```python
# 결측치가 있는 column 삭제

train = train.drop(columns = missing_columns)
test = test.drop(columns = missing_columns)
```

- 90% 이상 누락된 값을 가진 column들이 없기 때문에, 이번에는 어떠한 column들도 제거되지 않았음

  - feature selection을 위해서는 아마도 다른 방법을 적용해야 할 것 같음



```python
### 가공된 데이터 제장

train.to_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/3주차/data/train_bureau_raw.csv', index = False)
test.to_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/3주차/data/test_bureau_raw.csv', index = False)
```

## **3-5. 상관계수(Correlations)**

- target과 변수들 간의 상관계수

  - 새롭게 생성된 변수들 중 기존 훈련용 데이터(```application``` 데이터에 있던) 변수들보다 더 높은 상관계수를 가지는 변수를 찾을 수 있음



```python
# 데이터프레임상에서 모든 변수들 간의 상관계수를 계산

corrs = train.corr()
```


```python
corrs = corrs.sort_values('TARGET', ascending = False) # 내림차순 정렬

pd.DataFrame(corrs['TARGET'].head(10))
```


  <div id="df-2edf99c4-6ed7-4ea3-a21e-87279ef376cf">
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
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TARGET</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>bureau_DAYS_CREDIT_mean</th>
      <td>0.089729</td>
    </tr>
    <tr>
      <th>client_bureau_balance_MONTHS_BALANCE_min_mean</th>
      <td>0.089038</td>
    </tr>
    <tr>
      <th>DAYS_BIRTH</th>
      <td>0.078239</td>
    </tr>
    <tr>
      <th>bureau_CREDIT_ACTIVE_Active_count_norm</th>
      <td>0.077356</td>
    </tr>
    <tr>
      <th>client_bureau_balance_MONTHS_BALANCE_mean_mean</th>
      <td>0.076424</td>
    </tr>
    <tr>
      <th>bureau_DAYS_CREDIT_min</th>
      <td>0.075248</td>
    </tr>
    <tr>
      <th>client_bureau_balance_MONTHS_BALANCE_min_min</th>
      <td>0.073225</td>
    </tr>
    <tr>
      <th>client_bureau_balance_MONTHS_BALANCE_sum_mean</th>
      <td>0.072606</td>
    </tr>
    <tr>
      <th>bureau_DAYS_CREDIT_UPDATE_mean</th>
      <td>0.068927</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2edf99c4-6ed7-4ea3-a21e-87279ef376cf')"
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
          document.querySelector('#df-2edf99c4-6ed7-4ea3-a21e-87279ef376cf button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2edf99c4-6ed7-4ea3-a21e-87279ef376cf');
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
pd.DataFrame(corrs['TARGET'].dropna().tail(10))
```


  <div id="df-30d617ce-564a-4ba4-aebc-a35d0932f28a">
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
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>client_bureau_balance_MONTHS_BALANCE_count_min</th>
      <td>-0.048224</td>
    </tr>
    <tr>
      <th>client_bureau_balance_STATUS_C_count_norm_mean</th>
      <td>-0.055936</td>
    </tr>
    <tr>
      <th>client_bureau_balance_STATUS_C_count_max</th>
      <td>-0.061083</td>
    </tr>
    <tr>
      <th>client_bureau_balance_STATUS_C_count_mean</th>
      <td>-0.062954</td>
    </tr>
    <tr>
      <th>client_bureau_balance_MONTHS_BALANCE_count_max</th>
      <td>-0.068792</td>
    </tr>
    <tr>
      <th>bureau_CREDIT_ACTIVE_Closed_count_norm</th>
      <td>-0.079369</td>
    </tr>
    <tr>
      <th>client_bureau_balance_MONTHS_BALANCE_count_mean</th>
      <td>-0.080193</td>
    </tr>
    <tr>
      <th>EXT_SOURCE_1</th>
      <td>-0.155317</td>
    </tr>
    <tr>
      <th>EXT_SOURCE_2</th>
      <td>-0.160472</td>
    </tr>
    <tr>
      <th>EXT_SOURCE_3</th>
      <td>-0.178919</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-30d617ce-564a-4ba4-aebc-a35d0932f28a')"
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
          document.querySelector('#df-30d617ce-564a-4ba4-aebc-a35d0932f28a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-30d617ce-564a-4ba4-aebc-a35d0932f28a');
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
  


- target과 가장 큰 상관계수를 가지는 변수는 새롭게 **생성된** 변수

  - 그러나 상관계수가 높다는 것이 그 변수가 유용하다는 것을 의미하지는 않으며, 수백 개의 변수들을 생성했을 경우에는, 그저 random noise 때문에 상관관계에 있는 것처럼 보일 수도 있음을 주의해야 함



- 비판적으로 상관계수들을 들여다봤을 때, 그래도 새롭게 생성된 몇몇 변수들은 유용할 것처럼 보임

  - 변수들의 유용성을 평가하기 위해, 학습된 모델로부터 **feature Importance**를 살펴볼 예정

  - 새롭게 생성된 변수들에 대한 **kde 그래프**를 작성할 수 있음



```python
# 원문에는 client_bureau_balance_counts_mean 을 사용하고 있지만 해당 변수가 없어,
# 'client_bureau_balance_MONTHS_BALANCE_count_mean'으로 대체

kde_target(var_name = 'client_bureau_balance_MONTHS_BALANCE_count_mean', df = train)
```

<pre>
The correlation between client_bureau_balance_MONTHS_BALANCE_count_mean and the TARGET is -0.0802
Median value for loan that was not repaid = 19.3333
Median value for loan that was repaid =     25.1429

</pre>
<img src="data:image/png;base64,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"/>

- 이 변수는 각 고객의 **대출별 월별 기록**에 대한 평균을 의미합니다. 

  - 예를 들어, 만약 고객의 과거 월별로 3, 4, 5의 기록을 갖고 있는 3개의 대출을 갖고 있다면, 이 변수에 있어 데이터값은 4가 됨

- 분산 그래프에 기초하여, 과거 월별 평균이 높은 사람들이 Home Credit에서의 대출에서 상환을 잘하는 것으로 보임

  - 더 많은 신용 기록을 가지고 있는 고객들이 일반적으로 대출금을 상환할 가능성이 더 높다는 것을 나타낼 수 있음



```python
kde_target(var_name='bureau_CREDIT_ACTIVE_Active_count_norm', df=train)
```

<pre>
The correlation between bureau_CREDIT_ACTIVE_Active_count_norm and the TARGET is 0.0774
Median value for loan that was not repaid = 0.5000
Median value for loan that was repaid =     0.3636

</pre>
<img src="data:image/png;base64,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P0GW5VJfs2rVKst2o0ePdur47GUOtD3wwAMYM2YMIiMjrZYTEdmrW7du+OOPPxAeHg6g8L3U/BlCROQKDNAQOVB6ejpmz56Nrl27on79+oiMjET//v2xZMkSmEwm1de1adMGQUFBmDhxYon7L60WgnndrFmzABTOkR8/fjxat26N2rVrW25oikpJScGsWbPw8MMP45577kFoaChatGiBp59+Ghs2bChxPFlZWfj1118xefJkdO/eHZGRkQgJCUHjxo3x+OOPY968ecjMzFR9fVnmGDtjTva5c+ewadMmAIXBib///e82v7Zly5blLk563333Wf598+bNcu3DWfz9/bFo0SJLgcR58+ahoKDAzaMqvzVr1iA3NxcAMHLkSDRs2BCdO3cGUPqNvbvOj/Ly9/e3ZGPFxsaWOLXKzDy9qVq1aqVmcrnT5cuXceTIEQCwBNlGjBgBANi2bRvu3Llj036ys7OxcOFCDBkyBC1atEDt2rVRr149dO3aFa+//jp27NhhCViZO54ULS5qfr8q+l/RbihqXZzmzJlj2f7s2bOljvP1119HUFAQQkNDVb+3s2fP4q233kKnTp0QERGBunXron379nj11Vdx6tQpm34e5XXz5k18+OGHePjhh9GoUSOEhISgYcOG6NevH957770Sv8f4+Hj84x//QNeuXREZGYk6deqgbdu2ePnll3Hw4MESj1v8M05NSd20RJ87u3fvxtNPP205J1q1aoVXXnkF165dU3190SmSAwcOVJwXjqibUfwzPy8vD1999RV69eqFyMhI1KtXDz169MAXX3xheZ8rSVpaGmbNmoVevXqhQYMGCAsLQ6tWrfDcc89h27ZtJb62+GfwtWvX8Pe//x0dO3ZEvXr1EBQUZDnvRNu+8cYbuO+++1CnTh20adMGr776qqKg+Llz5/DKK6/gvvvus4ztzTffxO3bt8v6o7NLaGgo/vOf/1i+LvpvM1uuX5KTk/Hhhx9afl8hISFo0qQJOnfujGeeeQbff/89UlJSLNubz9t9+/YBAPbt26c4r4qf02W57iuti1NxV69etfq9NW3a1DI1W40jruuCgoKsrpMmTZqk+DkUfQ+wtR5UdnY25s2bh0cffRSNGjVC7dq10bx5c4wcORKrV68u8aFG8feUu3fvYvbs2ejSpQvq1auHyMhI9OvXDz/88EOlquFH7sEp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hyofBmiIHKRhw4bYtWsX5s+fjw0bNiA2NhYajQb33nsvnn76aTz77LMltqccOHAgtm/fjs8//xz79+9HWloaQkJC0KVLF0yaNAn3338/Zs2a5dAx9+3bFydPnsSPP/6Ibdu24fz587hz5w40Gg1CQkLQvHlzdOvWDQMHDhS2mJw/fz4eeugh/PDDDzhz5gwMBgNq166Nrl27YsKECTaPeebMmejYsSO++eYbnDp1Cnl5eQgPD0f//v0xefJkhzz9LU14eDiWLVuGU6dOYd26dfjzzz8RFxeHO3fuQKvVIjg4GG3atEHPnj0xbNgwq64m5dWxY0f07t0bO3fuxO7du3Ho0CGbp+XYS6vVIiAgAIGBgYiMjES7du3QtWtX9O/f36OnvqjJzc21zB8PCwsrc8eNJ598EufOnUNGRgbWr1+vyGRwx/nhKMOGDcO7775rmcc/atQoN4+oZOUNtAGFT/+//vpry36KBpV69uyJEydO4Pvvv8eWLVtw8eJF3L17F76+voiMjESnTp0wZMgQYbBl8eLFaN++PX777TdcvnwZmZmZpXYoKc2wYcPw/vvvW/bzyCOP2NxV680338Tw4cPxww8/YNeuXbh+/TrS09Ph6+uLOnXqoFWrVujVqxcGDRrk8HNx8uTJGDhwIL799lvs3LkTsbGxyM7ORvXq1dG0aVP06NFDNRNo+PDh6NKlCxYuXIg//vgDcXFxVp8b48ePt8pYEnnppZfQokULzJ8/H0eOHEFmZiZq166N3r17Y/LkyWjWrBk2bdrk0O9ZzfPPP4/Q0FD88MMPOH36NNLS0izThDxNz549cezYMSxcuBDbtm3DtWvXkJubi5CQEHTs2BHPPPOMpc5QVeHt7W35HGzUqBHat2+Phx56CD179rSrSO1rr72G1q1bY/fu3Th16hQSExNx+/ZtyLKM0NBQtGvXDsOGDcOTTz4pPE5AQAC2bduG//znP9i5cyfi4uJc3pnskUcewc6dO/HVV19h165dSE5ORkBAAB544AG8/PLLitozxTniuq5///747bffsHDhQhw/fhwpKSnIz88v1/fTunVrHDp0CN999x02btyIixcvIjMzE7Vq1cJ9992H4cOHY9iwYW4v0EwEAJJer3fP5EYiIiIiIiIiIgLAIsFERERERERERG7HAA0RERERERERkZsxQENERERERERE5GYVrxIkEdH/2Nr2t7jQ0FCEhoY6eDTkCAkJCdDr9WV+nU6ns7kdMzmfXq9HQkJCuV5rb5tuEuP7pWtlZWUhJiamXK9t0KABqlWr5uARERFRRcAiwURUYdna8aS4srRnJNeaOHGiVRcfW7mi9SjZbtmyZZg0aVK5Xmtvm24S4/ula+3ZswcDBw4s12vXr1+PHj16OHhERERUEXCKExERERERERGRmzGDhoiIiIiIiIjIzZhBQ0RERERERETkZgzQEBERERERERG5GQM0RC6Sm5uLa9euITc3191DoUqG5xY5A88rchaeW+QMPK/IWXhukSsxQEPkQkaj0d1DoEqK5xY5A88rchaeW+QMPK/IWXhukaswQENERERERERE5GYM0BARERERERERuRkDNEREREREREREbsYADRERERERERGRmzFAQ0RERERERETkZl7uHgARERERERF5DpPJhKysLLaWRuHPQqfT4e7du8jIyHD3cMiD+Pr6olq1atBoHJf3wgANERERERERASgMSKSmpiIgIAAhISGQJMndQ3Irk8kEg8EAnU7n0BtxqthkWUZubi5SU1MRHBzssHODZxgREREREREBALKyshAQEAA/P78qH5whUiNJEvz8/BAQEICsrCyH7ZcBGiIiIiIiIgIA5ObmwtfX193DIKoQfH19HToVkAEaIiIiIiIismDmDJFtHP23wgANEREREREREZGbMUBDRERERERERORmDNAQEREREREREbkZAzRERERERERERG7GAA0RERERERERkZt5uXsARFVBToGM7HwZsuzukRARERERUXFBQUFl2l6v11v+HRsbi3bt2sFkMuGDDz7Aa6+9JnzNnj17MHDgQKtlOp0OYWFh6NGjB6ZOnYrGjRuXeMwffvgB27Ztw6VLl6DX6+Hv74+GDRuic+fOGDFiBDp27Gj1mokTJ2LFihUlfi/z589HZGSkYmwl6datGzZu3Gjz9p4oKSkJM2fOxO+//w69Xo+IiAiMGjUKr7/+Ory9vd0yJgZoiJzsy9MZmH0iA9kFMjoF+WBBsBGNfd09KiIiIiIiMps2bZpiWVRUFNLT0zF16lRotVrVlspLly6FyWSCJElYunSpaoDGrF27dujfvz8AID09HQcPHsTy5cuxYcMG7NixA02bNlW8Zvfu3Rg/fjxSU1PRuHFjPPbYY6hduzaysrJw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"/>

- 해당 변수는 ```CREDIT_ACTIVE```의 값이 'Active'인 것의 개수를 전체 대출의 개수로 나눈 값

- 해당 변수의 경우 모든 곳에서 불규칙적임

- 상관관계 또한 매우 낮음


### **📌 Collinear Variables**

- target과 변수 간의 상관계수만 계산하는 것이 아닌, 각 변수 간 상관계수까지 계산할 수 있음

  - 이를 통해 제거해야 할 수도 있는 **collinear** 관계들을 가지는 변수들이 있는지 여부를 알려줌

- **0.8 이상**의 상관계수를 가지는 변수들을 찾아보자.



```python
# 임계값 설정
threshold = 0.8

# 상관계수가 높은 변수들을 저장하기 위한 빈 dictionary 생성
above_threshold_vars = {}

# 각각의 칼럼마다 임계치 이상의 변수들을 저장
for col in corrs:
    above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])
```

**상관계수가 높은 변수들 중 1개의 변수만 제거**



```python
# 제거할 column들 및 이미 검사된 column들의 목록을 저장위한 list 생성
cols_to_remove = []
cols_seen = []
cols_to_remove_pair = []

for key, value in above_threshold_vars.items():
    # 이미 검사된 column 저장
    cols_seen.append(key)
    for x in value:
        if x == key:
            next
        else:
            # 각 쌍 중 하나의 column만을 제거
            if x not in cols_seen:
                cols_to_remove.append(x)
                cols_to_remove_pair.append(key)
            
cols_to_remove = list(set(cols_to_remove))
print('Number of columns to remove: ', len(cols_to_remove))
```

<pre>
Number of columns to remove:  134
</pre>

```python
### 상관도가 높은 변수들을 제거

train_corrs_removed = train.drop(columns = cols_to_remove)
test_corrs_removed = test.drop(columns = cols_to_remove)

print('Training Corrs Removed Shape: ', train_corrs_removed.shape)
print('Testing Corrs Removed Shape: ', test_corrs_removed.shape)
```

<pre>
Training Corrs Removed Shape:  (307511, 199)
Testing Corrs Removed Shape:  (48744, 198)
</pre>

```python
train_corrs_removed.to_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/3주차/data/train_bureau_corrs_removed.csv', index = False)
test_corrs_removed.to_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/3주차/data/test_bureau_corrs_removed.csv', index = False)
```

# **4. 모델링(Modeling)**

- 해당 부분은 원문을 그대로 탑재하였습니다.

- 모델은 LGBM을 활용 + 교차 검증

- 위에서 가공한 데이터들로 모델을 학습/예측/평가 후 성능 비교



```python
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import gc

import matplotlib.pyplot as plt
```


```python
def model(features, test_features, encoding = 'ohe', n_folds = 5):
    
    """Train and test a light gradient boosting model using
    cross validation. 
    
    Parameters
    --------
        features (pd.DataFrame): 
            dataframe of training features to use 
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame): 
            dataframe of testing features to use
            for making predictions with the model. 
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        
    """
    
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = False, random_state = 50)
    
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics
```


```python
def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df
```

### Control



The first step in any experiment is establishing a control. For this we will use the function defined above (that implements a Gradient Boosting Machine model) and the single main data source (`application`). 



```python
train_control = pd.read_csv('../input/application_train.csv')
test_control = pd.read_csv('../input/application_test.csv')
```

Fortunately, once we have taken the time to write a function, using it is simple (if there's a central theme in this notebook, it's use functions to make things simpler and reproducible!). The function above returns a `submission` dataframe we can upload to the competition, a `fi` dataframe of feature importances, and a `metrics` dataframe with validation and test performance. 



```python
submission, fi, metrics = model(train_control, test_control)
```


```python
metrics
```

The control slightly overfits because the training score is higher than the validation score. We can address this in later notebooks when we look at regularization (we already perform some regularization in this model by using `reg_lambda` and `reg_alpha` as well as early stopping). 



We can visualize the feature importance with another function, `plot_feature_importances`. The feature importances may be useful when it's time for feature selection. 



```python
fi_sorted = plot_feature_importances(fi)
```


```python
submission.to_csv('control.csv', index = False)
```

__The control scores 0.745 when submitted to the competition.__


### Test One



Let's conduct the first test. We will just need to pass in the data to the function, which does most of the work for us.



```python
submission_raw, fi_raw, metrics_raw = model(train, test)
```


```python
metrics_raw
```

Based on these numbers, the engineered features perform better than the control case. However, we will have to submit the predictions to the leaderboard before we can say if this better validation performance transfers to the testing data. 



```python
fi_raw_sorted = plot_feature_importances(fi_raw)
```

Examining the feature improtances, it looks as if a few of the feature we constructed are among the most important. Let's find the percentage of the top 100 most important features that we made in this notebook. However, rather than just compare to the original features, we need to compare to the _one-hot encoded_ original features. These are already recorded for us in `fi` (from the original data). 



```python
top_100 = list(fi_raw_sorted['feature'])[:100]
new_features = [x for x in top_100 if x not in list(fi['feature'])]

print('%% of Top 100 Features created from the bureau data = %d.00' % len(new_features))
```

Over half of the top 100 features were made by us! That should give us confidence that all the hard work we did was worthwhile. 



```python
submission_raw.to_csv('test_one.csv', index = False)
```

__Test one scores 0.759 when submitted to the competition.__


### Test Two



That was easy, so let's do another run! Same as before but with the highly collinear variables removed. 



```python
submission_corrs, fi_corrs, metrics_corr = model(train_corrs_removed, test_corrs_removed)
```


```python
metrics_corr
```

These results are better than the control, but slightly lower than the raw features. 



```python
fi_corrs_sorted = plot_feature_importances(fi_corrs)
```


```python
submission_corrs.to_csv('test_two.csv', index = False)
```

__Test Two scores 0.753 when submitted to the competition.__


# Results



After all that work, we can say that including the extra information did improve performance! The model is definitely not optimized to our data, but we still had a noticeable improvement over the original dataset when using the calculated features. Let's officially summarize the performances:



| __Experiment__ | __Train AUC__ | __Validation AUC__ | __Test AUC__  |

|------------|-------|------------|-------|

| __Control__    | 0.815 | 0.760      | 0.745 |

| __Test One__   | 0.837 | 0.767      | 0.759 |

| __Test Two__   | 0.826 | 0.765      | 0.753 |





(Note that these scores may change from run to run of the notebook. I have not observed that the general ordering changes however.)



All of our hard work translates to a small improvement of 0.014 ROC AUC over the original testing data. Removing the highly collinear variables slightly decreases performance so we will want to consider a different method for feature selection. Moreover, we can say that some of the features we built are among the most important as judged by the model. 



In a competition such as this, even an improvement of this size is enough to move us up 100s of spots on the leaderboard. By making numerous small improvements such as in this notebook, we can gradually achieve better and better performance. I encourage others to use the results here to make their own improvements, and I will continue to document the steps I take to help others. 



## Next Steps



Going forward, we can now use the functions we developed in this notebook on the other datasets. There are still 4 other data files to use in our model! In the next notebook, we will incorporate the information from these other data files (which contain information on previous loans at Home Credit) into our training data. Then we can build the same model and run more experiments to determine the effect of our feature engineering. There is plenty more work to be done in this competition, and plenty more gains in performance to be had! I'll see you in the next notebook.

