---
layout: single
title:  "[ECC DS 2주차] 1. Data Preparation & Exploration"
categories: ML
tags: [ECC, DS, Porto Seguro] 
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


# **0. Introduction**


- PorteSeguro competition을 위해 좋은 insight를 얻는 것을 목표로 함

- 모델링을 위해 데이터를 준비하는 방법들

- 주요 섹션

  - 데이터 시각화

  - 메타데이터 정의

  - 기술통계량

  - 불균형 클래스 처리하기

  - 데이터 quality 확인

  - Exploratory data visualization

  - Feature engineering

  - Feature 선택

  - Feature scaling


# **1. Loading packages**



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer # version issue
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 100)
```

# **2. Loading data**



```python
train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/2주차/data/train.csv')
test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/2주차/data/test.csv')
```

# **3. Data at first sight**


- 대회 데이터 설명 발췌문

  - 유사한 그룹에 속하는 feature들은 feature name에 태깅되어 있음(ex> ind, reg, car, calc)

  - feature명에는 binary feature를 나타내는 postfix 빈과 범주형 feature를 나타내기 위한 cat이 포함됨

    - 이러한 지정이 없는 feature는 연속형 또는 순서형 feature이다.

  - **-1**: 결측치(missing value)

- target 열: 해당 정책 소유자에 대해 클레임이 제기되었는지의 여부


## **∎ 학습용 데이터**










```python
train.head()
```


  <div id="df-017a0273-fe42-431d-9bac-7ad766227c25">
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
      <th>id</th>
      <th>target</th>
      <th>ps_ind_01</th>
      <th>ps_ind_02_cat</th>
      <th>ps_ind_03</th>
      <th>ps_ind_04_cat</th>
      <th>ps_ind_05_cat</th>
      <th>ps_ind_06_bin</th>
      <th>ps_ind_07_bin</th>
      <th>ps_ind_08_bin</th>
      <th>ps_ind_09_bin</th>
      <th>ps_ind_10_bin</th>
      <th>ps_ind_11_bin</th>
      <th>ps_ind_12_bin</th>
      <th>ps_ind_13_bin</th>
      <th>ps_ind_14</th>
      <th>ps_ind_15</th>
      <th>ps_ind_16_bin</th>
      <th>ps_ind_17_bin</th>
      <th>ps_ind_18_bin</th>
      <th>ps_reg_01</th>
      <th>ps_reg_02</th>
      <th>ps_reg_03</th>
      <th>ps_car_01_cat</th>
      <th>ps_car_02_cat</th>
      <th>ps_car_03_cat</th>
      <th>ps_car_04_cat</th>
      <th>ps_car_05_cat</th>
      <th>ps_car_06_cat</th>
      <th>ps_car_07_cat</th>
      <th>ps_car_08_cat</th>
      <th>ps_car_09_cat</th>
      <th>ps_car_10_cat</th>
      <th>ps_car_11_cat</th>
      <th>ps_car_11</th>
      <th>ps_car_12</th>
      <th>ps_car_13</th>
      <th>ps_car_14</th>
      <th>ps_car_15</th>
      <th>ps_calc_01</th>
      <th>ps_calc_02</th>
      <th>ps_calc_03</th>
      <th>ps_calc_04</th>
      <th>ps_calc_05</th>
      <th>ps_calc_06</th>
      <th>ps_calc_07</th>
      <th>ps_calc_08</th>
      <th>ps_calc_09</th>
      <th>ps_calc_10</th>
      <th>ps_calc_11</th>
      <th>ps_calc_12</th>
      <th>ps_calc_13</th>
      <th>ps_calc_14</th>
      <th>ps_calc_15_bin</th>
      <th>ps_calc_16_bin</th>
      <th>ps_calc_17_bin</th>
      <th>ps_calc_18_bin</th>
      <th>ps_calc_19_bin</th>
      <th>ps_calc_20_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.7</td>
      <td>0.2</td>
      <td>0.718070</td>
      <td>10</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>2</td>
      <td>0.400000</td>
      <td>0.883679</td>
      <td>0.370810</td>
      <td>3.605551</td>
      <td>0.6</td>
      <td>0.5</td>
      <td>0.2</td>
      <td>3</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>5</td>
      <td>9</td>
      <td>1</td>
      <td>5</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.8</td>
      <td>0.4</td>
      <td>0.766078</td>
      <td>11</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>19</td>
      <td>3</td>
      <td>0.316228</td>
      <td>0.618817</td>
      <td>0.388716</td>
      <td>2.449490</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>5</td>
      <td>8</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>0</td>
      <td>5</td>
      <td>4</td>
      <td>9</td>
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
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.000000</td>
      <td>7</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>60</td>
      <td>1</td>
      <td>0.316228</td>
      <td>0.641586</td>
      <td>0.347275</td>
      <td>3.316625</td>
      <td>0.5</td>
      <td>0.7</td>
      <td>0.1</td>
      <td>2</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>7</td>
      <td>4</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.2</td>
      <td>0.580948</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>104</td>
      <td>1</td>
      <td>0.374166</td>
      <td>0.542949</td>
      <td>0.294958</td>
      <td>2.000000</td>
      <td>0.6</td>
      <td>0.9</td>
      <td>0.1</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>1</td>
      <td>8</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.7</td>
      <td>0.6</td>
      <td>0.840759</td>
      <td>11</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>82</td>
      <td>3</td>
      <td>0.316070</td>
      <td>0.565832</td>
      <td>0.365103</td>
      <td>2.000000</td>
      <td>0.4</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>10</td>
      <td>2</td>
      <td>12</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-017a0273-fe42-431d-9bac-7ad766227c25')"
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
          document.querySelector('#df-017a0273-fe42-431d-9bac-7ad766227c25 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-017a0273-fe42-431d-9bac-7ad766227c25');
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
train.tail()
```


  <div id="df-3baf6736-96c8-489d-9daf-8cf7c11c0428">
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
      <th>id</th>
      <th>target</th>
      <th>ps_ind_01</th>
      <th>ps_ind_02_cat</th>
      <th>ps_ind_03</th>
      <th>ps_ind_04_cat</th>
      <th>ps_ind_05_cat</th>
      <th>ps_ind_06_bin</th>
      <th>ps_ind_07_bin</th>
      <th>ps_ind_08_bin</th>
      <th>ps_ind_09_bin</th>
      <th>ps_ind_10_bin</th>
      <th>ps_ind_11_bin</th>
      <th>ps_ind_12_bin</th>
      <th>ps_ind_13_bin</th>
      <th>ps_ind_14</th>
      <th>ps_ind_15</th>
      <th>ps_ind_16_bin</th>
      <th>ps_ind_17_bin</th>
      <th>ps_ind_18_bin</th>
      <th>ps_reg_01</th>
      <th>ps_reg_02</th>
      <th>ps_reg_03</th>
      <th>ps_car_01_cat</th>
      <th>ps_car_02_cat</th>
      <th>ps_car_03_cat</th>
      <th>ps_car_04_cat</th>
      <th>ps_car_05_cat</th>
      <th>ps_car_06_cat</th>
      <th>ps_car_07_cat</th>
      <th>ps_car_08_cat</th>
      <th>ps_car_09_cat</th>
      <th>ps_car_10_cat</th>
      <th>ps_car_11_cat</th>
      <th>ps_car_11</th>
      <th>ps_car_12</th>
      <th>ps_car_13</th>
      <th>ps_car_14</th>
      <th>ps_car_15</th>
      <th>ps_calc_01</th>
      <th>ps_calc_02</th>
      <th>ps_calc_03</th>
      <th>ps_calc_04</th>
      <th>ps_calc_05</th>
      <th>ps_calc_06</th>
      <th>ps_calc_07</th>
      <th>ps_calc_08</th>
      <th>ps_calc_09</th>
      <th>ps_calc_10</th>
      <th>ps_calc_11</th>
      <th>ps_calc_12</th>
      <th>ps_calc_13</th>
      <th>ps_calc_14</th>
      <th>ps_calc_15_bin</th>
      <th>ps_calc_16_bin</th>
      <th>ps_calc_17_bin</th>
      <th>ps_calc_18_bin</th>
      <th>ps_calc_19_bin</th>
      <th>ps_calc_20_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>595207</th>
      <td>1488013</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>0.692820</td>
      <td>10</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>3</td>
      <td>0.374166</td>
      <td>0.684631</td>
      <td>0.385487</td>
      <td>2.645751</td>
      <td>0.4</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>12</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>595208</th>
      <td>1488016</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.7</td>
      <td>1.382027</td>
      <td>9</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>63</td>
      <td>2</td>
      <td>0.387298</td>
      <td>0.972145</td>
      <td>-1.000000</td>
      <td>3.605551</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>6</td>
      <td>8</td>
      <td>2</td>
      <td>12</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>595209</th>
      <td>1488017</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.2</td>
      <td>0.659071</td>
      <td>7</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>31</td>
      <td>3</td>
      <td>0.397492</td>
      <td>0.596373</td>
      <td>0.398748</td>
      <td>1.732051</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>4</td>
      <td>8</td>
      <td>0</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>595210</th>
      <td>1488021</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
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
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.4</td>
      <td>0.698212</td>
      <td>11</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>101</td>
      <td>3</td>
      <td>0.374166</td>
      <td>0.764434</td>
      <td>0.384968</td>
      <td>3.162278</td>
      <td>0.0</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
      <td>9</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>11</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>595211</th>
      <td>1488027</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>-1.000000</td>
      <td>7</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>34</td>
      <td>2</td>
      <td>0.400000</td>
      <td>0.932649</td>
      <td>0.378021</td>
      <td>3.741657</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2</td>
      <td>3</td>
      <td>10</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3baf6736-96c8-489d-9daf-8cf7c11c0428')"
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
          document.querySelector('#df-3baf6736-96c8-489d-9daf-8cf7c11c0428 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3baf6736-96c8-489d-9daf-8cf7c11c0428');
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
  


- **구성 요소**

  - 이항 변수(binary variables) -- yes or no

  - 카테고리의 값이 정수인 카테고리형 변수

  - 정수 또는 부동 소수점 값이 있는 기타 변수

  - **-1**(-1은 결측치를 의미)을 가지는 변수들

  - target 변수와 ID 변수



```python
### rows, cols 수 확인

train.shape
```

<pre>
(595212, 59)
</pre>
- 59개의 변수와 595212개의 관측치가 존재함




```python
### 중복된 데이터 제거

train.drop_duplicates()
train.shape
```

<pre>
(595212, 59)
</pre>

```python
### 테스트 데이터의 형태 확인

test.shape
```

<pre>
(892816, 58)
</pre>
- 1개의 변수가 없음

  - 이것은 target 변수  

  ~(당연히 테스트 데이터에는 없어야지..)~





- 나중에 14개의 범주형 변수들에 대한 dummy 변수를 만들 수 있음

  - 빈 변수는 이미 **이항 변수**이므로 중복값 제거가 필요하지 않음



```python
### 데이터 정보 확인

train.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 595212 entries, 0 to 595211
Data columns (total 59 columns):
 #   Column          Non-Null Count   Dtype  
---  ------          --------------   -----  
 0   id              595212 non-null  int64  
 1   target          595212 non-null  int64  
 2   ps_ind_01       595212 non-null  int64  
 3   ps_ind_02_cat   595212 non-null  int64  
 4   ps_ind_03       595212 non-null  int64  
 5   ps_ind_04_cat   595212 non-null  int64  
 6   ps_ind_05_cat   595212 non-null  int64  
 7   ps_ind_06_bin   595212 non-null  int64  
 8   ps_ind_07_bin   595212 non-null  int64  
 9   ps_ind_08_bin   595212 non-null  int64  
 10  ps_ind_09_bin   595212 non-null  int64  
 11  ps_ind_10_bin   595212 non-null  int64  
 12  ps_ind_11_bin   595212 non-null  int64  
 13  ps_ind_12_bin   595212 non-null  int64  
 14  ps_ind_13_bin   595212 non-null  int64  
 15  ps_ind_14       595212 non-null  int64  
 16  ps_ind_15       595212 non-null  int64  
 17  ps_ind_16_bin   595212 non-null  int64  
 18  ps_ind_17_bin   595212 non-null  int64  
 19  ps_ind_18_bin   595212 non-null  int64  
 20  ps_reg_01       595212 non-null  float64
 21  ps_reg_02       595212 non-null  float64
 22  ps_reg_03       595212 non-null  float64
 23  ps_car_01_cat   595212 non-null  int64  
 24  ps_car_02_cat   595212 non-null  int64  
 25  ps_car_03_cat   595212 non-null  int64  
 26  ps_car_04_cat   595212 non-null  int64  
 27  ps_car_05_cat   595212 non-null  int64  
 28  ps_car_06_cat   595212 non-null  int64  
 29  ps_car_07_cat   595212 non-null  int64  
 30  ps_car_08_cat   595212 non-null  int64  
 31  ps_car_09_cat   595212 non-null  int64  
 32  ps_car_10_cat   595212 non-null  int64  
 33  ps_car_11_cat   595212 non-null  int64  
 34  ps_car_11       595212 non-null  int64  
 35  ps_car_12       595212 non-null  float64
 36  ps_car_13       595212 non-null  float64
 37  ps_car_14       595212 non-null  float64
 38  ps_car_15       595212 non-null  float64
 39  ps_calc_01      595212 non-null  float64
 40  ps_calc_02      595212 non-null  float64
 41  ps_calc_03      595212 non-null  float64
 42  ps_calc_04      595212 non-null  int64  
 43  ps_calc_05      595212 non-null  int64  
 44  ps_calc_06      595212 non-null  int64  
 45  ps_calc_07      595212 non-null  int64  
 46  ps_calc_08      595212 non-null  int64  
 47  ps_calc_09      595212 non-null  int64  
 48  ps_calc_10      595212 non-null  int64  
 49  ps_calc_11      595212 non-null  int64  
 50  ps_calc_12      595212 non-null  int64  
 51  ps_calc_13      595212 non-null  int64  
 52  ps_calc_14      595212 non-null  int64  
 53  ps_calc_15_bin  595212 non-null  int64  
 54  ps_calc_16_bin  595212 non-null  int64  
 55  ps_calc_17_bin  595212 non-null  int64  
 56  ps_calc_18_bin  595212 non-null  int64  
 57  ps_calc_19_bin  595212 non-null  int64  
 58  ps_calc_20_bin  595212 non-null  int64  
dtypes: float64(10), int64(49)
memory usage: 267.9 MB
</pre>
- data type이 주로 정수형(integer) 또는 실수형(float)임을 확인할 수 있음

- 현재 데이터에 null 값이 존재하지 x

  - 결측치가 -1로 대체되었기 때문


# **4. Metadata**


- 데이터 관리를 용이하게 하기 위해 변수에 대한 메타 정보를 데이터 프레임에 저장

- 분석, 시각화, 모델링 등에 사용할 변수를 **선택**하려는 경우에 유용

- 저장할 내용들:

  - role: input, ID, target

  - level: nominal, interval, ordinal, binary

  - keep: True or False

  - dtype: int, float, str



```python
data = []

for f in train.columns:

    ### role
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'
         
    ### level
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif train[f].dtype == float:
        level = 'interval'
    elif train[f].dtype == int:
        level = 'ordinal'
        
    # Initialize keep to True for all variables except for id
    keep = True
    if f == 'id':
        keep = False
    
    ### dtype 
    dtype = train[f].dtype
    
    # Creating a Dict that contains all the metadata for the variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)
```


```python
meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)
```


```python
meta
```


  <div id="df-8a2f608b-011b-4746-a099-12dd90544b8c">
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
      <th>role</th>
      <th>level</th>
      <th>keep</th>
      <th>dtype</th>
    </tr>
    <tr>
      <th>varname</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>id</td>
      <td>nominal</td>
      <td>False</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>target</th>
      <td>target</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_01</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_02_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_03</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_04_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_05_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_06_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_07_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_08_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_09_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_10_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_11_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_12_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_13_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_14</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_15</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_16_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_17_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_18_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_reg_01</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_reg_02</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_reg_03</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_car_01_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_02_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_03_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_04_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_05_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_06_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_07_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_08_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_09_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_10_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_11_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_11</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_12</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_car_13</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_car_14</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_car_15</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_calc_01</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_calc_02</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_calc_03</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_calc_04</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_05</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_06</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_07</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_08</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_09</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_10</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_11</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_12</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_13</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_14</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_15_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_16_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_17_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_18_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_19_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_20_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8a2f608b-011b-4746-a099-12dd90544b8c')"
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
          document.querySelector('#df-8a2f608b-011b-4746-a099-12dd90544b8c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8a2f608b-011b-4746-a099-12dd90544b8c');
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
### 삭제되지 않은 모든 nominal variable 추출(예시)

meta[(meta.level == 'nominal') & (meta.keep)].index
```

<pre>
Index(['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat',
       'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat',
       'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
       'ps_car_10_cat', 'ps_car_11_cat'],
      dtype='object', name='varname')
</pre>

```python
### role과 level 별 변수의 수

pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()
```


  <div id="df-9105dd87-faed-4e9a-9c23-cbb01bb35e51">
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
      <th>role</th>
      <th>level</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id</td>
      <td>nominal</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>input</td>
      <td>binary</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>input</td>
      <td>interval</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>input</td>
      <td>nominal</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>input</td>
      <td>ordinal</td>
      <td>16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>target</td>
      <td>binary</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9105dd87-faed-4e9a-9c23-cbb01bb35e51')"
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
          document.querySelector('#df-9105dd87-faed-4e9a-9c23-cbb01bb35e51 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9105dd87-faed-4e9a-9c23-cbb01bb35e51');
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
  


# **5. Descriptive statistics**


- 데이터 프레임에 기술 방법(describe method)을 적용할 수 있음

- 그러나 범주형 변수와 id 변수에 대한 평균, 표준, ...을 계산하는 것은 그다지 의미가 없음

  - 범주형 변수 -> 나중에 시각적으로 살펴보자.

- meta file을 통해 기술통계량을 구할 변수를 쉽게 선택 가능

  - 명확하게 하기 위해 데이터 타입별로 작업 수행


## **5-1. Interval 타입**



```python
v = meta[(meta.level == 'interval') & (meta.keep)].index
train[v].describe()
```


  <div id="df-87e0c7c8-e0ad-4a42-a1a0-14d3298ae265">
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
      <th>ps_reg_01</th>
      <th>ps_reg_02</th>
      <th>ps_reg_03</th>
      <th>ps_car_12</th>
      <th>ps_car_13</th>
      <th>ps_car_14</th>
      <th>ps_car_15</th>
      <th>ps_calc_01</th>
      <th>ps_calc_02</th>
      <th>ps_calc_03</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.610991</td>
      <td>0.439184</td>
      <td>0.551102</td>
      <td>0.379945</td>
      <td>0.813265</td>
      <td>0.276256</td>
      <td>3.065899</td>
      <td>0.449756</td>
      <td>0.449589</td>
      <td>0.449849</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.287643</td>
      <td>0.404264</td>
      <td>0.793506</td>
      <td>0.058327</td>
      <td>0.224588</td>
      <td>0.357154</td>
      <td>0.731366</td>
      <td>0.287198</td>
      <td>0.286893</td>
      <td>0.287153</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>0.250619</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.400000</td>
      <td>0.200000</td>
      <td>0.525000</td>
      <td>0.316228</td>
      <td>0.670867</td>
      <td>0.333167</td>
      <td>2.828427</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.700000</td>
      <td>0.300000</td>
      <td>0.720677</td>
      <td>0.374166</td>
      <td>0.765811</td>
      <td>0.368782</td>
      <td>3.316625</td>
      <td>0.500000</td>
      <td>0.400000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.900000</td>
      <td>0.600000</td>
      <td>1.000000</td>
      <td>0.400000</td>
      <td>0.906190</td>
      <td>0.396485</td>
      <td>3.605551</td>
      <td>0.700000</td>
      <td>0.700000</td>
      <td>0.700000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.900000</td>
      <td>1.800000</td>
      <td>4.037945</td>
      <td>1.264911</td>
      <td>3.720626</td>
      <td>0.636396</td>
      <td>3.741657</td>
      <td>0.900000</td>
      <td>0.900000</td>
      <td>0.900000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-87e0c7c8-e0ad-4a42-a1a0-14d3298ae265')"
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
          document.querySelector('#df-87e0c7c8-e0ad-4a42-a1a0-14d3298ae265 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-87e0c7c8-e0ad-4a42-a1a0-14d3298ae265');
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
  


### **✅ reg 변수**


- ```ps_reg_03```에만 결측값이 있음

- 변수 간의 범위(최소 ~ 최대값)가 다름

  - 스케일링(ex> StandardScaler)을 적용할 수 있음

  - 사용할 분류기(classifier)에 따라 다름


### **✅ car 변수**


- ```ps_car_12```과 ```ps_car_15```에결측치 존재

- 스케일링 적용 가능



### **✅ calc 변수**

- 결측치가 존재하지 x

- 세 변수 모두 분포가 매우 유사함


> 전체적으로 interval 변수들의 범위가 다소 작다는 것을 확인할 수 있음

  - 데이터를 익명화하기 위해 어떤 변환(ex> log transformation)이 이미 적용된 것은 아닐까?


## **5-2. Ordinal 변수**



```python
v = meta[(meta.level == 'ordinal') & (meta.keep)].index
train[v].describe()
```


  <div id="df-c8bd62d1-11f5-4653-8b4c-58e75bbffcdd">
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
      <th>ps_ind_01</th>
      <th>ps_ind_03</th>
      <th>ps_ind_14</th>
      <th>ps_ind_15</th>
      <th>ps_car_11</th>
      <th>ps_calc_04</th>
      <th>ps_calc_05</th>
      <th>ps_calc_06</th>
      <th>ps_calc_07</th>
      <th>ps_calc_08</th>
      <th>ps_calc_09</th>
      <th>ps_calc_10</th>
      <th>ps_calc_11</th>
      <th>ps_calc_12</th>
      <th>ps_calc_13</th>
      <th>ps_calc_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.900378</td>
      <td>4.423318</td>
      <td>0.012451</td>
      <td>7.299922</td>
      <td>2.346072</td>
      <td>2.372081</td>
      <td>1.885886</td>
      <td>7.689445</td>
      <td>3.005823</td>
      <td>9.225904</td>
      <td>2.339034</td>
      <td>8.433590</td>
      <td>5.441382</td>
      <td>1.441918</td>
      <td>2.872288</td>
      <td>7.539026</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.983789</td>
      <td>2.699902</td>
      <td>0.127545</td>
      <td>3.546042</td>
      <td>0.832548</td>
      <td>1.117219</td>
      <td>1.134927</td>
      <td>1.334312</td>
      <td>1.414564</td>
      <td>1.459672</td>
      <td>1.246949</td>
      <td>2.904597</td>
      <td>2.332871</td>
      <td>1.202963</td>
      <td>1.694887</td>
      <td>2.746652</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>4.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>10.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.000000</td>
      <td>11.000000</td>
      <td>4.000000</td>
      <td>13.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>12.000000</td>
      <td>7.000000</td>
      <td>25.000000</td>
      <td>19.000000</td>
      <td>10.000000</td>
      <td>13.000000</td>
      <td>23.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c8bd62d1-11f5-4653-8b4c-58e75bbffcdd')"
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
          document.querySelector('#df-c8bd62d1-11f5-4653-8b4c-58e75bbffcdd button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c8bd62d1-11f5-4653-8b4c-58e75bbffcdd');
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
  


- 오직 ```ps_car_11```에서 결측치가 존재함

- 다른 범위를 가지는 것에 대해 **스케일링** 적용 가능


## **5-3. Binary 변수**



```python
v = meta[(meta.level == 'binary') & (meta.keep)].index
train[v].describe()
```


  <div id="df-68da0384-687f-4b0c-b8e6-78985623a5ca">
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
      <th>target</th>
      <th>ps_ind_06_bin</th>
      <th>ps_ind_07_bin</th>
      <th>ps_ind_08_bin</th>
      <th>ps_ind_09_bin</th>
      <th>ps_ind_10_bin</th>
      <th>ps_ind_11_bin</th>
      <th>ps_ind_12_bin</th>
      <th>ps_ind_13_bin</th>
      <th>ps_ind_16_bin</th>
      <th>ps_ind_17_bin</th>
      <th>ps_ind_18_bin</th>
      <th>ps_calc_15_bin</th>
      <th>ps_calc_16_bin</th>
      <th>ps_calc_17_bin</th>
      <th>ps_calc_18_bin</th>
      <th>ps_calc_19_bin</th>
      <th>ps_calc_20_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.036448</td>
      <td>0.393742</td>
      <td>0.257033</td>
      <td>0.163921</td>
      <td>0.185304</td>
      <td>0.000373</td>
      <td>0.001692</td>
      <td>0.009439</td>
      <td>0.000948</td>
      <td>0.660823</td>
      <td>0.121081</td>
      <td>0.153446</td>
      <td>0.122427</td>
      <td>0.627840</td>
      <td>0.554182</td>
      <td>0.287182</td>
      <td>0.349024</td>
      <td>0.153318</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.187401</td>
      <td>0.488579</td>
      <td>0.436998</td>
      <td>0.370205</td>
      <td>0.388544</td>
      <td>0.019309</td>
      <td>0.041097</td>
      <td>0.096693</td>
      <td>0.030768</td>
      <td>0.473430</td>
      <td>0.326222</td>
      <td>0.360417</td>
      <td>0.327779</td>
      <td>0.483381</td>
      <td>0.497056</td>
      <td>0.452447</td>
      <td>0.476662</td>
      <td>0.360295</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-68da0384-687f-4b0c-b8e6-78985623a5ca')"
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
          document.querySelector('#df-68da0384-687f-4b0c-b8e6-78985623a5ca button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-68da0384-687f-4b0c-b8e6-78985623a5ca');
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
  


- train 데이터에서 1(= true)의 비율은 3.645% -> 매우 불균형한 데이터

- 대부분의 변수들에 대한 값이 0임을 확인할 수 있음


# **6. Handling imbalanced classes**


-  target = 1인 레코드의 비율은 target = 0보다 훨씬 적음 -> imbalanced data

  - 정확도는 높지만 잡음이 많은 모델로 이어질 수 있음

- 해결 전략:

  - target = 1 oversampling

  - target = 0 undersampling

> 현재 train set이 상당히 크기에, **undersampling** 수행


## **📚참고자료**

[불균형 데이터 (imbalanced data) 처리를 위한 샘플링 기법](https://casa-de-feel.tistory.com/15)



```python
desired_apriori = 0.10 

### target 변수의 값마다 index 가져오기
idx_0 = train[train.target == 0].index # False
idx_1 = train[train.target == 1].index # True

### target 분포 확인
nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])

### 추가한 코드
print(nb_0)
print(nb_1)
```

<pre>
573518
21694
</pre>
- target 데이터가 굉장히 **불균형한(imbalanced)** 분포를 가지고 있음을 확인할 수 있다.



```python
### Undersampling 시킬 비율 선택

undersampling_rate = ((1 - desired_apriori)*nb_1) / (nb_0*desired_apriori) 
# undersampling 시켜야 할 비율 -> 해당 비율만큼 target = 0인 것들이 target = 1로 변경
undersampled_nb_0 = int(undersampling_rate * nb_0)

print('Rate to undersample records with target = 0: {}'.format(undersampling_rate))
print('Number of records with target = 0 after undersampling: {}'.format(undersampled_nb_0))
```

<pre>
Rate to undersample records with target = 0: 0.34043569687437886
Number of records with target = 0 after undersampling: 195246
</pre>
- 불균형 정도가 많이 감소됨을 확인할 수 있음  

  **(96% -> 33%)**



```python
### Undersampling 수행
# RandomSampling 방법을 채택
undersampled_idx = shuffle(idx_0, random_state = 37, n_samples = undersampled_nb_0) 
# 실행 시마다 결과가 달라지는 것을 방지하기 위해 random_state 사용

# 나머지 인덱스로 list 구성 => target = 1의 index
idx_list = list(undersampled_idx) + list(idx_1)

# undersampling된 데이터 반환(최종 데이터)
train = train.loc[idx_list].reset_index(drop=True)
```

# **7. Data Quality Checks**


## **7-1. 결측치 확인**

- 결측치는 ```-1```로 표기되어 있음



```python
vars_with_missing = [] # 결측치를 저장할 list

for f in train.columns: # 각 column별로
    missings = train[train[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        
        missings_perc = missings / train.shape[0] # 각 컬럼별 결측치의 비율
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))

print()
# 전체 결측치 개수        
print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))
```

<pre>
Variable ps_ind_02_cat has 103 records (0.05%) with missing values
Variable ps_ind_04_cat has 51 records (0.02%) with missing values
Variable ps_ind_05_cat has 2256 records (1.04%) with missing values
Variable ps_reg_03 has 38580 records (17.78%) with missing values
Variable ps_car_01_cat has 62 records (0.03%) with missing values
Variable ps_car_02_cat has 2 records (0.00%) with missing values
Variable ps_car_03_cat has 148367 records (68.39%) with missing values
Variable ps_car_05_cat has 96026 records (44.26%) with missing values
Variable ps_car_07_cat has 4431 records (2.04%) with missing values
Variable ps_car_09_cat has 230 records (0.11%) with missing values
Variable ps_car_11 has 1 records (0.00%) with missing values
Variable ps_car_14 has 15726 records (7.25%) with missing values

In total, there are 12 variables with missing values
</pre>
- ```ps_car_03_cat``` 및 ```ps_car_05_cat```에는 결측값이 있는 레코드의 비율이 많음

  - 해당 변수 제거

- 결측값이 있는 다른 **범주형** 변수의 경우 결측값 -1은 그대로 둘 수 있음

- ```ps_reg_03```(연속)는 전체 18% 데이터가 결측치

  - **평균**으로 대체

- ```ps_car_11```(순서)에는 결측값이 있는 레코드가 5개만 있음

  - **최빈값**으로 대체

- ```ps_car_12```(연속)에는 결측값이 있는 레코드가 1개만 있음

  - **평균**으로 대체

- ```ps_car_14```(연속)에는 모든 레코드의 7%에 대한 결측값이 있음

  - **평균**으로 대체



```python
### 결측값이 너무 많은 변수 삭제하기
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train.drop(vars_to_drop, inplace = True, axis = 1)
meta.loc[(vars_to_drop),'keep'] = False  # 메타데이터 변경하기

### 데이터 가공 - 결측치 채워넣기
mean_imp = SimpleImputer(missing_values = -1, strategy = 'mean')
mode_imp = SimpleImputer(missing_values = -1, strategy = 'most_frequent')
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()
```

## **7-2. 범주형 변수의 카디널리티(cardinality) 확인**

- 카디널리티(cardinality, 기수): 변수에 포함된 서로 다른 값의 수

- 나중에 범주형 변수에서 dummy 변수를 만들 것이기 때문에, 다른 값을 가진 변수의 개수를 확인할 필요가 있음

  - 해당 변수는 많은 dummy 변수를 초래하므로 다르게 처리해야 함




```python
v = meta[(meta.level == 'nominal') & (meta.keep)].index

for f in v:
    dist_values = train[f].value_counts().shape[0]
    print('Variable {} has {} distinct values'.format(f, dist_values))
```

<pre>
Variable ps_ind_02_cat has 5 distinct values
Variable ps_ind_04_cat has 3 distinct values
Variable ps_ind_05_cat has 8 distinct values
Variable ps_car_01_cat has 13 distinct values
Variable ps_car_02_cat has 3 distinct values
Variable ps_car_04_cat has 10 distinct values
Variable ps_car_06_cat has 18 distinct values
Variable ps_car_07_cat has 3 distinct values
Variable ps_car_08_cat has 2 distinct values
Variable ps_car_09_cat has 6 distinct values
Variable ps_car_10_cat has 3 distinct values
Variable ps_car_11_cat has 104 distinct values
</pre>
- ```ps_car_11_cat```가 많은 다양한 변수를 가지고 있음


- 평활화(Smoothing)는 Danielle Micci-Bareca에 의해 다음 논문과 같이 계산됨

  -  https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf



- parameters>

  - trn_series : 범주형 변수를 pd.Series로 훈련(train)

  - tst_series: 범주형 변수를 pd.Series로 평가(test)

  - target: target data를 pd.Series 형으로

  - min_samples_leaf(int):범주의 특성을 설명해 줄 수 있는 최소한의 표본

  - smoothing(int): 범주형 평균과 이전 평균의 균형을 맞추기 위한 평활(smoothing) 효과 작용



```python
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))
```


```python
def target_encode(trn_series = None, 
                  tst_series = None, 
                  target = None, 
                  min_samples_leaf = 1, 
                  smoothing = 1,
                  noise_level = 0):
   
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name

    '''[assert 문]
    -  어떤 조건이 참임을 확고히 하는 것
    -  해당 조건이 거짓이면 에러 상황 => 실행을 계속하지 못하게 함
    '''
    
    temp = pd.concat([trn_series, target], axis = 1)
    
    ### target 변수의 평균
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    
    ### 평활화 정도 계산
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    ### 모든 target 변수에 평활화 함수 적용하기
    prior = target.mean()
    # 카운트가 클수록 full_avg가 적게 고려됨
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis = 1, inplace = True)
    
    ### train, test set에 평활화 적용
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns = {'index': target.name, target.name: 'average'}),
        on = trn_series.name,
        how = 'left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # 인덱스 초기화(재설정)
    ft_trn_series.index = trn_series.index 
    
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # 인덱스 초기화(재설정)    
    ft_tst_series.index = tst_series.index
    
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
```


```python
### 범주형 변수 encoding
# 범주형 -> 수치형 변수(dummy 변수)

train_encoded, test_encoded = target_encode(train["ps_car_11_cat"], 
                             test["ps_car_11_cat"], 
                             target = train.target, 
                             min_samples_leaf = 100,
                             smoothing = 10,
                             noise_level = 0.01)
```


```python
### encoding된 값으로 변경
train['ps_car_11_cat_te'] = train_encoded
train.drop('ps_car_11_cat', axis = 1, inplace = True)

# 메타 데이터 update
meta.loc['ps_car_11_cat','keep'] = False

test['ps_car_11_cat_te'] = test_encoded
test.drop('ps_car_11_cat', axis=1, inplace = True)
```

# **8. Exploratory Data Visualization**


## **8-1. 범주형 변수**

- ```target = 1```인 고객들의 비율 파악



```python
v = meta[(meta.level == 'nominal') & (meta.keep)].index

for f in v:
    ### 기본 틀 설정
    plt.figure()
    fig, ax = plt.subplots(figsize = (20,10))

    ### target = 1의 비율 계산
    cat_perc = train[[f, 'target']].groupby([f],as_index = False).mean()
    cat_perc.sort_values(by = 'target', ascending = False, inplace = True) # 내림차순 정렬
    
    ### 막대그래프(Barplot) 그리기
    # target의 평균을 내림차순으로 정렬
    sns.barplot(ax = ax, x = f, y = 'target', data = cat_perc, order = cat_perc[f])
    plt.ylabel('% target', fontsize = 18)
    plt.xlabel(f, fontsize = 18)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
    plt.show()
```

<pre>
<Figure size 432x288 with 0 Axes>
</pre>
<img src="data:image/png;base64,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"/>

<pre>
<Figure size 432x288 with 0 Axes>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABKMAAAJdCAYAAADusrRCAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAAAt+klEQVR4nO3df7zmdV3n/+cLJxiBr4LO6AomU9gG4rpYQEWbaSttN7XdjPouFIq/GC1hGvpa5o9Fs9RKBhBCDTRXDVy072jRUqui4q8MxpYVQ1LjhwhqM6kFjAPKvPeP6zp4eTpzzrnOXOd9MWfu99vt3M6c9+fXa2Zut7nhw8/nc1VrLQAAAADQwz7THgAAAACAvYcYBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDerpj3A/cGaNWvaunXrpj0GAAAAwIrxqU99altrbe3sdTEqybp167Jly5ZpjwEAAACwYlTVLXOte0wPAAAAgG7EKAAAAAC6EaMAAAAA6EaMAgAAAKAbMQoAAACAbsQoAAAAALoRowAAAADoRowCAAAAoBsxCgAAAIBuxCgAAAAAuhGjAAAAAOhGjAIAAACgGzEKAAAAgG7EKAAAAAC6EaMAAAAA6EaMAgAAAKAbMQoAAACAbsQoAAAAALoRowAAAADoRowCAAAAoBsxCgAAAIBuxCgAAAAAuhGjAAAAAOhGjAIAAACgm1XTHoC5/fBvvH3aIwDM61Ove+a0RwAAAPZA7owCAAAAoBsxCgAAAIBuxCgAAAAAuhGjAAAAAOhGjAIAAACgGzEKAAAAgG7EKAAAAAC6EaMAAAAA6EaMAgAAAKAbMQoAAACAbsQoAAAAALoRowAAAADoRowCAAAAoBsxCgAAAIBuxCgAAAAAuhGjAAAAAOhGjAIAAACgGzEKAAAAgG7EKAAAAAC6EaMAAAAA6EaMAgAAAKAbMQoAAACAbsQoAAAAALoRowAAAADoRowCAAAAoBsxCgAAAIBuxCgAAAAAuhGjAAAAAOhGjAIAAACgGzEKAAAAgG7EKAAAAAC6mWqMqqp9qurMqrqhqnZU1a1VtamqDhjjHA+pqrOr6gvDc2ytqg9V1U8s5+wAAAAAjG/VlK9/bpINSd6TZFOSI4c/P76qntxa2znfwVV1WJIPJzkwyVuSfC7Jg5M8Lsmhyzc2AAAAAEsxtRhVVUclOSPJ5tbaiSPrNyU5P8lJSS5d4DR/ksHv4XGttS8v16wAAAAATMY0H9M7OUklOW/W+sVJtic5Zb6Dq+oJSf5Dkj9orX25qr6nqvZfjkEBAAAAmIxpxqhjk+xMcvXoYmttR5Jrh9vn85Th9y9W1eVJvpnkrqr6XFXNG7IAAAAAmI5pxqhDkmxrrd09x7bbkqypqn3nOf4Hh98vTvKQJKcmeU6Se5K8o6qePd/Fq2p9VW2pqi1bt24df3oAAAAAxjbNGLV/krlCVJLsGNlnV/6f4fc7kjyptXZJa+2tSX4iyTeSvKaqdvn7a61d1Fo7prV2zNq1a8ebHAAAAIAlmWaM2p5kv11sWz2yz658c/j9na21e2YWW2tfT/LnSf5NvnP3FAAAAAD3A9OMUbdn8CjeXEHq0Awe4btnjm0zvjT8/pU5ts18st7BuzEfAAAAABM2zRh1zfD6x40uVtXqJEcn2bLA8TMvPn/kHNtm1v5xN+YDAAAAYMKmGaMuS9KSbJy1floG74q6ZGahqg6vqiNm7ffeDN4XdUpVHTiy7yOS/FySz7XWvjDxqQEAAABYslXTunBr7bqqujDJ6VW1OckVSY5MsiHJVUkuHdn9yiSHJamR479eVS9K8kdJPllVf5xk3yS/Mvx+RpffCAAAAACLNrUYNbQxyc1J1id5apJtSS5IclZrbedCB7fWLqqqbUl+M8nvJNmZ5K+T/FJr7ePLNDMAAAAASzTVGNVauzfJpuHXfPutm2fb5iSbJzsZAAAAAMthmu+MAgAAAGAvI0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQzVRjVFXtU1VnVtUNVbWjqm6tqk1VdcAij2+7+LpzuWcHAAAAYHyrpnz9c5NsSPKeJJuSHDn8+fFV9eTW2s5FnOOjSS6atfatiU4JAAAAwERMLUZV1VFJzkiyubV24sj6TUnOT3JSkksXcaobW2t/sjxTAgAAADBJ03xM7+QkleS8WesXJ9me5JTFnqiq9q2qAyc3GgAAAADLYZox6tgkO5NcPbrYWtuR5Nrh9sX4hQzi1R1V9Y9VdUFVPXiSgwIAAAAwGdN8Z9QhSba11u6eY9ttSY6vqn1ba/fMc46rk7w7yReSPCjJU5KcnuQnq+r41touX2ReVeuTrE+SRz3qUUv8LQAAAAAwjmnGqP2TzBWikmTHyD67jFGttR+ZtfT2qvp0klcn+bXh910de1GGLz4/5phj2iJnBgAAAGA3TPMxve1J9tvFttUj+4zrdRkErKcuZSgAAAAAls80Y9TtSdZU1VxB6tAMHuGb7xG9ObXWvjVz7t2cDwAAAIAJm2aMumZ4/eNGF6tqdZKjk2xZykmHxz8yyVd3cz4AAAAAJmyaMeqyJC3Jxlnrp2XwrqhLZhaq6vCqOmJ0p6p66C7O+zsZvAvr8olNCgAAAMBETO0F5q2166rqwiSnV9XmJFckOTLJhiRXJbl0ZPcrkxyWpEbWXl5VP5rkQ0m+mOTADD5N70lJ/ibJBcv+mwAAAABgLNP8NL1kcFfUzUnWZ/DC8W0ZRKSzWms7Fzj2w0kek+TUJA9Ncm+Szyd5WZJzWms7dn0oAAAAANMw1RjVWrs3yabh13z7rZtj7c+S/NnyTAYAAADAcpjmO6MAAAAA2MuIUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQz1RhVVftU1ZlVdUNV7aiqW6tqU1UdsIRz7V9VN1ZVq6o/XI55AQAAANg9074z6twk5yS5PskZSd6dZEOSy6tq3NlelWTtZMcDAAAAYJJWTevCVXVUBgFqc2vtxJH1m5Kcn+SkJJcu8lw/lGRjkt9MsmniwwIAAAAwEdO8M+rkJJXkvFnrFyfZnuSUxZykqh4wPOavkmye4HwAAAAATNjU7oxKcmySnUmuHl1sre2oqmuH2xfjzCRHJDlxoR0BAAAAmK5p3hl1SJJtrbW759h2W5I1VbXvfCeoqu9L8ttJXtVau3mci1fV+qraUlVbtm7dOs6hAAAAACzRNGPU/knmClFJsmNkn/m8KcmNGbwEfSyttYtaa8e01o5Zu9Z7zwEAAAB6mOZjetuTPGwX21aP7DOnqjolyQlJntBa+9aEZwMAAABgGUzzzqjbM3gUb785th2awSN898x14PCYc5JckeQrVfXoqnp0ksOGuzx4uHbQMswNAAAAwBJNM0ZdM7z+caOLVbU6ydFJtsxz7AOTrE3y1CSfH/n68HD7KcOfnzfJgQEAAADYPdN8TO+yJC9NsjHJR0fWT8vgXVGXzCxU1eFJvqe1dsNw6a4kvzjHOdcmeUOSv0ryliSfnvjUAAAAACzZ1GJUa+26qrowyelVtTmDR+6OTLIhyVVJLh3Z/coMHsGr4bHfSvKns89ZVeuGv/yH1tq/2g4AAADAdE3zzqhkcFfUzUnWZ/DI3bYkFyQ5q7W2c3pjAQAAALAcphqjWmv3Jtk0/Jpvv3WLPN/NGd49BQAAAMD9zzRfYA4AAADAXkaMAgAAAKCbRceoqnpCVa2dZ/uaqnrCZMYCAAAAYCUa586oDyU5YZ7t/3G4DwAAAADMaZwYtdCLwR+QxCfgAQAAALBL474zqs2z7fgk23ZjFgAAAABWuFXzbayqX0vyayNL51XVq+fY9eAkD0ryxxOcDQAAAIAVZt4YleQbSW4Z/npdkn9K8tVZ+7Qkn0nyySTnTnA2AAAAAFaYeWNUa+1tSd6WJFV1U5Lfaq39eY/BAAAAAFh5Froz6j6tte9bzkEAAAAAWPnGfYF5quoJVfW7VXVxVR0xXDtwuH7QxCcEAAAAYMVYdIyqqgdU1WVJPpTkpUmek+SQ4eZvJ3lvkl+d9IAAAAAArBzj3Bn14iQnJvn1JEcmqZkNrbUdSd6T5CkTnQ4AAACAFWWcGPXMJG9vrb0+ybY5tn82yeETmQoAAACAFWmcGLUuyV/Ps/0bSQ7enWEAAAAAWNnGiVF3JHnIPNsfnWTr7o0DAAAAwEo2Toz6WJJTqqpmb6iqgzN4ofmHJjUYAAAAACvPODHq1Ul+IMkHkzxtuPbvq+r5Sf42yQFJfm+y4wEAAACwkqxa7I6ttS1VdWKSNyd563D57Aw+Ve8fkzy9tXb95EcEAAAAYKVYdIxKktba/6yqdUlOSHJkBiHq80n+V2tt++THAwAAAGAlGStGJUlr7e4kfzH8AgAAAIBFG+edUQAAAACwWxZ9Z1RV3bjALi3JN5N8Mcn7klzcWrtrN2YDAAAAYIUZ586oLyb5dpJ1SQ5O8o3h18HDtW9nEKN+NMk5ST5VVWsnNikAAAAAe7xxYtTGJA9J8qtJHtZa+6HW2g8lWZvk9OG25yZZk+SMJD+Q5FUTnRYAAACAPdo4LzA/O8llrbU3jS621r6d5A1V9dgkm1prJyS5sKp+LMlTJzcqAAAAAHu6ce6M+pEkn55n+6czeERvxieSPHwpQwEAAACwMo0To+5Ocuw8248b7jNjvyR3LmUoAAAAAFamcWLUnyd5dlX9VlXtP7NYVftX1UuSnDrcZ8bxST43mTEBAAAAWAnGeWfUi5I8Pslrkryqqm4frh8yPM91SX4jSapqdZIdSS6c3KgAAAAA7OkWHaNaa1+rqh9J8rwkT0vyfcNNVya5PMmbW2v3DPfdkeQZE54VAAAAgD3comJUVT0wyS8m+fvW2huSvGFZpwIAAABgRVrsO6PuTvLmDB7TAwAAAIAlWVSMaq3tTPLFJA9a3nEAAAAAWMnG+TS9tyV5RlXtt1zDAAAAALCyjfNpep9I8vNJrq2qNyT5fJLts3dqrX1kQrMBAAAAsMKME6PeP/Lr1ydps7bXcO0BuzsUAAAAACvTODHq2cs2BQAAAAB7hUXHqNba25ZzEAAAAABWvnFeYA4AAAAAu2Wcx/SSJFX18CTHJDk4c8Ss1trbJzAXAAAAACvQomNUVe2T5MIkz8v8d1SJUQAAAADMaZzH9F6U5PlJ3pnk1Aw+Pe+3krwwyeeTbElywqQHBAAAAGDlGCdGnZrkr1prz0zyl8O1T7XW3pTkh5OsGX4HAAAAgDmNE6O+P8lfDX+9c/j9e5KktXZXkrdm8AgfAAAAAMxpnBj1zSTfGv76ziQtycNGtn8lyfdOaC4AAAAAVqBxYtQtSQ5Pktbat5J8IcnPjGx/cpKvTm40AAAAAFaacWLUB5M8feTndyQ5uao+VFUfTvKLSd41wdkAAAAAWGFWjbHv2UneV1X7tdbuTvLaDB7TOyXJvUkuSvLKiU8IAAAAwIqx6BjVWvtyki+P/Hxvkg3DLwAAAABY0KIf06uqs6rqsfNsP6qqzprMWAAAAACsROO8M+qVSR43z/bHJnnFbk0DAAAAwIo2ToxayOok357g+QAAAABYYeZ9Z1RVPSjJQSNLD62qR82x60OS/HKSWyc3GgAAAAArzUIvMD8zycx7oFqS84Zfc6kkvzmRqQAAAABYkRaKUR8efq8MotR7knx61j4tyZ1JPtla+8REpwMAAABgRZk3RrXWrkpyVZJU1WFJ3tRa+5segwEAAACw8ix0Z9R9WmvPXs5BAAAAAFj5JvlpegAAAAAwLzEKAAAAgG7EKAAAAAC6EaMAAAAA6EaMAgAAAKCb3Y5RVbVmEoMAAAAAsPItKUZV1X5V9YdVdVeSr1bVN6vqzVV14ITnAwAAAGAFWbXE416X5GeSbEhya5LHJXl5BnHrOZMZDQAAAICVZt47o6rqsF1s+s9JTm2tvaW19r7W2tlJfjfJz45z8arap6rOrKobqmpHVd1aVZuq6oBFHPuDVXVJVX22qv65qrYPz3NOVT1inDkAAAAA6GOhx/T+rqp+rapq1vodSR45a+3QJHeNef1zk5yT5PokZyR5dwZ3W11eVQvN9sgkj0jyniQvSbIxyfuTrE/yqap62JizAAAAALDMFnpM75lJzk/yy1X13NbadcP1NyZ5a1U9NYPH9P5dkqckedliL1xVR2UQoDa31k4cWb9peM2Tkly6q+Nba1cmuXKO834kybuSPCvJHyx2HgAAAACW37x3H7XWNid5TJK/TXJNVb2mqvZrrb0hybOTPDzJzyV5YJLnttZ+f4xrn5ykkpw3a/3iJNuTnDLGuUbdMvx+8BKPBwAAAGCZLPgC89bavyR5QVX9SZKLkvxCVT2/tXZZkst249rHJtmZ5OpZ19tRVdcOty+oqlYnOTDJ6gzC2UwQu2I3ZgMAAABgGSz0Xqb7tNY+luToJO9M8pdV9ZaqOmg3rn1Ikm2ttbvn2HZbkjVVte8izvO8JFszeFzwfyU5KMkprbWP7sZsAAAAACyDRceoJGmt3dNae0WSH0pyRJIbquq/LvHa+yeZK0QlyY6RfRby3iQnJHl6klcl+UaSNQsdVFXrq2pLVW3ZunXrIi4DAAAAwO6aN0ZV1QOr6vVVdWtVfa2qLq+qR7fWrm+t/XgG8eePquovqup7x7z29iT77WLb6pF95tVa+1Jr7QOttfcOQ9mpSf6gql6ywHEXtdaOaa0ds3bt2rEGBwAAAGBpFrozalMGLyp/S5JXJnl0ksur6gFJMnyR+VFJvp3k76pqwxjXvj2DR/HmClKHZvAI3z1jnC/DmT6d5H8n+dVxjwUAAABgeS0Uo34+yWtaa69srZ2fwSfg/dsMXhSeJGmt3dZa+7kMotWLx7j2NcPrHze6OHwh+dFJtoxxrtkemOQhu3E8AAAAAMtgoRhVSdrIz23W9+9saO3/T3LkGNe+bHiejbPWT8vgXVGX3DdE1eFVdcR3DVb1b+YcuOpJSR6b5JNjzAIAAABAB6sW2P7eJC8dfqrd15O8IMnnk3x2rp1ba/+y2Au31q6rqguTnF5Vm5NckUHM2pDkqiSXjux+ZZLDMohjM95YVY9I8sEkt2TwnqkfTnJSkjuS/H+LnQUAAACAPhaKUb+ewfugfiWDR9/+OsnG1tq9E7r+xiQ3J1mf5KlJtiW5IMlZrbWdCxz7ziTPTPKMJGszuMvqliR/lOR1rbUvTmhGAAAAACZk3hjVWrsryQuHXxM3jFqbhl/z7bdujrV3JXnXcswFAAAAwPJY6J1RAAAAADAxYhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdTDVGVdU+VXVmVd1QVTuq6taq2lRVByzi2H9bVa+qqk9W1daquqOqrq2qly3meAAAAAD6m/adUecmOSfJ9UnOSPLuJBuSXF5VC832nCRnJvmHJK9K8htJ/j7J7yb5RFU9cLmGBgAAAGBpVk3rwlV1VAYBanNr7cSR9ZuSnJ/kpCSXznOKP03y2tbaP4+svamqPp/kZUmem+QPJz44AAAAAEs2zTujTk5SSc6btX5xku1JTpnv4NballkhasZlw++P3d0BAQAAAJisacaoY5PsTHL16GJrbUeSa4fbl+KRw+9fXfJkAAAAACyLacaoQ5Jsa63dPce225Ksqap9xzlhVT0gyX9L8u3M/4gfAAAAAFMwzRi1f5K5QlSS7BjZZxznJfmxJGe11v5+vh2ran1VbamqLVu3bh3zMgAAAAAsxTRj1PYk++1i2+qRfRalqn4nyelJLmqtvXah/VtrF7XWjmmtHbN27drFXgYAAACA3TDNGHV7Bo/izRWkDs3gEb57FnOiqnplkpcneWuSF0xsQgAAAAAmapox6prh9Y8bXayq1UmOTrJlMScZhqhXJHlbkue11tpEpwQAAABgYqYZoy5L0pJsnLV+WgbvirpkZqGqDq+qI2afoKrOyiBEvSPJc1prO5dtWgAAAAB226ppXbi1dl1VXZjk9KranOSKJEcm2ZDkqnz3p+FdmeSwJDWzUFUvTPLbSb6Y5ANJfqmqRg7JV1tr71/W3wQAAAAAY5lajBramOTmJOuTPDXJtiQXZPBpeAvd5XTs8PujMnhEb7arkohRAAAAAPcjU41RrbV7k2wafs2337o51p6V5FnLMRcAAAAAy2Oa74wCAAAAYC8jRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAAAAAN2IUQAAAAB0I0YBAAAA0I0YBQAAAEA3YhQAAAAA3YhRAAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANDNVGNUVe1TVWdW1Q1VtaOqbq2qTVV1wCKPf0lVvbuqbqyqVlU3L/PIAAAAAOyGVVO+/rlJNiR5T5JNSY4c/vz4qnpya23nAse/JsnXkvxtkoOWcU4AWJIvvurfTXsEgHk96qzrpj0CAHuZqcWoqjoqyRlJNrfWThxZvynJ+UlOSnLpAqc5vLV24/C4zyQ5cJnGBQAAAGACpvmY3slJKsl5s9YvTrI9ySkLnWAmRAEAAACwZ5hmjDo2yc4kV48uttZ2JLl2uB0AAACAFWSaMeqQJNtaa3fPse22JGuqat/OMwEAAACwjKYZo/ZPMleISpIdI/ssi6paX1VbqmrL1q1bl+syAAAAAIyYZozanmS/XWxbPbLPsmitXdRaO6a1dszatWuX6zIAAAAAjJhmjLo9g0fx5gpSh2bwCN89nWcCAAAAYBmtmuK1r0ny00mOS/LRmcWqWp3k6CQfmc5YAAAA3+3HL/jxaY8AMK+Pn/HxaY+waNO8M+qyJC3Jxlnrp2XwrqhLZhaq6vCqOqLfaAAAAAAsh6ndGdVau66qLkxyelVtTnJFkiOTbEhyVZJLR3a/MslhSWr0HFX1jOF6kqxNsm9VvXz48y2ttXcs428BAAAAgDFN8zG9ZHBX1M1J1id5apJtSS5IclZrbecijn9ukp+ctfY7w+9XJRGjAAAAAO5HphqjWmv3Jtk0/Jpvv3W7WH/i5KcCAAAAYLlM851RAAAAAOxlxCgAAAAAuhGjAAAAAOhGjAIAAACgGzEKAAAAgG7EKAAAAAC6EaMAAAAA6EaMAgAAAKAbMQoAAACAbsQoAAAAALoRowAAAADoRowCAAAAoBsxCgAAAIBuxCgAAAAAuhGjAAAAAOhGjAIAAACgGzEKAAAAgG7EKAAAAAC6EaMAAAAA6EaMAgAAAKAbMQoAAACAbsQoAAAAALoRowAAAADoRowCAAAAoBsxCgAAAIBuxCgAAAAAuhGjAAAAAOhGjAIAAACgGzEKAAAAgG7EKAAAAAC6EaMAAAAA6EaMAgAAAKAbMQoAAACAbsQoAAAAALoRowAAAADoRowCAAAAoBsxCgAAAIBuxCgAAAAAuhGjAAAAAOhGjAIAAACgGzEKAAAAgG7EKAAAAAC6EaMAAAAA6EaMAgAAAKAbMQoAAACAbsQoAAAAALoRowAAAADoRowCAAAAoBsxCgAAAIBuxCgAAAAAuhGjAAAAAOhGjAIAAACgGzEKAAAAgG7EKAAAAAC6EaMAAAAA6EaMAgAAAKAbMQoAAACAbsQoAAAAALoRowAAAADoRowCAAAAoBsxCgAAAIBuxCgAAAAAuhGjAAAAAOhGjAIAAACgGzEKAAAAgG7EKAAAAAC6mXqMqqp9qurMqrqhqnZU1a1VtamqDuhxPAAAAAD9TD1GJTk3yTlJrk9yRpJ3J9mQ5PKqWsx8u3s8AAAAAJ2smubFq+qoDALS5tbaiSPrNyU5P8lJSS5druMBAAAA6Gvadw6dnKSSnDdr/eIk25OcsszHAwAAANDRtGPUsUl2Jrl6dLG1tiPJtcPty3k8AAAAAB1NO0YdkmRba+3uObbdlmRNVe27jMcDAAAA0NFU3xmVZP8kc4WkJNkxss89kz6+qtYnWT/88c6q+vsFp4U925ok26Y9BCtHnX3qtEeAvZV/z5msV9S0J4C9lX/PmajacL/89/ywuRanHaO2J3nYLratHtln4se31i5KctFCA8JKUVVbWmvHTHsOAHaPf88BVgb/nrM3m/Zjerdn8CjdfnNsOzSDR/B2dVfUJI4HAAAAoKNpx6hrhjMcN7pYVauTHJ1kyzIfDwAAAEBH045RlyVpSTbOWj8tg3c9XTKzUFWHV9URSz0e8FgqwArh33OAlcG/5+y1qrU23QGqLkhyepL3JLkiyZFJNiT5eJKfaq3tHO53c5LDWmu1lOMBAAAAmL77Q4x6QAZ3Nq1Psi6DTxO4LMlZrbU7R/a7OXPHqEUdDwAAAMD0TT1GAQAAALD3mPY7o4AOquq4qjq/qj5eVXdWVauqZ017LgAWVlUvqap3V9WNw3+/b572TACMr6r2qaozq+qGqtpRVbdW1aaqOmDas0FvYhTsHZ6S5IVJDkryf6Y7CgBjek2Sn0ryD0m+PuVZAFi6c5Ock+T6JGckeXcG7zu+vKr8b3P2KqumPQDQxRuTvK61dldV/UKS46c9EACLdnhr7cYkqarPJDlwyvMAMKaqOiqDALW5tXbiyPpNSc5PclKSS6c0HnSnvsJeoLX21dbaXdOeA4DxzYQoAPZoJyepJOfNWr84yfYkp/QeCKZJjAIAAIDldWySnUmuHl1sre1Icu1wO+w1xCgAAABYXock2dZau3uObbclWVNV+3aeCaZGjAIAAIDltX+SuUJUkuwY2Qf2Cl5gDitEVT0gydpZy99srf3zNOYBAADusz3Jw3axbfXIPrBXcGcUrBzfm+TLs75eP9WJAACAJLk9g0fx9ptj26EZPMJ3T+eZYGrcGQUrx1eSnDBr7fZpDAIAAHyXa5L8dJLjknx0ZrGqVic5OslHpjMWTIcYBSvE8JM4PjDtOQAAgH/lsiQvTbIxIzEqyWkZvCvqkinMBFMjRsFeoKoOS/KM4Y9HDb//bFU9cvjrd7TWbuk/GQALqapnJDls+OPaJPtW1cuHP9/SWnvHdCYDYLFaa9dV1YVJTq+qzUmuSHJkkg1Jrkpy6TTng96qtTbtGYBlVlVPTPKheXZ5Umvtw12GAWAsVfXhJD+5i81Xtdae2G8aAJZq+IFDG5OsT7IuybYM7pg6q7V25/Qmg/7EKAAAAAC68Wl6AAAAAHQjRgEAAADQjRgFAAAAQDdiFAAAAADdiFEAAAAAdCNGAQAAANCNGAUAAABAN2IUAMAiVdUTq6pV1bOW4dzPGp77iZM+NwDA/YkYBQCwh6uqfarqzKq6oap2VNWtVbWpqg5YxLGXDSPYZ3rMuhjD6PfKqjpo2rMAAJMnRgEALN5HkjwwyTumPcgs5yY5J8n1Sc5I8u4kG5JcXlW7/O+9qnpakl9I8s0eQ47hiUlekeSg6Y4BACyHVdMeAABgT9Fa25lkx7TnGFVVR2UQoDa31k4cWb8pyflJTkpy6RzHHZjkDUkuTPKf+0wLAODOKABgDzXyjqUnDx/puqWq7q6qT1fVSbP2Pb6q/rKqvjJ8jO22qrqiqn50zGv+q3dGja5V1bOr6u+Gc9xSVb+5i/OcNnyk7u6q+kJVbUxSS/hjSJKTh8eeN2v94iTbk5yyi+NeneQBSV6+xOvep6oeVFWvrqrPDv98/6mqPjb691BVR1TVG4Z/PndU1faq+lRVPW/Wuf57BndFJclNwz/bVlWv3N05AYD7B3dGAQB7ut9PckAGd/kkybOTvLOqVrfW/ntV/WCS9yf5SpLXJ/lqkocn+Q9J/n2ST05ojhcMz/uWJN/IIAL9flV9qbV2351Jw/B0bpL/k+SlSfZP8qIk/7jE6x6bZGeSq0cXW2s7qura4fbvUlXHJTk9ycmttX+pWmoHS4bvdfpYkqOS/GmSN2YQuR6f5GlJ/sdw1ycmeUKSv0hyUwZ/Z7+Y5OKqWttae+1wvz9K8qAkT09yZpJtw/VPL3lIAOB+RYwCAPZ0a5I8rrX2z0lSVW/KIFycU1WXJflPGQSfk1trV+/6NLvtUUmOHJnjj5PcksEjdJcO1w7K4I6kzyY5vrW2fbj+1iQ3LPG6hyTZ1lq7e45ttyU5vqr2ba3dM7zWqiRvTvK+1tq7lnjNUa/JIEQ9v7V20eiGWe+rekdr7U2ztp+b5INJfquqzm6tfau19tdV9ekMYtR7W2s3T2BGAOB+xGN6AMCe7o0zAShJhr9+U5KDM7gbZ2bbf6mq1cs4x1tnzbE9g7uufmBkn5/OIIxdOBOihvt+KcklS7zu/knmClHJd95vtf/I2m8keXSSFy7xevcZxqaTknx2dohK7nvH1syv7xo5bnVVPTTJQ5K8L4M7oY7Y3XkAgD2DGAUA7Ok+O8fa9cPv35/BY2IfyOCRuK9V1Qer6sVVddiE57hxjrV/SvLQkZ+/f/h9rrugrp9jbTG2J9lvF9tWj+yTqnp0krOSvLq1Nte841qTQfS7dqEdq+rAqjq7qr6Ywaf3bUuyNYM7xTI8DwCwFxCjAIAVrbV2d2vthCQ/kuS1Se5N8qokN1TV0yd4qXsneK5x3J5kTVXNFaQOzeARvnuGP29K8rUk76mqR898ZfDqhn2HPz9imea8NMmvJ7kiyS8n+ZkkJ2Tw/qzEf5cCwF7DO6MAgD3dkUn+bNbaY4bf77v7Z/i+qKuTpKq+N8n/TvK7Sd7TYcYZM/MckeTKWdsek6W5JoPH/45L8tGZxeEjiUcn+cjIvodl8I6pv9vFuT6f5H9m8OLxxdiW5OsZvAh+l4bvynpaBu+NesGsbU+e45C2yOsDAHsg/w8UALCn+5WqevDMD8NfvyCDT7S7qqrWzHHMlzJ4ROwhXSb8jvdn8IjaC6vqvvc4VdUjk/zSEs95WQbxZuOs9dMyeFfU6LuoXpTBJ9jN/tqa5Nbhr1+bRRq+E+qdSR5TVc+dvb2+8zF9M3eN1aztj0jyvDlOfefwe++/HwCgA3dGAQB7um1J/mb4iXRJ8uwMPtnuea217VX1mqr66SR/keSmDILIz2Zwd9If9By0tfb1qvpvSc5O8omqensGwegFGdyV9PglnPO6qrowyelVtTmDx+COTLIhyVUZfpLfcN8PzHWOqjo7yZ2ttT8d9/pJXp7kp5K8efjn/LEM/owfn8F/az6jtXZHVb0vySlV9c0M7uY6LMnzM/g7eeisc35y+P33q+qSDF7E/pnW2meWMB8AcD8jRgEAe7oXJ/mJDD4d7uFJPpfkl1trMxHmvUkekeT/HW7/Zgbh57Qkb+k9bGttU1XdmcH7k16bwR1JZ2fwqX9/vMTTbkxyc5L1SZ6aQaC7IMlZo59otxyGge3HMnhB/M8neXqSOzJ4IfsFI7uekuT3MgiBp2bwd/CyJN9K8tZZ5/x4Vb04g0h3cQb/zfrbScQoAFgBqjWP5AMAe56qelYGEeNJrbUPT3caAAAWyzujAAAAAOjGY3oAwF6tqvbN4l6UvbW1du/Cu+2+ac9UVQ9M8uCF9mutfWXS1wYAVj4xCgDY2x2f5EOL2O/7MngvUw/Tnum/ZtZ7nHahFt4FAOC7eWcUALBXq6qDk/zwInb9WGttx3LPk0x/pqp6RJKjFtpvV5/OBwAwHzEKAAAAgG68wBwAAACAbsQoAAAAALoRowAAAADoRowCAAAAoBsxCgAAAIBu/i+ACy3qCe/1rQAAAABJRU5ErkJggg=="/>

<pre>
<Figure size 432x288 with 0 Axes>
</pre>
<img src="data:image/png;base64,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"/>

<pre>
<Figure size 432x288 with 0 Axes>
</pre>
<img src="data:image/png;base64,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"/>

<pre>
<Figure size 432x288 with 0 Axes>
</pre>
<img src="data:image/png;base64,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"/>

<pre>
<Figure size 432x288 with 0 Axes>
</pre>
<img src="data:image/png;base64,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6pSvv7K19tapvQkAAAAAVszYywb3S1JJTpjTfnKSq5LsvxLXT5Ya3qyqapn1AgAAALAZjR1e7ZPk+iTnzG5sra1Pcv7k/LSv3znJFUl+mOSKqjqjqvbYiNoBAAAAWGFj73m1U5J1rbVr5jl3SZL7VdVWrbVrp3T9N5J8IsnnklyX5N5JDk7ykKr67dbaBZvyZgAAAACYrrHDq22TzBc8Jcn6WX02FF4t6/rW2lPn9HlnVb07yceSHJfkYfPdqKoOSnJQkuyyyy4bGA4AAACAaRt72eBVSbbewLltZvVZqevTWvt4kn9L8uCquskG+pzUWtu7tbb3jjvuuNDtAAAAAJiiscOrS5PsUFXzBVA7Z1gSuKFZV9O4fsZFSbZIsv0S+gIAAACwmYwdXn1mUsO9ZjdW1TZJ9kpy7gpfP+MOSX6a5HtL7A8AAADAZjB2eHVakpbksDntT8+wV9XbZhqqavd5ngq4nOtvXlVbzC2gqh6Z5P5JPjh5SiEAAAAAnRh1w/bW2gVV9dokB1fVGUnem2TPJIcmOTvJqbO6fzjJrklqI69/cJLjquqsJF/PMNPqXkn2T7IuvxyAAQAAADCysZ82mAyh0UUZnub3yAxB0olJjmytXT/F67+cYRnho5LcKsmNk/x3ktcnObq1dskmvxMAAAAApmr08Kq1dl2SYyevhfrttonXfzHJ4zeuSgAAAADGMPaeVwAAAACwQcIrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALolvAIAAACgW8IrAAAAALo1enhVVTeqqsOr6ktVtb6qLq6qY6vqpitxfVU9oqo+WVVXVtX3qur0qrrtdN8VAAAAANMweniV5PgkxyX5QpJDkpye5NAkZ1XVUupb8vVV9UdJ3pPkJkmem+TvkjwgySeqaqepvBsAAAAApmbLMQevqjtnCJzOaK09dlb7N5K8OskTkpw6jeur6sZJTkxycZL/1Vq7YtL+viTnJTkqyUFTfHsAAAAAbKKxZ17tl6SSnDCn/eQkVyXZf4rXPzDJTkneMBNcJUlr7fwkH0vyx5OACwAAAIBOjB1e7ZPk+iTnzG5sra1Pcv7k/LSun/n6U/Pc59NJbpbkjksrGwAAAIDNYezwaqck61pr18xz7pIkO1TVVlO6fqdZ7fP1TZKdl1AzAAAAAJtJtdbGG7zqa0lu3FrbZZ5zb07y5CTbt9Z+sKnXV9UbkxyQZPfW2tfn9D0gyRuTPKa1duY89zooP98P605JvrzkN7n67JBk3dhFsNF8fquXz2518/mtXj671c3nt7r5/FYvn93q5vNbvW7on92urbUd5zsx6obtGfal+rUNnNtmVp9pXD9z3Hq5Y7XWTkpy0gJ13GBU1bmttb3HroON4/NbvXx2q5vPb/Xy2a1uPr/Vzee3evnsVjef3+q1lj+7sZcNXpphad98gdLOGZYEXjul6y+d1T5f32T+JYUAAAAAjGTs8OozkxruNbuxqrZJsleSc6d4/Wcmx/vOc5/7JPlRkq8srWwAAAAANoex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"/>

<pre>
<Figure size 432x288 with 0 Axes>
</pre>
<img src="data:image/png;base64,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"/>

<pre>
<Figure size 432x288 with 0 Axes>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABLoAAAJdCAYAAAA1CR5gAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAAA89klEQVR4nO3de9StdVkv/O8FCAhu88CqHRRgaEGaSYLH/SYVtttSlpklJqUiZFsgqN3J3OSrpp0AhTBf0MwD+JKGFmYHNSXzsBGNxANqCEigvq48cligcL1/zPux6eN8TmvNZ028/XzGmGM+83dfv/u+5sMYz1jjy/373dXdAQAAAIBvdLssugEAAAAAmAdBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFEQdAEAAAAwCrstuoGx22efffrAAw9cdBsAAAAAo/He9753a3dvWT4u6NpkBx54YC699NJFtwEAAAAwGlV1zaxxSxcBAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFEQdAEAAAAwCoIuAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAo7LboBtj5Hvjrr1h0CwAreu8f/cKiWwAAAL5BLfyOrqrapapOqaorqmpbVV1bVadV1d7rmPvdVfXsqnp3VX2mqr5UVZdV1e+sNL+qvqeqXl9Vn6uqG6vq7VX1wyvUfktVnVVV1w29fbCqfrmqake/NwAAAADztfCgK8kZSU5P8qEkJyZ5TZKTklxUVWv195QkpyS5Msmzk/x6ko8keW6Sd1bVnaeLq+qgJO9M8tAkfzjU3yXJ31fVkctqd0/ypiRPS3LB0NtHkrwoye9u53cFAAAAYJMsdOliVd03kwDpwu5+7NT4VUnOTPL4JOevcorXJnl+d39hauzFVfWxJL+T5NgkfzJ17PlJ7pbkgd192XCtVyT5YJKzq+rg7u6h9qlJDk9yUnefNYydW1V/meQZVfWy7r5mO742AAAAAJtg0Xd0HZ2kkrxg2fi5SW5K8sTVJnf3pctCriUXDO/3WxoYljI+OsnblkKu4Rw3JHlJku/OJNha8oShh3OXnfsFSe6U5OdW6w0AAACAnWvRQdfhSW5Pcsn0YHdvS3JZvjZ42ojvGN4/PTV2/yR7JHnXjPp3T/WTYcnkDyT5l6GXaZck6R3oDQAAAIBNsOiga98kW7v7lhnHrkuyz7BX1rpV1a5J/neSr+Rrlz3uO3XeWddKkv2G97snufOs2qHXrVO1AAAAANwBLDro2ivJrJArSbZN1WzECzLZbP7U7v7Ismtlhestv9ZqtUv1K/ZVVcdX1aVVdelnPvOZdTUNAAAAwI5ZdNB1UybLCWfZc6pmXarqOUlOSHJOdz9/xrWywvWWX2u12qX6Ffvq7nO6+7DuPmzLli1rNw4AAADADlt00HV9JssTZwVK+2WyrPHW9Zyoqp6V5JlJXpbkaStca+m8s66V/OdSxc8luXlW7dDrPpm9BBIAAACABVl00PWeoYcHTQ9W1Z5JHpDk0vWcZAi5fjfJy5M8tbt7RtnlmSxFfOiMYw8Z3i9Nku6+Pcn7khw6I4R7UCZPilxXbwAAAADsHIsOui7I5AmGJy8bPy6TPbDOWxqoqoOq6uDlJ6iqUzMJuV6Z5ClDSPV1uvuGJBclOaKqvn9q/l2SPDXJx/K1T3989dDD8ctOdXImG91fsOa3AwAAAGCn2W2RF+/uy6vq7CQnVNWFSd6Y5JAkJyW5OF/71MS3JDkgk7upkiRV9fQk/3eSTyR5c5InVNXUlHy6u9809fm3k/xIkn+oqjOSfDGTUG2/JEctuxPs3CRPTnJ6VR2Y5MNJHpXkMUme291X79CXBwAAAGCuFhp0DU5OcnUmd04dlWRrkrMyeWrizLuzphw+vO+fybLF5S5O8tWgq7v/raoenuT3k/xWkt0zWaL4Y9395umJ3X1rVR2Z5LlJjk5yzyRXJjkxydnr/3oAAAAA7AwLD7q6+7Ykpw2v1eoOnDH2pCRP2uD1PpzkJ9dZ+/lMnuJ4wkauAQAAAMDOt+g9ugAAAABgLgRdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFEQdAEAAAAwCoIuAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFEQdAEAAAAwCoIuAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIzCQoOuqtqlqk6pqiuqaltVXVtVp1XV3uuc/9tV9Zqq+nhVdVVdvULdgcPx1V4/v876D8zp6wMAAAAwR7st+PpnJDkpyeuSnJbkkOHzoVV1ZHffvsb85yX5bJL3JbnbKnWfSXLMCsf+JMmdk/z9jGOvS3LhsrHPr9ETAAAAAAuwsKCrqu6b5MQkF3b3Y6fGr0pyZpLHJzl/jdMc1N0fH+Z9IMldZhV1941JXjWjh4cm+ZYkr+3urTOmvr+7v24eAAAAAHc8i1y6eHSSSvKCZePnJrkpyRPXOsFSyLUDnjq8v2Slgqras6r22sHrAAAAALDJFhl0HZ7k9iSXTA9297Yklw3HN01V3SXJzya5JsmbVij7tUxCtxuH/cOeXVV7bGZfAAAAAGyfRe7RtW+Srd19y4xj1yV5WFXt3t23btL1fy6TpY5/PGMvsNuT/GOS12cShG3JJBT730keWlU/1t23bVJfAAAAAGyHRQZdeyWZFXIlybapms0Kup6aSaD1suUHuvsTSX5k2fBLq+qcJMdlsn/YeSuduKqOT3J8kuy///7z6hcAAACAVSxy6eJNSVZaBrjnVM3cVdX3JnlIkjcNodZ6/d7wftRqRd19Tncf1t2HbdmyZXvbBAAAAGADFhl0XZ9knxX2vNovk2WNm3U317HD+4qb0K/g2iS3Jdlnvu0AAAAAsKMWGXS9Z7j+g6YHq2rPJA9IculmXLSqdk9yTJLPJPmrDU7/riS7Jvn0vPsCAAAAYMcsMui6IEknOXnZ+HGZ7M311T2wquqgqjp4Ttd9dCaby7+yu788q6Cq7jljbJckzx0+XjSnXgAAAACYk4VtRt/dl1fV2UlOqKoLk7wxySFJTkpycZLzp8rfkuSAJDV9jqo6ZhhPJuHV7lX1zOHzNd39yhmXXs+yxXOr6q5J3pnJcsV9kjw2yQMzuQvstev6kgAAAADsNIt86mIyuZvr6kyeUHhUkq1Jzkpyanffvo75xyZ5xLKx5wzvFyf5mqCrqr4zyY8meWd3f3iV8/5NJssbj09yj0yeDvnBJE9P8uJ19gYAAADATrTQoKu7b0ty2vBare7AFcaP2OD1rs1kj6216l6a5KUbOTcAAAAAi7XIPboAAAAAYG4EXQAAAACMgqALAAAAgFEQdAEAAAAwCoIuAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFEQdAEAAAAwCoIuAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMwsKDrqrapapOqaorqmpbVV1bVadV1d7rnP/bVfWaqvp4VXVVXb1K7Z8PNbNePzOjfo+qenZVXVVVt1TVlVX1zKq60w58ZQAAAAA2wW6LbiDJGUlOSvK6JKclOWT4fGhVHdndt68x/3lJPpvkfUnuts5rHjNj7JIZYxck+ckkf5bkXUkemuQ5Se6d5EnrvBYAAAAAO8FCg66qum+SE5Nc2N2PnRq/KsmZSR6f5Pw1TnNQd398mPeBJHdZ67rd/ap19PaoTEKu07v714bhl1TV55P8alWd093vXOs8AAAAAOwci166eHSSSvKCZePnJrkpyRPXOsFSyLURNXHXqlrt+z9heF/e29LnNXsDAAAAYOdZdNB1eJLbs2zZYHdvS3LZcHwzfGF43VxVb6qqB6/Q23Xdfe2y3q5Ncv0m9gYAAADAdlh00LVvkq3dfcuMY9cl2aeqdp/j9T6VyZ5gv5zkMZns73VYkrdX1ZEzertuhfNcl2S/OfYFAAAAwA5a9Gb0eyWZFXIlybapmlvncbHu/q1lQ6+vqvMzuXvsT5PcZwO97bXSdarq+CTHJ8n++++/ve0CAAAAsAGLvqPrpiR7rHBsz6maTdPdH0vyF0nuXVXfvYHeVuyru8/p7sO6+7AtW7bMr1kAAAAAVrTooOv6TJYnzgqU9stkWeNc7uZaw9XD+z5TY9dn5eWJ+2XlZY0AAAAALMCig673DD08aHqwqvZM8oAkl+6kPpaWLH56auw9Sfarqu+cLhw+77sTewMAAABgHRYddF2QpJOcvGz8uEz2wDpvaaCqDqqqg7f3QlW19xCgLR8/NMnjkny4u6+cOvTq4X15b0ufzwsAAAAAdxgL3Yy+uy+vqrOTnFBVFyZ5Y5JDkpyU5OIk50+VvyXJAUlq+hxVdcwwniRbkuxeVc8cPl/T3a8cfr5Pkr+tqtcn+ViSG5N8f5KnJLktw+bxU739TVW9IcmvVtW3JHlXkocmOTbJq7r7n3fw6wMAAAAwR4t+6mIyuUPq6kyCpqOSbE1yVpJTu/v2dcw/Nskjlo09Z3i/OMlS0PWpJG9O8kNJfj7JnZN8MpO7yp7f3VfMOPfjkjwzyROTHJPJvlynJvn9dfQFAAAAwE608KCru29LctrwWq3uwBXGj1jndT6VSVi1kd62ZRJ0PXOtWgAAAAAWa9F7dAEAAADAXAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFEQdAEAAAAwCoIuAAAAAEZB0AUAAADAKOy26AYAgO3ziWd/36JbAFjR/qdevugWAPgm5I4uAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFFYaNBVVbtU1SlVdUVVbauqa6vqtKrae53zf7uqXlNVH6+qrqqrV6jbs6qOq6q/qqqrq+rmYc6rq+qQGfUHDueb9frADn5tAAAAADbBbgu+/hlJTkryuiSnJTlk+HxoVR3Z3bevMf95ST6b5H1J7rZK3YFJzknyz0lemuT6JN+V5JeT/HRV/Vh3v3XGvNcluXDZ2OfX6AkAAACABVhY0FVV901yYpILu/uxU+NXJTkzyeOTnL/GaQ7q7o8P8z6Q5C4r1H0myaHdfdmyHs5L8i9J/ijJYTPmvb+7X7X2twEAAABg0Ra5dPHoJJXkBcvGz01yU5InrnWCpZBrHXX/sTzkGsY/lOQDSe630txh2eNe67kOAAAAAIuzyKDr8CS3J7lkerC7tyW5bDi+qapqlyTfnuTTK5T8Wiah243D/mHPrqo9NrsvAAAAADZukUHXvkm2dvctM45dl2Sfqtp9k3t4WiZB18uXjd+e5B+TPCPJTyV5apIPJfnfSd5QVbtucl8AAAAAbNAiN6PfK8mskCtJtk3V3LoZF6+qhyU5Pcm/ZrKp/Vd19yeS/MiyKS+tqnOSHJfJ/mHnrXLu45McnyT777//HLsGAAAAYCWLvKPrpiQrLQPcc6pm7qrqgUn+JpOnLx41LJdcj98b3o9arai7z+nuw7r7sC1btuxApwAAAACs1yKDruszWZ44K+zaL5NljXO/m6uqfiDJm5J8IckPdfd1G5h+bZLbkuwz774AAAAA2DGLDLreM1z/QdODVbVnkgckuXTeFxxCrjcn+VImIdc1GzzFdyXZNStvXg8AAADAgiwy6LogSSc5edn4cZnszfXVPbCq6qCqOnhHLlZVh2ZyJ9cNmYRcV61Se88ZY7skee7w8aId6QUAAACA+VvYZvTdfXlVnZ3khKq6MMkbkxyS5KQkFyc5f6r8LUkOSFLT56iqY4bxJNmSZPeqeubw+ZrufuVQd0AmIdfdk5yZ5GHDZvTTXtfdNw4/n1tVd03yzkyWK+6T5LFJHpjkr5K8dke+OwAAAADzt8inLiaTu7muzuQJhUcl2ZrkrCSndvft65h/bJJHLBt7zvB+cZJXDj/fK8nSXVrPWuFc90qyFHT9TZJjhr7ukcnTIT+Y5OlJXrzO3gAAAADYiRYadHX3bUlOG16r1R24wvgR67zO27LsbrA16l+a5KXrrQcAAABg8Ra5RxcAAAAAzI2gCwAAAIBRWHfQVVU/WFVbVjm+T1X94HzaAgAAAICN2cgdXW9N8shVjv/IUAMAAAAAO91Ggq61NnPfNYmnEQIAAACwEBvdo6tXOfawJFt3oBcAAAAA2G67rXawqn4lya9MDb2gqn5vRundk9w1yZ/NsTcAAAAAWLdVg64kn09yzfDzgUn+I8mnl9V0kg8keXeSM+bYGwAAAACs26pBV3e/PMnLk6SqrkryW9391zujMQAAAADYiLXu6Pqq7r7XZjYCAAAAADtio5vRp6p+sKqeW1XnVtXBw9hdhvG7zb1DAAAAAFiHdQddVbVrVV2Q5K1JnpHkKUn2HQ5/Jcnrk/zPeTcIAAAAAOuxkTu6fjPJY5P8apJDktTSge7eluR1SR411+4AAAAAYJ02EnT9QpJXdPcLk2ydcfzDSQ6aS1cAAAAAsEEbCboOTPKuVY5/Psndd6QZAAAAANheGwm6vpTkHqscv3eSz+xYOwAAAACwfTYSdP1zkidWVS0/UFV3z2Rz+rfOqzEAAAAA2IiNBF2/l+Q+Sf4xyY8PY99fVb+U5H1J9k7y+/NtDwAAAADWZ7f1Fnb3pVX12CQvSfKyYfiPM3n64v+X5DHd/aH5twgAAAAAa1t30JUk3f03VXVgkkcmOSSTkOtjSf6+u2+af3sAAAAAsD4bCrqSpLtvSfKG4QUAAAAAdwgb2aMLAAAAAO6w1n1HV1V9fI2STnJzkk8k+Yck53b3jTvQGwAAAACs20bu6PpEkq8kOTDJ3ZN8fnjdfRj7SiZB10OSnJ7kvVW1ZW6dAgAAAMAqNhJ0nZzkHkn+Z5Jv7e4f6O4fSLIlyQnDsWOT7JPkxCT3SfLsuXYLAAAAACvYyGb0f5zkgu5+8fRgd38lyYuq6n5JTuvuRyY5u6oemuSo+bUKAAAAACvbyB1dD07y/lWOvz+TZYtL3pnk27anKQAAAADYqI0EXbckOXyV4w8aapbskeSG7WkKAAAAADZqI0HXXyd5clX9VlXttTRYVXtV1W8n+cWhZsnDknx0Pm0CAAAAwOo2skfX/0pyaJLnJXl2VV0/jO87nOfyJL+eJFW1Z5JtSc6eX6sAAAAAsLJ1B13d/dmqenCSpyb58ST3Gg69JclFSV7S3bcOtduSHDPnXgEAAABgResKuqrqzkkel+Qj3f2iJC/a1K4AAAAAYIPWu0fXLUleksnSRQAAAAC4w1lX0NXdtyf5RJK7bm47AAAAALB9NvLUxZcnOaaq9tisZgAAAABge23kqYvvTPLTSS6rqhcl+ViSm5YXdfc/zak3AAAAAFi3jQRdb5r6+YVJetnxGsZ23dGmAAAAAGCjNhJ0PXnTugAAAACAHbTuoKu7X76ZjQAAAADAjtjIZvQAAAAAcIe1kaWLSZKq+rYkhyW5e2YEZd39ijn0BQAAAAAbsu6gq6p2SXJ2kqdm9TvBBF0AAAAA7HQbWbr4v5L8UpJXJ/nFTJ6y+FtJnp7kY0kuTfLIeTcIAAAAAOuxkaDrF5P8XXf/QpK/Hcbe290vTvLAJPsM7wAAAACw020k6PquJH83/Hz78H6nJOnuG5O8LJNljQAAAACw020k6Lo5yZeHn29I0km+der4p5J855z6AgAAAIAN2UjQdU2Sg5Kku7+c5N+S/NjU8SOTfHp+rQEAAADA+m0k6PrHJI+Z+vzKJEdX1Vur6m1JHpfkL+bYGwAAAACs224bqP3jJP9QVXt09y1Jnp/J0sUnJrktyTlJnjX3DgEAAABgHdYddHX3J5N8curzbUlOGl4AAAAAsFDrXrpYVadW1f1WOX7fqjp1ow1U1S5VdUpVXVFV26rq2qo6rar2Xuf8366q11TVx6uqq+rqNeofXFVvrqovVdUXq+rvquoBK9TuW1WvqKrPVNXNVXVpVT1uo98RAAAAgM23kT26npXk/qscv1+S392OHs5IcnqSDyU5MclrMrlL7KKqWk9/z0vyw0muTPK51Qqr6iFJLk5yrySnDv3eJ8nbq+r7ltXeI8k/J/npJH+a5FcyedrkX1TVk9f75QAAAADYOTayR9da9kzylY1MqKr7ZhJuXdjdj50avyrJmUken+T8NU5zUHd/fJj3gSR3WaX2zCS3JvnB7r5umPMXST6c5LQkPzpV+1uZBGKP7u6LhtqXJnlXkj+uqtd09w3r/a4AAAAAbK5V75iqqrtW1f5Vtf8wdM+lz8teD0jy80mu3eD1j05SSV6wbPzcJDdlstH9qpZCrrVU1b2THJ7kNUsh1zD/ukzuIjuyqv7r1JQnJLlyKeQaam9LclaSeyR51HquCwAAAMDOsdbSwFOSXDW8OpNA6qoZr/cmOTLJizd4/cOT3J7kkunB7t6W5LLh+LwsnetdM469O5PA7YFJUlXfnmS/YXxW7fT5AAAAALgDWGvp4tuG98pkT6vXJXn/sprOZO+qd3f3Ozd4/X2TbO3uW2Ycuy7Jw6pq9+6+dYPnXelaS+edda1kEm5ttBYAAACAO4BVg67uvjiTzdtTVQckeXF3/585Xn+vJLNCriTZNlUzj6Brr+F91vW2LavZSO3XqarjkxyfJPvvv/9KZQAAAADM0bqfutjdT55zyJVM9uHaY4Vje07VzOtaWeF6y6+1kdqv093ndPdh3X3Yli1bNtwoAAAAABu37qBrk1yfZJ+qmhUo7ZfJssZ53M21dK2l8866VvKfyxI3UgsAAADAHcCig673DD08aHqwqvZM8oAkl875Wkny0BnHHpLJXmPvTZLu/mQmQdZDVqjNnHsDAAAAYActOui6IJOA6eRl48dlsgfWeUsDVXVQVR28vRfq7n/LJJx6XFUtbTaf4efHJfnH7v7U1JRXJzmoqn5iqnbXJCcm+XySN25vLwAAAADM31pPXdxU3X15VZ2d5ISqujCT8OiQJCdlsgn++VPlb0lyQCZPgPyqqjpmGE+SLUl2r6pnDp+v6e5XTpX/SpK3Jnl7VZ01jJ2YSeD3a8va+/1MArDzq+r0TO7wOjrJ4Ume2t1f2r5vDQAAAMBmWGjQNTg5ydWZPKXwqCRbk5yV5NTuvn0d849N8ohlY88Z3i9O8tWgq7vfWVVHJHnu8Ook70zyuO7+1+kTdPd/VNXDMwm8np7kLkk+lOTx3X3Bur8dAAAAADvFDgddVbVPd2/d3vndfVuS04bXanUHrjB+xAav964kP7LO2uuSHLOR8wMAAACwGNu1R1dV7VFVf1JVNyb5dFXdXFUvqaq7zLk/AAAAAFiX7b2j64+S/Fgme2ldm+T+SZ6ZSXD2lPm0BgAAAADrt2rQVVUHdPc1Mw49OsnPd/c7hs//UFVJ8ptz7g8AAAAA1mWtO7o+WFW/k+TM7u6p8S8l+Y5ltfsluXGezQEAAIzdw896+KJbAFjRO058x9pFdyBrBV2/kOTMJD9fVcd29+XD+J8meVlVHZXJ0sXvS/KoJL+zaZ0CAAAAwCpW3Yy+uy9M8r1J3pfkPVX1vKrao7tflOTJSb4tyU8luXOSY7v7Dza5XwAAAACYac3N6Lv7i0meVlWvSnJOkp+pql/q7guSXLDZDQIAAADAeqx6R9e07v7nJA9I8uokf1tVL62qu21SXwAAAACwIesOupKku2/t7t9N8gNJDk5yRVX93KZ0BgAAAAAbsGrQVVV3rqoXVtW1VfXZqrqoqu7d3R/q7ocneXaS/6eq3lBV37lzWgYAAACAr7fWHV2nZbLp/EuTPCvJvZNcVFW7JsmwKf19k3wlyQer6qTNaxUAAAAAVrZW0PXTSZ7X3c/q7jOTHJ3kuzN5EmOSpLuv6+6fyiQQ+83NahQAAAAAVrNW0FVJeupzL3v/zwPdf5nkkDn1BQAAAAAbstsax1+f5BlVtXuSzyV5WpKPJfnwrOLu/uJcuwMAAACAdVor6PrVTPbf+uUkd07yriQnd/dtm90YAAAAAGzEqkFXd9+Y5OnDCwAAAADusNbaowsAAAAAviEIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFEQdAEAAAAwCoIuAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGIWFBl1VtUtVnVJVV1TVtqq6tqpOq6q95zm/qo6oql7j9fB11r9h3r8HAAAAAHbcbgu+/hlJTkryuiSnJTlk+HxoVR3Z3bfPaf6HkxwzY/4eSc5JsjXJJTOOn5Pk7cvG/n2tLwUAAADAzrewoKuq7pvkxCQXdvdjp8avSnJmkscnOX8e87v700leNeMcR2dyV9sruvvLMy7zru7+unkAAAAA3PEscuni0UkqyQuWjZ+b5KYkT9zk+Uny1OH9JSsVVNXeVbXnOs4FAAAAwAItMug6PMntWbZksLu3JblsOL5p86vqXkl+KMk/d/dHVih7YZIbktxcVR+tql+pqlqjLwAAAAAWYJFB175Jtnb3LTOOXZdkn6rafRPnPyWTO8Jm3c315SR/neQ3kjw6ydOSfD6Tu8f+bJVzAgAAALAgi9yMfq8ks0KqJNk2VXPrvOdX1a5JnpTki0les/x4d78jyU8um3NukjcmeVJVvWSomamqjk9yfJLsv//+K5UBAAAAMEeLvKPrpkyeejjLnlM1mzH/vyf5jiSv7u7VrvFVwxMcnz98PGqN2nO6+7DuPmzLli3rOT0AAAAAO2iRQdf1mSwvnBVW7ZfJssSV7uba0fnHDu8rbkK/gquH9302OA8AAACATbbIoOs9w/UfND04POHwAUku3Yz5VfWtSX4iyb9291rXWO4+w/unNzgPAAAAgE22yKDrgiSd5ORl48dlsrfWeUsDVXVQVR28vfOX+YUkd0ry0pUaq6p7zhjbI8mzho8XrTQXAAAAgMVY2Gb03X15VZ2d5ISqujCTjd4PSXJSkouTnD9V/pYkB2TylMTtmT/t2Ew2q3/VKu39XVVdn+S9mSyR3DfJEzO5o+us7r5kg18XAAAAgE22yKcuJpO7sa7O5AmFRyXZmuSsJKcOm7/PdX5VPSzJwUnO7+7PrXLe1yb5qSQnJrlbkhuT/EuS3+3uV6+jLwAAAAB2soUGXd19W5LThtdqdQfuyPyp+ndm6q6wVer+IMkfrOecAAAAANwxLHKPLgAAAACYG0EXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFEQdAEAAAAwCoIuAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFEQdAEAAAAwCoIuAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgIugAAAAAYBUEXAAAAAKOw8KCrqnapqlOq6oqq2lZV11bVaVW197znV9XbqqpXeB02o/5bquqsqrpuOPcHq+qXq6rm8d0BAAAAmJ/dFt1AkjOSnJTkdUlOS3LI8PnQqjqyu2+f8/ytSU6ZcZ6PT3+oqt2TvCnJoUnOSvLhJP8jyYuSfFuSZ63z+wEAAACwEyw06Kqq+yY5McmF3f3YqfGrkpyZ5PFJzp/z/Bu7+1XraO+pSQ5PclJ3nzWMnVtVf5nkGVX1su6+Zh3nAQAAAGAnWPTSxaOTVJIXLBs/N8lNSZ64GfOH5Y53XWMJ4hOGc5y7bPwFSe6U5OfW6A0AAACAnWjRQdfhSW5Pcsn0YHdvS3LZcHze8/dLckOSLyS5oaourKqDpwuqapckP5DkX4ZzTbskSa+jNwAAAAB2okXv0bVvkq3dfcuMY9cleVhV7d7dt85p/lVJ3pHk/UluS/LgJCck+ZGq+m/dfflQd/ckdx7O8TW6+5aq2ppJYAYAAADAHcSig669kswKqZJk21TNSkHXhuZ395OX1by2qv46yduSnJ7kkVNzssa591rhWKrq+CTHJ8n++++/UhkAAAAAc7TopYs3JdljhWN7TtVs1vx099uT/FOSH6qqOy+bs9q5Vzxvd5/T3Yd192FbtmxZ7fIAAAAAzMmig67rk+xTVbMCpf0yWZa40t1c85i/5Ooku2ayZDFJPpfk5sxYnjhca5/MWNYIAAAAwOIsOuh6z9DDg6YHq2rPJA9Icukmz19ynyRfSfLZJOnu25O8L8mhM0K0B2XypMf1nhsAAACAnWDRQdcFmTzB8ORl48dlsgfWeUsDVXXQ8qcjbnD+t1TVrssbqKqjkjw8yZuWPWHx1cM5jl825eRMQrELVv5aAAAAAOxsC92Mvrsvr6qzk5xQVRcmeWOSQ5KclOTiJOdPlb8lyQGZ3E21PfN/KMnpVXVRko9nElY9KMkTk2zN14dl5yZ58jDnwCQfTvKoJI9J8tzuvnoHvz4AAAAAc7Topy4mk4Dp6kzunDoqk9DprCSnDksI5zX/I5ksN/zxJN+W5E5J/j3Ji5M8r7u/Zs+t7r61qo5M8twkRye5Z5Irk5yY5OwNf0sAAAAANtXCg67uvi3JacNrtboDd3D+h5P87AZ7+3ySE4YXAAAAAHdgi96jCwAAAADmQtAFAAAAwCgIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFEQdAEAAAAwCoIuAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFEQdAEAAAAwCoIuAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgsNOiqql2q6pSquqKqtlXVtVV1WlXtPc/5VXX3qvqVqvqHoebmqvpIVZ1TVd8547xHVFWv8HrDvL4/AAAAAPOz24Kvf0aSk5K8LslpSQ4ZPh9aVUd29+1zmv/g4fhbkvxJkq1J7pfkl5L8bFU9rLs/NOP85yR5+7Kxf9/YVwQAAABgZ1hY0FVV901yYpILu/uxU+NXJTkzyeOTnD+n+Vck+Z7uvnLZOf4myZuSPDvJz8y4zLu6+1Ub/3YAAAAA7GyLXLp4dJJK8oJl4+cmuSnJE+c1v7uvXh5yDeNvTvLZTO7umqmq9q6qPdfoBQAAAIAFW2TQdXiS25NcMj3Y3duSXDYc38z5qapvSfJfknx6hZIXJrkhyc1V9dFhn69a67wAAAAA7HyLDLr2TbK1u2+Zcey6JPtU1e6bOD9JfifJnZK8fNn4l5P8dZLfSPLoJE9L8vlM7h77szXOCQAAAMACLHIz+r2SzAqpkmTbVM2tmzG/qn4myf9K8ndJXjZ9rLvfkeQnl9Wfm+SNSZ5UVS8ZamaqquOTHJ8k+++//0plAAAAAMzRIu/ouinJHisc23OqZu7zq+pRSc5L8t4kP9fdvXqryfAEx+cPH49ao/ac7j6suw/bsmXLWqcGAAAAYA4WGXRdn8nywllh1X6ZLEtc6W6u7Z5fVT+W5MIkH0zyo939xQ30fPXwvs8G5gAAAACwEywy6HrPcP0HTQ8OTzh8QJJL5z1/CLlen+SKJEd29+c22PN9hveVNq8HAAAAYEEWGXRdkKSTnLxs/LhM9tY6b2mgqg6qqoO3d/5wjh9N8rokH0nyI9392ZUaq6p7zhjbI8mzho8XrTQXAAAAgMVY2Gb03X15VZ2d5ISqujCTjd4PSXJSkouTnD9V/pYkBySp7ZlfVYcl+ath/suS/I+qyrTuftXUx7+rqusz2cPr+kye8PjETO7oOqu7L9nhXwAAAAAAc7XIpy4mk7uxrs7kCYVHJdma5Kwkpw6bv89r/v3ynxvUn7HCuaaDrtcm+akkJya5W5Ibk/xLkt/t7levoy8AAAAAdrKFBl3dfVuS04bXanUH7uD8P0/y5xvo6w+S/MF66wEAAABYvEXu0QUAAAAAcyPoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFEQdAEAAAAwCoIuAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAAABGQdAFAAAAwCgIugAAAAAYBUEXAAAAAKMg6AIAAABgFARdAAAAAIyCoAsAAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGAVBFwAAAACjIOgCAAAAYBQEXQAAAACMgqALAAAAgFEQdAEAAAAwCoIuAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUFh50VdUuVXVKVV1RVduq6tqqOq2q9t6M+VX1qKp6Z1XdWFWfrarXVNW9Vqj9nqp6fVV9bqh/e1X98I58XwAAAAA2x8KDriRnJDk9yYeSnJjkNUlOSnJRVa2nv3XPr6qfTvKGJHdO8utJ/ijJDyZ5R1Xtu6z2oCTvTPLQJH841N8lyd9X1ZHb9U0BAAAA2DS7LfLiVXXfTMKpC7v7sVPjVyU5M8njk5w/j/lVdackZyW5Nsn/1d03DON/m+S9SZ6V5Pip0z8/yd2SPLC7LxtqX5Hkg0nOrqqDu7u3/9sDAAAAME+LvqPr6CSV5AXLxs9NclOSJ85x/iOS7JvkJUshV5IMIdbbkvzcEIZlWPb46CRvWwq5htobkrwkyXcnOXyN3gAAAADYiRYddB2e5PYkl0wPdve2JJdl7TBpI/OXfn7XjPO8O8ldMwmwkuT+SfZYpXb6fAAAAADcASw66No3ydbuvmXGseuS7FNVu89p/r5T47Nqk2S/7agFAAAA4A5goXt0JdkryayQKkm2TdXcOof5ew2fZ9VP12aDtV+nqo7Pf+73dUNVfWSlWhiJfZJsXXQTjEP98S8uugX4ZubvOfPzu7XoDuCbmb/nzE2ddIf9e37ArMFFB103JfnWFY7tOVUzj/lL73vMufbrdPc5Sc5Z6TiMTVVd2t2HLboPAHaMv+cA4+DvOd/MFr108fpMlhfOCpT2y2RZ4kp3c210/vVT47Nqk/9clriRWgAAAADuABYddL1n6OFB04NVtWeSByS5dI7z3zO8P3TGeR6S5ItJPjp8vjyTZYsr1WYdvQEAAACwEy066LogSSc5edn4cZnsgXXe0kBVHVRVB2/v/CQXJ/lkkqdW1V2mzvv9SY5I8pru/nKSdPcNSS5KcsRwfKn2LkmemuRjWfakR/gmZ6kuwDj4ew4wDv6e802runuxDVSdleSEJK9L8sYkhyQ5Kck7kvxwd98+1F2d5IDuru2ZP9Q+LpNw7F+TnJvkrklOySQse2B3XzdVe+9MwqwvJzkjkzu+jkvyfUmO6u6/n+fvAQAAAIAdc0cIunbN5I6s45McmMmTIS5IcupwZ9VS3dWZHXSta/5U/Y8neWaS+2eyPPEtSX6zu6+cUXtIkt9P8ogkuyd5X5Jndfebt/8bAwAAALAZFh50AQAAAMA8LHqPLuAbXFU9qKrOrKp3VNUNVdVV9aRF9wXA2qpql6o6paquqKptVXVtVZ1WVXsvujcA1q+qfruqXlNVHx/+PX71onuCRRF0ATvqUUmenuRumex/B8A3jjOSnJ7kQ0lOTPKaTPY6vaiq/DsR4BvH85L8cJIrk3xuwb3AQu226AaAb3h/muSPuvvGqvqZJA9bdEMArK2q7ptJuHVhdz92avyqJGcmeXyS8xfUHgAbc1B3fzxJquoDSe6y4H5gYfyfOmCHdPenu/vGRfcBwIYdnaSSvGDZ+LlJbkryxJ3dEADbZynkAgRdAADfrA5PcnuSS6YHu3tbksuG4wAA31AEXQAA35z2TbK1u2+Zcey6JPtU1e47uScAgB0i6AIA+Oa0V5JZIVeSbJuqAQD4hmEzemBNVbVrki3Lhm/u7i8soh8A5uKmJN+6wrE9p2oAAL5huKMLWI/vTPLJZa8XLrQjAHbU9ZksT9xjxrH9MlnWeOtO7gkAYIe4owtYj08leeSysesX0QgAc/OeJD+a5EFJ3r40WFV7JnlAkn9aTFsAANtP0AWsaXgC15sX3QcAc3VBkmckOTlTQVeS4zLZm+u8BfQEALBDBF3ADqmqA5IcM3y87/D+E1X1HcPPr+zua3Z+ZwCsprsvr6qzk5xQVRcmeWOSQ5KclOTiJOcvsj8A1q+qjklywPBxS5Ldq+qZw+druvuVi+kMdr7q7kX3AHwDq6ojkrx1lZIf6u637ZRmANiQ4WEjJyc5PsmBSbZmcqfXqd19w+I6A2AjquptSR6xwuGLu/uIndcNLJagCwAAAIBR8NRFAAAAAEZB0AUAAADAKAi6AAAAABgFQRcAAAAAoyDoAgAAAGAUBF0AAAAAjIKgCwAAAIBREHQBAAAAMAqCLgAAVlRV+1bVK6rqM1V1c1VdWlWPm1H351XVq7w+toj+l6uqk6vqSYvuAwDYHNXdi+4BAIA7oKq6R5JLk3xrktOT/HuSJyR5RJKndPfLpmofmuSgGaf54SRPTnJGd//qpje9hqq6OsnV3X3EglsBADaBoAsA4JtEVe2aZI/uvmmd9X+Y5NeTPLq7L5o6x7syCbUO6O4b1jjH3yf50ST36+4P7kj/8yDoAoBxs3QRAGBKVT1pWGp3ZFU9q6quqapbqur9VfX4ZbUPq6q/rapPVdW2qrquqt5YVQ/ZjuvuXlW/UVWXVdVNVfWFYZngCVM1+1bVaUPN54ZrfqiqfnMIoFb6Hv+7qq5Msi3Jz26grSckuXIp5EqS7r4tyVlJ7pHkUWt8pwOSHJnk3dsbclXVoVX1mqr69PDf4dqqenVVHTRV83NV9ddV9YmhZmtVvb6q7r/sXJ3kgCSPWLas8sDt6Q0AuOPZbdENAADcQf1Bkr2TvGj4/OQkr66qPbv7z6vqe5K8KcmnkrwwyaeTfFuS/5bk+5O8e70Xqqrdk/x9kiOS/EOSV2USSn1fkp9O8idD6f2Hz69LcmWSOyX5sSS/n+S7kvzSjNP/8VB3bpIvJvnIOnv69iT7JTlvxuGl73Z4kr9Y5TRPzuR/rL5kPdec0cOPJ/nLJDcO5/i3JP81yX9Pcr9MfgdJckKS/0hyTib/PQ5KcnySd1TVD3T30v5gxyQ5I8nWJL83danPbE9/AMAdj6WLAABTho3KX5bkE0nu391fGMa/Jcn7k/yXTAKg4zIJuB7c3Zfs4DV/I5Ng7fnd/Yxlx3bp7tuHn++cZFsv+wdcVb0yk7uvvqO7P7nse3w0yaHrXa44dc4HZrI/1x92928uO7ZXJuHTq7v7CSvM3yXJVZnc+fXtay1xnDF/ryTXJOmh/+uWn3/q97J3d9+47PghSS5L8tLu/p9T41fH0kUAGC1LFwEAZvvTpZArSYafX5zk7pncebV07Ceras8dvNbPJ/lckmcvP7AU5gw/37wUcg1LHe9RVftkcjfYLkkOW+F7bCjkGuw1vN8y49i2ZTWzPDLJ/kku2GjINfjvSfZJctrykCv5ut/LjUlSE3cdfiefyeTutQdvx7UBgG9Qgi4AgNk+PGPsQ8P7dyX5f5O8Ockzkny2qv5x2CvrgO241n2SXNHd21YrqqrdquqZVfXRTMKm/8gk0HnlUHL3GdM+uh39JMlSOLbHjGN7LquZ5djhfbuWLWbyO0mSf1mrcNjH6w1JvpRJAPmZ4fV9mf07AQBGStAFALAduvuW7n5kJncMPT/JbZnckXVFVT1mky57epLnJHlfJvtfPSqTO6eWlhbO+rfd9tzNlSTXD+/7zTi2NPZ1d1olSVXdM8lPJvlAd697r7LtUVX7J/mnJIdm8rt5TCZPeXxkkg/Gv3cB4JuKzegBAGY7JMlfLRv73uH940sDw/5clyRJVX1nJncgPTeTDePX66NJDq6qPbp71lLBJcck+afuXv70x3tv4Frr0t2frKrrksx6guTS2KUrTP+FJLsneekOtLB0J9oDMtmgfyWPSXKXJI/u7rdOHxgCt+W/TxvUAsCI+T9cAACz/fKwAX2Sr25G/7Qkn09y8bAP1HL/nsmSuXts8FrnZbLE7pnLD1RVTX28LUktO753klM2eL31enWSg6rqJ6aut2uSEzP5PbxxhXnHJrk1/7mkcnv8QyZPR/y14QmQX2Pq93Lb0tCy48dl8oTG5W7Ixv/7AADfINzRBQAw29Yk/6eqXjZ8fnImm6s/tbtvqqrnVdWPJnlDJk8XrCQ/keTgJH+4wWu9cJj7zKo6PJOQZ1uS+yb5niRHDnWvTfJLVXVBJvuDfVuSp2SyV9dm+P0kj0tyflWdnslSxaOTHJ7J7+FLyydU1YOHvv+iu7e7r+F3fGwm3/kDVfWSJP+WZEsmG9Wfnskdd3+byfLMV1bVn2Syqf/DM1nWeWW+/t+7705ybFU9J5N92G5PctHypzYCAN+YBF0AALP9ZpL/K8nTMwmUPprk57v7/OH465N8e5KfHY7fnORjSY7LBpfsdfetQ2j2a0mekOR5mQRdH0vysqnSX81kw/WfzWQPrGuTnJPkPZkEX3PV3f9RVQ/PJPB6eiZLBD+U5PHdfcEK03Z0E/rp6/91Vf23TDb8PzbJf0ny6SRvT3L5UHNlVf2PTH5nz8jkDq93JHlEkj9JcuCy0/5OJnd0PT3J3TIJKO+VRNAFACNQwxOqAQBIUlVPyiRc+qHufttiuwEAYCPs0QUAAADAKFi6CACwCapq96xv0/PPdPdta5fNx7Cp/p3XKLu1uz+7Sde/RyZPZFzNzd39hc24PgAwboIuAIDN8bAkb11H3b2SXL25rXyNFyb5xTVqLk5yxCZd/8JM9s9azcuTPGmTrg8AjJg9ugAANkFV3T3JA9dR+s/dvW2z+1lSVd+bZN81yj7X3e/dpOs/MMnd1yi7vrs/tBnXBwDGTdAFAAAAwCjYjB4AAACAURB0AQAAADAKgi4AAAAARkHQBQAAAMAoCLoAAAAAGIX/H+j6y5I4OUj/AAAAAElFTkSuQmCC"/>

<pre>
<Figure size 432x288 with 0 Axes>
</pre>
<img src="data:image/png;base64,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"/>

<pre>
<Figure size 432x288 with 0 Axes>
</pre>
<img src="data:image/png;base64,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"/>

<pre>
<Figure size 432x288 with 0 Axes>
</pre>
<img src="data:image/png;base64,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"/>

- 결측값을 최빈값 등으로 대체하는 대신 별도의 범주 값으로 유지하는 것이 좋음

- 결측값을 가진 고객들이 보험금 청구를 요청할 확률이 훨씬 높은(경우에 따라 훨씬 낮은) 것으로 보임


## **8-2. 구간 변수**

- 변수들 간의 **상관계수** 확인

- ```heatmap```은 변수들 간의 상관 관계를 시각화하는 좋은 방법



```python
### heatmap 시각화를 위한 함수

def corr_heatmap(v):
    ### 변수들 간의 상관계수
    correlations = train[v].corr() 

    ### 시각화
    cmap = sns.diverging_palette(220, 10, as_cmap = True) # colormap 시각화 

    fig, ax = plt.subplots(figsize = (10,10))
    sns.heatmap(correlations, cmap = cmap, vmax = 1.0, center = 0, 
                fmt='.2f', square=True, linewidths=.5, annot=True, 
                cbar_kws={"shrink": .75})
    
    plt.show()
```


```python
v = meta[(meta.level == 'interval') & (meta.keep)].index
corr_heatmap(v)
```

<img src="data:image/png;base64,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"/>

- 해당 변수 사이에는 강력한 상관 관계가 존재함

  - ```ps_reg_02``` & ```ps_reg_03``` (0.7)

  - ```ps_car_12``` & ```ps_car13``` (0.67)

  - ```ps_car_12``` & ```ps_car14``` (0.58)

  - ```ps_car_13``` & ```ps_car15``` (0.67)


- ```Seaborn```에는 변수 간의 (선형) 관계를 시각화할 수 있는 유용한 도구가 있음

  - ```pairplot```을 사용하여 변수 간의 관계를 시각화할 수 있음

  - 상관성이 높은 변수를 개별적으로 살펴보자!



```python
### 처리 속도로 인해 sample을 추출하여 관찰

s = train.sample(frac = 0.1)
```

### **✅ ps_reg_02 & ps_reg_03**



```python
sns.lmplot(x = 'ps_reg_02', y = 'ps_reg_03', data = s, hue = 'target', 
           palette = 'Set1', scatter_kws = {'alpha':0.3})
plt.show()
```

<img src="data:image/png;base64,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"/>

- 회귀선에서 알 수 있듯이 변수 사이에는 선형 관계가 있음

- 색상 매개변수 덕분에 target = 0과 target = 1의 회귀선이 동일함을 알 수 있음


### **✅ ps_car_12 and ps_car_13**



```python
sns.lmplot(x = 'ps_car_12', y = 'ps_car_13', data = s, hue = 'target', 
           palette = 'Set1', scatter_kws = {'alpha':0.3})
plt.show()
```

<img src="data:image/png;base64,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"/>

### **✅ ps_car_12 and ps_car_14**



```python
sns.lmplot(x = 'ps_car_12', y = 'ps_car_14', data  =s, hue = 'target', 
           palette = 'Set1', scatter_kws = {'alpha':0.3})
plt.show()
```

<img src="data:image/png;base64,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"/>

### **✅ ps_car_13 and ps_car_15**



```python
sns.lmplot(x = 'ps_car_15', y = 'ps_car_13', data = s, hue = 'target', 
           palette = 'Set1', scatter_kws = {'alpha':0.3})
plt.show()
```

<img src="data:image/png;base64,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"/>

- 변수에 대한 **주성분 분석(PCA)**을 수행하여 치수를 줄일 수 있음

  - 상관된 변수의 수가 다소 적기에, 그냥 냅두자!


## **8-3. 순서형 변수** 



```python
### 상관계수

v = meta[(meta.level == 'ordinal') & (meta.keep)].index
corr_heatmap(v)
```

<img src="data:image/png;base64,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"/>

- 상관 관계가 높은 변수들이 보이지 않음

  - 대신, target 값으로 group화 시 분포가 어떻게 형성되는지 확인할 수 있음


# **9. Feature engineering**


## **9-1. Dummy 변수 생성**



- 범주형 변수의 값은 순서나 크기를 나타내지 않음

  - 카테고리 2는 카테고리 1의 값의 두 배가 아님

- 이러한 문제를 해결하기 위해 **dummy 변수**를 활용

  - 원래 변수의 범주에 대해 생성된 다른 더미 변수에서 파생될 수 있음 -> 첫 번째 더미 변수를 drop



```python
v = meta[(meta.level == 'nominal') & (meta.keep)].index
print('Before dummification we have {} variables in train'.format(train.shape[1]))

train = pd.get_dummies(train, columns = v, drop_first = True)
print('After dummification we have {} variables in train'.format(train.shape[1]))
```

<pre>
Before dummification we have 57 variables in train
After dummification we have 109 variables in train
</pre>
- 더미 변수 생성으로 인해 52개의 변수가 추가적으로 생성되었다.


## **9-2. interaction 변수 생성**



```python
v = meta[(meta.level == 'interval') & (meta.keep)].index
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
interactions = pd.DataFrame(data=poly.fit_transform(train[v]), columns=poly.get_feature_names_out(v))
interactions.drop(v, axis=1, inplace=True)  # Remove the original columns

# Concat the interaction variables to the train data
print('Before creating interactions we have {} variables in train'.format(train.shape[1]))
train = pd.concat([train, interactions], axis=1)
print('After creating interactions we have {} variables in train'.format(train.shape[1]))
```

<pre>
Before creating interactions we have 109 variables in train
After creating interactions we have 164 variables in train
</pre>
- train 데이터에 interaction 변수가 추가됨

- ```get_feature_names_out()``` 메서드 덕분에 열 이름을 새 변수에 할당할 수 있음


# **10. Feature selection**


## **10-1. 분산이 낮거나 0인 feature 제거**


- Sklearn은 이를 위한 편리한 방법을 가지고 있음

  - 분산 문턱값(Variance Threshold)

- 기본적으로 분산이 0인 형상을 제거

  - 이전 단계에서 zero-variance 변수가 없음을 확인

  - 분산이 **1% 미만**인 feature를 제거하면 31개의 변수가 제거됨



```python
selector = VarianceThreshold(threshold = .01)
selector.fit(train.drop(['id', 'target'], axis = 1)) 

f = np.vectorize(lambda x : not x) # boolean 값 요소를 전환

v = train.drop(['id', 'target'], axis = 1).columns[f(selector.get_support())]

### 분신이 threshold보다 작은 변수 추출
print('{} variables have too low variance.'.format(len(v)))
print('These variables are {}'.format(list(v)))
```

<pre>
28 variables have too low variance.
These variables are ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_12', 'ps_car_14', 'ps_car_11_cat_te', 'ps_ind_05_cat_2', 'ps_ind_05_cat_5', 'ps_car_01_cat_1', 'ps_car_01_cat_2', 'ps_car_04_cat_3', 'ps_car_04_cat_4', 'ps_car_04_cat_5', 'ps_car_04_cat_6', 'ps_car_04_cat_7', 'ps_car_06_cat_2', 'ps_car_06_cat_5', 'ps_car_06_cat_8', 'ps_car_06_cat_12', 'ps_car_06_cat_16', 'ps_car_06_cat_17', 'ps_car_09_cat_4', 'ps_car_10_cat_1', 'ps_car_10_cat_2', 'ps_car_12^2', 'ps_car_12 ps_car_14', 'ps_car_14^2']
</pre>
- 분산을 기준으로 선택한다면 많은 변수를 잃게 될 것임

  - 하지만 변수가 많지 않기 때문에, 분류기가 선택하도록 할 것임

  - 변수가 많은 데이터 세트의 경우 데이터 처리 시간을 줄일 수 있음



- 사이킷런에는 다른 feature 선택 방법으로 ```SelectFromModel```을 제공함



```python
### 전체 변수 활용
# RandomForest 활용

X_train = train.drop(['id', 'target'], axis = 1)
y_train = train['target']

feat_labels = X_train.columns

rf = RandomForestClassifier(n_estimators = 1000, random_state = 0, n_jobs=-1)

rf.fit(X_train, y_train)
importances = rf.feature_importances_

indices = np.argsort(rf.feature_importances_)[::-1] # 역순으로 요소들을 정렬(내림차순 정렬)

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))
```

<pre>
 1) ps_car_11_cat_te               0.021066
 2) ps_car_13                      0.017323
 3) ps_car_12 ps_car_13            0.017271
 4) ps_car_13^2                    0.017238
 5) ps_car_13 ps_car_14            0.017201
 6) ps_reg_03 ps_car_13            0.017106
 7) ps_car_13 ps_car_15            0.016783
 8) ps_reg_01 ps_car_13            0.016777
 9) ps_reg_03 ps_car_14            0.016258
10) ps_reg_03 ps_car_12            0.015567
11) ps_reg_03 ps_car_15            0.015179
12) ps_car_14 ps_car_15            0.015002
13) ps_car_13 ps_calc_02           0.014744
14) ps_reg_01 ps_reg_03            0.014728
15) ps_car_13 ps_calc_01           0.014714
16) ps_reg_02 ps_car_13            0.014641
17) ps_car_13 ps_calc_03           0.014635
18) ps_reg_01 ps_car_14            0.014488
19) ps_reg_03^2                    0.014317
20) ps_reg_03                      0.014165
21) ps_reg_03 ps_calc_03           0.013774
22) ps_reg_03 ps_calc_02           0.013765
23) ps_reg_03 ps_calc_01           0.013694
24) ps_calc_10                     0.013626
25) ps_car_14 ps_calc_02           0.013594
26) ps_car_14 ps_calc_01           0.013580
27) ps_car_14 ps_calc_03           0.013519
28) ps_calc_14                     0.013414
29) ps_car_12 ps_car_14            0.012919
30) ps_ind_03                      0.012910
31) ps_car_14^2                    0.012752
32) ps_car_14                      0.012751
33) ps_reg_02 ps_car_14            0.012746
34) ps_calc_11                     0.012582
35) ps_reg_02 ps_reg_03            0.012560
36) ps_ind_15                      0.012182
37) ps_car_12 ps_car_15            0.010928
38) ps_car_15 ps_calc_03           0.010903
39) ps_car_15 ps_calc_02           0.010843
40) ps_car_15 ps_calc_01           0.010814
41) ps_car_12 ps_calc_01           0.010496
42) ps_calc_13                     0.010441
43) ps_car_12 ps_calc_03           0.010354
44) ps_car_12 ps_calc_02           0.010304
45) ps_reg_02 ps_car_15            0.010219
46) ps_reg_01 ps_car_15            0.010213
47) ps_calc_02 ps_calc_03          0.010074
48) ps_calc_01 ps_calc_03          0.010060
49) ps_calc_01 ps_calc_02          0.010000
50) ps_calc_07                     0.009818
51) ps_calc_08                     0.009797
52) ps_reg_01 ps_car_12            0.009466
53) ps_reg_02 ps_car_12            0.009299
54) ps_reg_02 ps_calc_01           0.009288
55) ps_reg_02 ps_calc_03           0.009221
56) ps_reg_02 ps_calc_02           0.009146
57) ps_reg_01 ps_calc_03           0.009059
58) ps_calc_06                     0.009044
59) ps_reg_01 ps_calc_02           0.009037
60) ps_reg_01 ps_calc_01           0.009012
61) ps_calc_09                     0.008800
62) ps_ind_01                      0.008605
63) ps_calc_05                     0.008318
64) ps_calc_04                     0.008128
65) ps_reg_01 ps_reg_02            0.008024
66) ps_calc_12                     0.007976
67) ps_car_15                      0.006139
68) ps_car_15^2                    0.006136
69) ps_calc_03                     0.006017
70) ps_calc_01^2                   0.006009
71) ps_calc_03^2                   0.005974
72) ps_calc_01                     0.005964
73) ps_calc_02^2                   0.005945
74) ps_calc_02                     0.005919
75) ps_car_12^2                    0.005356
76) ps_car_12                      0.005345
77) ps_reg_02^2                    0.004986
78) ps_reg_02                      0.004973
79) ps_reg_01                      0.004159
80) ps_reg_01^2                    0.004139
81) ps_car_11                      0.003798
82) ps_ind_05_cat_0                0.003557
83) ps_ind_17_bin                  0.002843
84) ps_calc_17_bin                 0.002674
85) ps_calc_16_bin                 0.002590
86) ps_calc_19_bin                 0.002548
87) ps_calc_18_bin                 0.002504
88) ps_ind_16_bin                  0.002403
89) ps_car_01_cat_11               0.002393
90) ps_ind_04_cat_0                0.002380
91) ps_ind_04_cat_1                0.002359
92) ps_ind_07_bin                  0.002333
93) ps_car_09_cat_2                0.002312
94) ps_ind_02_cat_1                0.002275
95) ps_car_01_cat_7                0.002130
96) ps_calc_20_bin                 0.002095
97) ps_car_09_cat_0                0.002090
98) ps_ind_02_cat_2                0.002088
99) ps_ind_06_bin                  0.002058
100) ps_car_06_cat_1                0.002007
101) ps_calc_15_bin                 0.001989
102) ps_car_07_cat_1                0.001957
103) ps_ind_08_bin                  0.001937
104) ps_car_09_cat_1                0.001804
105) ps_car_06_cat_11               0.001804
106) ps_ind_18_bin                  0.001719
107) ps_ind_09_bin                  0.001719
108) ps_car_01_cat_10               0.001605
109) ps_car_01_cat_9                0.001595
110) ps_car_01_cat_4                0.001545
111) ps_car_01_cat_6                0.001544
112) ps_car_06_cat_14               0.001532
113) ps_ind_05_cat_6                0.001494
114) ps_ind_02_cat_3                0.001430
115) ps_car_07_cat_0                0.001372
116) ps_car_01_cat_8                0.001345
117) ps_car_08_cat_1                0.001343
118) ps_car_02_cat_1                0.001328
119) ps_car_02_cat_0                0.001307
120) ps_car_06_cat_4                0.001241
121) ps_ind_05_cat_4                0.001199
122) ps_ind_02_cat_4                0.001163
123) ps_car_01_cat_5                0.001143
124) ps_car_06_cat_6                0.001105
125) ps_car_06_cat_10               0.001063
126) ps_ind_05_cat_2                0.001036
127) ps_car_04_cat_1                0.001030
128) ps_car_04_cat_2                0.000992
129) ps_car_06_cat_7                0.000986
130) ps_car_01_cat_3                0.000896
131) ps_car_09_cat_3                0.000878
132) ps_car_01_cat_0                0.000877
133) ps_ind_14                      0.000854
134) ps_car_06_cat_15               0.000847
135) ps_car_06_cat_9                0.000791
136) ps_ind_05_cat_1                0.000750
137) ps_car_06_cat_3                0.000711
138) ps_car_10_cat_1                0.000696
139) ps_ind_12_bin                  0.000684
140) ps_ind_05_cat_3                0.000665
141) ps_car_09_cat_4                0.000623
142) ps_car_01_cat_2                0.000553
143) ps_car_04_cat_8                0.000550
144) ps_car_06_cat_17               0.000512
145) ps_car_06_cat_16               0.000475
146) ps_car_04_cat_9                0.000443
147) ps_car_06_cat_12               0.000427
148) ps_car_06_cat_13               0.000403
149) ps_car_01_cat_1                0.000381
150) ps_ind_05_cat_5                0.000312
151) ps_car_06_cat_5                0.000273
152) ps_ind_11_bin                  0.000215
153) ps_car_04_cat_6                0.000201
154) ps_ind_13_bin                  0.000152
155) ps_car_04_cat_3                0.000149
156) ps_car_06_cat_2                0.000143
157) ps_car_04_cat_5                0.000097
158) ps_car_06_cat_8                0.000094
159) ps_car_04_cat_7                0.000080
160) ps_ind_10_bin                  0.000074
161) ps_car_10_cat_2                0.000060
162) ps_car_04_cat_4                0.000045
</pre>
- ```SelectFromModel```을 활용하면 사용할 사전 적합 분류기와 피쳐 중요도에 대한 임계값을 지정할 수 있음

- ```get_support``` 방법 적용 시 train 데이터의 변수 수를 제한할 수 있음



```python
### SelectFromModel 적용
# 특정 feature만 선택적으로 활용

sfm = SelectFromModel(rf, threshold = 'median', prefit = True)

print('Number of features before selection: {}'.format(X_train.shape[1]))

# 변수 선택
n_features = sfm.transform(X_train).shape[1]
print('Number of features after selection: {}'.format(n_features))

selected_vars = list(feat_labels[sfm.get_support()])
```

<pre>
Number of features before selection: 162
</pre>
<pre>
/usr/local/lib/python3.9/dist-packages/sklearn/base.py:432: UserWarning: X has feature names, but SelectFromModel was fitted without feature names
  warnings.warn(
</pre>
<pre>
Number of features after selection: 81
</pre>

```python
train = train[selected_vars + ['target']]
```

# **11. Feature scaling**


- train 데이터의 범위를 조정해주기 위해 StandardScaler를 적용할 수 있다.

---

**📌 통계학과 교수님 설명**  

- 모델 알고리즘들 중 데이터들 간의 **거리**에 따라 성능이 좌우되는 모델 말고는 굳이 안해도 된다고 합니다.  

(오히려 표준화 이후 나중에 결과 해석이 어렵다고 합니다.)

  - 꼭 해야하는 것들(거리 기반 알고리즘): 릿지(Ridge), 라쏘(Lasso), SVM 등



```python
scaler = StandardScaler()
scaler.fit_transform(train.drop(['target'], axis=1))
```

<pre>
array([[-0.45941104, -1.26665356,  1.05087653, ..., -0.72553616,
        -1.01071913, -1.06173767],
       [ 1.55538958,  0.95034274, -0.63847299, ..., -1.06120876,
        -1.01071913,  0.27907892],
       [ 1.05168943, -0.52765479, -0.92003125, ...,  1.95984463,
        -0.56215309, -1.02449277],
       ...,
       [-0.9631112 ,  0.58084336,  0.48776003, ..., -0.46445747,
         0.18545696,  0.27907892],
       [-0.9631112 , -0.89715418, -1.48314775, ..., -0.91202093,
        -0.41263108,  0.27907892],
       [-0.45941104, -1.26665356,  1.61399304, ...,  0.28148164,
        -0.11358706, -0.72653353]])
</pre>
# **📚 References**

- [기술블로그_불균형 데이터 처리](https://casa-de-feel.tistory.com/15)

- [사이킷런 공식 API](https://scikit-learn.org/stable/modules/classes.html)

