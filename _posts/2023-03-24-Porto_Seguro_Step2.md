---
layout: single
title:  "[ECC DS 2주차] 2. Interactive Porto Insights - A plot.ly Tutorial"
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

- 운전자가 내년에 보험 청구를 시작할 확률을 예측하는 프로젝트

- Python 동적 시각화 라이브러리인 ```plot.ly```를 활용

- 해당 노트북에서 활용한 ```plot.ly```의 여러 기능들

  - 단순 가로 barplot: target 변수 분포를 검사하는 데 사용

  - 상관계수 heatmap: 여러 feature들 간의 상관 관계 확인

  -산점도 plot: RandomForest 및 GradientBoosting 모델에서 생성된 feature 중요도 비교

  - 수직 barplot: 여러 feature들을 대상으로 중요도가 높은 순서대로 내림차순 정렬

  - 3D 산점도 plot



## **📌 해당 노트북의 목표**

1. 데이터 품질 점검: 모든 결측값/Null값(-1인 값) 시각화 및 평가

2. feature 검사 및 필터링

- 대상 변수에 대한 상관관계 및 형상 상호 정보 그림

- 이항, 범주형 및 기타 변수의 검사

3. 

- 학습 모델을 통한 feature 중요도 순위 매기기

- 학습 과정에 기반하여 feature들을 순위화 할 수 있도록 도와주는 n building Random Forest와 Gradient Boosted model


# **1. Import Libraries & Data Loading**


**plotly 결과 출력 관련**  

[Reference](https://physikk.tistory.com/15)



```python
### 관련 라이브러리 import

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go
import plotly.tools as tls

import plotly.io as pio
pio.renderers.default = "svg"

import chart_studio
chart_studio.tools.set_credentials_file(username = 'username', api_key = 'api_key')

import warnings
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings('ignore')
```

```python
### Colab Notebook에서 render 할 수 있도록 해주는 함수

def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))
```


```python
### 데이터 불러오기

train = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/2주차/data/train.csv")
train.head()
```


  <div id="df-9994fbc6-7efc-46db-b4e3-62c60ed16167">
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
      <th>...</th>
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
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
<p>5 rows × 59 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9994fbc6-7efc-46db-b4e3-62c60ed16167')"
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
          document.querySelector('#df-9994fbc6-7efc-46db-b4e3-62c60ed16167 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9994fbc6-7efc-46db-b4e3-62c60ed16167');
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
### train 데이터의 행, 열 개수 파악하기

rows = train.shape[0]
columns = train.shape[1]
print("The train dataset contains {0} rows and {1} columns".format(rows, columns))
```

<pre>
The train dataset contains 595212 rows and 59 columns
</pre>
# **2. 데이터 확인하기(Data Quality Check)**



## **2-1. Null값/결측치 확인**



```python
### Null값 확인
# 모든 열에 대해 isnull 검사를 실행하기 위해 any()를 두 번 적용

train.isnull().any().any()
```

<pre>
False
</pre>
- Null 값 검사에서 False를 반환하지만 사실 ```-1``` 또한 결측치를 의미한다는 점을 주의해야 함



```python
### 결측치 확인
# 모든 -1을 null로 쉽게 대체할 수 있도록 값에 -1이 포함된 열 확인하기

train_copy = train
train_copy = train_copy.replace(-1, np.NaN) # -1 -> NaN
```

## **2-2. 결측치 시각화**

- ```missingno``` 패키지 활용



```python
import missingno as msno

# 열별 null 또는 결측값
msno.matrix(df = train_copy.iloc[:,2:39], figsize = (20, 14), color = (0.42, 0.1, 0.05))
```
![image](https://user-images.githubusercontent.com/98953721/227312503-7707fd7d-c3be-4888-8150-a0485240f219.png)


- 수직의 어두운 빨간색 띠(누락되지 않은 데이터)에 겹쳐진 빈 흰색 띠는 특정 열의 데이터의 **결측**을 반영

- 해당 경우 전체 59개 feature 중 7개의 feature가 실제로 null 값을 포함하고 있음을 알 수 있음

  - 결측값이 있는 열은 실제로 총 13개

    - 결측 행렬 그림이 하나의 그림에 약 40개의 홀수 형상에만 적합할 수 있기 때문 -> 일부 열이 생략된 상태

    - 모든 null을 시각화 하려면 ```figsize``` 인수를 변경하고 데이터 프레임을 분할



- 제외된 7개의 컬럼:

  - ps_ind_05_cat

  - ps_reg_03

  - ps_car_03_cat

  - ps_car_05_cat

  - ps_car_07_cat

  - ps_car_09_cat

  - ps_car_14

- 대부분의 결측값은 _cat이 붙은 열에서 발생

- ```ps_reg_03```, ```ps_car_03_cat```, ```ps_car_05_cat``` 열의 경우 대부분의 값이 누락됨  

  ->  Null에 대해 -1을 전체적으로 대체하는 것은 그다지 좋아 보이지 x



## **2-3. Target 변수 확인하기**

- target 값은 클래스/라벨/정답이라는 이름으로 제공되며, 학습된 함수가 일반화 및 예측을 잘 할 수 있기를 바람

- 데이터를 목표값에 가장 잘 매핑하는 함수를 학습하기 위해 주어진 데이터(우리의 경우 id 열을 제외한 모든 train 데이터)와 함께 지도 학습 모델에 새로운 보이지 않는 데이터와 함께 사용



```python
configure_plotly_browser_state() # 매 그래프 출력 시 호출해준다.

data = [go.Bar(
            x = train["target"].value_counts().index.values,
            y = train["target"].value_counts().values,
            text ='Distribution of target variable'
    )]

layout = go.Layout(
    title='Target variable distribution'
)

fig = go.Figure(data = data, layout = layout)

py.iplot(fig, filename = 'basic-bar')
```

{% include plotly/ECC/Porto_Seguro/Target_variable_distribution.html %}

- target 변수가 굉장히 **불균형한** 분포를 가지고 있음을 확인할 수 있음


## **2-4. 데이터 타입(dtype) 확인**

- train 데이터를 구성하는 데이터 유형 확인

  - 정수, 문자 또는 실수

- Python 시퀀스에서 고유한 유형의 카운트를 가져오기 위해 ```Collections``` 모듈을 가져올 때 ```Counter()``` 메서드를 활용



```python
Counter(train.dtypes.values)
```

<pre>
Counter({dtype('int64'): 49, dtype('float64'): 10})
</pre>
- train data는 총 59개의 컬럼을 가지고 있음

  - 정수/실수 2개의 데이터 타입으로 구성됨

- 데이터에 ```_bin```, ```_cat``` 및 ```_reg```와 같은 접미사가 붙어있음

  - ```_bin```: 이진형(binary) feature

  - ```_cat```: 범주형 feature

  - ```_reg```: 연속/순서형 feature



```python
train_float = train.select_dtypes(include = ['float64'])
train_int = train.select_dtypes(include = ['int64'])
```

## **2-5. 상관계수 Plot**



### **a) 실수형 feature들의 상관계수 시각화**

- ```sns.heatmap()``` 활용

- pandas 데이터 프레임에는 **Pearson 상관관계**를 계산하는 ```corr()``` 방법이 내장되어 있음



```python
colormap = plt.cm.magma
plt.figure(figsize = (16,12))
plt.title('Pearson correlation of continuous features', y = 1.05, size = 15)
sns.heatmap(train_float.corr(),linewidths = 0.1,vmax = 1.0, square = True, 
            cmap = colormap, linecolor = 'white', annot = True)
```

![image](https://user-images.githubusercontent.com/98953721/227313287-9517dce3-c1ce-4736-aa11-c2b3bf72f8f3.png)

- 대부분의 feature들 간의 상관계수가 0임을 확인할 수 있음

- 양의 상관관계를 가지는 feature들의 쌍

  - (ps_reg_01, ps_reg_03)

  - (ps_reg_02, ps_reg_03)

  - (ps_car_12, ps_car_13)

  - (ps_car_13, ps_car_15)



### **b) 정수형 feature들의 상관계수 시각화**

-  ```Plotly``` 라이브러리를 사용하여 상관 관계 값의 열 지도를 **대화식**으로 생성

- 이전의 plotly plot과 마찬가지로 ```go()```를 호출하여 heatmap 객체를 생성

  - x축과 y축은 열 이름을 사용

  - 상관 관계 값은 z축에서 제공




```python
### 정적 heatmap

train_int = train_int.drop(["id", "target"], axis = 1)
colormap = plt.cm.bone
plt.figure(figsize = (21,16))
plt.title('Pearson correlation of categorical features', y = 1.05, size = 15)
sns.heatmap(train_int.corr(),linewidths = 0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=False)
```

![image](https://user-images.githubusercontent.com/98953721/227313481-8a60cc2f-ee5c-4f5f-bd63-f5ca10080a09.png)

```python
### 동적 heatmap 생성

configure_plotly_browser_state() # 매 그래프 출력 시 호출해준다.

data = [
    go.Heatmap(
        z= train_int.corr().values,
        x=train_int.columns.values,
        y=train_int.columns.values,
        colorscale='Viridis',
        reversescale = False,
        opacity = 1.0 )
]

layout = go.Layout(
    title='Pearson Correlation of Integer-type features',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 700)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')
```

{% include plotly/ECC/Porto_Seguro/pearson.html %}


- correlation plot에서 값이 0인 셀이 상당히 많이 관찰됨

  - 서로 선형 상관 관계가 전혀 없는 feature들이 많음

- 주성분 분석(PCA)과 같은 차원 축소 변환을 수행하려는 경우에는 어느 정도의 상관 관계가 필요

  


## **2-6. 상호 정보 Plot**

- target 변수와 target 변수가 계산되는 해당 형상 사이의 상호 정보를 검사할 수 있게 해주는 유용한 도구

- 분류 문제의 경우 Sklearn의 ```mutual_info_classif()``` 메서드를 호출

  - k - 근접 이웃 거리의 엔트로피 추정에 기반한 비모수 방법에 의존

  - 두 랜덤 변수 사이의 의존성을 측정

    - 0(랜덤 변수가 서로 독립적인 경우)에서 더 높은 값(일부 종속성을 나타냄)까지 범위를 가짐

  - target의 정보가 feature 내에 얼마나 포함될 수 있는지 파악할 수 있음



```python
mf = mutual_info_classif(train_float.values, train.target.values,
                         n_neighbors = 3, random_state = 17)
print(mf)
```

<pre>
[0.02599971 0.00767074 0.00617141 0.01855302 0.00158483 0.00338192
 0.01668813 0.0134428  0.01334669 0.01348572]
</pre>

## **2-7. 이진(binary) 변수 확인**

- 이진 변수: 값으로 1 또는 0 중 하나만 사용하는 변수

- 이진 값을 포함하는 모든 열을 저장한 다음 다음과 같이 **vertical barplot**을 생성



```python
bin_col = [col for col in train.columns if '_bin' in col] # 이진 변수 추출
zero_list = [] # 값이 0
one_list = [] # 값이 1
for col in bin_col:
    zero_list.append((train[col] == 0).sum())
    one_list.append((train[col] == 1).sum())
```


```python
### 시각화

configure_plotly_browser_state() # 매 그래프 출력 시 호출해준다.

# 0
trace1 = go.Bar(
    x = bin_col,
    y = zero_list ,
    name = 'Zero count'
)
# 1
trace2 = go.Bar(
    x = bin_col,
    y = one_list,
    name = 'One count'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode = 'stack',
    title = 'Count of 1 and 0 in binary variables'
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='stacked-bar')
```

{% include plotly/ECC/Porto_Seguro/Count_of_binary_variables.html %}

- ```ps_ind_10_bin```, ```ps_ind_11_bin```, ```ps_ind_12_bin```,```ps_ind_13_bin```에서 0의 값을 가지는 변수들의 비율이 높음을 확인할 수 있음



## **2-8. 범주형(categorical) 변수와 순서형(ordinal) 변수 확인**


### **✅ 랜덤 포레스트를 통한 기능 중요도**

- train 데이터를 RandomForestClassifier로 맞추고 모델이 훈련을 마친 후 feature들의 순위 파악

- 유용한 기능 중요도를 얻는 데 많은 매개 변수 조정이 필요하지 않고 불균형 feature에 대해서도 매우 강력한 앙상블 모델(Bootstrap 집계 하에 적용된 약한 의사 결정 트리 학습자의 앙상블)을 사용하는 빠른 방법



```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 150, max_depth = 8, min_samples_leaf = 4, 
                            max_features = 0.2, n_jobs = -1, random_state = 0)
rf.fit(train.drop(['id', 'target'],axis = 1), train.target)
features = train.drop(['id', 'target'],axis = 1).columns.values
print("----- Training Done -----")
```

<pre>
----- Training Done -----
</pre>

**시각화**

- RandomForest를 학습시킨 후 ```feature_importances_``` 속성을 호출하여 feature 중요도 목록을 얻고 plotly의 산점도 plot을 시각화

- ```Scatter``` 명령을 실행하고 이전의 plotly plot에 따라 y축과 x축을 정의해야 함

  - ```marker``` 속성: 점의 크기, 색상 및 척도를 정의/제어






```python
### 시각화(산점도)

configure_plotly_browser_state() # 매 그래프 출력 시 호출해준다.

trace = go.Scatter(
    y = rf.feature_importances_,
    x = features,
    mode = 'markers',
    marker = dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 13,
        #size= rf.feature_importances_,
        #color = np.random.randn(500), #set color equal to a variable
        color = rf.feature_importances_,
        colorscale='Portland',
        showscale=True
    ),
    text = features
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
     xaxis= dict(
         ticklen= 5,
         showgrid=False,
        zeroline=False,
        showline=False
     ),
    yaxis=dict(
        title= 'Feature Importance',
        showgrid=False,
        zeroline=False,
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')
```

{% include plotly/ECC/Porto_Seguro/RandomForest_dot.html %}

```python
### 시각화(barplot)

configure_plotly_browser_state() # 매 그래프 출력 시 호출해준다.

x, y = (list(x) for x in zip(*sorted(zip(rf.feature_importances_, features), 
                                                            reverse = False)))
trace2 = go.Bar(
    x = x, # feature importance
    y = y, # feature name
    marker = dict(
        color = x,
        colorscale = 'Viridis',
        reversescale = True
    ),
    name = 'Random Forest Feature importance',
    orientation = 'h',
)

layout = dict(
    title = 'Barplot of Feature importances',
     width = 900, height = 2000,
    yaxis = dict(
        showgrid = False,
        showline = False,
        showticklabels = True,
      # domain=[0, 0.85],
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)

py.iplot(fig1, filename='plots')
```

{% include plotly/ECC/Porto_Seguro/RandomForest_bar.html %}

**의사 결정 트리 시각화**

- 단순하게 하기 위해 ```Decisiontree(max_depth = 3)```를 적합

- ```sklearn.export_graphviz```에서 그래프 시각화 속성으로 내보내기를 사용



```python
from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import re

decision_tree = tree.DecisionTreeClassifier(max_depth = 3)
decision_tree.fit(train.drop(['id', 'target'],axis=1), train.target)

# Export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 4,
                              impurity = False,
                              feature_names = train.drop(['id', 'target'],axis=1).columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png",)
```

<img src="data:image/png;base64,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"/>

### **✅ Gradient Boosting을 통한 기능 중요도**

- 각 단계에서 손실 함수의 기울기(기본적으로 Sklearn 구현의 편차로 설정됨)에 적합한 전진 단계별 방식으로 진행됨



```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators = 100, max_depth = 3, min_samples_leaf = 4, 
                                max_features = 0.2, random_state = 0)
gb.fit(train.drop(['id', 'target'],axis = 1), train.target)
features = train.drop(['id', 'target'],axis = 1).columns.values
print("----- Training Done -----")
```

<pre>
----- Training Done -----
</pre>

```python
### 시각화(산점도)

configure_plotly_browser_state() # 매 그래프 출력 시 호출해준다.

trace = go.Scatter(
    y = gb.feature_importances_,
    x = features,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 13,
        #size= rf.feature_importances_,
        #color = np.random.randn(500), #set color equal to a variable
        color = gb.feature_importances_,
        colorscale='Portland',
        showscale=True
    ),
    text = features
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Machine Feature Importance',
    hovermode= 'closest',
     xaxis= dict(
         ticklen= 5,
         showgrid=False,
        zeroline=False,
        showline=False
     ),
    yaxis=dict(
        title= 'Feature Importance',
        showgrid=False,
        zeroline=False,
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')
```

{% include plotly/ECC/Porto_Seguro/GBM_dot.html %}

```python
### 시각화(barplot)

configure_plotly_browser_state() # 매 그래프 출력 시 호출해준다.

x, y = (list(x) for x in zip(*sorted(zip(gb.feature_importances_, features), 
                                                            reverse = False)))
trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Viridis',
        reversescale = True
    ),
    name='Gradient Boosting Classifer Feature importance',
    orientation='h',
)

layout = dict(
    title='Barplot of Feature importances',
     width = 900, height = 2000,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')
```

{% include plotly/ECC/Porto_Seguro/GBM_bar.html %}

- RandomForest와 GradientBoost 학습 모델에서 모두 ```ps_car_13``` feature를 가장 중요한 특징으로 선택함


# **3. 결론**

- Null 값과 데이터 품질을 검사하고, feature들 간의 선형 상관관계를 조사하여 Porto Seguro 데이터 세트를 상당히 광범위하게 검사함

- 일부 feature의 분포를 검사하고 모델이 중요하다고 생각하는 기능을 식별하기 위해 몇 가지 학습 모델(RandomForest 및 GradientBoosting 분류기)을 구현
