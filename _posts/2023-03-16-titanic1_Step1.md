- (이유한님) 캐글 코리아 캐글 스터디 커널 커리큘럼
- 1st level. Titanic: Machine Learning from Disaster
  - 타이타닉 튜토리얼 1_Exploratory data analysis, visualization, machine learning

# **0. Import libraries**


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib의 기본 scheme 말고 seaborn scheme을 세팅
# 일일이 graph의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편리
plt.style.use('seaborn')
sns.set(font_scale =  2.5) 
import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
```

## **📌 진행 프로세스**  
**1. 데이터셋 확인**  
- null data를 확인하고, 향후 수정

**2. 탐색적 데이터 분석(exploratory data analysis)**  
- 여러 feature 들을 개별적으로 분석하고, feature들 간의 상관관계를 확인
- 여러 시각화 툴을 사용하여 insight 얻기

**3. feature engineering**  
- 모델을 세우기에 앞서, 모델의 성능을 높일 수 있도록 feature 들을 engineering
- one-hot encoding, class로 나누기, 구간으로 나누기, 텍스트 데이터 처리 등

**4. model 만들기**  
- sklearn을 사용해 모델 생성
  - 파이썬에서 머신러닝을 할 때는 sklearn을 사용하면 수많은 알고리즘을 일관된 문법으로 사용할 수 있음
- 딥러닝을 위해 tensorflow, pytorch 등을 사용

**5. 모델 학습 및 예측**  
- train set을 가지고 모델을 학습시킨 후, test set을 가지고 prediction 수행

**6. 모델 평가**  
- 예측 성능이 원하는 수준인지 판단
- 풀려는 문제에 따라 모델을 평가하는 방식도 달라짐
- 학습된 모델이 어떤 것을 학습 하였는지 확인

# **1. 데이터셋 확인**

- 파이썬에서 테이블화 된 데이터를 다루는 데 가장 최적화되어 있으며, 많이 쓰이는 라이브러리는 ```pandas```
- ```pandas```를 사용하여 데이터셋의 간단한 통계적 분석부터, 복잡한 처리들을 간단한 메소드를 사용하여 해낼 수 있음
- 파이썬으로 데이터 분석을 한다고 하면 반드시 능숙해져야 할 라이브러리
- 캐글에서 데이터셋은 보통 train, test set으로 나뉘어 있음


```python
### 코랩에서 파일을 불러오기 위한 코드
from google.colab import drive
drive.mount('/content/drive') 
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
df_train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/1주차/data/train.csv')
df_test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/1주차/data/test.csv')
```


```python
### 파일의 일부만 확인

df_train.head()
```





  <div id="df-b3724163-d1fa-4981-a16e-7dbdd9289471">
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b3724163-d1fa-4981-a16e-7dbdd9289471')"
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
          document.querySelector('#df-b3724163-d1fa-4981-a16e-7dbdd9289471 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b3724163-d1fa-4981-a16e-7dbdd9289471');
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




- 우리가 다루는 문제에서 feature는 Pclass, Age, SibSp, Parch, Fare 이며, 예측하려는 target label 은 Survived 이다.

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAioAAAHACAYAAACMB0PKAAAgAElEQVR4nO3df3BdZ33n8Y/tq1TySusowQpyiNK1UoeNvCg/idg4xd7iFqeYLQkjWjPANuluaKuZMB1ti5fOMJluxrug6SYddQYYYjAMptEQ0yZdm0HZ2kuURlniJKJRiAX2rgWJiEysuLrkCkvx3T+kK18dnef8uufHc+99v2Y81r333Oc859dzvuf5ddcUi8WiAAAALLQ26wwAAACYEKgAAABr5cpf/PznP88qHwAAAJKkt73tbct/U6MCAACsRaACAACsRaACAACsRaACAACsRaACAACsRaACAACsRaACAACsVaOByoIWFrLOAwAAqJQ1gUr+5WENfWtIo694LDM5prEfjGnynE9iP35M/+3+fToyWfbehQVN/3hMYz84qakCUQwAANUgpUBlRqNf2ad9n3tcE6V3Rvdr3+f2a/T1xddvTr2ksefHdOoNcyonR4Y09MiQRn7qta68Rr57XIX223V7x9Jb5yd06L9/Vg99ZUhDj+zXN546W/kmAQCAxOX8F4nDvObO5ZU/N6v50juFWeXPzerZw0OabJIKP5v1TqJwXC/8cPHPUz86KXV1ui839aRGX5E6dl+v5qW38s8d1fG8pPZtuvuuLq1f3xbDNk3r2IMPafhMl/Y8sEddMaQIAABWSilQKTmhb39unx6TpEJekjT18pim/L62MKVjX3lMExea1LS+oML3v6kDHX+sT9zYumrRqbFxzahD27Y2L7/3ZmFOktTWdZM62+MIUiRpVmfzMSUFAABcpRyotGjj5g61Sir8bEITUy3a+an7tH2jNH30IT30xPTFRS8sqDDzqiZ+MKJjx8Y1vSC13vJR/fFvzevbf3lQ448O6LMjXdrxGzt08+aNam7KScrr9OSMtKFLm5bilInH9unR5xcjiunvPax939+qu/50t7ZIyp86pke/dVQT5xakXLO2/OZ/0Edva1dO0sLr4xr57lGNvDylwoJWfv76qPY//IROF6Tl4GvjDn3y93uk0f36wvde0xW//knd3dN6MQ8vS1t/Z692b1ls9vrC96Qdv/8B6e8f1uM/zmvLh+/XJ27IaeH1MT32jUM6/tqCtDantls+qns+sEXN1vQmAgAgPSkHKpu07cO96tJiYDIxNa2jX96np9dpuYZl2dqzeubr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)

- ```pd.DataFrame.describe()```: 각 feature가 가진 통계치들을 반환


```python
df_train.describe()
```





  <div id="df-59ec1308-8a42-43d2-a176-d7072bded2d4">
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-59ec1308-8a42-43d2-a176-d7072bded2d4')"
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
          document.querySelector('#df-59ec1308-8a42-43d2-a176-d7072bded2d4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-59ec1308-8a42-43d2-a176-d7072bded2d4');
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
df_test.describe()
```





  <div id="df-973fcd2b-6012-4dd9-ba9f-3dadef3b3272">
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>332.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>417.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1100.500000</td>
      <td>2.265550</td>
      <td>30.272590</td>
      <td>0.447368</td>
      <td>0.392344</td>
      <td>35.627188</td>
    </tr>
    <tr>
      <th>std</th>
      <td>120.810458</td>
      <td>0.841838</td>
      <td>14.181209</td>
      <td>0.896760</td>
      <td>0.981429</td>
      <td>55.907576</td>
    </tr>
    <tr>
      <th>min</th>
      <td>892.000000</td>
      <td>1.000000</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>996.250000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.895800</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1100.500000</td>
      <td>3.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1204.750000</td>
      <td>3.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1309.000000</td>
      <td>3.000000</td>
      <td>76.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-973fcd2b-6012-4dd9-ba9f-3dadef3b3272')"
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
          document.querySelector('#df-973fcd2b-6012-4dd9-ba9f-3dadef3b3272 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-973fcd2b-6012-4dd9-ba9f-3dadef3b3272');
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




- PassenserID 숫자와 다른, 그러니까 null data가 존재하는 열(feature)가 있는 것을 확인할 수 있음
- 이를 좀 더 보기 편하도록 그래프로 시각화해서 살펴보자.

## **1-1. Null data check**

- 각 컬럼별로 전체 데이터 중 결측치(NaN)의 비율 구하기
- ```pd.isnull()```: 배열 형태 객체에 결측치가 있는지 확인해주는 함수


```python
### train data

for col in df_train.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(msg)
```

    column: PassengerId	 Percent of NaN value: 0.00%
    column:   Survived	 Percent of NaN value: 0.00%
    column:     Pclass	 Percent of NaN value: 0.00%
    column:       Name	 Percent of NaN value: 0.00%
    column:        Sex	 Percent of NaN value: 0.00%
    column:        Age	 Percent of NaN value: 19.87%
    column:      SibSp	 Percent of NaN value: 0.00%
    column:      Parch	 Percent of NaN value: 0.00%
    column:     Ticket	 Percent of NaN value: 0.00%
    column:       Fare	 Percent of NaN value: 0.00%
    column:      Cabin	 Percent of NaN value: 77.10%
    column:   Embarked	 Percent of NaN value: 0.22%
    


```python
### test data

for col in df_test.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))
    print(msg)
```

    column: PassengerId	 Percent of NaN value: 0.00%
    column:     Pclass	 Percent of NaN value: 0.00%
    column:       Name	 Percent of NaN value: 0.00%
    column:        Sex	 Percent of NaN value: 0.00%
    column:        Age	 Percent of NaN value: 20.57%
    column:      SibSp	 Percent of NaN value: 0.00%
    column:      Parch	 Percent of NaN value: 0.00%
    column:     Ticket	 Percent of NaN value: 0.00%
    column:       Fare	 Percent of NaN value: 0.24%
    column:      Cabin	 Percent of NaN value: 78.23%
    column:   Embarked	 Percent of NaN value: 0.00%
    

- Train, Test set 에서 Age(둘다 약 20%), Cabin(둘다 약 80%), Embarked(Train만 0.22%)에 null data가 존재하는 것을 볼 수 있음
- ```MSNO```라는 라이브러리를 사용하면 null data의 존재를 더 쉽게 볼 수 있음

**train set**


```python
msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
```




    <AxesSubplot:>




    
![png](output_20_1.png)
    



```python
msno.bar(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
```




    <AxesSubplot:>




    
![png](output_21_1.png)
    


**test set**


```python
msno.bar(df=df_test.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
```




    <AxesSubplot:>




    
![png](output_23_1.png)
    


## **1-2. Target Label 확인**

- target label이 어떤 **distribution**을 가지고 있는지 확인해 봐야 함
- 지금과 같은 **binary classification** 문제의 경우에서, 1과 0의 분포가 어떠냐에 따라 모델의 평가 방법이 달라질 수 있음


```python
### target label의 분포 시각화

f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=df_train, ax=ax[1])
ax[1].set_title('Count plot - Survived')

plt.show()
```


    
![png](output_26_0.png)
    


- 죽은 사람이 많음
  - 38.4% 가 살아남았음(Survived = 1)
- target label의 분포가 제법 균일(balanced)함
  - 불균일한 경우, 예를 들어서 100중 1이 99, 0이 1개인 경우에는 만약 모델이 모든것을 1이라 해도 정확도가 99%가 나오게 됩니다.
    - 0을 찾는 문제라면 이 모델은 원하는 결과를 줄 수 없게 됨



# **2. EDA(Exploratory Data Analysis)**

- 많은 데이터 안에 숨겨진 사실을 찾기 위해선 적절한 ***시각화**가 필요
- 시각화 라이브러리는 ```matplotlib```, ```seaborn```, ```plotly``` 등이 있음
  - 특정 목적에 맞는 소스 코드를 정리해 두어 필요할 때마다 참고하면 편함

## **2-1. Pclass**
- Pclass에 따른 생존률의 차이를 살펴볼 예정

- Pclass는 ordinal, 서수형 데이터
  - 카테고리이면서, 순서가 있는 데이터 타입
- 엑셀의 피벗 차트와 유사한 작업을 수행하기 위해 ```pd.DataFrame.groupby()```와 ```pd.DataFrame.pivot()```을 활용
- 'Pclass', 'Survived'를 가져온 후, pclass로 묶기 
  - 그러고 나면 각 pclass 마다 0, 1이 count가 되는데, 이를 평균내면 각 pclass 별 생존률이 나옴
- 아래와 같이 ```count()``` 를 하면 각 class 에 몇 명이 있는 지 확인할 수 있으며, ```sum()```을 하면 216 명 중 생존한(survived = 1) 사람의 총합을 주게 됨
- ```pd.crosstab```을 사용하면 좀 더 위 과정을 좀 더 수월하게 볼 수 있음


```python
### 각 클래스 내의 인원수 확인
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = True).count() 
```





  <div id="df-465ea4f6-3fda-470f-93b4-4e3eac888865">
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
      <th>Survived</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>216</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
    </tr>
    <tr>
      <th>3</th>
      <td>491</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-465ea4f6-3fda-470f-93b4-4e3eac888865')"
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
          document.querySelector('#df-465ea4f6-3fda-470f-93b4-4e3eac888865 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-465ea4f6-3fda-470f-93b4-4e3eac888865');
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
### 각 클래스 내의 사람들 중 생존자 수 파악

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
```





  <div id="df-da1b2119-4ebf-40ef-b600-359454e8866d">
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
      <th>Survived</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>136</td>
    </tr>
    <tr>
      <th>2</th>
      <td>87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>119</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-da1b2119-4ebf-40ef-b600-359454e8866d')"
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
          document.querySelector('#df-da1b2119-4ebf-40ef-b600-359454e8866d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-da1b2119-4ebf-40ef-b600-359454e8866d');
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




- ```as_index = True``` 옵션: 집계된 출력의 경우 그룹 레이블을 인덱스로 사용하여 개체를 반환


```python
### pd.crosstab 사용

pd.crosstab(df_train['Pclass'], df_train['Survived'], margins = True).style.background_gradient(cmap = 'summer_r')
```




<style type="text/css">
#T_d46bf_row0_col0, #T_d46bf_row1_col1, #T_d46bf_row1_col2 {
  background-color: #ffff66;
  color: #000000;
}
#T_d46bf_row0_col1 {
  background-color: #cee666;
  color: #000000;
}
#T_d46bf_row0_col2 {
  background-color: #f4fa66;
  color: #000000;
}
#T_d46bf_row1_col0 {
  background-color: #f6fa66;
  color: #000000;
}
#T_d46bf_row2_col0 {
  background-color: #60b066;
  color: #f1f1f1;
}
#T_d46bf_row2_col1 {
  background-color: #dfef66;
  color: #000000;
}
#T_d46bf_row2_col2 {
  background-color: #90c866;
  color: #000000;
}
#T_d46bf_row3_col0, #T_d46bf_row3_col1, #T_d46bf_row3_col2 {
  background-color: #008066;
  color: #f1f1f1;
}
</style>
<table id="T_d46bf" class="dataframe">
  <thead>
    <tr>
      <th class="index_name level0" >Survived</th>
      <th id="T_d46bf_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_d46bf_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_d46bf_level0_col2" class="col_heading level0 col2" >All</th>
    </tr>
    <tr>
      <th class="index_name level0" >Pclass</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_d46bf_level0_row0" class="row_heading level0 row0" >1</th>
      <td id="T_d46bf_row0_col0" class="data row0 col0" >80</td>
      <td id="T_d46bf_row0_col1" class="data row0 col1" >136</td>
      <td id="T_d46bf_row0_col2" class="data row0 col2" >216</td>
    </tr>
    <tr>
      <th id="T_d46bf_level0_row1" class="row_heading level0 row1" >2</th>
      <td id="T_d46bf_row1_col0" class="data row1 col0" >97</td>
      <td id="T_d46bf_row1_col1" class="data row1 col1" >87</td>
      <td id="T_d46bf_row1_col2" class="data row1 col2" >184</td>
    </tr>
    <tr>
      <th id="T_d46bf_level0_row2" class="row_heading level0 row2" >3</th>
      <td id="T_d46bf_row2_col0" class="data row2 col0" >372</td>
      <td id="T_d46bf_row2_col1" class="data row2 col1" >119</td>
      <td id="T_d46bf_row2_col2" class="data row2 col2" >491</td>
    </tr>
    <tr>
      <th id="T_d46bf_level0_row3" class="row_heading level0 row3" >All</th>
      <td id="T_d46bf_row3_col0" class="data row3 col0" >549</td>
      <td id="T_d46bf_row3_col1" class="data row3 col1" >342</td>
      <td id="T_d46bf_row3_col2" class="data row3 col2" >891</td>
    </tr>
  </tbody>
</table>




- ```margins = True``` 옵션: 행/열 합계 추가

- grouped 객체에 mean() 을 하게 되면, 각 클래스별 생존률을 얻을 수 있음
- class = 1이면 아래와 같음
  $$\frac{80}{(80+136)}≈0.63$$


```python
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by = 'Survived', ascending = False).plot.bar()
```




    <AxesSubplot:xlabel='Pclass'>




    
![png](output_39_1.png)
    


- 보다시피, Pclass 가 좋을수록(1st) 생존률이 높은 것을 확인할 수 있음
- ```sns.countplot```을 이용하면 특정 label에 따른 개수를 확인해볼 수 있dma


```python
### sns.countplot 확인

y_position = 1.02
f, ax = plt.subplots(1, 2, figsize = (18, 8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax = ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')

sns.countplot('Pclass', hue = 'Survived', data = df_train, ax = ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y = y_position)
plt.show()
```


    
![png](output_41_0.png)
    


- 클래스가 높을수록 생존 확률이 높은걸 확인할 수 있음
- Pclass 1, 2, 3 순서대로 63%, 48%, 25% 이다.
- 이를 통해 생존에 **Pclass**가 큰 영향을 미친다고 생각해 볼 수 있으며, 나중에 모델을 세울 때 이 feature를 사용하는 것이 좋을 것이라 판단할 수 있음

## **2-2. Sex**
- **성별**로 생존률이 어떻게 달라지는 지 확인

**pandas groupby 와 seaborn countplot 을 사용해서 시각화**


```python
f, ax = plt.subplots(1, 2, figsize = (18, 8))
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index = True).mean().plot.bar(ax = ax[0])
ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue = 'Survived', data = df_train, ax = ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()
```


    
![png](output_45_0.png)
    


- 여자가 생존할 확률이 높음


```python
### 결과 집계(groupby)

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
```





  <div id="df-029c6152-abdc-4c47-b005-787e8119051f">
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
      <th>Sex</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>0.742038</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>0.188908</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-029c6152-abdc-4c47-b005-787e8119051f')"
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
          document.querySelector('#df-029c6152-abdc-4c47-b005-787e8119051f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-029c6152-abdc-4c47-b005-787e8119051f');
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
### 피벗 테이블 만들기

pd.crosstab(df_train['Sex'], df_train['Survived'], margins  = True).style.background_gradient(cmap = 'summer_r')
```




<style type="text/css">
#T_37d74_row0_col0, #T_37d74_row0_col2, #T_37d74_row1_col1 {
  background-color: #ffff66;
  color: #000000;
}
#T_37d74_row0_col1 {
  background-color: #77bb66;
  color: #f1f1f1;
}
#T_37d74_row1_col0 {
  background-color: #2c9666;
  color: #f1f1f1;
}
#T_37d74_row1_col2 {
  background-color: #8bc566;
  color: #000000;
}
#T_37d74_row2_col0, #T_37d74_row2_col1, #T_37d74_row2_col2 {
  background-color: #008066;
  color: #f1f1f1;
}
</style>
<table id="T_37d74" class="dataframe">
  <thead>
    <tr>
      <th class="index_name level0" >Survived</th>
      <th id="T_37d74_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_37d74_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_37d74_level0_col2" class="col_heading level0 col2" >All</th>
    </tr>
    <tr>
      <th class="index_name level0" >Sex</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_37d74_level0_row0" class="row_heading level0 row0" >female</th>
      <td id="T_37d74_row0_col0" class="data row0 col0" >81</td>
      <td id="T_37d74_row0_col1" class="data row0 col1" >233</td>
      <td id="T_37d74_row0_col2" class="data row0 col2" >314</td>
    </tr>
    <tr>
      <th id="T_37d74_level0_row1" class="row_heading level0 row1" >male</th>
      <td id="T_37d74_row1_col0" class="data row1 col0" >468</td>
      <td id="T_37d74_row1_col1" class="data row1 col1" >109</td>
      <td id="T_37d74_row1_col2" class="data row1 col2" >577</td>
    </tr>
    <tr>
      <th id="T_37d74_level0_row2" class="row_heading level0 row2" >All</th>
      <td id="T_37d74_row2_col0" class="data row2 col0" >549</td>
      <td id="T_37d74_row2_col1" class="data row2 col1" >342</td>
      <td id="T_37d74_row2_col2" class="data row2 col2" >891</td>
    </tr>
  </tbody>
</table>




- **Pclass**와 마찬가지로, **Sex**도 예측 모델에 쓰일 중요한 feature임을 알 수 있음

## **2-3. Both Sex and Pclass**
- Sex, Pclass **두 가지**에 관하여 생존이 어떻게 달라지는 지 확인

- ```sns.factorplot```을 이용하여 손쉽게 3개의 차원으로 이루어진 그래프를 그릴 수 있음


```python
sns.factorplot('Pclass', 'Survived', hue = 'Sex', data = df_train, size = 6, aspect = 1.5)
```




    <seaborn.axisgrid.FacetGrid at 0x7f008c0b8070>




    
![png](output_52_1.png)
    


- 모든 클래스에서 female이 살 확률이 male 보다 높은 걸 알 수 있음
- 남자, 여자 상관없이 클래스가 좋을수록(숫자가 작을수록) 살 확률 높음

- 위 그래프는 hue 대신 column으로 하면 아래와 같아짐


```python
sns.factorplot(x = 'Sex', y = 'Survived', col = 'Pclass',
              data = df_train, satureation = .5,
               size = 9, aspect = 1)
```




    <seaborn.axisgrid.FacetGrid at 0x7f008c290130>




    
![png](output_55_1.png)
    


**```sns.factorplot()```의 parameters**  
- hue: 색 부호화를 위해 column명을 가져옴
  - 어느 column의 값을 기준으로 색을 구분할 것인가
- aspect: 가로, 세로 비율



## **2-4. Age**
- Age feature 살펴보기


```python
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))
print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))
print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))
```

    제일 나이 많은 탑승객 : 80.0 Years
    제일 어린 탑승객 : 0.4 Years
    탑승객 평균 나이 : 29.7 Years
    


```python
### 생존에 따른 age의 histogram

fig, ax = plt.subplots(1, 1, figsize = (9, 5))

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax = ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax = ax)

plt.legend(['Survived == 1', 'Survived == 0'])
plt.show()
```


    
![png](output_59_0.png)
    


- 생존자 중 나이가 **어린** 경우가 많음

**```sns.kdeplot()```**  
- 커널 밀도 추정을 사용하여 일변량(univariate) 또는 이변량(bivariate) 분포를 표시하는 함수


```python
### Age distribution within classes

plt.figure(figsize = (8, 6))

df_train['Age'][df_train['Pclass'] == 1].plot(kind = 'kde')
df_train['Age'][df_train['Pclass'] == 2].plot(kind = 'kde')
df_train['Age'][df_train['Pclass'] == 3].plot(kind = 'kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])
```




    <matplotlib.legend.Legend at 0x7f008c51ebe0>




    
![png](output_62_1.png)
    


- Class가 좋을수록 나이 많은 사람의 비중이 커짐


```python
### 나이대에 따른 생존률의 변화
# 나이 범위를 점점 넓혀가며 생존률 변화 확인

cummulate_survival_ratio = []
for i in range(1, 80):
  cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))
    
plt.figure(figsize = (7, 7))
plt.plot(cummulate_survival_ratio)
plt.title('Survival rate change depending on range of Age', y = 1.02)
plt.ylabel('Survival rate')
plt.xlabel('Range of Age(0~x)')
plt.show()

```


    
![png](output_64_0.png)
    


- 나이가 어릴수록 생존률이 확실히 높은 것을 확인할 수 있음
- **나이**가 중요한 feature 로 쓰일 수 있음을 확인할 수 있음

## **2-5. Pclass, Sex, Age**
- Sex, Pclass, Age, Survived **모두**에 대해 시각화
- ```sns.violinplot```을 통해 여러 변수들에 대한 시각화 수행
  - x축: 우리가 나눠서 보고 싶어하는 case(여기선 Pclass, Sex)
  - y축: 보고 싶어하는 distribution(여기서는 Age)


```python
f,ax = plt.subplots(1,2,figsize = (18,8))

sns.violinplot("Pclass","Age", hue = "Survived", data = df_train, scale = 'count', split = True,ax = ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=df_train, scale='count', split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))

plt.show()
```


    
![png](output_67_0.png)
    


- 왼쪽 그림은 Pclass 별로 Age의 distribution 이 어떻게 다른지, 거기에 생존 여부에 따라 구분한 그래프임
- 오른쪽 그림도 마찬가지로 Sex, 생존에 따른 distribution이 어떻게 다른지 보여주는 그래프임
- 생존만 봤을 때, 모든 클래스에서 나이가 어릴 수록 생존을 많이 한것을 볼 수 있음
- 오른쪽 그림에서 보면, 명확히 여자가 생존을 많이 한것을 볼 수 있음
  - 여성과 아이를 먼저 챙긴 것을 볼 수 있음 

## **2-6. Embarked**
- 탑승한 항구
- 탑승한 곳에 따르 생존률 파악


```python
f, ax = plt.subplots(1, 1, figsize = (7, 7))
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
```




    <AxesSubplot:xlabel='Embarked'>




    
![png](output_70_1.png)
    


- 조금의 차이는 있지만 생존률은 대체로 비슷함
  - 그래도 C가 제일 높음
- 모델에 큰 영향을 미치지 않을 것으로 예상됨
  - but 일단 사용
  - 모델을 만들고 나면 우리가 사용한 feature들이 얼마나 중요한 역할을 했는지 확인해볼 수 있음



```python
### 다른 feature들로 split

f,ax = plt.subplots(2, 2, figsize = (20,15))

sns.countplot('Embarked', data = df_train, ax = ax[0,0])
ax[0,0].set_title('(1) No. Of Passengers Boarded')

sns.countplot('Embarked', hue='Sex', data=df_train, ax=ax[0,1])
ax[0,1].set_title('(2) Male-Female Split for Embarked')

sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax[1,0])
ax[1,0].set_title('(3) Embarked vs Survived')

sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax[1,1])
ax[1,1].set_title('(4) Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2, hspace=0.5) 
plt.show()
```


    
![png](output_72_0.png)
    


- Figure(1): 전체적으로 봤을 때, S에서 가장 많은 사람이 탑승
- Figure(2): C와 Q 는 남녀의 비율이 비슷하고, S는 남자가 더 많음
- Figure(3): 생존 확률이 S인 경우 많이 낮은 걸 볼 수 있음
- Figure(4): 
  - Class로 split 해서 보니, C가 생존 확률이 높은건 클래스가 높은 사람이 많이 타서 그러함
  - S는 3rd class 가 많아서 생존 확률이 낮게 나옴

## **2-7. Family - SibSp(형제 자매) + Parch(부모, 자녀)**
- SibSp와 Parch를 합하면 Family가 될 것임


```python
# 자신을 포함해야하니 1을 더함
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 
```


```python
print("Maximum size of Family: ", df_train['FamilySize'].max())
print("Minimum size of Family: ", df_train['FamilySize'].min())
```

    Maximum size of Family:  11
    Minimum size of Family:  1
    


```python
### FamilySize와 생존의 관계

f,ax = plt.subplots(1, 3, figsize = (40,10))

sns.countplot('FamilySize', data = df_train, ax = ax[0])
ax[0].set_title('(1) No. Of Passengers Boarded', y = 1.02)

sns.countplot('FamilySize', hue = 'Survived', data = df_train, ax = ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilySize',  y = 1.02)

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)

plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
plt.show()
```


    
![png](output_77_0.png)
    


- Figure (1): 
  - 가족 크기가 1 ~ 11까지 있음을 볼 수 있음
  - 대부분 1명이고 그 다음으로 2, 3, 4명입니다.
- Figure (2), (3):
  - 가족 크기에 따른 생존비교
  - 가족이 4명인 경우가 가장 생존 확률이 높음
  - 가족 수가 많아질수록(5, 6, 7, 8, 11) 생존 확률이 낮아짐
  - 가족수가 너무 작아도(1), 너무 커도(5, 6, 8, 11) 생존 확률이 작음
  - 3 ~ 4명 선에서 생존확률이 높은 걸 확인할 수 있음

## **2-8. Fare**
- Fare는 탑승 요금
- contious feature임



```python
### histogram

fig, ax = plt.subplots(1, 1, figsize = (8, 8))
g = sns.distplot(df_train['Fare'], color = 'b', label = 'Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc = 'best')
```


    
![png](output_80_0.png)
    


- distribution이 매우 비대칭인 것을 알 수 있음 -> **high skewness**
  - 만약 이대로 모델에 넣어준다면 자칫 모델이 잘못 학습할 수도 있음 
  - 몇 개 없는 outlier에 대해서 너무 민감하게 반응한다면, 실제 예측 시에 좋지 못한 결과를 부를 수 있음

- outlier의 영향을 줄이기 위해 Fare에 **log**를 취함

- DataFrame의 특정 columns에 공통된 작업(함수)를 적용하고 싶으면 아래의 ```map``` 또는 ```apply```를 사용하여 매우 손쉽게 적용할 수 있음

- 우리가 지금 원하는 것은 Fare columns의 데이터 모두를 log 변환하는 것
  - 파이썬의 간단한 ```lambda 함수```를 이용해 간단한 로그를 적용하는 함수를 **map**에 인수로 넣어주면, Fare columns 데이터에 그대로 적용됨



```python
# test set 에 있는 nan value를 평균값으로 치환
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean() 

### 로그 변환
df_train['Fare'] = df_train['Fare'].map(lambda x: np.log(x) if x > 0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda x: np.log(x) if x > 0 else 0)
```


```python
### 시각화

fig, ax = plt.subplots(1, 1, figsize = (8, 8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc ='best')
```


    
![png](output_83_0.png)
    


- log를 취하니, 비대칭성이 많이 사라진 것을 볼 수 있음
  - 이런 작업을 사용해 모델이 좀 더 좋은 성능을 내도록 할 수 있음


**feature engineering**  
- 모델을 학습시키기 위해, 그리고 그 모델의 성능을 높이기 위해 feature들에 여러 조작을 가하거나, 새로운 feature를 추가하는 것

## **2-9. Cabin**
- NaN이 대략 80%
  - 생존에 영향을 미칠 중요한 정보를 얻어내기가 쉽지는 않음
  - 모델에 적용 x


```python
df_train.head()
```





  <div id="df-198abaec-75f1-4b44-89a2-074f80dd96a0">
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>FamilySize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>1.981001</td>
      <td>NaN</td>
      <td>S</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>4.266662</td>
      <td>C85</td>
      <td>C</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>2.070022</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>3.972177</td>
      <td>C123</td>
      <td>S</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>2.085672</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-198abaec-75f1-4b44-89a2-074f80dd96a0')"
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
          document.querySelector('#df-198abaec-75f1-4b44-89a2-074f80dd96a0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-198abaec-75f1-4b44-89a2-074f80dd96a0');
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




## **2-10. Ticket**
- NaN은 없음
- string data이기에 모델에 적용 전 전처리 필요



```python
df_train['Ticket'].value_counts()
```




    347082      7
    CA. 2343    7
    1601        7
    3101295     6
    CA 2144     6
               ..
    9234        1
    19988       1
    2693        1
    PC 17612    1
    370376      1
    Name: Ticket, Length: 681, dtype: int64



- ticket number는 매우 다양함
  - 특징을 이끌어내어 생존률과 연관지을 수 있음


**개인적인 생각**
- 티켓 번호가 너무 다양하다.
  - 탑승객이 481명인데 제일 많은 티켓 번호의 개수가 7이다.
  - 그냥 단체손님인 것 같다.



# **📚References**
- [Pandas API](https://pandas.pydata.org/docs/reference/index.html#)
- [Seaborn API](https://seaborn.pydata.org/api.html)

