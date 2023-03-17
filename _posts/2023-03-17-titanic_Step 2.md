---
layout: single
title:  "[ECC DS 1주차] EDA To Prediction(DieTanic)"
categories: ML
tags: [ECC, DS, titanic1] 
author_profile: false
---


# **0. 프로젝트 소개**

- **노트북의 목표**: 예측 모델링 문제에서 워크 플로우가 어떻게 작동하는지에 대한 아이디어를 제공
- feature 확인 방법, 새로운 feature의 추가 및 Machine Learning 개념 적용


# **1. Exploratory Data Analysis(EDA)**


```python
### Import Libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 현재 코랩에서는 0.12.0 버전

plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
```

**❌ Version Issue**
- seaborn version issue로 인해 원본 노트북에서 일부 함수 변경(factorplot -> pointplot, catplot)
- plotting을 수행할 때 몇몇 함수는 위치 매개변수 지정이 잘 되지 x -> 키워드 매개변수로 변경


```python
### 데이터 준비하기

data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/1주차/data/train.csv')
```


```python
### 일부 데이터 확인

data.head()
```





  <div id="df-8c6048f2-6a2b-477d-9738-815ee016434e">
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-8c6048f2-6a2b-477d-9738-815ee016434e')"
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
          document.querySelector('#df-8c6048f2-6a2b-477d-9738-815ee016434e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8c6048f2-6a2b-477d-9738-815ee016434e');
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
### 결측치 확인

data.isnull().sum() 
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64



- Age, Cabin, Embarked에 결측치가 존재한다.

## **1-1. Features**


```python
### Survived

f,ax = plt.subplots(1,2,figsize = (18,8))

data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')

sns.countplot(x = 'Survived', data = data, ax = ax[1])
ax[1].set_title('Survived')

plt.show()
```


    
![png](output_10_0.png)
    


- 사고에서 살아남은 승객(Survived = 1)이 많지 않은 것은 분명하다.
  - 891명의 승객 중 350명만이 살아남음

### **📌 Feature의 종류**

**1. 범주형 변수(Categorical Features)**  
- 두 개 이상의 범주가 있는 변수
  - 해당 피쳐의 각 값은 범주별로 분류할 수 있음
- 변수를 분류하거나 순서를 지정할 수 없음
- Sex, Embarked가 해당됨

**2. 순서형 변수(Ordinal Features)**  
- 범주형 값과 유사하지만 값 사이에 상대적 순서 또는 정렬이 가능하다는 점이 차이
  - ex> 높이(높이), 중간(중간), 짧은 값과 같은 기능이 있는 경우 높이는 순서형 변수
- 변수에 상대적인 정렬을 사용할 수 있음
- PClass가 해당됨

**3. 연속형 변수(Continous Feature)**  
- feature가 두 점 사이 또는 feature column의 최소값 또는 최대값 사이의 값을 취할 수 있는 경우
- Age가 해당됨

## **1-2. 다양한 feature들 사이의 관계**



### **Sex**
- Categorical 변수


```python
data.groupby(['Sex','Survived'])['Survived'].count()
```




    Sex     Survived
    female  0            81
            1           233
    male    0           468
            1           109
    Name: Survived, dtype: int64




```python
f,ax = plt.subplots(1,2,figsize = (18,8))

data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax = ax[0])
ax[0].set_title('Survived vs Sex')

sns.countplot(x = 'Sex',hue = 'Survived',data = data,ax = ax[1])
ax[1].set_title('Sex:Survived vs Dead')

plt.show()
```


    
![png](output_17_0.png)
    


- 배에 타고 있던 남성의 수는 여성의 수보다 훨씬 많다.
  - 그러나 구조된 여성의 수는 구조된 남성의 거의 두 배임
- 여성의 생존율은 약 75%인 반면 남성의 생존율은 약 18~19%임
- Sex는 모델링 시 굉장히 중요한 변수인 것 같음

### **PClass**
- 순서형 변수


```python
pd.crosstab(data.Pclass,data.Survived,margins = True).style.background_gradient(cmap = 'summer_r')
```




<style type="text/css">
#T_b92d8_row0_col0, #T_b92d8_row1_col1, #T_b92d8_row1_col2 {
  background-color: #ffff66;
  color: #000000;
}
#T_b92d8_row0_col1 {
  background-color: #cee666;
  color: #000000;
}
#T_b92d8_row0_col2 {
  background-color: #f4fa66;
  color: #000000;
}
#T_b92d8_row1_col0 {
  background-color: #f6fa66;
  color: #000000;
}
#T_b92d8_row2_col0 {
  background-color: #60b066;
  color: #f1f1f1;
}
#T_b92d8_row2_col1 {
  background-color: #dfef66;
  color: #000000;
}
#T_b92d8_row2_col2 {
  background-color: #90c866;
  color: #000000;
}
#T_b92d8_row3_col0, #T_b92d8_row3_col1, #T_b92d8_row3_col2 {
  background-color: #008066;
  color: #f1f1f1;
}
</style>
<table id="T_b92d8" class="dataframe">
  <thead>
    <tr>
      <th class="index_name level0" >Survived</th>
      <th id="T_b92d8_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_b92d8_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_b92d8_level0_col2" class="col_heading level0 col2" >All</th>
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
      <th id="T_b92d8_level0_row0" class="row_heading level0 row0" >1</th>
      <td id="T_b92d8_row0_col0" class="data row0 col0" >80</td>
      <td id="T_b92d8_row0_col1" class="data row0 col1" >136</td>
      <td id="T_b92d8_row0_col2" class="data row0 col2" >216</td>
    </tr>
    <tr>
      <th id="T_b92d8_level0_row1" class="row_heading level0 row1" >2</th>
      <td id="T_b92d8_row1_col0" class="data row1 col0" >97</td>
      <td id="T_b92d8_row1_col1" class="data row1 col1" >87</td>
      <td id="T_b92d8_row1_col2" class="data row1 col2" >184</td>
    </tr>
    <tr>
      <th id="T_b92d8_level0_row2" class="row_heading level0 row2" >3</th>
      <td id="T_b92d8_row2_col0" class="data row2 col0" >372</td>
      <td id="T_b92d8_row2_col1" class="data row2 col1" >119</td>
      <td id="T_b92d8_row2_col2" class="data row2 col2" >491</td>
    </tr>
    <tr>
      <th id="T_b92d8_level0_row3" class="row_heading level0 row3" >All</th>
      <td id="T_b92d8_row3_col0" class="data row3 col0" >549</td>
      <td id="T_b92d8_row3_col1" class="data row3 col1" >342</td>
      <td id="T_b92d8_row3_col2" class="data row3 col2" >891</td>
    </tr>
  </tbody>
</table>





```python
f,ax = plt.subplots(1,2,figsize = (18,8))

data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')

sns.countplot(x = 'Pclass',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead')

plt.show()
```


    
![png](output_21_0.png)
    


- PClass가 1인 승객들이 우선적으로 구조되었음을 짐작할 수 있음
  - PClass가 3인 승객 수가 훨씬 더 많았지만, 여전히 그들 중 생존자 수는 25% 정도로 매우 낮음
  - Pclass가 1인 경우 생존율이 약 63%인 반면 Pclass가 2인 경우 약 48%임


### **Sex + Pclass**


```python
pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True).style.background_gradient(cmap='summer_r')
```




<style type="text/css">
#T_df578_row0_col0, #T_df578_row0_col1, #T_df578_row0_col3, #T_df578_row3_col2 {
  background-color: #ffff66;
  color: #000000;
}
#T_df578_row0_col2, #T_df578_row1_col2 {
  background-color: #f1f866;
  color: #000000;
}
#T_df578_row1_col0 {
  background-color: #96cb66;
  color: #000000;
}
#T_df578_row1_col1 {
  background-color: #a3d166;
  color: #000000;
}
#T_df578_row1_col3 {
  background-color: #cfe766;
  color: #000000;
}
#T_df578_row2_col0 {
  background-color: #a7d366;
  color: #000000;
}
#T_df578_row2_col1, #T_df578_row2_col3 {
  background-color: #85c266;
  color: #000000;
}
#T_df578_row2_col2 {
  background-color: #6eb666;
  color: #f1f1f1;
}
#T_df578_row3_col0 {
  background-color: #cde666;
  color: #000000;
}
#T_df578_row3_col1 {
  background-color: #f0f866;
  color: #000000;
}
#T_df578_row3_col3 {
  background-color: #f7fb66;
  color: #000000;
}
#T_df578_row4_col0, #T_df578_row4_col1, #T_df578_row4_col2, #T_df578_row4_col3 {
  background-color: #008066;
  color: #f1f1f1;
}
</style>
<table id="T_df578" class="dataframe">
  <thead>
    <tr>
      <th class="blank" >&nbsp;</th>
      <th class="index_name level0" >Pclass</th>
      <th id="T_df578_level0_col0" class="col_heading level0 col0" >1</th>
      <th id="T_df578_level0_col1" class="col_heading level0 col1" >2</th>
      <th id="T_df578_level0_col2" class="col_heading level0 col2" >3</th>
      <th id="T_df578_level0_col3" class="col_heading level0 col3" >All</th>
    </tr>
    <tr>
      <th class="index_name level0" >Sex</th>
      <th class="index_name level1" >Survived</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_df578_level0_row0" class="row_heading level0 row0" rowspan="2">female</th>
      <th id="T_df578_level1_row0" class="row_heading level1 row0" >0</th>
      <td id="T_df578_row0_col0" class="data row0 col0" >3</td>
      <td id="T_df578_row0_col1" class="data row0 col1" >6</td>
      <td id="T_df578_row0_col2" class="data row0 col2" >72</td>
      <td id="T_df578_row0_col3" class="data row0 col3" >81</td>
    </tr>
    <tr>
      <th id="T_df578_level1_row1" class="row_heading level1 row1" >1</th>
      <td id="T_df578_row1_col0" class="data row1 col0" >91</td>
      <td id="T_df578_row1_col1" class="data row1 col1" >70</td>
      <td id="T_df578_row1_col2" class="data row1 col2" >72</td>
      <td id="T_df578_row1_col3" class="data row1 col3" >233</td>
    </tr>
    <tr>
      <th id="T_df578_level0_row2" class="row_heading level0 row2" rowspan="2">male</th>
      <th id="T_df578_level1_row2" class="row_heading level1 row2" >0</th>
      <td id="T_df578_row2_col0" class="data row2 col0" >77</td>
      <td id="T_df578_row2_col1" class="data row2 col1" >91</td>
      <td id="T_df578_row2_col2" class="data row2 col2" >300</td>
      <td id="T_df578_row2_col3" class="data row2 col3" >468</td>
    </tr>
    <tr>
      <th id="T_df578_level1_row3" class="row_heading level1 row3" >1</th>
      <td id="T_df578_row3_col0" class="data row3 col0" >45</td>
      <td id="T_df578_row3_col1" class="data row3 col1" >17</td>
      <td id="T_df578_row3_col2" class="data row3 col2" >47</td>
      <td id="T_df578_row3_col3" class="data row3 col3" >109</td>
    </tr>
    <tr>
      <th id="T_df578_level0_row4" class="row_heading level0 row4" >All</th>
      <th id="T_df578_level1_row4" class="row_heading level1 row4" ></th>
      <td id="T_df578_row4_col0" class="data row4 col0" >216</td>
      <td id="T_df578_row4_col1" class="data row4 col1" >184</td>
      <td id="T_df578_row4_col2" class="data row4 col2" >491</td>
      <td id="T_df578_row4_col3" class="data row4 col3" >891</td>
    </tr>
  </tbody>
</table>





```python
sns.pointplot(x = 'Pclass', y = 'Survived',hue='Sex',data = data) 
plt.show()
```


    
![png](output_25_0.png)
    


- 범주형 값들을 쉽게 분리하여 파악하기 위해 ```pointplot()```을 사용
- Pclass = 1인 여성의 생존율이 약 95~96%(94명 중 3명만 사망)
  - PClass와 상관없이 구조 과정에서 여성에게
  우선 순위가 부여된 것은 분명함
- PClass도 중요한 feature라고 생각할 수 있음

### **Age**
- 연속형 변수


```python
print('Oldest Passenger was of:',data['Age'].max(),'Years')
print('Youngest Passenger was of:',data['Age'].min(),'Years')
print('Average Age on the ship:',data['Age'].mean(),'Years')
```

    Oldest Passenger was of: 80.0 Years
    Youngest Passenger was of: 0.42 Years
    Average Age on the ship: 29.69911764705882 Years
    


```python
f,ax = plt.subplots(1,2,figsize=(18,8))

sns.violinplot(x = "Pclass",y = "Age", hue="Survived", data=data,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

sns.violinplot(x = "Sex",y = "Age", hue="Survived", data=data,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))

plt.show()
```


    
![png](output_29_0.png)
    


**✔ Observations**  
- 어린이의 수는 PClass에 따라 증가하며 10세 미만의 승객(즉, 어린이)의 생존율은 PClass와 상관없이 양호한 것으로 보임
- Pclass = 1에서 20-50세 승객의 생존 가능성은 높고 여성이 훨씬 더 높음
- 남성의 경우, 생존 가능성은 나이가 증가함에 따라 감소




**✔ 결측치(NaN) 처리**  
- Age feature에는 177개의 null 값이 있음
- NaN 값을 대체하기 위해 데이터 세트의 평균 연령을 할당할 수 있음
  - but 사람들의 연령은 매우 다양함
- 이름 앞의 붙은 키워드(Mr, Mrs.)를 통해 그룹화 후 각 그룹의 평균값을 할당 가능


```python
### 이름에서 키워드 추출하기

data['Initial'] = 0
for i in data:
    data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.') # 이름에서 .(dot) 앞에 부분만 추출
```


```python
### Sex와 함께 이니셜 확인하기

pd.crosstab(data.Initial,data.Sex).T.style.background_gradient(cmap='summer_r') 
```




<style type="text/css">
#T_9e9c3_row0_col0, #T_9e9c3_row0_col1, #T_9e9c3_row0_col3, #T_9e9c3_row0_col4, #T_9e9c3_row0_col5, #T_9e9c3_row0_col7, #T_9e9c3_row0_col8, #T_9e9c3_row0_col12, #T_9e9c3_row0_col15, #T_9e9c3_row0_col16, #T_9e9c3_row1_col2, #T_9e9c3_row1_col6, #T_9e9c3_row1_col9, #T_9e9c3_row1_col10, #T_9e9c3_row1_col11, #T_9e9c3_row1_col13, #T_9e9c3_row1_col14 {
  background-color: #ffff66;
  color: #000000;
}
#T_9e9c3_row0_col2, #T_9e9c3_row0_col6, #T_9e9c3_row0_col9, #T_9e9c3_row0_col10, #T_9e9c3_row0_col11, #T_9e9c3_row0_col13, #T_9e9c3_row0_col14, #T_9e9c3_row1_col0, #T_9e9c3_row1_col1, #T_9e9c3_row1_col3, #T_9e9c3_row1_col4, #T_9e9c3_row1_col5, #T_9e9c3_row1_col7, #T_9e9c3_row1_col8, #T_9e9c3_row1_col12, #T_9e9c3_row1_col15, #T_9e9c3_row1_col16 {
  background-color: #008066;
  color: #f1f1f1;
}
</style>
<table id="T_9e9c3" class="dataframe">
  <thead>
    <tr>
      <th class="index_name level0" >Initial</th>
      <th id="T_9e9c3_level0_col0" class="col_heading level0 col0" >Capt</th>
      <th id="T_9e9c3_level0_col1" class="col_heading level0 col1" >Col</th>
      <th id="T_9e9c3_level0_col2" class="col_heading level0 col2" >Countess</th>
      <th id="T_9e9c3_level0_col3" class="col_heading level0 col3" >Don</th>
      <th id="T_9e9c3_level0_col4" class="col_heading level0 col4" >Dr</th>
      <th id="T_9e9c3_level0_col5" class="col_heading level0 col5" >Jonkheer</th>
      <th id="T_9e9c3_level0_col6" class="col_heading level0 col6" >Lady</th>
      <th id="T_9e9c3_level0_col7" class="col_heading level0 col7" >Major</th>
      <th id="T_9e9c3_level0_col8" class="col_heading level0 col8" >Master</th>
      <th id="T_9e9c3_level0_col9" class="col_heading level0 col9" >Miss</th>
      <th id="T_9e9c3_level0_col10" class="col_heading level0 col10" >Mlle</th>
      <th id="T_9e9c3_level0_col11" class="col_heading level0 col11" >Mme</th>
      <th id="T_9e9c3_level0_col12" class="col_heading level0 col12" >Mr</th>
      <th id="T_9e9c3_level0_col13" class="col_heading level0 col13" >Mrs</th>
      <th id="T_9e9c3_level0_col14" class="col_heading level0 col14" >Ms</th>
      <th id="T_9e9c3_level0_col15" class="col_heading level0 col15" >Rev</th>
      <th id="T_9e9c3_level0_col16" class="col_heading level0 col16" >Sir</th>
    </tr>
    <tr>
      <th class="index_name level0" >Sex</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
      <th class="blank col7" >&nbsp;</th>
      <th class="blank col8" >&nbsp;</th>
      <th class="blank col9" >&nbsp;</th>
      <th class="blank col10" >&nbsp;</th>
      <th class="blank col11" >&nbsp;</th>
      <th class="blank col12" >&nbsp;</th>
      <th class="blank col13" >&nbsp;</th>
      <th class="blank col14" >&nbsp;</th>
      <th class="blank col15" >&nbsp;</th>
      <th class="blank col16" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_9e9c3_level0_row0" class="row_heading level0 row0" >female</th>
      <td id="T_9e9c3_row0_col0" class="data row0 col0" >0</td>
      <td id="T_9e9c3_row0_col1" class="data row0 col1" >0</td>
      <td id="T_9e9c3_row0_col2" class="data row0 col2" >1</td>
      <td id="T_9e9c3_row0_col3" class="data row0 col3" >0</td>
      <td id="T_9e9c3_row0_col4" class="data row0 col4" >1</td>
      <td id="T_9e9c3_row0_col5" class="data row0 col5" >0</td>
      <td id="T_9e9c3_row0_col6" class="data row0 col6" >1</td>
      <td id="T_9e9c3_row0_col7" class="data row0 col7" >0</td>
      <td id="T_9e9c3_row0_col8" class="data row0 col8" >0</td>
      <td id="T_9e9c3_row0_col9" class="data row0 col9" >182</td>
      <td id="T_9e9c3_row0_col10" class="data row0 col10" >2</td>
      <td id="T_9e9c3_row0_col11" class="data row0 col11" >1</td>
      <td id="T_9e9c3_row0_col12" class="data row0 col12" >0</td>
      <td id="T_9e9c3_row0_col13" class="data row0 col13" >125</td>
      <td id="T_9e9c3_row0_col14" class="data row0 col14" >1</td>
      <td id="T_9e9c3_row0_col15" class="data row0 col15" >0</td>
      <td id="T_9e9c3_row0_col16" class="data row0 col16" >0</td>
    </tr>
    <tr>
      <th id="T_9e9c3_level0_row1" class="row_heading level0 row1" >male</th>
      <td id="T_9e9c3_row1_col0" class="data row1 col0" >1</td>
      <td id="T_9e9c3_row1_col1" class="data row1 col1" >2</td>
      <td id="T_9e9c3_row1_col2" class="data row1 col2" >0</td>
      <td id="T_9e9c3_row1_col3" class="data row1 col3" >1</td>
      <td id="T_9e9c3_row1_col4" class="data row1 col4" >6</td>
      <td id="T_9e9c3_row1_col5" class="data row1 col5" >1</td>
      <td id="T_9e9c3_row1_col6" class="data row1 col6" >0</td>
      <td id="T_9e9c3_row1_col7" class="data row1 col7" >2</td>
      <td id="T_9e9c3_row1_col8" class="data row1 col8" >40</td>
      <td id="T_9e9c3_row1_col9" class="data row1 col9" >0</td>
      <td id="T_9e9c3_row1_col10" class="data row1 col10" >0</td>
      <td id="T_9e9c3_row1_col11" class="data row1 col11" >0</td>
      <td id="T_9e9c3_row1_col12" class="data row1 col12" >517</td>
      <td id="T_9e9c3_row1_col13" class="data row1 col13" >0</td>
      <td id="T_9e9c3_row1_col14" class="data row1 col14" >0</td>
      <td id="T_9e9c3_row1_col15" class="data row1 col15" >6</td>
      <td id="T_9e9c3_row1_col16" class="data row1 col16" >1</td>
    </tr>
  </tbody>
</table>





```python
### 잘못 표기된 이니셜 변경
# Mile이나 Mme 등

data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace = True)
```


```python
### 이니셜 별 평균 나이

data.groupby('Initial')['Age'].mean() 
```




    Initial
    Master     4.574167
    Miss      21.860000
    Mr        32.739609
    Mrs       35.981818
    Other     45.888889
    Name: Age, dtype: float64




```python
### 결측치 처리(NaN 채우기)
# 평균 연령의 올림 값을 활용

data.loc[(data.Age.isnull()) & (data.Initial=='Mr'),'Age'] = 33
data.loc[(data.Age.isnull()) & (data.Initial=='Mrs'),'Age'] = 36
data.loc[(data.Age.isnull()) & (data.Initial=='Master'),'Age'] = 5
data.loc[(data.Age.isnull()) & (data.Initial=='Miss'),'Age'] = 22
data.loc[(data.Age.isnull()) & (data.Initial=='Other'),'Age'] = 46
```


```python
data.Age.isnull().any() 

# 결측치 처리 완료!
```




    False




```python
### 시각화

f,ax = plt.subplots(1,2,figsize=(20,10))

data[data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1 = list(range(0,85,5))
ax[0].set_xticks(x1)

data[data['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)

plt.show()
```


    
![png](output_38_0.png)
    


**✔ Observations**  
- 유아(연령이 5세 미만)는 생존률이 높음
- 가장 나이가 많은 승객은 구조되었음(80세)
- 가장 많이 사망한 승객들의 연령대는 30 ~ 40대

### **Embarked**  
- 범주형 변수


```python
pd.crosstab([data.Embarked,data.Pclass],[data.Sex,data.Survived],margins=True).style.background_gradient(cmap='summer_r')
```




<style type="text/css">
#T_9e67b_row0_col0, #T_9e67b_row1_col2 {
  background-color: #fcfe66;
  color: #000000;
}
#T_9e67b_row0_col1 {
  background-color: #d2e866;
  color: #000000;
}
#T_9e67b_row0_col2 {
  background-color: #f2f866;
  color: #000000;
}
#T_9e67b_row0_col3 {
  background-color: #d8ec66;
  color: #000000;
}
#T_9e67b_row0_col4, #T_9e67b_row2_col3 {
  background-color: #e8f466;
  color: #000000;
}
#T_9e67b_row1_col0, #T_9e67b_row3_col0, #T_9e67b_row3_col1, #T_9e67b_row3_col2, #T_9e67b_row3_col3, #T_9e67b_row3_col4, #T_9e67b_row4_col0, #T_9e67b_row4_col2, #T_9e67b_row4_col3, #T_9e67b_row4_col4 {
  background-color: #ffff66;
  color: #000000;
}
#T_9e67b_row1_col1, #T_9e67b_row6_col0 {
  background-color: #f9fc66;
  color: #000000;
}
#T_9e67b_row1_col3, #T_9e67b_row1_col4 {
  background-color: #fbfd66;
  color: #000000;
}
#T_9e67b_row2_col0, #T_9e67b_row5_col1 {
  background-color: #e6f266;
  color: #000000;
}
#T_9e67b_row2_col1 {
  background-color: #f0f866;
  color: #000000;
}
#T_9e67b_row2_col2 {
  background-color: #eef666;
  color: #000000;
}
#T_9e67b_row2_col4, #T_9e67b_row7_col0 {
  background-color: #edf666;
  color: #000000;
}
#T_9e67b_row4_col1 {
  background-color: #fefe66;
  color: #000000;
}
#T_9e67b_row5_col0 {
  background-color: #e3f166;
  color: #000000;
}
#T_9e67b_row5_col2 {
  background-color: #ecf666;
  color: #000000;
}
#T_9e67b_row5_col3 {
  background-color: #f8fc66;
  color: #000000;
}
#T_9e67b_row5_col4 {
  background-color: #ebf566;
  color: #000000;
}
#T_9e67b_row6_col1 {
  background-color: #cde666;
  color: #000000;
}
#T_9e67b_row6_col2 {
  background-color: #e4f266;
  color: #000000;
}
#T_9e67b_row6_col3 {
  background-color: #bede66;
  color: #000000;
}
#T_9e67b_row6_col4 {
  background-color: #dbed66;
  color: #000000;
}
#T_9e67b_row7_col1 {
  background-color: #bdde66;
  color: #000000;
}
#T_9e67b_row7_col2 {
  background-color: #d3e966;
  color: #000000;
}
#T_9e67b_row7_col3, #T_9e67b_row8_col1 {
  background-color: #dcee66;
  color: #000000;
}
#T_9e67b_row7_col4 {
  background-color: #d1e866;
  color: #000000;
}
#T_9e67b_row8_col0 {
  background-color: #52a866;
  color: #f1f1f1;
}
#T_9e67b_row8_col2 {
  background-color: #81c066;
  color: #000000;
}
#T_9e67b_row8_col3 {
  background-color: #b0d866;
  color: #000000;
}
#T_9e67b_row8_col4 {
  background-color: #9acc66;
  color: #000000;
}
#T_9e67b_row9_col0, #T_9e67b_row9_col1, #T_9e67b_row9_col2, #T_9e67b_row9_col3, #T_9e67b_row9_col4 {
  background-color: #008066;
  color: #f1f1f1;
}
</style>
<table id="T_9e67b" class="dataframe">
  <thead>
    <tr>
      <th class="blank" >&nbsp;</th>
      <th class="index_name level0" >Sex</th>
      <th id="T_9e67b_level0_col0" class="col_heading level0 col0" colspan="2">female</th>
      <th id="T_9e67b_level0_col2" class="col_heading level0 col2" colspan="2">male</th>
      <th id="T_9e67b_level0_col4" class="col_heading level0 col4" >All</th>
    </tr>
    <tr>
      <th class="blank" >&nbsp;</th>
      <th class="index_name level1" >Survived</th>
      <th id="T_9e67b_level1_col0" class="col_heading level1 col0" >0</th>
      <th id="T_9e67b_level1_col1" class="col_heading level1 col1" >1</th>
      <th id="T_9e67b_level1_col2" class="col_heading level1 col2" >0</th>
      <th id="T_9e67b_level1_col3" class="col_heading level1 col3" >1</th>
      <th id="T_9e67b_level1_col4" class="col_heading level1 col4" ></th>
    </tr>
    <tr>
      <th class="index_name level0" >Embarked</th>
      <th class="index_name level1" >Pclass</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_9e67b_level0_row0" class="row_heading level0 row0" rowspan="3">C</th>
      <th id="T_9e67b_level1_row0" class="row_heading level1 row0" >1</th>
      <td id="T_9e67b_row0_col0" class="data row0 col0" >1</td>
      <td id="T_9e67b_row0_col1" class="data row0 col1" >42</td>
      <td id="T_9e67b_row0_col2" class="data row0 col2" >25</td>
      <td id="T_9e67b_row0_col3" class="data row0 col3" >17</td>
      <td id="T_9e67b_row0_col4" class="data row0 col4" >85</td>
    </tr>
    <tr>
      <th id="T_9e67b_level1_row1" class="row_heading level1 row1" >2</th>
      <td id="T_9e67b_row1_col0" class="data row1 col0" >0</td>
      <td id="T_9e67b_row1_col1" class="data row1 col1" >7</td>
      <td id="T_9e67b_row1_col2" class="data row1 col2" >8</td>
      <td id="T_9e67b_row1_col3" class="data row1 col3" >2</td>
      <td id="T_9e67b_row1_col4" class="data row1 col4" >17</td>
    </tr>
    <tr>
      <th id="T_9e67b_level1_row2" class="row_heading level1 row2" >3</th>
      <td id="T_9e67b_row2_col0" class="data row2 col0" >8</td>
      <td id="T_9e67b_row2_col1" class="data row2 col1" >15</td>
      <td id="T_9e67b_row2_col2" class="data row2 col2" >33</td>
      <td id="T_9e67b_row2_col3" class="data row2 col3" >10</td>
      <td id="T_9e67b_row2_col4" class="data row2 col4" >66</td>
    </tr>
    <tr>
      <th id="T_9e67b_level0_row3" class="row_heading level0 row3" rowspan="3">Q</th>
      <th id="T_9e67b_level1_row3" class="row_heading level1 row3" >1</th>
      <td id="T_9e67b_row3_col0" class="data row3 col0" >0</td>
      <td id="T_9e67b_row3_col1" class="data row3 col1" >1</td>
      <td id="T_9e67b_row3_col2" class="data row3 col2" >1</td>
      <td id="T_9e67b_row3_col3" class="data row3 col3" >0</td>
      <td id="T_9e67b_row3_col4" class="data row3 col4" >2</td>
    </tr>
    <tr>
      <th id="T_9e67b_level1_row4" class="row_heading level1 row4" >2</th>
      <td id="T_9e67b_row4_col0" class="data row4 col0" >0</td>
      <td id="T_9e67b_row4_col1" class="data row4 col1" >2</td>
      <td id="T_9e67b_row4_col2" class="data row4 col2" >1</td>
      <td id="T_9e67b_row4_col3" class="data row4 col3" >0</td>
      <td id="T_9e67b_row4_col4" class="data row4 col4" >3</td>
    </tr>
    <tr>
      <th id="T_9e67b_level1_row5" class="row_heading level1 row5" >3</th>
      <td id="T_9e67b_row5_col0" class="data row5 col0" >9</td>
      <td id="T_9e67b_row5_col1" class="data row5 col1" >24</td>
      <td id="T_9e67b_row5_col2" class="data row5 col2" >36</td>
      <td id="T_9e67b_row5_col3" class="data row5 col3" >3</td>
      <td id="T_9e67b_row5_col4" class="data row5 col4" >72</td>
    </tr>
    <tr>
      <th id="T_9e67b_level0_row6" class="row_heading level0 row6" rowspan="3">S</th>
      <th id="T_9e67b_level1_row6" class="row_heading level1 row6" >1</th>
      <td id="T_9e67b_row6_col0" class="data row6 col0" >2</td>
      <td id="T_9e67b_row6_col1" class="data row6 col1" >46</td>
      <td id="T_9e67b_row6_col2" class="data row6 col2" >51</td>
      <td id="T_9e67b_row6_col3" class="data row6 col3" >28</td>
      <td id="T_9e67b_row6_col4" class="data row6 col4" >127</td>
    </tr>
    <tr>
      <th id="T_9e67b_level1_row7" class="row_heading level1 row7" >2</th>
      <td id="T_9e67b_row7_col0" class="data row7 col0" >6</td>
      <td id="T_9e67b_row7_col1" class="data row7 col1" >61</td>
      <td id="T_9e67b_row7_col2" class="data row7 col2" >82</td>
      <td id="T_9e67b_row7_col3" class="data row7 col3" >15</td>
      <td id="T_9e67b_row7_col4" class="data row7 col4" >164</td>
    </tr>
    <tr>
      <th id="T_9e67b_level1_row8" class="row_heading level1 row8" >3</th>
      <td id="T_9e67b_row8_col0" class="data row8 col0" >55</td>
      <td id="T_9e67b_row8_col1" class="data row8 col1" >33</td>
      <td id="T_9e67b_row8_col2" class="data row8 col2" >231</td>
      <td id="T_9e67b_row8_col3" class="data row8 col3" >34</td>
      <td id="T_9e67b_row8_col4" class="data row8 col4" >353</td>
    </tr>
    <tr>
      <th id="T_9e67b_level0_row9" class="row_heading level0 row9" >All</th>
      <th id="T_9e67b_level1_row9" class="row_heading level1 row9" ></th>
      <td id="T_9e67b_row9_col0" class="data row9 col0" >81</td>
      <td id="T_9e67b_row9_col1" class="data row9 col1" >231</td>
      <td id="T_9e67b_row9_col2" class="data row9 col2" >468</td>
      <td id="T_9e67b_row9_col3" class="data row9 col3" >109</td>
      <td id="T_9e67b_row9_col4" class="data row9 col4" >889</td>
    </tr>
  </tbody>
</table>





```python
### Embarked에 따른 생존률

sns.pointplot(x = 'Embarked',y = 'Survived',data = data)
fig = plt.gcf()
fig.set_size_inches(5,3)
plt.show()
```


    
![png](output_42_0.png)
    


- 항구 C에서의 생존률이 0.55 정도로 가장 높음
- 항구 S에서의 생존률이 가장 낮음


```python
f,ax = plt.subplots(2,2,figsize = (20,15))

sns.countplot(x = 'Embarked',data = data,ax = ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')

sns.countplot(x = 'Embarked',hue = 'Sex',data = data,ax = ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')

sns.countplot(x = 'Embarked',hue = 'Survived',data = data,ax = ax[1,0])
ax[1,0].set_title('Embarked vs Survived')

sns.countplot(x = 'Embarked',hue = 'Pclass',data = data,ax = ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')

plt.subplots_adjust(wspace = 0.2,hspace = 0.5)
plt.show()
```


    
![png](output_44_0.png)
    


**✔ Observations**  
- S에서 탑승한 승객들 중 대다수는 Pclass = 3 출신
- C에서 온 승객들은 그들 중 상당한 비율이 살아남음
  - Pclass = 1과 Pclass = 2 승객 전원을 구조한 것일 수 있음
- Embarked =  S는 대부분 부자들이 탑승한 항구로 보임
  - 여전히 이곳에서는 생존 가능성이 낮음
  - PClass = 3 승객의 81%가 정도 살아남지 못함
- Port Q had almost 95% of the passengers were from Pclass3


```python
sns.catplot(x = 'Pclass',y = 'Survived',hue = 'Sex',col = 'Embarked',data = data, kind = 'point')
plt.show()
```


    
![png](output_46_0.png)
    


**✔ Observations**  
- PClass에 관계없이 PClass = 1과 PClass = 2의 여성의 생존 확률은 거의 1이다.
- Pclass = 3에 대해 Embarked = S는 남녀 모두 생존율이 매우 낮음
- Embarked = Q는 거의 모두 PClass = 3 출신
  - 남성에게 가장 불운한 것으로 보임


**✔ 결측치(NaN) 처리**  
- 많은 승객들이 S 항구에서 탑승하였음
  - NaN을 S로 대체


```python
data['Embarked'].fillna('S',inplace = True)
```


```python
data.Embarked.isnull().any()

# 결측치 처리가 정상적으로 수행됨
```




    False



### **SibSip**
- 이산 변수(Discrete Feature)
- 혼자 탔는지 아니면 그의 가족 구성원과 함께 탔는지
- 형제 => 형제, 자매, 의붓동생, 의붓언니
- 배우자 => 남편, 아내


```python
pd.crosstab([data.SibSp],data.Survived).style.background_gradient(cmap='summer_r')
```




<style type="text/css">
#T_80024_row0_col0, #T_80024_row0_col1 {
  background-color: #008066;
  color: #f1f1f1;
}
#T_80024_row1_col0 {
  background-color: #c4e266;
  color: #000000;
}
#T_80024_row1_col1 {
  background-color: #77bb66;
  color: #f1f1f1;
}
#T_80024_row2_col0, #T_80024_row4_col0 {
  background-color: #f9fc66;
  color: #000000;
}
#T_80024_row2_col1 {
  background-color: #f0f866;
  color: #000000;
}
#T_80024_row3_col0, #T_80024_row3_col1 {
  background-color: #fbfd66;
  color: #000000;
}
#T_80024_row4_col1 {
  background-color: #fcfe66;
  color: #000000;
}
#T_80024_row5_col0, #T_80024_row5_col1, #T_80024_row6_col1 {
  background-color: #ffff66;
  color: #000000;
}
#T_80024_row6_col0 {
  background-color: #fefe66;
  color: #000000;
}
</style>
<table id="T_80024" class="dataframe">
  <thead>
    <tr>
      <th class="index_name level0" >Survived</th>
      <th id="T_80024_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_80024_level0_col1" class="col_heading level0 col1" >1</th>
    </tr>
    <tr>
      <th class="index_name level0" >SibSp</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_80024_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_80024_row0_col0" class="data row0 col0" >398</td>
      <td id="T_80024_row0_col1" class="data row0 col1" >210</td>
    </tr>
    <tr>
      <th id="T_80024_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_80024_row1_col0" class="data row1 col0" >97</td>
      <td id="T_80024_row1_col1" class="data row1 col1" >112</td>
    </tr>
    <tr>
      <th id="T_80024_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_80024_row2_col0" class="data row2 col0" >15</td>
      <td id="T_80024_row2_col1" class="data row2 col1" >13</td>
    </tr>
    <tr>
      <th id="T_80024_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_80024_row3_col0" class="data row3 col0" >12</td>
      <td id="T_80024_row3_col1" class="data row3 col1" >4</td>
    </tr>
    <tr>
      <th id="T_80024_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_80024_row4_col0" class="data row4 col0" >15</td>
      <td id="T_80024_row4_col1" class="data row4 col1" >3</td>
    </tr>
    <tr>
      <th id="T_80024_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_80024_row5_col0" class="data row5 col0" >5</td>
      <td id="T_80024_row5_col1" class="data row5 col1" >0</td>
    </tr>
    <tr>
      <th id="T_80024_level0_row6" class="row_heading level0 row6" >8</th>
      <td id="T_80024_row6_col0" class="data row6 col0" >7</td>
      <td id="T_80024_row6_col1" class="data row6 col1" >0</td>
    </tr>
  </tbody>
</table>





```python
### 시각화

f,ax = plt.subplots(1,2,figsize = (20,8))

sns.barplot(x = 'SibSp',y = 'Survived',data = data, ax = ax[0])
ax[0].set_title('SibSp vs Survived')

sns.pointplot(x = 'SibSp',y = 'Survived',data = data, ax = ax[1])
ax[1].set_title('SibSp vs Survived')
plt.close(2)

plt.show()
```


    
![png](output_53_0.png)
    



```python
### PClass에 따른 자식 수

pd.crosstab(data.SibSp,data.Pclass).style.background_gradient(cmap='summer_r')
```




<style type="text/css">
#T_f20b1_row0_col0, #T_f20b1_row0_col1, #T_f20b1_row0_col2 {
  background-color: #008066;
  color: #f1f1f1;
}
#T_f20b1_row1_col0 {
  background-color: #7bbd66;
  color: #000000;
}
#T_f20b1_row1_col1 {
  background-color: #8ac466;
  color: #000000;
}
#T_f20b1_row1_col2 {
  background-color: #c6e266;
  color: #000000;
}
#T_f20b1_row2_col0, #T_f20b1_row4_col2 {
  background-color: #f6fa66;
  color: #000000;
}
#T_f20b1_row2_col1 {
  background-color: #eef666;
  color: #000000;
}
#T_f20b1_row2_col2 {
  background-color: #f8fc66;
  color: #000000;
}
#T_f20b1_row3_col0, #T_f20b1_row3_col2 {
  background-color: #fafc66;
  color: #000000;
}
#T_f20b1_row3_col1 {
  background-color: #fdfe66;
  color: #000000;
}
#T_f20b1_row4_col0, #T_f20b1_row4_col1, #T_f20b1_row5_col0, #T_f20b1_row5_col1, #T_f20b1_row5_col2, #T_f20b1_row6_col0, #T_f20b1_row6_col1 {
  background-color: #ffff66;
  color: #000000;
}
#T_f20b1_row6_col2 {
  background-color: #fefe66;
  color: #000000;
}
</style>
<table id="T_f20b1" class="dataframe">
  <thead>
    <tr>
      <th class="index_name level0" >Pclass</th>
      <th id="T_f20b1_level0_col0" class="col_heading level0 col0" >1</th>
      <th id="T_f20b1_level0_col1" class="col_heading level0 col1" >2</th>
      <th id="T_f20b1_level0_col2" class="col_heading level0 col2" >3</th>
    </tr>
    <tr>
      <th class="index_name level0" >SibSp</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f20b1_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_f20b1_row0_col0" class="data row0 col0" >137</td>
      <td id="T_f20b1_row0_col1" class="data row0 col1" >120</td>
      <td id="T_f20b1_row0_col2" class="data row0 col2" >351</td>
    </tr>
    <tr>
      <th id="T_f20b1_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_f20b1_row1_col0" class="data row1 col0" >71</td>
      <td id="T_f20b1_row1_col1" class="data row1 col1" >55</td>
      <td id="T_f20b1_row1_col2" class="data row1 col2" >83</td>
    </tr>
    <tr>
      <th id="T_f20b1_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_f20b1_row2_col0" class="data row2 col0" >5</td>
      <td id="T_f20b1_row2_col1" class="data row2 col1" >8</td>
      <td id="T_f20b1_row2_col2" class="data row2 col2" >15</td>
    </tr>
    <tr>
      <th id="T_f20b1_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_f20b1_row3_col0" class="data row3 col0" >3</td>
      <td id="T_f20b1_row3_col1" class="data row3 col1" >1</td>
      <td id="T_f20b1_row3_col2" class="data row3 col2" >12</td>
    </tr>
    <tr>
      <th id="T_f20b1_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_f20b1_row4_col0" class="data row4 col0" >0</td>
      <td id="T_f20b1_row4_col1" class="data row4 col1" >0</td>
      <td id="T_f20b1_row4_col2" class="data row4 col2" >18</td>
    </tr>
    <tr>
      <th id="T_f20b1_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_f20b1_row5_col0" class="data row5 col0" >0</td>
      <td id="T_f20b1_row5_col1" class="data row5 col1" >0</td>
      <td id="T_f20b1_row5_col2" class="data row5 col2" >5</td>
    </tr>
    <tr>
      <th id="T_f20b1_level0_row6" class="row_heading level0 row6" >8</th>
      <td id="T_f20b1_row6_col0" class="data row6 col0" >0</td>
      <td id="T_f20b1_row6_col1" class="data row6 col1" >0</td>
      <td id="T_f20b1_row6_col2" class="data row6 col2" >7</td>
    </tr>
  </tbody>
</table>




**✔ Observartions**  
- barplot 및 factorplot은 승객이 혼자 탑승한 경우(SibSp = 0) 생존율이 34.5%임을 보여줌
  - 형제자매의 수가 증가하면 그래프는 대략 감소
  - 즉, 만약 가족을 태운다면, 자신을 먼저 구하는 대신 그들을 구하기 위해 노력할 것
-  구성원이 5-8명인 가족의 생존율은 0%
  - 이유: PClass
  - crosstab => SibSp > 3를 가진 사람이 모두 Pclass = 3임을 파악할 수 있음

### **Parch**  



```python
pd.crosstab(data.Parch,data.Pclass).style.background_gradient(cmap='summer_r')
```




<style type="text/css">
#T_43192_row0_col0, #T_43192_row0_col1, #T_43192_row0_col2 {
  background-color: #008066;
  color: #f1f1f1;
}
#T_43192_row1_col0 {
  background-color: #cfe766;
  color: #000000;
}
#T_43192_row1_col1 {
  background-color: #c2e066;
  color: #000000;
}
#T_43192_row1_col2 {
  background-color: #dbed66;
  color: #000000;
}
#T_43192_row2_col0 {
  background-color: #dfef66;
  color: #000000;
}
#T_43192_row2_col1 {
  background-color: #e1f066;
  color: #000000;
}
#T_43192_row2_col2 {
  background-color: #e3f166;
  color: #000000;
}
#T_43192_row3_col0, #T_43192_row4_col1, #T_43192_row5_col0, #T_43192_row5_col1, #T_43192_row6_col0, #T_43192_row6_col1, #T_43192_row6_col2 {
  background-color: #ffff66;
  color: #000000;
}
#T_43192_row3_col1 {
  background-color: #fcfe66;
  color: #000000;
}
#T_43192_row3_col2, #T_43192_row4_col0, #T_43192_row4_col2 {
  background-color: #fefe66;
  color: #000000;
}
#T_43192_row5_col2 {
  background-color: #fdfe66;
  color: #000000;
}
</style>
<table id="T_43192" class="dataframe">
  <thead>
    <tr>
      <th class="index_name level0" >Pclass</th>
      <th id="T_43192_level0_col0" class="col_heading level0 col0" >1</th>
      <th id="T_43192_level0_col1" class="col_heading level0 col1" >2</th>
      <th id="T_43192_level0_col2" class="col_heading level0 col2" >3</th>
    </tr>
    <tr>
      <th class="index_name level0" >Parch</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_43192_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_43192_row0_col0" class="data row0 col0" >163</td>
      <td id="T_43192_row0_col1" class="data row0 col1" >134</td>
      <td id="T_43192_row0_col2" class="data row0 col2" >381</td>
    </tr>
    <tr>
      <th id="T_43192_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_43192_row1_col0" class="data row1 col0" >31</td>
      <td id="T_43192_row1_col1" class="data row1 col1" >32</td>
      <td id="T_43192_row1_col2" class="data row1 col2" >55</td>
    </tr>
    <tr>
      <th id="T_43192_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_43192_row2_col0" class="data row2 col0" >21</td>
      <td id="T_43192_row2_col1" class="data row2 col1" >16</td>
      <td id="T_43192_row2_col2" class="data row2 col2" >43</td>
    </tr>
    <tr>
      <th id="T_43192_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_43192_row3_col0" class="data row3 col0" >0</td>
      <td id="T_43192_row3_col1" class="data row3 col1" >2</td>
      <td id="T_43192_row3_col2" class="data row3 col2" >3</td>
    </tr>
    <tr>
      <th id="T_43192_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_43192_row4_col0" class="data row4 col0" >1</td>
      <td id="T_43192_row4_col1" class="data row4 col1" >0</td>
      <td id="T_43192_row4_col2" class="data row4 col2" >3</td>
    </tr>
    <tr>
      <th id="T_43192_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_43192_row5_col0" class="data row5 col0" >0</td>
      <td id="T_43192_row5_col1" class="data row5 col1" >0</td>
      <td id="T_43192_row5_col2" class="data row5 col2" >5</td>
    </tr>
    <tr>
      <th id="T_43192_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_43192_row6_col0" class="data row6 col0" >0</td>
      <td id="T_43192_row6_col1" class="data row6 col1" >0</td>
      <td id="T_43192_row6_col2" class="data row6 col2" >1</td>
    </tr>
  </tbody>
</table>




- 가족 구성원의 수가 많아질수록 주로 PClass = 3에 속함


```python
### 시각화

f,ax = plt.subplots(1,2,figsize = (20,8))

sns.barplot(x = 'Parch',y = 'Survived',data=data,ax=ax[0])
ax[0].set_title('Parch vs Survived')

sns.pointplot(x = 'Parch',y = 'Survived',data=data,ax=ax[1])
ax[1].set_title('Parch vs Survived')
plt.close(2)

plt.show()
```


    
![png](output_59_0.png)
    


### **Fare**
- 연속형 변수



```python
print('Highest Fare was:',data['Fare'].max())
print('Lowest Fare was:',data['Fare'].min())
print('Average Fare was:',data['Fare'].mean())
```

    Highest Fare was: 512.3292
    Lowest Fare was: 0.0
    Average Fare was: 32.204207968574636
    


```python
f,ax = plt.subplots(1,3,figsize=(20,8))

sns.distplot(data[data['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')

sns.distplot(data[data['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')

sns.distplot(data[data['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()
```


    
![png](output_62_0.png)
    


- Pclass = 1의 승객 요금의 분산이 큰 것으로 보이며, 기준이 감소함에 따라 분산이 계속해서 감소하고 있음
- 연속 변수이기에, binning을 사용하여 이산 값으로 변환할 수 있음


## **1-3. 경향성 파악**

### **전체 변수들에 대한 요약**
- Sex: 여성이 남성에 비해 생존 가능성이 높음
- PClass
  - 등급이 좋을수록 생존률이 높아지는 추세를 보임
  - PClass = 3의 생존률이 매우 낮음
  - 여성의 경우 PClass = 1에서의 생존률은 거의 1이고, PClass = 2의 경우도 생존률이 높음
- Age
  - 5-10세 미만 어린이들의 경우 생존 가능성이 높음
  - 15세-30세 승객들이 많이 죽음
- Embarked
  - PClass = 1 승객 대부분이 S에서 탑승하였다만, C에서의 생존률이 훨씬 좋음
  - Q에서 탑승한 손님들은 모두 PClass = 3
- Parch + SibSp
  - 1-2명의 형제자매가 있거나, 배우자가 있거나, 1-3명의 부모가 있는 승객들이 혼자 타거나 대가족인 경우보다 생존률이 높음

### **Feature들 간의 상관게수**


```python
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) # correlation matrix

fig = plt.gcf()
fig.set_size_inches(10,8)

plt.show()
```


    
![png](output_67_0.png)
    


- 각 feature들 간의 큰 상관관계는 없음을 파악할 수 있음
- 가장 큰 상관관계
  - SibSp와 Parch(0.41)
> 모든 feature를 활용할 수 있음

**✔ Heatmap에 대한 해석**
- 알파벳이나 문자열 사이에서 상관관계를 파악할 수 없음
  - numeric feature들만 비교됨

- 양의 상관관계(positive correlation)
  - feature A의 증가가 feature B의 증가로 이어지는 경우
  - 1은 완전한 양의 상관 관계를 의미

- 음의 상관관계(negative correlation)
  - feature A의 증가가 feature B의 감소로 이어지는 경우
  - -1은 완전한 양의 상관 관계를 의미






**✔ 다중공선성(multicollinearity)**  
- 두 특징이 매우/ 완벽하게 상관됨
  - 한 feature의 증가가 다른 feature의 증가로 이어짐
  - 즉, 두 feature 모두 매우 유사한 정보를 포함하고 있으며 정보의 차이가 거의/ 전혀 없음
- 두 개의 feature들이 중복되기에, 둘 다 사용하는 대신 하나만 사용해도 무방함


# **2. 특성 공학(Feature Engineering) & 데이터 클렌징**
- 모든 feature들이 중요한 것은 아님
  - 제거해야 할 중복된 성격의 feature들이 많이 있을 수 있음
- 다른 feature들을 관찰하거나 정보를 추출하여 새로운 feature로 가져오거나 추가할 수 있음

## **2-1. 새로운 feature 추가**

### **Age_band**


**✔ Age feature의 문제점**  
- Age는 연속형 변수
  - ML에서 문제가 생길 수 있음
- **binning** 또는 **정규화**를 통해 범주형 변수로 변환 필요
  - binning을 활용
  - 연령 범위를 단일 빈으로 group화 하거나 단일 값 할당
  - 0 - 80세를 5개의 bin으로 나누기


```python
### 연령대 나누기

data['Age_band'] = 0
data.loc[data['Age']<= 16,'Age_band'] = 0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[data['Age']>64,'Age_band']=4
```


```python
data.head(2)
```





  <div id="df-3a3c6d40-752c-436c-a24f-bf576f7f9e8b">
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
      <th>Initial</th>
      <th>Age_band</th>
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
      <td>Mr</td>
      <td>1</td>
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
      <td>Mrs</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3a3c6d40-752c-436c-a24f-bf576f7f9e8b')"
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
          document.querySelector('#df-3a3c6d40-752c-436c-a24f-bf576f7f9e8b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3a3c6d40-752c-436c-a24f-bf576f7f9e8b');
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
### 각 연령대에 속하는 승객 수

data['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')
```




<style type="text/css">
#T_74fdc_row0_col0 {
  background-color: #ffff66;
  color: #000000;
}
#T_74fdc_row1_col0 {
  background-color: #d8ec66;
  color: #000000;
}
#T_74fdc_row2_col0 {
  background-color: #40a066;
  color: #f1f1f1;
}
#T_74fdc_row3_col0 {
  background-color: #289366;
  color: #f1f1f1;
}
#T_74fdc_row4_col0 {
  background-color: #008066;
  color: #f1f1f1;
}
</style>
<table id="T_74fdc" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_74fdc_level0_col0" class="col_heading level0 col0" >Age_band</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_74fdc_level0_row0" class="row_heading level0 row0" >1</th>
      <td id="T_74fdc_row0_col0" class="data row0 col0" >382</td>
    </tr>
    <tr>
      <th id="T_74fdc_level0_row1" class="row_heading level0 row1" >2</th>
      <td id="T_74fdc_row1_col0" class="data row1 col0" >325</td>
    </tr>
    <tr>
      <th id="T_74fdc_level0_row2" class="row_heading level0 row2" >0</th>
      <td id="T_74fdc_row2_col0" class="data row2 col0" >104</td>
    </tr>
    <tr>
      <th id="T_74fdc_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_74fdc_row3_col0" class="data row3 col0" >69</td>
    </tr>
    <tr>
      <th id="T_74fdc_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_74fdc_row4_col0" class="data row4 col0" >11</td>
    </tr>
  </tbody>
</table>





```python
sns.catplot(x = 'Age_band', y = 'Survived',data=data,col='Pclass', kind = 'point')
plt.show()
```


    
![png](output_78_0.png)
    


- PClass와 상관 없이 나이가 증가할수록 생존률이 낮아짐

### **Family_Size & Alone**
- Parch + SibSp
- 생존률이 가족 구성원 수와 관련이 있는지 확인
- 단독으로 승객이 혼자인지 아닌지를 나타낼 수 있음


```python
data['Family_Size'] = 0
data['Family_Size'] = data['Parch'] + data['SibSp'] #family size

data['Alone'] = 0
data.loc[data.Family_Size == 0,'Alone'] = 1 #Alone

### 시각화

f,ax = plt.subplots(1,2,figsize = (18,6))
sns.pointplot(x = 'Family_Size',y = 'Survived',data = data, ax = ax[0])
ax[0].set_title('Family_Size vs Survived')
sns.pointplot(x = 'Alone',y = 'Survived',data = data, ax = ax[1])
ax[1].set_title('Alone vs Survived')
plt.close(2)
plt.close(3)
plt.show()
```


    
![png](output_81_0.png)
    


- Family_Size = 0: 승객이 **혼자**임을 의미
- 혼자이거나 family_size = 0이면 생존률이 매우 낮음
- family_size > 4인 경우도 생존률 감소



```python
sns.catplot(x = 'Alone',y = 'Survived',data=data,hue='Sex',col='Pclass', kind = 'point')
plt.show()
```


    
![png](output_83_0.png)
    


- 가족이 있는 사람보다 혼자인 여성의 확률이 높은 Pclass = 3을 제외하고는 Sex, Pclass 구분 없이 혼자 있는 것이 위험함


### **Fare_Range**
- Fare 또한 연속형 변수 -> 전처리 필요
- 전처리를 위해 ```pandas.qcut()```을 활용

- ```pd.qcut(data, bins)```
  - 통과한 bin의 수에 따라 값을 분할/배열
  - 5개의 bin에 대해 전달 시 값이 5개의 bin 또는 값 범위로 균등하게 배열됨


```python
data['Fare_Range'] = pd.qcut(data['Fare'], 4)
data.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
```




<style type="text/css">
#T_a0632_row0_col0 {
  background-color: #ffff66;
  color: #000000;
}
#T_a0632_row1_col0 {
  background-color: #b9dc66;
  color: #000000;
}
#T_a0632_row2_col0 {
  background-color: #54aa66;
  color: #f1f1f1;
}
#T_a0632_row3_col0 {
  background-color: #008066;
  color: #f1f1f1;
}
</style>
<table id="T_a0632" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_a0632_level0_col0" class="col_heading level0 col0" >Survived</th>
    </tr>
    <tr>
      <th class="index_name level0" >Fare_Range</th>
      <th class="blank col0" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a0632_level0_row0" class="row_heading level0 row0" >(-0.001, 7.91]</th>
      <td id="T_a0632_row0_col0" class="data row0 col0" >0.197309</td>
    </tr>
    <tr>
      <th id="T_a0632_level0_row1" class="row_heading level0 row1" >(7.91, 14.454]</th>
      <td id="T_a0632_row1_col0" class="data row1 col0" >0.303571</td>
    </tr>
    <tr>
      <th id="T_a0632_level0_row2" class="row_heading level0 row2" >(14.454, 31.0]</th>
      <td id="T_a0632_row2_col0" class="data row2 col0" >0.454955</td>
    </tr>
    <tr>
      <th id="T_a0632_level0_row3" class="row_heading level0 row3" >(31.0, 512.329]</th>
      <td id="T_a0632_row3_col0" class="data row3 col0" >0.581081</td>
    </tr>
  </tbody>
</table>




- fare_range가 증가함에 따라 생존률이 증가
- Fare_range 값을 Age_band와 같이 범주형 값으로 변경


```python
data['Fare_cat'] = 0
data.loc[data['Fare'] <= 7.91,'Fare_cat'] = 0
data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454),'Fare_cat'] = 1
data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31),'Fare_cat'] = 2
data.loc[(data['Fare'] > 31) & (data['Fare'] <= 513),'Fare_cat'] = 3
```


```python
sns.pointplot(x = 'Fare_cat',y = 'Survived',data=data,hue='Sex')
plt.show()
```


    
![png](output_89_0.png)
    


- Fare_cat이 증가함에 따라 생존률이 증가함
- Sex와 더불어 모델링 시 중요한 feature로 예상됨

## **2-2. feature 변환**
- 모델에 적합할 수 있는 형태로 feature들을 변환

### **String -> 수치형**
- ML 모형은 수치형 변수들만 처리 가능


```python
data['Sex'].replace(['male','female'],[0,1],inplace = True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace = True)
data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace = True)
```

## **2-3. 불필요한 feature 제거**
- Name: 범주형 변수로 변환 불가
- Age: Age_band 변수로 대체
- Ticket: 범주형 변수로 변환하기엔 너무 다채로움
- Fare: Fare_cat 변수로 대체
- Cabin: 많은 결측치(NaN), 한 승객이 여러 개의 Cabin
- Fare_range: fare_cat 변수로 대체
- PassengerId: 범주형 변수로 변환 불가


```python
data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis = 1,inplace = True)
```


```python
### 최종 변수들의 heatmap

sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})

fig = plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()
```


    
![png](output_96_0.png)
    


- 몇몇 변수들이 **양의 상관관계**를 가짐을 확인할 수 있음
  - SibSp & Family_Size
  - Parch & Family_Size
- 변수들 간의 **음의 상관관계**도 확인할 수 있음
  - Alone & Family_Size

# **3. 예측적 모델링**
- 분류 알고리즘을 사용하여 승객의 생존 여부를 예측
- 활용 알고리즘들
  - 서포트 벡터 머신(Support Vector Machine)
  - 로지스틱 회귀(LogisticRegression)
  - 결정 트리(Decision Tree)
  - K-최근접 이웃(K-Nearest Neighbors)
  -가우스 나이브 베이즈(Gauss Naive Bayes)
  - 랜덤 포레스트(RandomForest)
  - 로지스틱 회귀(LogisticRegression)


```python
### ML에 필요한 모든 library import

from sklearn.linear_model import LogisticRegression 
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import train_test_split 
from sklearn import metrics # accuracy measure
from sklearn.metrics import confusion_matrix # for confusion matrix
```


```python
### 데이터 분할

train, test = train_test_split(data,test_size = 0.3,random_state = 0,stratify = data['Survived'])

train_X = train[train.columns[1:]]
train_Y = train[train.columns[:1]]
test_X = test[test.columns[1:]]
test_Y = test[test.columns[:1]]

X = data[data.columns[1:]]
Y = data['Survived']
```

## **3-1. 기본 알고리즘**

### **Support Vector Machines**
- C: 마진 오류를 얼마나 허용할 것인가
  - 클수록 마진이 넓어지고 오류 증가
  - 작을수록 마진이 좁아지고 오류 감소
- kernel: 커널 함수 종류
  - 'linear', 'poly', 'rbf', 'sigmoid'
- gamma: 커널 계수 지정
  - kernel이 'poly', 'rbf', 'sigmoid'일 때 유효

---
※ Reference: 핸즈온 머신러닝


```python
### Radial Support Vector Machines(rbf-SVM)

model = svm.SVC(kernel = 'rbf',C = 1,gamma = 0.1) # 모델 객체 생성

model.fit(train_X,train_Y) # 학습
prediction1 = model.predict(test_X) # 예측
print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,test_Y)) # 평가
```

    Accuracy for rbf SVM is  0.835820895522388
    


```python
### Linear Support Vector Machine(linear-SVM)

model = svm.SVC(kernel = 'linear',C = 0.1,gamma = 0.1)
model.fit(train_X,train_Y)
prediction2=model.predict(test_X)
print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2,test_Y))
```

    Accuracy for linear SVM is 0.8171641791044776
    

### **Logistic Regression**


```python
model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3 = model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
```

    The accuracy of the Logistic Regression is 0.8134328358208955
    

### **결정 트리**


```python
model = DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction4 = model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction4,test_Y))
```

    The accuracy of the Decision Tree is 0.8059701492537313
    

### **K-Nearest Neighbours(KNN)**



```python
model = KNeighborsClassifier() 
model.fit(train_X,train_Y)
prediction5 = model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction5,test_Y))
```

    The accuracy of the KNN is 0.8134328358208955
    


```python
### n_neignbors 값을 변경하며 KNN 모델의 정확도 확인하기

a_index = list(range(1,11))
a = pd.Series() # 정확도가 저장될 Series 객체
x = [0,1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):
    model = KNeighborsClassifier(n_neighbors = i)  
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    a = a.append(pd.Series(metrics.accuracy_score(prediction,test_Y)))

plt.plot(a_index, a)
plt.xticks(x)
fig = plt.gcf()
fig.set_size_inches(12,6)
plt.show()

print()
print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())
```


    
![png](output_111_0.png)
    


    
    Accuracies for different values of n are: [0.73134328 0.76119403 0.79477612 0.80597015 0.81343284 0.80223881
     0.82835821 0.83208955 0.84701493 0.82835821] with the max value as  0.8470149253731343
    

### **Gaussian Naive Bayes**


```python
model = GaussianNB()
model.fit(train_X,train_Y)
prediction6 = model.predict(test_X)
print('The accuracy of the NaiveBayes is',metrics.accuracy_score(prediction6,test_Y))
```

    The accuracy of the NaiveBayes is 0.8134328358208955
    

### **Random Forests**


```python
model = RandomForestClassifier(n_estimators = 100)
model.fit(train_X,train_Y)
prediction7 = model.predict(test_X)
print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction7,test_Y))
```

    The accuracy of the Random Forests is 0.8059701492537313
    

- 모델의 정확성이 분류기의 성능을 결정하는 유일한 요소는 아님
  - 분류기가 자체 교육에 사용할 모든 인스턴스를 결정할 수는 없음
  - train 및 test 데이터가 변경됨에 따라 정확도도 변경됨
> 모형 분산(model variance)

- 일반화된 모델을 얻기 위해 **교차 검증**을 활용

## **3-2. 교차 검증(Cross Validation)**
- 많은 경우 데이터가 불균형함
  - 최대한 데이터 세트의 모든 인스턴스에서 알고리즘을 train하고 test 해야 함
  - 데이터 세트에 대해 알려진 모든 정확도의 평균을 얻기





### **K-Fold 교차 검증**  
- 먼저 데이터 세트를 k-부분 집합으로 나누기
- 데이터 세트를 (k = 5) 부분으로 나눈다고 가정하면 test를 위해 1개의 부분을 정하고 4개의 부분에 걸쳐 알고리즘을 train
- 각 반복에서 test 부분을 변경하고 다른 부분에 대해 알고리즘을 훈련
  - 이후 각 반복마다 얻은 정확도와 오차를 평균
- 알고리즘이 일부 train 데이터에 적합하지 않거나(underfitting) 지나치게 적합되는 것(overfitting) 방지


```python
### import libraries
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score # score evaluation
from sklearn.model_selection import cross_val_predict # prediction

kfold = KFold(n_splits = 10, random_state = 22, shuffle = True) # k=10, split the data into 10 equal parts
xyz = []
accuracy = []
std = []
classifiers = ['Linear Svm','Radial Svm','Logistic Regression','KNN',
               'Decision Tree','Naive Bayes','Random Forest']

models = [svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),
          KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),
          RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy") # 교차 검증 수행
    cv_result = cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)

new_models_dataframe2 = pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2
```





  <div id="df-70e785d9-34c7-428f-b604-dae06b8c0332">
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
      <th>CV Mean</th>
      <th>Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Linear Svm</th>
      <td>0.784607</td>
      <td>0.057841</td>
    </tr>
    <tr>
      <th>Radial Svm</th>
      <td>0.828377</td>
      <td>0.057096</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.799176</td>
      <td>0.040154</td>
    </tr>
    <tr>
      <th>KNN</th>
      <td>0.808140</td>
      <td>0.035630</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.805855</td>
      <td>0.042848</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.795843</td>
      <td>0.054861</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.811486</td>
      <td>0.049518</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-70e785d9-34c7-428f-b604-dae06b8c0332')"
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
          document.querySelector('#df-70e785d9-34c7-428f-b604-dae06b8c0332 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-70e785d9-34c7-428f-b604-dae06b8c0332');
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
### 시각화

plt.subplots(figsize = (12,6))
box = pd.DataFrame(accuracy,index = [classifiers])
box.T.boxplot()
```




    <Axes: >




    
![png](output_120_1.png)
    



```python
new_models_dataframe2['CV Mean'].plot.barh(width = 0.8)

plt.title('Average CV Mean Accuracy')
fig = plt.gcf()
fig.set_size_inches(8,5)
plt.show()
```


    
![png](output_121_0.png)
    


- 분류 정확도는 불균형으로 인해 때때로 오해의 소지가 있을 수 있음
- 모델이 어디서 잘못되었는지, 또는 모델이 어떤 클래스를 잘못 예측했는지를 보여주는 **오차 행렬**(confusion matrix)의 도움으로 요약된 결과를 얻을 수 있음

### **오차 행렬** 
- 분류기에 의해 만들어진 정확한 분류와 잘못된 분류의 수를 제공


```python
f,ax = plt.subplots(3,3,figsize=(12,10))

y_pred = cross_val_predict(svm.SVC(kernel = 'rbf'),X,Y,cv = 10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Matrix for rbf-SVM')

y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Matrix for Linear-SVM')

y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')
ax[0,2].set_title('Matrix for KNN')

y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Matrix for Random-Forests')

y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Matrix for Logistic Regression')

y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')
ax[1,2].set_title('Matrix for Decision Tree')

y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')
ax[2,0].set_title('Matrix for Naive Bayes')

plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()
```


    
![png](output_124_0.png)
    


**✔ 오차 행렬 해석**  
- 왼쪽 대각선: 각 클래스에 대해 수행된 **정확한** 예측의 수/ 오른쪽 대각선: **잘못된** 예측의 수를 표시

-  rbf-SVM에 대한 첫 번째 그림 해석  
  - 정확한 예측의 수는 491(사망자의 경우) + 247(생존자의 경우)이며 평균 CV 정확도는 (491+247)/891 = 82.8%
  - 오류--> 58명의 사망자를 생존자로 잘못 분류하고 95명은 사망자를 생존자로 잘못 분류
  - 모든 행렬을 살펴본 후, 우리는 rbf-SVM이 사망한 승객을 정확하게 예측할 가능성이 더 높지만, 나이브 베이즈는 생존한 승객을 정확하게 예측할 가능성이 더 높다고 말할 수 있음

**✔ 하이퍼 파라미터 튜닝**  
- 하이퍼 파라미터: 서로 다른 분류기에 대한 서로 다른 매개 변수
- 이를 조정하여 알고리즘의 학습 속도 변경 등 더 나은 모델을 얻을 수 있음 => 튜닝



```python
### SVM 하이퍼 파라미터 튜닝

from sklearn.model_selection import GridSearchCV

C = [0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel = ['rbf','linear']
hyper = {'kernel':kernel,'C':C,'gamma':gamma}

# 하이퍼 파라미터 튜닝
gd = GridSearchCV(estimator = svm.SVC(),param_grid = hyper,verbose = True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)
```

    Fitting 5 folds for each of 240 candidates, totalling 1200 fits
    0.8282593685267716
    SVC(C=0.4, gamma=0.3)
    

- C = 0.4, gamma = 0.3일 때 정확도가 82.82%로 가장 좋은 성능


```python
### RandomForest 하이퍼 파라미터 튜닝

n_estimators = range(100,1000,100)
hyper = {'n_estimators':n_estimators}

gd = GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)
```

    Fitting 5 folds for each of 9 candidates, totalling 45 fits
    0.819327098110602
    RandomForestClassifier(n_estimators=300, random_state=0)
    

- n_estimators = 300일 때 정확도가 81.9% 정도로 가장 좋은 성능

## **3-3. 앙상블(Ensembling)**
- 다양한 단순한 모델들이 결합하여 **하나**의 강력한 모델을 만드는 것
  - 모델의 정확도나 성능을 높이는 좋은 방법
- 방법
  - Voting Classifier
  - Bagging
  - Boosting

### **VotingClassifier**
- 다양한 기계 학습 모델의 예측을 결합하는 가장 간단한 방법
- 모든 하위 모델의 예측을 기반으로 **평균** 예측 결과를 제공
  - 이때 하위 모델 또는 기본 모델은 모두 **다른** 유형


```python
from sklearn.ensemble import VotingClassifier

ensemble_lin_rbf = VotingClassifier(estimators = [('KNN',KNeighborsClassifier(n_neighbors=10)),
                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                             ], 
                       voting = 'soft').fit(train_X,train_Y) # soft voting
print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))
cross = cross_val_score(ensemble_lin_rbf,X,Y, cv = 10, scoring = "accuracy")
print('The cross validated score is',cross.mean())
```

    The accuracy for ensembled model is: 0.8171641791044776
    The cross validated score is 0.8249188514357053
    

### **Bagging**
- 일반적인 앙상블 방식
- 데이터 세트의 작은 파티션(일부분)에 **유사한** 분류기를 적용한 다음 모든 예측의 평균을 취함으로써 작동
- 평균화로 인해 분산이 감소됨


**✔ Baged KNN**  
- 배깅은 분산이 **높은** 모형에서 가장 잘 작동
ex> 의사 결정 트리, 랜덤 포레스트
- n_neighbors로 작은 값을 가지는 KNN과 함께 사용할 수 있음


```python
from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(base_estimator = KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,test_Y))

result = cross_val_score(model,X,Y,cv = 10,scoring='accuracy')
print('The cross validated score for bagged KNN is:',result.mean())
```

    The accuracy for bagged KNN is: 0.832089552238806
    The cross validated score for bagged KNN is: 0.8104244694132333
    

**✔ Bagged DecisionTree**  



```python
model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(train_X,train_Y)
prediction = model.predict(test_X)
print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(prediction,test_Y))

result = cross_val_score(model,X,Y,cv = 10,scoring = 'accuracy')
print('The cross validated score for bagged Decision Tree is:',result.mean())
```

    The accuracy for bagged Decision Tree is: 0.8208955223880597
    The cross validated score for bagged Decision Tree is: 0.8171410736579275
    

### **Boosting**
- 분류기의 **순차 학습**을 사용하는 앙상블 기술
  - 약한 모델을 단계적으로 향상시키는 것
-  먼저 전체 데이터 세트에 대해 훈련
  - 일부 인스턴스는 맞지만 일부 인스턴스는 틀림
- 이후 다음 반복에서 잘못 예측된 사례에 데 집중하거나 비중을 두어 잘못된 예측을 바로잡으려 노력
  - 정확도가 한계에 다다를 때까지 새로운 분류기를 추가시키며 학습

**✔ AdaBoost(Adaptive Boosting)**  
- 약한 분류기: 의사 결정 트리
  - ```base_estimator``` 옵션에서 다른 알고리즘으로 변경 가능


```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators = 200,random_state = 0,learning_rate = 0.1)
result = cross_val_score(ada,X,Y,cv = 10,scoring = 'accuracy')
print('The cross validated score for AdaBoost is:',result.mean())
```

    The cross validated score for AdaBoost is: 0.8249188514357055
    

**✔ Stochastic Gradient Boosting**  
- 약한 분류기: 의사 결정 트리


```python
from sklearn.ensemble import GradientBoostingClassifier

grad = GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result = cross_val_score(grad,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())
```

    The cross validated score for Gradient Boosting is: 0.8115230961298376
    

**✔ XGBoost** 


```python
import xgboost as xg

xgboost = xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
result = cross_val_score(xgboost,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())
```

    The cross validated score for XGBoost is: 0.8160299625468165
    

- AdaBoost에서 가장 좋은 성능



```python
### AdaBoost - hyper parameter tuning

n_estimators = list(range(100,1100,100))
learn_rate = [0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper = {'n_estimators':n_estimators,'learning_rate':learn_rate}

gd = GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(X,Y)

print(gd.best_score_)
print(gd.best_estimator_)
```

    Fitting 5 folds for each of 120 candidates, totalling 600 fits
    0.8293892411022534
    AdaBoostClassifier(learning_rate=0.1, n_estimators=100)
    

- learning_rate = 0.1, n_estimators = 100일 때 정확도가 82.94% 정도로 가장 좋은 성능을 보임

### **최적 모형에 대한 오차 행렬**


```python
ada = AdaBoostClassifier(n_estimators=100,random_state=0,learning_rate=0.1)
result = cross_val_predict(ada,X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,result),cmap = 'winter',annot=True,fmt='2.0f')
plt.show()
```


    
![png](output_150_0.png)
    


## **3-4. Feature 중요도**


```python
f,ax = plt.subplots(2,2,figsize=(15,12))

### RandomForest
model = RandomForestClassifier(n_estimators=500,random_state=0) 
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])
ax[0,0].set_title('Feature Importance in Random Forests')

### AdaBoost
model=AdaBoostClassifier(n_estimators=100,learning_rate=0.1,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#ddff11')
ax[0,1].set_title('Feature Importance in AdaBoost')

### Gradient Boosting
model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')
ax[1,0].set_title('Feature Importance in Gradient Boosting')

### XGBoost
model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='#FD0F00')
ax[1,1].set_title('Feature Importance in XgBoost')

plt.show()
```


    
![png](output_152_0.png)
    


**✔ Observations**  
- 공통적으로 중요하다고 나타나는 feature들: Initial, Fare_cat, Pclass, Family_Size
- Sex 기능은 중요하지 않은 feature로 보임
  - RandomForest에서만 Sex가 중요해 보임
  - 그러나 많은 분류기에서 맨 위에 Initial feature가 있음을 확인할 수 있는데, 이는 둘 다 성별을 언급

