---
layout: single
title:  "[ECC DS 10주차] 회귀 2_캐글 주택 가격:고급 회귀 기법"
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


# **0. Introduction**


- 미국 아이오아 주의 에임스 지방의 ```주택 가격 정보```를 가지고 있음

- 성능 평가 지표: ```RMSLE```

  - 가격이 비싼 주택일수록 예측 결과 오류가 전체 오류에 미치는 비중이 높으므로 이를 상쇄하기 위해 오류 값을 변환한 RMSLE를 이용

- [대회 링크](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)


**📌 Data Descrpition**  

[대회 안내](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)


# **1. 데이터 사전 처리(Preprocessing)**



```python
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

## **1-1. 데이터 확인**



```python
house_df_org = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ECC 48기 데과B/10주차/data/house_price.csv')
house_df = house_df_org.copy() # 데이터 가공을 여러 번 하기 위해 원본을 복사해 둠
house_df.head(3)
```

<pre>
   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
0   1          60       RL         65.0     8450   Pave   NaN      Reg   
1   2          20       RL         80.0     9600   Pave   NaN      Reg   
2   3          60       RL         68.0    11250   Pave   NaN      IR1   

  LandContour Utilities  ... PoolArea PoolQC Fence MiscFeature MiscVal MoSold  \
0         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   
1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      5   
2         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      9   

  YrSold  SaleType  SaleCondition  SalePrice  
0   2008        WD         Normal     208500  
1   2007        WD         Normal     181500  
2   2008        WD         Normal     223500  

[3 rows x 81 columns]
</pre>
- target 값은 맨 마지막 칼럼인 ```SalesPrice```임



```python
### 데이터 세트의 전체 크기와 칼럼의 타입, Null이 있는 칼럼과 건수를 내림차순으로 출력

print('데이터 세트의 Shape:', house_df.shape)
print('\n전체 feature들의 type \n',house_df.dtypes.value_counts())

isnull_series = house_df.isnull().sum()
print('\nNull 컬럼과 그 건수:\n ', isnull_series[isnull_series > 0].sort_values(ascending = False))
```

<pre>
데이터 세트의 Shape: (1460, 81)

전체 feature들의 type 
 object     43
int64      35
float64     3
dtype: int64

Null 컬럼과 그 건수:
  PoolQC          1453
MiscFeature     1406
Alley           1369
Fence           1179
FireplaceQu      690
LotFrontage      259
GarageType        81
GarageYrBlt       81
GarageFinish      81
GarageQual        81
GarageCond        81
BsmtExposure      38
BsmtFinType2      38
BsmtFinType1      37
BsmtCond          37
BsmtQual          37
MasVnrArea         8
MasVnrType         8
Electrical         1
dtype: int64
</pre>
- 데이터 세트는 1460개의 레코드와 81개의 피처로 구성돼 있음

- 피처의 타입은 숫자형, 문자형 등 다양함

  - 43개의 문자형 컬럼 + 37개의 숫자형 컬럼 + target 컬럼

- 데이터 양에 비해 Null 값이 많은 피처도 존재

  - Null 값이 너무 많은 피처는 drop




```python
### target 값의 분포 확인
# 분포도가 정규 분포인지 확인

plt.title('Original Sale Price Histogram')
sns.distplot(house_df['SalePrice'])
```

<pre>
<Axes: title={'center': 'Original Sale Price Histogram'}, xlabel='SalePrice', ylabel='Density'>
</pre>
<pre>
<Figure size 640x480 with 1 Axes>
</pre>
- 데이터 값의 분포가 중심에서 왼쪽으로 치우친 형태로, 정규 분포에서 벗어나 있음

- 정규 분포가 아닌 결과값을 정규 분포 형태로 변환하기 위해 ```로그 변환(Log Transformation)```을 적용

  - ```np.log1p()```를 이용해 로그 변환한 결괏값을 기반으로 학습

  - 예측 시는 다시 결괏값을 ```expm1()```으로 환원



```python
### 로그 변환 시 분포

plt.title('Log Transformed Sale Price Histogram')
log_SalePrice = np.log1p(house_df['SalePrice'])
sns.distplot(log_SalePrice)
```

<pre>
<Axes: title={'center': 'Log Transformed Sale Price Histogram'}, xlabel='SalePrice', ylabel='Density'>
</pre>
<pre>
<Figure size 640x480 with 1 Axes>
</pre>
- ```SalesPrice```를  로그 변환해 정규 분포 형태로 결괏값이 분포하게 되었음



## **1-2. 로그 변환 및 전처리**

- target 변수인 ```SalesPrice```는 로그 변환

- Null 값이 많은 피처들은 삭제

  - ```PoolQC```, ```MiscFeature```, ```Alley```, ```Fence```, ```FireplaceQu```

- ```Id```는 단순 식별자이므로 삭제

- ```LotFrontage```는 Null 값이 많으므로 평균값으로 대체


### **a) 숫자형 변수 전처리**



```python
### 로그 변환 및 전처리

# SalePrice 로그 변환
original_SalePrice = house_df['SalePrice']
house_df['SalePrice'] = np.log1p(house_df['SalePrice'])

# Null 이 너무 많은 컬럼들과 불필요한 컬럼 삭제
house_df.drop(['Id','PoolQC' , 'MiscFeature', 'Alley', 'Fence','FireplaceQu'], axis=1 , inplace=True)
# Drop 하지 않는 숫자형 Null컬럼들은 평균값으로 대체
house_df.fillna(house_df.mean(),inplace = True)

# Null 값이 있는 피처명과 타입을 추출
null_column_count = house_df.isnull().sum()[house_df.isnull().sum() > 0]
print('## Null 피처의 Type :\n', house_df.dtypes[null_column_count.index])
```

<pre>
## Null 피처의 Type :
 MasVnrType      object
BsmtQual        object
BsmtCond        object
BsmtExposure    object
BsmtFinType1    object
BsmtFinType2    object
Electrical      object
GarageType      object
GarageFinish    object
GarageQual      object
GarageCond      object
dtype: object
</pre>
- 숫자형 피처들의 경우 결측치가 더이상 존재하지 않음


### **b) 문자형 변수 전처리**


- 문자형 피처는 모두 One-hot Encoding으로 변환

  - ```pd.get_dummies()``` 활용

  - ```pd.get_dummies()```는 자동으로 문자열 피처를 원-핫 인코딩 변환하면서 Null 값은 'None' 칼럼으로 대체해주기 때문에 별도의 Null값 대체 로직이 필요 x



```python
### 문자형 변수 전처리

print('get_dummies() 수행 전 데이터 Shape:', house_df.shape)
house_df_ohe = pd.get_dummies(house_df)
print('get_dummies() 수행 후 데이터 Shape:', house_df_ohe.shape)

null_column_count = house_df_ohe.isnull().sum()[house_df_ohe.isnull().sum() > 0]
print('## Null 피처의 Type :\n', house_df_ohe.dtypes[null_column_count.index])
```

<pre>
get_dummies() 수행 전 데이터 Shape: (1460, 75)
get_dummies() 수행 후 데이터 Shape: (1460, 271)
## Null 피처의 Type :
 Series([], dtype: object)
</pre>
- 원-핫 인코딩 후 피처가 75개에서 271개로 증가함

- Null 값을 가진 피처는 더이상 존재하지 x


# **2. 선형 회귀 모델의 학습/예측/평가**


- 해당 대회에서는 성능 평가 지표르 ```RMSLE```를 채택함

- 이미 target 값인 ```SalesPrice```가 로그 변환됨

  - 예측값 역시 로그 변환된 ```SalesPrice``` 값을 기반으롶 예측하므로 원본 SalesPrice가 ```로그 변환```된 값임

> 예측 결과 오류에 RMSE만 적용하면 RMSLE가 자동으로 측정됨



```python
### 성능 평가를 위한 함수

from sklearn.metrics import mean_squared_error

# 단일 모델의 RMSE 값 측정
def get_rmse(model):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test , pred)
    rmse = np.sqrt(mse)
    print('{0} 로그 변환된 RMSE: {1}'.format(model.__class__.__name__,np.round(rmse, 3)))
    return rmse

# get_rmse()를 이용해 여러 모델의 RMSE 값을 반환
def get_rmses(models):
    rmses = [ ]
    for model in models:
        rmse = get_rmse(model)
        rmses.append(rmse)
    return rmses
```

## **2-1. 기본 선형 회귀 모델**



```python
### 학습/예측/평가

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop('SalePrice',axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, 
                                                    test_size = 0.2, random_state = 156)

# LinearRegression, Ridge, Lasso 학습/예측/평가
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)

ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)

models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)
```

<pre>
LinearRegression 로그 변환된 RMSE: 0.132
Ridge 로그 변환된 RMSE: 0.128
Lasso 로그 변환된 RMSE: 0.176
</pre>
<pre>
[0.13189576579154297, 0.12750846334053004, 0.17628250556471403]
</pre>

```python
### 피처 중요도 시각화 함수

def get_top_bottom_coef(model):
    # coef_ 속성을 기반으로 Series 객체를 생성. index는 컬럼명. 
    coef = pd.Series(model.coef_, index = X_features.columns)
    
    # + 상위 10개, - 하위 10개 coefficient 추출하여 반환.
    coef_high = coef.sort_values(ascending = False).head(10)
    coef_low = coef.sort_values(ascending = False).tail(10)
    
    return coef_high, coef_low
```


```python
### 모델별 피처 중요도 시각화

def visualize_coefficient(models):
    # 3개 회귀 모델의 시각화를 위해 3개의 컬럼을 가지는 subplot 생성
    fig, axs = plt.subplots(figsize = (24,10), nrows = 1, ncols = 3)
    fig.tight_layout() 
    
    # 입력 인자로 받은 list 객체인 models에서 차례로 model을 추출하여 회귀 계수 시각화
    for i_num, model in enumerate(models):
        # 상위 10개, 하위 10개 회귀 계수를 구하고, 이를 판다스 concat으로 결합
        coef_high, coef_low = get_top_bottom_coef(model)
        coef_concat = pd.concat([coef_high, coef_low])
        
        # 순차적으로 ax subplot에 barchar로 표현
        # 한 화면에 표현하기 위해 tick label 위치와 font 크기 조정

        axs[i_num].set_title(model.__class__.__name__+' Coeffiecents', size = 25)
        axs[i_num].tick_params(axis = "y", direction = "in", pad = -120)
        for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):
            label.set_fontsize(22)
        sns.barplot(x = coef_concat.values, y = coef_concat.index, ax = axs[i_num])
```


```python
# 앞 예제에서 학습한 lr_reg, ridge_reg, lasso_reg 모델의 회귀 계수 시각화

models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)
```

<pre>
<Figure size 2400x1000 with 3 Axes>
</pre>
- ```OLS 기반```의 LinearRegression과 Ridge의 경우는 회귀 계수가 유사한 형태로 분포돼 있음

- 그러나 Lasso는 전체적으로 회귀 계수 값이 매우 작고, 그 중 ```YearBuilt```만 가장 크고 다른 피처의 회귀 계수는 너무 작음



## **2-2. 교차 검증**



```python
from sklearn.model_selection import cross_val_score

def get_avg_rmse_cv(models):
    for model in models:
        # 분할하지 않고 전체 데이터로 cross_val_score( ) 수행
        # 모델별 CV RMSE값과 평균 RMSE 출력
        rmse_list = np.sqrt(- cross_val_score(model, X_features, y_target,
                                             scoring = "neg_mean_squared_error", cv = 5))
        rmse_avg = np.mean(rmse_list)
        print('\n{0} CV RMSE 값 리스트: {1}'.format(model.__class__.__name__, np.round(rmse_list, 3)))
        print('{0} CV 평균 RMSE 값: {1}'.format(model.__class__.__name__, np.round(rmse_avg, 3)))

# 앞 예제에서 학습한 lr_reg, ridge_reg, lasso_reg 모델의 CV RMSE값 출력           
models = [lr_reg, ridge_reg, lasso_reg]
get_avg_rmse_cv(models)
```

<pre>

LinearRegression CV RMSE 값 리스트: [0.135 0.165 0.168 0.111 0.198]
LinearRegression CV 평균 RMSE 값: 0.155

Ridge CV RMSE 값 리스트: [0.117 0.154 0.142 0.117 0.189]
Ridge CV 평균 RMSE 값: 0.144

Lasso CV RMSE 값 리스트: [0.161 0.204 0.177 0.181 0.265]
Lasso CV 평균 RMSE 값: 0.198
</pre>
- 데이터의 구성과 관계없이 라쏘의 성능이 OLS 모델이나 릿지 모델에 비해 떨어짐

  - 교차 검증을 수행한 뒤에도 별다른 성능의 개선이 x


## **2-3. 하이퍼 파라미터 튜닝**



```python
from sklearn.model_selection import GridSearchCV

def print_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid = params, 
                              scoring = 'neg_mean_squared_error', cv = 5)
    grid_model.fit(X_features, y_target)
    rmse = np.sqrt(-1* grid_model.best_score_)
    print('{0} 5 CV 시 최적 평균 RMSE 값: {1}, 최적 alpha:{2}'.format(model.__class__.__name__,
                                        np.round(rmse, 4), grid_model.best_params_))
    return grid_model.best_estimator_

ridge_params = {'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20]}
lasso_params = {'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10]}
best_rige = print_best_params(ridge_reg, ridge_params)
best_lasso = print_best_params(lasso_reg, lasso_params)
```

<pre>
Ridge 5 CV 시 최적 평균 RMSE 값: 0.1418, 최적 alpha:{'alpha': 12}
Lasso 5 CV 시 최적 평균 RMSE 값: 0.142, 최적 alpha:{'alpha': 0.001}
</pre>
- 라쏘 모델의 경우 alpha 값 최적화 이후 예측 성능이 많이 좋아짐



```python
### 최적 alpha 값을 설정한 뒤, 다시 모델 학습/예측/평가 수행해보기


# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행. 
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)

ridge_reg = Ridge(alpha = 12)
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha = 0.001)
lasso_reg.fit(X_train, y_train)

# 모든 모델의 RMSE 출력
models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

# 모든 모델의 회귀 계수 시각화 
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)
```

<pre>
LinearRegression 로그 변환된 RMSE: 0.132
Ridge 로그 변환된 RMSE: 0.124
Lasso 로그 변환된 RMSE: 0.12
</pre>
<pre>
<Figure size 2400x1000 with 3 Axes>
</pre>
- alpha 값 최적화 후 테스트 데이터 세트의 예측 성능이 더 좋아짐

- 모델별 회귀 계수도 많이 달라짐

- 기존에는 라쏘 모델의 회귀 계수가 나머지 두 개 모델과 많은 차이가 있었지만, 이번에는 릿지와 라쏘 모델에서 비슷한 피처의 회귀 계수가 높음

- 다만 라쏘 모델의 경우는 릿지에 비해 동일한 피처라도 회귀 계수의 값이 상당히 작음


# **3. 추가적인 데이터 가공**


## **3-1. 피처 데이터 세트의 데이터 분포도**



```python
from scipy.stats import skew

# object가 아닌 숫자형 피쳐의 컬럼 index 객체 추출.
features_index = house_df.dtypes[house_df.dtypes != 'object'].index
# house_df에 컬럼 index를 [ ]로 입력하면 해당하는 컬럼 데이터 셋 반환. apply lambda로 skew( )호출 
skew_features = house_df[features_index].apply(lambda x : skew(x))
# skew 정도가 1 이상인 컬럼들만 추출. 
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top.sort_values(ascending=False))
```

<pre>
MiscVal          24.451640
PoolArea         14.813135
LotArea          12.195142
3SsnPorch        10.293752
LowQualFinSF      9.002080
KitchenAbvGr      4.483784
BsmtFinSF2        4.250888
ScreenPorch       4.117977
BsmtHalfBath      4.099186
EnclosedPorch     3.086696
MasVnrArea        2.673661
LotFrontage       2.382499
OpenPorchSF       2.361912
BsmtFinSF1        1.683771
WoodDeckSF        1.539792
TotalBsmtSF       1.522688
MSSubClass        1.406210
1stFlrSF          1.375342
GrLivArea         1.365156
dtype: float64
</pre>
- 일반적으로 skew() 함수의 반환 값이 ```1 이상```인 경우를 왜곡 정도가 높다고 판단함

  - 상황에 따라 편차는 있음

  - 여기서는 1 이상의 값을 반환하는 피처만 추출해 왜곡 정도를 완화하기 위해 로그 변환을 적용

- skew()를 적용하는 숫자형 피처에서 ```원-핫 인코딩``` 된 카테고리 숫자형 피처는 제외해야 함

  - 코드성 피처이므로 인코딩 시 당연히 왜곡될 가능성이 높지만, 이는 왜곡과는 무관함



```python
### 왜곡이 심한 변수들을 로그 변환

house_df[skew_features_top.index] = np.log1p(house_df[skew_features_top.index])
```


```python
### 왜곡 정도를 다시 확인

# object가 아닌 숫자형 피쳐의 컬럼 index 객체 추출.
features_index = house_df.dtypes[house_df.dtypes != 'object'].index
# house_df에 컬럼 index를 [ ]로 입력하면 해당하는 컬럼 데이터 셋 반환. apply lambda로 skew( )호출 
skew_features = house_df[features_index].apply(lambda x : skew(x))
# skew 정도가 1 이상인 컬럼들만 추출. 
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top.sort_values(ascending=False))
```

<pre>
PoolArea         14.348342
3SsnPorch         7.727026
LowQualFinSF      7.452650
MiscVal           5.165390
BsmtHalfBath      3.929022
KitchenAbvGr      3.865437
ScreenPorch       3.147171
BsmtFinSF2        2.521100
EnclosedPorch     2.110104
dtype: float64
</pre>
- 여전히 높은 왜곡 정도를 가진 피처가 있지만, 더 이상 로그 변환을 하더라도 개선하기는 어렵기에 그대로 유지



```python
### 데이터 재가공

# Skew가 높은 피처들을 로그 변환 했으므로 다시 원-핫 인코딩 적용 및 피처/타겟 데이터 셋 생성
house_df_ohe = pd.get_dummies(house_df)
y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop('SalePrice',axis = 1, inplace = False)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, 
                                                    test_size = 0.2, random_state = 156)

# 피처들을 로그 변환 후 다시 최적 하이퍼 파라미터와 RMSE 출력
ridge_params = {'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20]}
lasso_params = {'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10]}
best_ridge = print_best_params(ridge_reg, ridge_params)
best_lasso = print_best_params(lasso_reg, lasso_params)
```

<pre>
Ridge 5 CV 시 최적 평균 RMSE 값: 0.1275, 최적 alpha:{'alpha': 10}
Lasso 5 CV 시 최적 평균 RMSE 값: 0.1252, 최적 alpha:{'alpha': 0.001}
</pre>
- 릿지 모델의 경우 최적 alpha 값이 12에서 10으로 변경됨

- 두 모델 모두 피처의 로그 변환 이전과 비교해 RMSE 값이 향상됨




```python
### 피처 중요도 시각화

# 앞의 최적화 alpha 값으로 학습 데이터로 학습, 테스트 데이터로 예측/평가 수행
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)

ridge_reg = Ridge(alpha = 10)
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha = 0.001)
lasso_reg.fit(X_train, y_train)

# 모든 모델의 RMSE 출력
models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

# 모든 모델의 회귀 계수 시각화 
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)
```

<pre>
LinearRegression 로그 변환된 RMSE: 0.128
Ridge 로그 변환된 RMSE: 0.122
Lasso 로그 변환된 RMSE: 0.119
</pre>
<pre>
<Figure size 2400x1000 with 3 Axes>
</pre>
- 세 모델 모두 ```GrLivArea```, 즉 주거 공간 크기가 회귀 계수가 가장 높은 피처가 됨


## **3-2. 이상치 데이터 처리**

- 회귀 계수가 높은 피처, 즉 예측에 많은 영향을 미치는 중요 피처의 이상치 데이터의 처리가 중요함



```python
### GrLivArea 피처의 데이터 분포도 살펴보기

plt.scatter(x = house_df_org['GrLivArea'], y = house_df_org['SalePrice'])
plt.ylabel('SalePrice', fontsize=15)
plt.xlabel('GrLivArea', fontsize=15)
plt.show()
```

<pre>
<Figure size 640x480 with 1 Axes>
</pre>
- 일반적으로 주거 공간이 큰 집일수록 가격이 비싸기 때문에 GrLivArea 피처는 SalesPrice와 양의 상관도가 매우 높음

- 이상치가 존재함을 확인할 수 있음

  - GrLivArea가 가장 큰 데도 불구하고 가격이 매우 낮음

  - GrLivArea가 4000평방피트 이상임에도 가격이 500,000 달러 이하인 데이터는 모두 이상치로 간주하고 삭제 



```python
### 이상치 데이터 삭제

# GrLivArea와 SalePrice 모두 로그 변환되었으므로 이를 반영한 조건 생성
cond1 = house_df_ohe['GrLivArea'] > np.log1p(4000)
cond2 = house_df_ohe['SalePrice'] < np.log1p(500000)
outlier_index = house_df_ohe[cond1 & cond2].index

print('아웃라이어 레코드 index :', outlier_index.values)
print('아웃라이어 삭제 전 house_df_ohe shape:', house_df_ohe.shape)

# DataFrame의 index를 이용하여 아웃라이어 레코드 삭제. 
house_df_ohe.drop(outlier_index , axis = 0, inplace = True)
print('아웃라이어 삭제 후 house_df_ohe shape:', house_df_ohe.shape)
```

<pre>
아웃라이어 레코드 index : [ 523 1298]
아웃라이어 삭제 전 house_df_ohe shape: (1460, 271)
아웃라이어 삭제 후 house_df_ohe shape: (1458, 271)
</pre>

```python
### 데이터 다시 생성

y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop('SalePrice',axis = 1, inplace = False)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, 
                                                    test_size = 0.2, random_state = 156)
```


```python
### 모델 최적화

ridge_params = {'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20]}
lasso_params = {'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10]}
best_ridge = print_best_params(ridge_reg, ridge_params)
best_lasso = print_best_params(lasso_reg, lasso_params)
```

<pre>
Ridge 5 CV 시 최적 평균 RMSE 값: 0.1125, 최적 alpha:{'alpha': 8}
Lasso 5 CV 시 최적 평균 RMSE 값: 0.1122, 최적 alpha:{'alpha': 0.001}
</pre>
- 단 두 개의 이상치 데이터만 제거하였음에도 불구하고 예측 수치가 매우 크게 향상됨

- 릿지 모델의 경우 최적 alpha 값이 12에서 8로 변경됨

  - 평균 RMSE가 0.1275에서 0.1125로 개선됨

- 라쏘 모델의 경우 평균 RMSE가 0.1252에서 0.1122로 개선됨  


- 이상치를 찾는 것은 쉽지 않지만. 회귀에 중요한 영향을 미치는 피처들 위주로 ```이상치 데이터```를 찾으려는 노력은 중요함

- 보통 머신러닝 프로세스 중에서 데이터의 가공은 알고리즘을 적용하기 이전에 수행함

  - 하지만 이것이 머신러닝 알고리즘을 적용하기 이전에 완벽하게 데이터의 선처리 작업을 수행하라는 의미는 아님

  - 일단 대략의 데이터 가공과 모델 최적화를 수행한 뒤 다시 이에 기반한 여러 가지 기법의 데이터 가공과 하이퍼 파라미터 기반의 모델 최적화를 ```반복적```으로 수행하는 것이 바람직한 머신러닝 모델 생성 과정임



```python
# 앞의 최적화한 alpha 값으로 학습 데이터로 학습, 테스트 데이터로 예측/평가 수행
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)

ridge_reg = Ridge(alpha = 8)
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha = 0.001)
lasso_reg.fit(X_train, y_train)

# 모든 모델의 RMSE 출력
models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

# 모든 모델의 회귀 계수 시각화 
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)
```

<pre>
LinearRegression 로그 변환된 RMSE: 0.129
Ridge 로그 변환된 RMSE: 0.103
Lasso 로그 변환된 RMSE: 0.1
</pre>
<pre>
<Figure size 2400x1000 with 3 Axes>
</pre>
# **4. 회귀 트리 학습/예측/평가**


- 수행 시간 이슈로 하이퍼 파라미터 설정을 미리 해두고 수행



```python
from xgboost import XGBRegressor

xgb_params = {'n_estimators':[1000]}
xgb_reg = XGBRegressor(n_estimators = 1000, learning_rate = 0.05, 
                       colsample_bytree = 0.5, subsample = 0.8)
best_xgb = print_best_params(xgb_reg, xgb_params)
```

<pre>
XGBRegressor 5 CV 시 최적 평균 RMSE 값: 0.1182, 최적 alpha:{'n_estimators': 1000}
</pre>

```python
from lightgbm import LGBMRegressor

lgbm_params = {'n_estimators':[1000]}
lgbm_reg = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=4, 
                         subsample=0.6, colsample_bytree=0.4, reg_lambda=10, n_jobs=-1)
best_lgbm = print_best_params(lgbm_reg, lgbm_params)
```

<pre>
LGBMRegressor 5 CV 시 최적 평균 RMSE 값: 0.1163, 최적 alpha:{'n_estimators': 1000}
</pre>

```python
### 피처 중요도 시각화

# 중요도 기준으로 상위 20개의 피처명과 그때의 중요도 값을 Series로 반환하는 함수
def get_top_features(model):
    ftr_importances_values = model.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values, index = X_features.columns)
    ftr_top20 = ftr_importances.sort_values(ascending = False)[:20]
    return ftr_top20

def visualize_ftr_importances(models):
    # 2개 회귀 모델의 시각화를 위해 2개의 컬럼을 가지는 subplot 생성
    fig, axs = plt.subplots(figsize = (24,10), nrows = 1, ncols = 2)
    fig.tight_layout() 
    
    # 입력 인자로 받은 list 객체인 models에서 차례로 model을 추출하여 피처 중요도 시각화
    for i_num, model in enumerate(models):
        # 중요도 상위 20개의 피처명과 그때의 중요도 값 추출 
        ftr_top20 = get_top_features(model)
        axs[i_num].set_title(model.__class__.__name__ + ' Feature Importances', size = 25)
        
        #font 크기 조정
        for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):
            label.set_fontsize(22)
        sns.barplot(x = ftr_top20.values, y = ftr_top20.index, ax = axs[i_num])

# 앞 예제에서 print_best_params( )가 반환한 GridSearchCV로 최적화 된 모델의 피처 중요도 시각화    
models = [best_xgb, best_lgbm]
visualize_ftr_importances(models)
```

<pre>
<Figure size 2400x1000 with 2 Axes>
</pre>
# **5. 회귀 모델들의 예측 결과 혼합을 통한 최종 예측**


- 여러 모델의 예측치에 가중치를 두어 최종 예측치를 도출

- 예측 성능이 조금 좋은 쪽에 주로 더 큰 가중치를 둠


**1. Ridge + Lasso**



```python
def get_rmse_pred(preds):
    for key in preds.keys():
        pred_value = preds[key]
        mse = mean_squared_error(y_test, pred_value)
        rmse = np.sqrt(mse)
        print('{0} 모델의 RMSE: {1}'.format(key, rmse))

# 개별 모델의 학습
ridge_reg = Ridge(alpha = 8)
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha = 0.001)
lasso_reg.fit(X_train, y_train)

# 개별 모델 예측
ridge_pred = ridge_reg.predict(X_test)
lasso_pred = lasso_reg.predict(X_test)

# 개별 모델 예측값 혼합으로 최종 예측값 도출
pred = 0.4 * ridge_pred + 0.6 * lasso_pred
preds = {'최종 혼합': pred,
         'Ridge': ridge_pred,
         'Lasso': lasso_pred}
         
#최종 혼합 모델, 개별모델의 RMSE 값 출력
get_rmse_pred(preds)
```

<pre>
최종 혼합 모델의 RMSE: 0.10007930884470506
Ridge 모델의 RMSE: 0.1034517754660323
Lasso 모델의 RMSE: 0.10024170460890035
</pre>
- 최종 혼합 모델의 RMSE가 개별 모뎋보다 성능 면에서 약간 개선됨



**2. XGBoost + LightGBM**



```python
xgb_reg = XGBRegressor(n_estimators = 1000, learning_rate = 0.05, 
                       colsample_bytree = 0.5, subsample = 0.8)
lgbm_reg = LGBMRegressor(n_estimators = 1000, learning_rate = 0.05, num_leaves = 4, 
                         subsample = 0.6, colsample_bytree = 0.4, reg_lambda = 10, n_jobs = -1)
xgb_reg.fit(X_train, y_train)
lgbm_reg.fit(X_train, y_train)

xgb_pred = xgb_reg.predict(X_test)
lgbm_pred = lgbm_reg.predict(X_test)

pred = 0.5 * xgb_pred + 0.5 * lgbm_pred
preds = {'최종 혼합': pred,
         'XGBM': xgb_pred,
         'LGBM': lgbm_pred}
        
get_rmse_pred(preds)
```

<pre>
최종 혼합 모델의 RMSE: 0.10129327758047968
XGBM 모델의 RMSE: 0.10617576258589495
LGBM 모델의 RMSE: 0.10382510019327311
</pre>
- 혼합 모델의 RMSE가 개별 모델의 RMSE보다 조금 향상됨


# **6. 스태킹 모델을 통한 회귀 예측**


- 스태킹 모델은 두 종류의 모델을 필요로 함

  1. 개별적인 ```기반``` 모델

  2. 개별 기반 모델의 예측 데이터를 학습 데이터로 만들어서 학습하는 최종 ```메타``` 모델 

- 스태킹 모델의 핵심은 여러 개별 모델의 예측 데이터를 각각 스태킹 형태로 결합해 최종 메타 모델의 학습용 피처 데이터 세트와 테스트용 피처 데이터 세트를 만드는 것임

  - 최종 메타 모델이 학습할 피처 데이터 세트는 원본 학습 피처 세트로 학습한 개별 모델의 예측값을 스태킹 형태로 결합한 것



```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# 개별 기반 모델에서 최종 메타 모델이 사용할 학습/테스트용 데이터를 생성하기 위한 함수 
def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds):
    # 지정된 n_folds값으로 KFold 생성
    kf = KFold(n_splits = n_folds, shuffle = False)
    # 추후에 메타 모델이 사용할 학습 데이터 반환을 위한 넘파이 배열 초기화 
    train_fold_pred = np.zeros((X_train_n.shape[0], 1))
    test_pred = np.zeros((X_test_n.shape[0], n_folds))
    print(model.__class__.__name__ , ' model 시작 ')
    
    for folder_counter, (train_index, valid_index) in enumerate(kf.split(X_train_n)):
        # 입력된 학습 데이터에서 기반 모델이 학습/예측할 폴드 데이터 셋 추출 
        print('\t 폴드 세트: ', folder_counter,' 시작 ')
        X_tr = X_train_n[train_index] 
        y_tr = y_train_n[train_index] 
        X_te = X_train_n[valid_index]  
        
        # 폴드 세트 내부에서 다시 만들어진 학습 데이터로 기반 모델의 학습 수행
        model.fit(X_tr, y_tr)       
        # 폴드 세트 내부에서 다시 만들어진 검증 데이터로 기반 모델 예측 후 데이터 저장.
        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1,1)
        # 입력된 원본 테스트 데이터를 폴드 세트 내 학습된 기반 모델에서 예측 후 데이터 저장. 
        test_pred[:, folder_counter] = model.predict(X_test_n)
            
    # 폴드 세트 내에서 원본 테스트 데이터를 예측한 데이터를 평균하여 테스트 데이터로 생성 
    test_pred_mean = np.mean(test_pred, axis = 1).reshape(-1,1)    
    
    #train_fold_pred는 최종 메타 모델이 사용하는 학습 데이터, test_pred_mean은 테스트 데이터
    return train_fold_pred , test_pred_mean
```


```python
### 1. 개별 모델 학습

# get_stacking_base_datasets( )은 넘파이 ndarray를 인자로 사용하므로 DataFrame을 넘파이로 변환
X_train_n = X_train.values
X_test_n = X_test.values
y_train_n = y_train.values

# 각 개별 기반(Base)모델이 생성한 학습용/테스트용 데이터 반환. 
ridge_train, ridge_test = get_stacking_base_datasets(ridge_reg, X_train_n, y_train_n, X_test_n, 5)
lasso_train, lasso_test = get_stacking_base_datasets(lasso_reg, X_train_n, y_train_n, X_test_n, 5)
xgb_train, xgb_test = get_stacking_base_datasets(xgb_reg, X_train_n, y_train_n, X_test_n, 5)  
lgbm_train, lgbm_test = get_stacking_base_datasets(lgbm_reg, X_train_n, y_train_n, X_test_n, 5)
```

<pre>
Ridge  model 시작 
	 폴드 세트:  0  시작 
	 폴드 세트:  1  시작 
	 폴드 세트:  2  시작 
	 폴드 세트:  3  시작 
	 폴드 세트:  4  시작 
Lasso  model 시작 
	 폴드 세트:  0  시작 
	 폴드 세트:  1  시작 
	 폴드 세트:  2  시작 
	 폴드 세트:  3  시작 
	 폴드 세트:  4  시작 
XGBRegressor  model 시작 
	 폴드 세트:  0  시작 
	 폴드 세트:  1  시작 
	 폴드 세트:  2  시작 
	 폴드 세트:  3  시작 
	 폴드 세트:  4  시작 
LGBMRegressor  model 시작 
	 폴드 세트:  0  시작 
	 폴드 세트:  1  시작 
	 폴드 세트:  2  시작 
	 폴드 세트:  3  시작 
	 폴드 세트:  4  시작 
</pre>

```python
### 2. 최종 메타 모델 학습/예측/평가

# 개별 모델이 반환한 학습 및 테스트용 데이터 세트를 Stacking 형태로 결합
Stack_final_X_train = np.concatenate((ridge_train, lasso_train, 
                                      xgb_train, lgbm_train), axis = 1)
Stack_final_X_test = np.concatenate((ridge_test, lasso_test, 
                                     xgb_test, lgbm_test), axis = 1)

# 최종 메타 모델은 라쏘 모델을 적용
meta_model_lasso = Lasso(alpha = 0.0005)

# 기반 모델의 예측값을 기반으로 새롭게 만들어진 학습 및 테스트용 데이터로 예측하고 RMSE 측정
meta_model_lasso.fit(Stack_final_X_train, y_train)
final = meta_model_lasso.predict(Stack_final_X_test)
mse = mean_squared_error(y_test , final)
rmse = np.sqrt(mse)
print('스태킹 회귀 모델의 최종 RMSE 값은:', rmse)
```

<pre>
스태킹 회귀 모델의 최종 RMSE 값은: 0.09751138566662847
</pre>
- 가장 좋은 성능 평가를 보여줌

- 스태킹 모델은 분류뿐만 아니라 회귀에서도 효과적으로 활용될 수 있음

