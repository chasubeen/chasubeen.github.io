---
layout: single
title:  "[ECC DS 4주차] 노트북 Review & 개념 정리"
categories: ML
tags: [ECC, DS, Costa Rica] 
author_profile: false
---

<span style="background-color:#FFE6E6"> 📢 해당 포스트는 [ECC DS 4주차] 1. A Complete Introduction Walkthrough 에 대한 추가적인 개념정리입니다. </span>  
[캐글 노트북 필사](https://chasubeen.github.io/ml/Costa-Rica_1/)

# **1️⃣ Macro F1-score**
<span style="background-color:#B2C248"> [References_대회에서 자주 사용되는 평가산식들](https://dacon.io/en/forum/405817) </span>

## **1. 오차(Error)**
### **1-1. 정확도의 함정**
- 음성(negative, 0)보다 양성(positive, 1) target이 많은 데이터의 경우 정확도만 본다면 **무조건 양성**으로 예측하는 분류기가 성능이 더 좋음
- 따라서, 정확도(accuracy)만 보고 분류기의 성능을 판별하는 것은 이와 같은 ```정확도의 함정```에 빠질 수 있음

### **1-2. 오차 행렬(confusion matrix)**
- **코드**  

```Python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,pred)
```

```
array([[ 1,  0],
       [ 2, 88]])
```

- **결과 해석**  
<img src = "https://user-images.githubusercontent.com/98953721/230944118-30d404f9-d84a-4665-ac25-fa8091c90cd8.png" width = 300 height = 250>

<img src = "https://user-images.githubusercontent.com/98953721/230944666-1588445b-46b3-44ea-8f49-61d84c254aba.png" width = 350 height = 200>

- **용어 정리**
  - ```정밀도(precision)```
    - 양성 예측 정확도
    - TP/(TP+FP)
    - __무조건 양성__으로 판단하면 좋은 정밀도를 얻기에, 유용하지는 않은 방법임
    
  - ```재현율(recall)```
  	- 정확하게 감지한 양성 샘플의 비율
  	- __민감도(sensitivity)__ 또는 __True Positive Rate(TPR)__ 이라고도 부름
  	- TP/(TP+FN)
  	
  - ```F1 score```
  	- 정밀도와 재현율의 조화 평균을 나타내는 지표
  	
		<img src = "https://user-images.githubusercontent.com/98953721/230946709-358968a3-5fda-416f-8b3b-dfd1fd9db015.png" width = 400 height = 80>
		
		- 0에서 1사이의 값을 가지며, 1에 가까울수록 좋음
		- accuracy와 달리 클래스 데이터가 __불균형__ 할 때도 사용하기 좋음
		
		- ```Macro F1```
			- 클래스별/레이블별 F1-score의 __평균__  
			- 모든 class의 값에 __동등한__ 중요도를 부여
				- 비교적 적은 클래스(rare classes)에서 성능이 좋지 않다면, Macro-F1의 값은 낮게 나타남
				- 각 레이블의 발생 빈도를 계산에 반영하려면 **weighted** 옵션을 활용
		 
# **2️⃣ 상관관계(Correlations)**
<span style="background-color:#B2C248"> [References_상관 분석(Correlation Analysis)](https://bioinformaticsandme.tistory.com/58) </span>

## **0. 상관분석(Correlation Analysis)**
- 두 변수 간에 어떠한__선형적__ 관계를 가지는지를 분석하는 기법
- ```상관계수(correlation coefficient)```
	- (X와 Y가 함께 변하는 정도) / (X와 Y가 각각 변하는 정도)
	- X와 Y가 완전히 동일하면 +1, 반대 방향으로 완전히 동일하면 -1
	- cf> r = 0이라는 것은 X와 Y가 전혀 상관이 없다는 것이 아니라 '__선형__ 의 상관관계는 아니다'라고 해석하는 것이 더 적절함
		- 두 변수가 곡선 관계라면 상관계수로 설명할 수 없음 
	
## **1. 피어슨 상관계수(Pearson Correlation Coefficient)**
- 상관분석에서 가장 기본적으로 사용되는 상관계수
- __연속형 변수__ 의 상관관계를 측정하는 데 사용
- 모수적 검정(parametric test)

## **2. 스피어만 상관계수(Spearman Correlation Coefficient)**
- 스피어만 상관 계수(ρ: rho)
- 변수의 값 대신 __순위__ 를 활용한 상관계수 
- 비모수적 검정(non-parametric test)
- 데이터 내의 편차와 에러에 민감함
	- 다중 비교에서의 해결법: 본페로니 교정  
	
	<span style="background-color:#B2C248"> [References_본페로니 교정(Bonferroni correction)](https://velog.io/@chulhongsung/%EB%8B%A4%EC%A4%91%EA%B2%80%EC%A0%95%EB%AC%B8%EC%A0%9C%EC%99%80-%ED%95%B4%EA%B2%B0%EB%B2%95-%EB%B3%B8%ED%8E%98%EB%A1%9C%EB%8B%88-%EA%B5%90%EC%A0%95) </span>
	
# **3️⃣ Baseline Models**
- 기계 학습 알고리즘에서는 어떤 모델이 주어진 데이터 세트에 가장 잘 작동할 지를 미리 알 수 없음
	- 여러 모델을 시도해 보아야 함 
- 해당 대회에서 사용딘 모델들을 정리해 보자!

## **1. RandomForestClassifier**
- 지난 주 개념정리 부분을 참고해 주세요.

<span style="background-color:#B2C248"> [References](https://chasubeen.github.io/ml/%EB%AA%A8%EB%8D%B8-%EA%B0%9C%EB%85%90%EC%A0%95%EB%A6%AC/) </span>

- 데이터의 반복 __복원__ 추출 + random하게 변수 추출 -> 다양한 모델 생성
	- 모델 간의 공분산을 줄이는 효과
- 모델링 기준을 찾기 위해 활용됨
	- 트리 기반 모델의 ```기능 중요도(feature importances)```를 통해 중요한 변수들을 파악할 수 있음

## **2. Linear SVC**
### **2-1. 서포트 벡터 머신(Support Vector Machine)**
- 여러 집단들을 가장 잘 분류할 수 있는 최적의 선(```결정 경계```)를 찾기 위한 모델
	- ```비확률적``` __이진__ 선형 분류 모델을 만들 때 활용 
	- 경계로 표현하는 데이터들 중 가장 ```큰``` 폭을 가지는 경계를 찾는 알고리즘
	- 서포트 벡터: 영역의 __경계__ 부분의 데이터를 기준으로 한 평행한 두 직선
	- 마진: 두 집단 사이의 거리 -> 최적의 결정 경계는 마진을 최대화하는 경계
	
	<img src = "https://user-images.githubusercontent.com/98953721/230955875-f2d11163-45a1-44ce-9510-c3adf94640a1.png" width = 600 height = 150>

- 변수의 스케일(범위)에 따라 데이터의 위치가 달라짐 -> 결정 경계도 달라짐
	- 적절한 스케일링이 필요
	
	<img src = "https://user-images.githubusercontent.com/98953721/230957025-8638d63c-f62f-48e7-b0a0-ad2b15c29ef3.png" width = 600 height = 150>

### **2-2. SVC 클래스**
- 코드
```Python
from sklearn.svm import SVC
```

- 주요 파라미터
	- ```C```: 마진 오류를 얼마나 허용할 것인가
		- 값이 클수록 마진이 넓어지고 마진 오류 증가
		- 값이 작을수록 마진이 좁아지고 마진 오류 감소
		 
	- ```kernel```: 커널 함수 종류 지정
		- 'linear', 'poly', 'rbf', 'sigmoid'
		 
	- ```gamma```:  커널 계수 지정
		- 'poly', 'rbf', 'sigmoid'일 때 유효 

## **3. GaussianNB(Gaussian Naive Bayes)**
- ```Naive Bayes```: 확률(Bayes Theorem)을 이용해서 가장 합리적인 예측값을 계산하는 방식
- ```정규분포(가우시안 분포)```를 가정한 표본들을 대상으로 조건부 독립을 나타내, 항상 같은 분모를 갖는 조건 하에서, 분자의 값이 가장 큰 경우(= 확률이 가장 __높은__ 경우)를 ```선택```하는 것
- 설명변수가 __연속형__ 변수일 때 활용

<span style="background-color:#B2C248"> [NaiveBayes(나이브 베이즈) 모델](https://todayisbetterthanyesterday.tistory.com/17) </span>

## **4. MLPClassifier(Multi-Layer Perceptron Classifier)**
- ```다중 신경망``` 분류 알고리즘을 저장하고 있는 모듈
- 라이브러리 import

```Python
from sklearn.neural_network import MLPClassifier
```

- 모델 구현(해당 노트북에서..)

```
from sklearn.neural_network import MLPClassifier

model_results = cv_model(train_set, train_labels,
                        MLPClassifier(hidden_layer_sizes = (32, 64, 128, 64, 32)),
                        'MLP', model_results)
```

- ```model_results``` 변수에 모델 학습 결과를 저장
- ```hidden_layer_sizes``` 파라미터: 5개의 은닉층을 만들고 각 계층별로 지정된 개수만큼의 노드를 할당
	
<span style="background-color:#B2C248"> [다층 퍼셉트론](https://ko.d2l.ai/chapter_deep-learning-basics/mlp.html) </span>

## **5. LinearDiscriminantAnalysis(선형 판별 분석, Linear Discriminant Analysis)**
- ```분류``` 모델과 ```차원 축소```까지 동시에 사용하는 알고리즘
- __선형__ 으로 데이터를 분할 하는 방법
	- 직선을 이용해 데이터를 분할
	- 기본적으로 ```베이즈 정리```를 활용하여 선형 판별 함수를 구함
- 입력 데이터의 target(label) 클래스를 __최대한__ 으로 분리할 수 있는 축을 찾음
	- 클래스 간 분산(between-class-scatter)과 클래스 내부 분산(within-class-scatter)의 비율을 __최대화__ 하는 방식으로 차원 축소를 적용
	- 클래스 __간__ 분산은 최대한 크게, 클래스 __내부__ 분산은 최대한 작게 분리
	
<span style="background-color:#B2C248"> [선형판별분석(LDA)](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-18-%EC%84%A0%ED%98%95%ED%8C%90%EB%B3%84%EB%B6%84%EC%84%9DLDA) </span>

## **6. RidgeClassifierCV(RidgeClassifierCV)**
- 릿지 분류기 + 교차 검증
- ```릿지(Ridge)```
	- __규제__ 의 일종으로, 학습이 과적합되는 것을 방지하고자 일종의 penalty를 부여하는 것임
		- 규제 강도를 크게 하면 가중치가 더 많이 감소되고(규제를 중요시함), 규제 강도를 작게 하면 가중치가 증가함(규제를 중요시하지 않음)
	- 각 가중치 제곱의 합에 규제 강도를 곱한 값($Error=MAE+αw^2$)

## **7. K-NeighborsClassifier**
- 최근접 이웃 알고리즘

<img src = "https://user-images.githubusercontent.com/98953721/231062830-39134fa1-74fe-49a2-baf7-917658cf7716.png" width = 300 height = 250>

## **8. Extra Trees Classifier**
- 극도로 ```무작위화(Extremely Randomized)``` 된 기계 학습 방법
- 데이터 샘플 수와 특성 설정까지 랜덤
- 랜덤 포레스트(RandomForest)와 동일한 원리를 이용 -> 많은 특성을 공유함
- 랜덤 포레스트에 비해 속도가 빠르고 성능도 미세하게 높음
- Bootstrap 샘플링을 사용하지 않고 전체 특성 중 일부를 랜덤하게 선택해 노드 분할에 사용
	- 무작위 분할 중 가장 좋은 것을 분할 규칙으로 선택

<span style="background-color:#B2C248"> [Extra Trees 정리](https://velog.io/@nata0919/Extra-Trees-%EC%A0%95%EB%A6%AC) </span>


# **4️⃣ Model Update**
## **1. LGBM(Light Gradient Boosting Machine)**
- 지난 주 개념정리 부분을 참고해 주세요.

<span style="background-color:#B2C248"> [References](https://chasubeen.github.io/ml/%EB%AA%A8%EB%8D%B8-%EA%B0%9C%EB%85%90%EC%A0%95%EB%A6%AC/) </span>

# **5️⃣ 모델 최적화_HyperOpt**
## **1. 베이지안 최적화**
- ```objective function(목적 함수)```를 최대/최소로 하는 최적 해를 찾는 기법

## **2. HyperOpt**
- ```베이지안 최적화```의 접근 방식을 취하여 하이퍼 파라미터를 최적화하는 기법
	- ```목적 함수```와 ```하이퍼 파라미터``` 쌍을 대상으로 여러 모델들을 만들어 평가 -> 순차적으로 업데이트 하면서 최적의 조합을 찾아내는 방식
- 단계>  
	1. 목적 함수 정의: 최대화/최소화 하고 싶은 것
	2. 도메인 영역: 최적화의 대상을 지정
	3. 알고리즘 지정(최적화 지정): 과거 모델링 결과를 활용하여 다음 하이퍼 파라미터 값을 제안
	4. 최적화 수행
		- ```fmin()``` 함수
		- 지정해 주는 알고리즘과 최대 반복 횟수 등을 변경해 보면서 성능 차이를 모니터링
	
<span style="background-color:#B2C248"> [HyperOpt를 활용한 하이퍼 파라미터 튜닝](https://teddylee777.github.io/machine-learning/hyper-opt/#hyperopt-%EC%84%A4%EC%B9%98) </span>

# **6️⃣ 차원 축소(Dimension Reduction)**
- 이후 내용 추가할 예정..





