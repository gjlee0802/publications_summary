# SOTTA: Robust Test-Time Adaptation on Noisy Data Streams
~~~
SoTTA: Robust Test-Time Adaptation on Noisy Data Streams
Taesik Gong*, Yewon Kim*, Taeckyung Lee*, Sorn Chottananurak, and Sung-Ju Lee
Conference on Neural Information Processing Systems (NeurIPS), 2023.
~~~
## 1. Intro
### 1.1. TTA(Test-time adaptation)이란?
- Test-time adaptation(TTA): 모델이 훈련된 후에도 새로운 데이터나 환경에 적응하여 성능을 향상시키는 방법  
### 1.2. 문제
실제 데이터에서는 예기치 못한 테스트 샘플이 등장할 수 있는데 이는 현재 TTA 알고리즘에 새로운 위협으로 다가올 수 있음. 즉, 테스트 데이터는 실제 환경에서 예기치 않게 다양할 수 있으며, 관련 데이터뿐만 아니라 모델의 범위를 벗어난 외부 요소도 포함할 수 있음.  
이러한 문제를 개선하기 위해 논문에서는 SoTTA를 제안함.

### 1.3. SoTTA의 핵심 요소
SoTTA의 핵심 요소는 2가지임.  
- input-wise robustness : 높은 신뢰도의 균등 표본 추출(잡음이 많은 표본의 영향을 효과적으로 걸러냄)
- parameter-wise robustness : Entropy-sharpness 최소화(노이즈가 많은 샘플의 큰 그라디언트에 대해 모델 파라미터의 견고성을 향상시킴)

~~~
머신러닝에서 Robust란?
머신러닝에서 일반화(generalization)는 일부 특정 데이터만 잘 설명하는(=overfitting) 것이 아니라 범용적인 데이터도 적합한 모델을 의미한다. 즉, 잘 일반화하기 위해서는 이상치나 노이즈가 들어와도 크게 흔들리지 않아야(=robust) 한다.
~~~

### 1.4. 현 TTA의 직관적 해결책

TTA가 노이지한 샘플들에 대항하는 직관적인 해결책은 테스트 스트림에서 노이즈가 많은 샘플들을 선별하는 것일 것임. 현재 아래의 방법들이 있음.  
- OOD(Out-of-distribution) detection: 샘플이 훈련 데이터와 동일한 분포에서 추출되었는지 여부를 탐지
- OSDA(Open-set domain adaptation)와 UDA(Universal domain adaptation): 훈련 데이터에 없는 테스트 데이터에 알 수 없는 클래스가 있다고 가정하여 적응 시나리오를 일반화

그러나 이러한 방법에는 전체 훈련 데이터 배치와 레이블이 지정되지 않은 대상 데이터에 대한 액세스가 필요하며, 이는 개인 정보 보호 문제로 인해 모델이 테스트 시간에 훈련 데이터에 액세스할 수 없고 많은 데이터 배치를 저장할 수 없는 경우가 많다는 점에서 TTA 설정을 준수하지 않는 경우가 많음.  

### 1.5. input-wise robustness & parameter-wise robustness

- input-wise robustness는 모델이 양성 샘플로만 훈련되도록 노이즈가 많은 샘플을 필터링하는 것을 목표로 함.
- input-wise robustness는 **HUS(High-confidence Uniform class Sampling)**을 통해 모델을 업데이트할 때 노이즈 많은 샘플을 피하도록 함.
- parameter-wise robustness는 노이즈가 많은 샘플로 인한 큰 기울기로 인해 모델이 표류하는 것을 방지하는 방식으로 모델 가중치 업데이트하는 것을 추구함.
- parameter-wise robustness는 **ESM(Entropy-sharpness minimization)**을 통해 노이즈가 많은 샘플로 인해 발생하는 가중치 교란에 대해 손실 환경을 더 부드럽게 만들고 파라미터를 탄력적으로 조정함.

### 1.6. TTA 벤치마크
아래 세 가지 벤치 마크들을 네 가지의 다양한 수준의 분포 변화가 있는 노이즈가 많은 시나리오(Near, Far, Attack, Nosie) SoTTA를 평가함.  
- CIFAR10-C
- CIFAR100-C
- ImageNet-C

## 2. 사전 정의
### 2.1. noisy 테스트 샘플 정의
target 데이터 분포에 포함되지 않은 샘플을 나타내기 위해 noisy 테스트 샘플을 정의함.  
TTA는 일반적으로 손상된 샘플과 같은 OOD 샘플에 적응하는 것을 목표로 하기 때문에 `noisy` 용어를 사용하여 OOD(Out-of-Distribution)와 구별함.

### 2.2. 시나리오
- Benign: 잡음 표본이 없는 일반적인 TTA 학습 설정
- Near: target 분포에서 semantic shift
- Far: covariate shift이 명백한 severer shift
- Attack: 혼란을 동반하는 지능적으로 생성된 적대적 공격
- Noise: 무작위의(랜덤한) 노이즈
~~~
* covariate shift : 같은 object인데, 다른 각도/형태로 보이는 이미지 사이의 차이가 발생하는 것.
* semantic shift : 다른 object 사이에 차이가 발생하는 것.
~~~


## 3. 방법론
### 3.1. Problem
- 이전의 TTA 방법은 테스트 샘플이 양성이고 들어오는 테스트 샘플 배치에 맹목적으로 적응한다고 가정. 
- 테스트 시간 동안 노이즈가 많은 샘플이 존재하면 아직 문헌에서 탐구되지 않은 TTA 알고리즘에 대한 성능이 크게 저하될 수 있음.
~~~
TTA가 적용되는 상황
- TTA가 원본 데이터에 액세스할 수 없음
- 대상 test 데이터에 라벨이 주어지지 않음
- 모델은 지속적으로 개선되므로 바람직한 해결책은 다양한 모델에 적용되어야 함.
~~~
- 현존하는 해결책들은 이러한 상황에 적용하기 어려움.
- 예를 들어, OOD(Out-of-Distribution) 탐지 연구는 모델이 테스트 시간에 고정되어 있다는 가정 하에 구축되며, OSDA(Open-set Domain Adaptation) 방법은 학습을 위해 레이블이 지정된 소스와 레이블이 지정되지 않은 대상 데이터가 필요.

### 3.2. Challenge
- 이 문제를 해결하기 위해 SoTTA를 제안. 
- SoTTA는 (i) 모델을 업데이트할 때 노이즈 샘플을 선택하지 않는 HUS(high-confidence uniform-class sampling)을 통해 입력별 견고성을 달성하고, (ii) 노이즈 샘플로 인한 가중치 혼란에 대해 파라미터를 탄력적으로 만드는 ESM(Entropy-sharpness minimization)를 통해 파라미터별 견고성(robustness)을 달성
### (i) Input-wise robustness via HUS(High-confidence Uniform class Sampling)

Adaptation을 위해 샘플을 선택할 때 노이즈 샘플을 필터링하여 노이즈 샘플에 대한 입력별 견고성(robustness)을 보장하는 것임.  
레이블 없이 노이즈 샘플을 찾는 것이 어렵기 때문에 노이즈 샘플에 대한 모델 예측의 경험적 관찰을 기반으로 한 아이디어.  
세운 가설: 노이즈가 많은 샘플이 분포 변화로 인해 양성 샘플과 특성을 구별했으며, 이는 모델의 예측 출력을 통해 관찰할 수 있다는 것.  
양성 샘플을 식별하는 대용으로 작동하는 두 가지 유형의 특징을 조사함.
- 샘플들의 신뢰 분포
- 예측된 클래스 분포

![SoTTA_Figure5](/SoTTA_img/SoTTA_Figure5.PNG)  

#### 관찰: 분포 비교 분석

첫째, 양성 샘플들에 비해 샘플들의 신뢰도가 상대적으로 낮음.  
분포 이동(shift)이 심할수록 신뢰도가 떨어지는데(예: Far가 Near보다 신뢰도가 낮음), 이는 사전 학습된 모델이 분포 외 데이터보다 목표 분포에 대해 더 높은 신뢰도를 보인다는 이전 연구 결과와도 일치함.

둘째, noisy 샘플은 예측 측면에서 왜곡되는 경우가 많으며, 모델이 올바르게 분류되지 않도록 하는 것이 목표인 Attck을 제외하고는 더 심각한 이동 Shift(예: Noise)에서 이러한 현상이 두드러진다는 것을 발견함.  

이러한 skewed 분포는 p(y)의 바람직하지 않은 편향을 초래할 수 있으며 따라서 엔트로피 최소화와 같은 TTA 목표에 부정적인 영향을 미칠 수 있음. 

#### HUS Solution

메모리에서 예측된 클래스의 균형을 유지하면서 신뢰도 있는 샘플을 유지함. 그런 다음 메모리에서 선택된 샘플은 Adaptation에 사용됨.

타깃 test 샘플 x가 주어지면 HUS는 test 샘플 x의 신뢰도를 측정함. 구체적으로 각 test 샘플 x의 신뢰도 `C(x; θ)`를 다음과 같이 정의함.

![Confidence_function](/SoTTA_img/SoTTA_Confidence_formula.PNG)  

샘플의 신뢰도가 미리 정의된 임계값 `C0`보다 높으면 저장함. 이러한 방식으로 적응에 사용되는 메모리에서 신뢰도가 높은 샘플만 유지하므로 신뢰도가 낮은 노이즈 샘플의 영향을 줄임.  

또한, **메모리에 데이터를 저장하는 동안 클래스 간의 균형을 유지**함.  
**특히, 현재 test 샘플의 예측 클래스가 메모리에서 가장 유력한(널리 사용되는) 클래스가 아니라면 HUS는 가장 널리 사용되는 클래스의 한 무작위 샘플을 새 샘플로 무작위로 대체함.**  

**반대로, 현재 샘플이 메모리에서 가장 유력한(널리 사용되는) 클래스에 속할 경우 HUS는 같은 클래스에 있는 랜덤 샘플 하나를 현재 샘플로 대체함.**  
이 전략을 사용하면 클래스를 균일하게 유지할 수 있으며, 이는 노이즈가 많은 샘플을 필터링할 수 있을 뿐만 아니라 적응에 사용할 때 샘플 간의 클래스 편향을 제거하는 데 효과적이며, 이는 TTA에 유용하다는 것을 발견함.  

우리는 이 두 가지 메모리 관리 전략을 통해 적응을 위한 노이즈 샘플의 영향을 효과적으로 줄일 수 있을 뿐만 아니라 편향되고 신뢰도가 낮은 샘플로 인한 모델 성능 저하를 방지하여 양성만 있는 경우의 모델 성능을 향상시킬 수 있음을 발견함.  

메모리에 저장된 샘플을 사용하여 **이전 TTA 방법**에 따라 BN(Batch Normalization) 계층의 정규화 통계 및 아핀 파라미터를 업데이트함. 이는 계산 효율적일 뿐만 아니라 전체 계층을 업데이트하는 것과 비슷한 성능 향상을 보였다고 알려져 있음.  

적응을 위해 노이즈가 많은 샘플을 사용하는 것을 피하는 것을 목표로 하지만, 예를 들어 양성 샘플이나 이상치(outliers)와 유사한 경우 몇 가지 노이즈가 많은 샘플을 메모리에 여전히 저장할 수 있음.  

메모리 내 샘플의 시간적 분산에 robust하기 위해 메모리 내 샘플의 통계를 직접 사용하는 대신 **EMA(Exponential Moving Average, 지수 이동 평균)**을 사용하여 BN 통계(평균과 분산)를 업데이트함.  


~~~
* 이전 TTA 방법에 해당하는 논문은 아래를 참고

[29] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and
Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. In The Eleventh
International Conference on Learning Representations, 2023.

[38] Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent:
Fully test-time adaptation by entropy minimization. In International Conference on Learning
Representations, 2021.

[44] Longhui Yuan, Binhui Xie, and Shuang Li. Robust test-time adaptation in dynamic scenarios.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), pages 15922–15932, June 2023.
~~~
~~~
* EMA(Exponential Moving Average) 개념에 대한 자료는 아래를 참고

- https://ganghee-lee.tistory.com/26

지수 이동평균. 이 개념을 설명하기 앞서 먼저 average와 moving average를 비교해보자. 
average는 동일시점에서 산출되는 평균 값인데 반해 moving average는 시간이라는 개념이 도입됐을때 산출되는 평균 값이다. 
예를들어, 동전을 던지는데 9번 연속 앞면이 나오면 다음 번에는 높은 확률로 뒷면이 나올거라 기대하는 것과 같다. 
시간이라는 개념이 도입될 경우 최근의 정보가 더 많은 영향력을 미칠 수 있기 때문이다. 
여기서 exponential moving average란 최근 data에 지수적으로 높은 가중치를 주는 것이다.
(ex. Momentum optimizer를 보면 오래된 data일수록 베타를 계속 곱하기때문에 지수적으로 가중치가 낮아진다.)

- https://dev-jm.tistory.com/12
- https://taek-guen.tistory.com/22
~~~

### (ii) Parameter-wise robustness via entropy-sharpness minimization

![SoTTA_Figure6](/SoTTA_img/SoTTA_Figure6.PNG)  

#### 관찰: 
noisy 샘플을 사용한 적응은 모델이 양성 샘플에 적응하는 것을 방해하는 경우가 많다는 것을 관찰함.  

여기서 핵심 질문은 모델이 노이즈가 많은 샘플에 과적합되는 것을 방지하는 방법임.  
그림 6b는 다음 단락에서 설명하는 Entropy-sharpness minimization(ESM) 결과를 보여줌.  
ESM을 사용하면 noisy 데이터의 기울기 규범이 높게 유지되며 의도한 대로 적응한 후 양성 샘플의 정확도가 향상됨.  

#### Solution
모델 매개변수를 noisy 샘플과의 adaptation에 robust하게 만들기 위해서는 모델이 noisy 샘플로 인해 예상치 못한 모델 성능 저하에 회복(resilient)이 되도록 엔트로피 손실(Entropy-Loss) landscape를 더 부드럽게 해야 함.  
이를 위해 **순진한 엔트로피 손실(Naive entropy-loss)과 엔트로피 손실의 sparpness(선명도)를 공동으로 최소화**하여 noisy 샘플에서 오는 큰 기울기에 의한 가중치 혼란에 대응하여 손실 landscape를 robust하게 만듦.  

구체적으로는 다음과 같이 naive entropy minimization(E, 순수 엔트로피 최소화)를 entropy-sharpness minimization(ESM) 로 대체함.  

![SoTTA_ESM_Formula1](/SoTTA_img/SoTTA_ESM_formula1.PNG)  

여기서 entropy-sharpness(엔트로피 선명도) ES(x, θ)는 L2-norm constraint(L2 제약 조건) ρ을 가진 가중치 혼란 주변의 maximum objective(최대 목적)로 정의됨.  
이 joint optimization problem(공동 최적화 문제)를 해결하기 위해 [29]와 유사한 **sharpness aware minimization(선명도 인식 최소화)** [4]를 따르며,  
이는 원래 확률적 경사 하강법(SGD)과 같은 표준 최적화 알고리즘보다 모델의 generalizability(일반화 가능성)을 향상시키는 것을 목표로 함.  
~~~
[4] Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. In International Conference on Learning
Representations, 2021.

[29] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and
Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. In The Eleventh
International Conference on Learning Representations, 2023
~~~
