# SOTTA: Robust Test-Time Adaptation on Noisy Data Streams

## Intro
### TTA(Test-time adaptation)이란?
- Test-time adaptation(TTA): 모델이 훈련된 후에도 새로운 데이터나 환경에 적응하여 성능을 향상시키는 방법  
### 문제
실제 데이터에서는 예기치 못한 테스트 샘플이 등장할 수 있는데 이는 현재 TTA 알고리즘에 새로운 위협으로 다가올 수 있음. 즉, 테스트 데이터는 실제 환경에서 예기치 않게 다양할 수 있으며, 관련 데이터뿐만 아니라 모델의 범위를 벗어난 외부 요소도 포함할 수 있음.  
이러한 문제를 개선하기 위해 논문에서는 SoTTA를 제안함.

### SoTTA의 핵심 요소
SoTTA의 핵심 요소는 2가지임.  
- input-wise robustness : 높은 신뢰도의 균등 표본 추출(잡음이 많은 표본의 영향을 효과적으로 걸러냄)
- parameter-wise robustness : Entropy-sharpness 최소화(노이즈가 많은 샘플의 큰 그라디언트에 대해 모델 파라미터의 견고성을 향상시킴)

~~~
머신러닝에서 Robust란?
머신러닝에서 일반화(generalization)는 일부 특정 데이터만 잘 설명하는(=overfitting) 것이 아니라 범용적인 데이터도 적합한 모델을 의미한다. 즉, 잘 일반화하기 위해서는 이상치나 노이즈가 들어와도 크게 흔들리지 않아야(=robust) 한다.
~~~

### 현 TTA의 직관적 해결책

TTA가 노이지한 샘플들에 대항하는 직관적인 해결책은 테스트 스트림에서 노이즈가 많은 샘플들을 선별하는 것일 것임. 현재 아래의 방법들이 있음.  
- OOD(Out-of-distribution) detection: 샘플이 훈련 데이터와 동일한 분포에서 추출되었는지 여부를 탐지
- OSDA(Open-set domain adaptation)와 UDA(Universal domain adaptation): 훈련 데이터에 없는 테스트 데이터에 알 수 없는 클래스가 있다고 가정하여 적응 시나리오를 일반화

그러나 이러한 방법에는 전체 훈련 데이터 배치와 레이블이 지정되지 않은 대상 데이터에 대한 액세스가 필요하며, 이는 개인 정보 보호 문제로 인해 모델이 테스트 시간에 훈련 데이터에 액세스할 수 없고 많은 데이터 배치를 저장할 수 없는 경우가 많다는 점에서 TTA 설정을 준수하지 않는 경우가 많음.  

### input-wise robustness & parameter-wise robustness

- input-wise robustness는 모델이 양성 샘플로만 훈련되도록 노이즈가 많은 샘플을 필터링하는 것을 목표로 함.
- input-wise robustness는 **HUS(High-confidence Uniform class Sampling)**을 통해 모델을 업데이트할 때 노이즈 많은 샘플을 피하도록 함.
- parameter-wise robustness는 노이즈가 많은 샘플로 인한 큰 기울기로 인해 모델이 표류하는 것을 방지하는 방식으로 모델 가중치 업데이트하는 것을 추구함.
- parameter-wise robustness는 **ESM(Entropy-sharpness minimization)**을 통해 노이즈가 많은 샘플로 인해 발생하는 가중치 교란에 대해 손실 환경을 더 부드럽게 만들고 파라미터를 탄력적으로 조정함.

### TTA 벤치마크
아래 세 가지 벤치 마크들을 네 가지의 다양한 수준의 분포 변화가 있는 노이즈가 많은 시나리오(Near, Far, Attack, Nosie) SoTTA를 평가함.  
- CIFAR10-C
- CIFAR100-C
- ImageNet-C

## 사전 정의
### noisy 테스트 샘플 정의
target 데이터 분포에 포함되지 않은 샘플을 나타내기 위해 noisy 테스트 샘플을 정의함.  
TTA는 일반적으로 손상된 샘플과 같은 OOD 샘플에 적응하는 것을 목표로 하기 때문에 `noisy` 용어를 사용하여 OOD(Out-of-Distribution)와 구별함.

### 시나리오
- Benign: 잡음 표본이 없는 일반적인 TTA 학습 설정
- Near: target 분포에서 semantic shift
- Far: covariate shift이 명백한 severer shift
- Attack: 혼란을 동반하는 지능적으로 생성된 적대적 공격
- Noise: 무작위의(랜덤한) 노이즈
~~~
* covariate shift : 같은 object인데, 다른 각도/형태로 보이는 이미지 사이의 차이가 발생하는 것.
* semantic shift : 다른 object 사이에 차이가 발생하는 것.
~~~


## 방법론
TODO