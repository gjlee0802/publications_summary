# Knocker: Vibroacoustic-based Object Recognition with Smartphones

~~~
Knocker: Vibroacoustic-based Object Recognition with Smartphones
Taesik Gong, Hyunsung Cho, Bowon Lee, and Sung-Ju Lee
ACM Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT) 2019 (UbiComp '19).
~~~
https://nmsl.kaist.ac.kr/pdf/IMWUT19_Knocker.pdf

## Introduction

![Knocker_Overview](/Knocker_img/Knocker_system_overview.PNG)  

- 멀티모달 센서 데이터로부터 SVM(Support Vector Machine)을 활용하여 classification 수행함.

- 소리만 사용하면 소음에 취약하기 때문에 노크 소리 외에도 물체별로 고유하고 소음에 강한 가속도계 및 자이로스코프 값을 활용하여 물체를 식별.

## Knock Detection, 노크 감지
### Peak detection(피크 감지) 프로세스
- 피크 감지 사용자가 물체를 두드리면 소리와 가속도계의 진폭에 갑자기 피크가 나타나고 시간이 지남에 따라 값이 감소. 

- 노크의 이러한 특성을 활용하여 스마트폰에 내장된 마이크와 가속도계를 모니터링하여 노크를 감지함. 

- 소리와 가속도계 모두에 미리 정의된 임계치 이상의 피크가 있으면 노커는 이러한 피크를 노크의 신호로 간주하고 추가 계산을 위해 센서의 스트림(소리, 가속도계 및 자이로스코프 값)을 버퍼링하기 시작함.

### Response Pruning(응답 가지치기) 프로세스
- Response Pruning: 일련의 원시 데이터에서 노이즈(예: 소리의 경우 주변 소음, 가속도계 및 자이로스코프의 경우 신체 움직임)를 배제하고 계산을 최소화하기 위해 노크 관련 응답만 추출해야 함. Response Pruning(응답 가지치기)라고 불리는 이 프로세스는 소리 응답을 가속도계 및 자이로스코프와 일치시킴.

- 정렬 과정에서 피크를 활용하여 노크가 시작될 때부터 소리에 대한 샘플 4,096개와 가속도계와 자이로스코프에 대한 샘플 32개를 추출. 23개의 물체를 대상으로 노크 응답의 지속 시간을 조사하는 실험 연구를 기반으로 이 값을 선택함. 지속 시간이 물체마다 다르며 대략 20ms에서 80ms 사이임을 발견했음.

- 내장 마이크의 경우 48kHz, 가속도계 및 자이로스코프의 경우 400Hz의 일반적인 샘플링 속도가 주어지면, 이는 소리의 경우 85ms, 가속도계 및 자이로스코프의 경우 80ms와 일치함. 이 설정은 노크 응답을 충분히 포착하고 노크와 관련 없는 데이터와 계산 오버헤드를 최소화함.


### Knock Validation 프로세스
- 피크 감지 접근법은 가속도계 값과 소리 모두에서 우연히 변경될 수 있음. 예를 들어, 시끄러운 환경에서 스마트폰을 휘두르고 있는 사용자는 Knocker를 작동시킬 수 있음. 이 잘못된 positive는 사용자 경험에 해를 끼칠 수 있음. 잘못된 positive를 줄이기 위해, Knocker는 현재 반응이 실제 노크에서 비롯된 것인지 평가함.  

- 실제 노크와 의사 노크(pseudo knock, 물체를 대상으로 휘두르는 것처럼 허공에 휘두름)의 반응을 비교함. 그림 3은 실제 노크와 의사 노크의 가속도계 반응이 다르다는 것을 보여줌. 실제 노크는 가속도계 값의 급격한 변화로 인해 고주파 성분이 더 많은 반면, 의사 노크(pseudo knock)는 저주파 성분이 더 많음.

- 고주파 성분의 합(15Hz 이상)과 저주파 성분의 합(15Hz 미만)의 비율을 정의하고 이 비율을 사용하여 입력의 현재 스트림이 노크인지 여부를 다음과 같이 검사함:

    ![Knocker_Formula](/Knocker_img/Knocker_Formula.PNG)  

- 여기서 fi는 가속도계 응답에서 나오는 스펙트럼의 각 요소이고, FSumh는 고주파 성분의 합이며, FSuml은 저주파 성분의 합. 실험에서 비율이 2보다 크면 노크가 유효한 것으로 간주함

## Classification, 분류
### 특징 추출
- 음향에 크기 스펙트럼(magnitude spectrum), 로그 크기 스펙트럼(log magnitude spectrum) 및  mel-frequency cepstral coefficients(MFCC)의 세 가지 유형의 특징을 사용함. 고속 푸리에 변환(FFT)에서 파생된 크기 스펙트럼 외에도 로그 크기 스펙트럼도 사용함. 크기 스펙트럼은 물체당 특정 주파수에서 두드러진 피크를 효과적으로 나타내는 반면, 로그 크기 스펙트럼은 물체당 고유한 정보를 운반할 수 있는 상대적으로 낮은 전력으로 음향 신호의 주파수 내용을 향상시킴. 실험에 따르면 두 가지 특징 세트를 모두 사용하면 개별 특징보다 더 높은 분류 정확도를 얻을 수 있음.

- 자동 음성 인식에 일반적으로 사용되는 MFCC는 인간의 청각 시스템에 따라 주파수 대역을 간격을 두어 파생된 널리 사용되는 기능임. MFCC를 사용하는 데 있어 직관적인 이유는 인간이 서로 다른 물체에서 독특한 노크 소리를 식별할 수 있다는 것임.

- 길이-256 FFT로 노크의 방향을 고려하여 가속도계의 x축과 자이로스코프의 z축의 크기 스펙트럼(magnitude spectrum)을 사용함. 더 높은 주파수 분해능을 얻기 위해 자이로스코프 신호의 32개 샘플에 제로 패딩을 적용함. 
~~~
제로 패딩과 주파수 분해능에 대한 참고자료
https://www.bitweenie.com/listings/fft-zero-padding/
~~~


![Knocker_Table1](/Knocker_img/Knocker_Table1.PNG)  

- 이러한 특징은 표 1에 요약되어 있음. 특징 집합은 통계적 특징(평균, 분산 등), 쌍별 밴드 비율(pair-wise band ratio), 도함수와 같은 특징 선택 프로세스와 다양한 특징 집합을 탐색한 후 결정되었음. 그러나 계산 오버헤드를 감수하면서 정확도를 저하시키거나 정확도를 약간만 향상시킴.

### 분류기
- Weka machine learning toolkit에서 제공하는 sequential minimal optimization(SMO) 기반의 Support Vector Machine(c = 1.0, ϵ = 0.01, E = 1.0인 다항식 커널)을 분류기로 사용함.

- SVM은 분류를 위한 최적의 초평면을 구축하는 데 널리 사용되는 기계 학습 기법. 딥러닝 기법에 비해 학습 데이터와 런타임 복잡성이 적고 실험에서 다른 분류기보다 성능이 뛰어나기 때문에 SVM을 채택함. 

- 또한 SVM 모델은 정규화로 인한 높은 차원성으로 인해 과적합에 대한 저항력이 있는 것으로 잘 알려져 있음. 우리는 과적합을 최소화하기 위해 정규화의 하이퍼파라미터를 조정함.

## 한계점
- 노크 영역

- 항상 듣기

- 콘텐츠 양: Knocker는 객체의 특징적인 노크 응답을 분석하여 객체를 식별하기 때문에 콘텐츠 양의 변화와 같이 그 자체에 변화가 있을 수 있는 객체를 정확하게 식별하는 것이 어려울 수 있음. 내용물의 양이 변할 때(예를 들어, 약병에 든 알약의 양이 변할 때) 노크 반응이 약간 다르다는 것을 관찰함.

- 스마트폰에 손상

## 향후 연구
이 논문에서는 상용 스마트폰을 사용한 객체 식별의 기술적 측면에 초점을 맞추고 Knocker의 가능한 적용을 제안합니다. 향후 작업으로 식별 후 발생하는 사용자 상호 작용을 연구하는 것은 사용자가 실제로 일상 생활에서 Knocker를 어떻게 활용하는지에 대한 귀중한 통찰력을 추가할 것입니다. 예를 들어, 사람들이 다른 방법보다 Knocker를 사용하기를 선호하는 객체와 기능, 사용자가 동일한 모델의 두 객체 인스턴스를 구별하기를 원하는지 여부, 사용자가 훈련 과정에 대해 어떻게 느끼는지 등을 발견하는 것이 흥미로울 것입니다. 시각 장애인과 같은 특정 그룹의 사용자의 경우 시각 기반 방법이 카메라 정렬 문제로 어려움을 겪기 때문에 Knocker는 객체에서 정보를 검색하고 상호 작용할 수 있는 새로운 가능성을 열 수 있습니다.

