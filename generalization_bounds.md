# Generalization bounds에 대해 이해하기 위한 내용 정리

## - KL divergence(KLD)
~~~
참고 자료
https://hwiyong.tistory.com/408
~~~

![KLD](generalization_bounds_img/KLD_formula.png)

KLD(이하 KL-Divergence)는 P 분포와 Q 분포가 얼마나 다른지를 측정하는 방법.  
여기서 통계적으로 P는 사후, Q는 사전분포를 의미함.  

텐서플로우 공식 문서에 정의되어있는 용어로 설명해보면, KLD는 y_true(P)가 가지는 분포값과 y_pred(Q)가 가지는 분포값이 얼마나 다른지를 확인하는 방법.

KLD는 값이 낮을수록 두 분포가 유사하다라고 해석함.  
정보이론에서 흔히 볼 수 있는 엔트로피(Entropy) 또한, 값이 낮을수록 랜덤성이 낮다고 해석하는 것과 비슷함.  
두 가지의 해석 방법이 비슷한 것은 바로 KLD에 크로스-엔트로피(Cross-Entropy) 개념이 이미 포함되어 있기 때문임.  

### KLD와 Cross Entropy 관계

정보이론에서 정보량은 효과적으로 표현하기 위해 로그를 사용하여 표현함.  
우리가 아는 엔트로피는 평균 정보량을 나타내므로 아래와 같이 표현됨.  

![Entropy](generalization_bounds_img/entropy_formula.png)

KLD와 어떤 관련성이 있을까?  

아래와 같이 생각해보자.  
~~~
p : 실제 세계에서 관찰하여 얻어낸 확률 ; 실제 확률분포 P
q : 모델이 예측한 확률 ; 확률분포 P로 근사될 분포 Q
~~~

앞에서 소개했던 KLD 식을 다시 생각해보면 log의 성질에 의해 왼쪽의 항처럼 분해하여 생각할 수 있음.  

![KLD_and_CrossEntropy](generalization_bounds_img/KLD_and_CrossEntropy.png)

이는 아래와 같이 해석됨.  
~~~
KL-Divergence = Cross-Entropy - Entropy
~~~

결과적으로 모델이 예측한 확률분포(Q)의 정보량과 실제 확률분포(P) 정보량의 차이를 의미함.  
**이에 대한 차이(정보량)를 분포가 유사한지에 대한 정도로 다시 해석**할 수 있음.  

### KLD와 가우시안 분포
~~~
참고 자료
https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
https://simpling.tistory.com/33
~~~

두 개의 서로 다른 Gaussian 분포를 가정했을 때 KL-divergence(Kullback–Leibler divergence, KLD)를 구하는 유도과정에 대해 알아봄.  

아래처럼 유도과정을 정리함.  
<img src="https://github.com/gjlee0802/publications_summary/assets/49184890/5396369b-a39e-4da0-852d-40e9c8f50bd5" width="400" height="600"/>  
<img src="https://github.com/gjlee0802/publications_summary/assets/49184890/a7529796-ce90-4ec7-94ce-14d7962a8b82" width="400" height="600"/>  
<img src="https://github.com/gjlee0802/publications_summary/assets/49184890/92d229eb-a94c-455a-b373-638afb05db4e" width="400" height="600"/>  
<img src="https://github.com/gjlee0802/publications_summary/assets/49184890/d74067dc-bed2-416c-b821-08e72afa2a0c" width="400" height="600"/>  
<img src="https://github.com/gjlee0802/publications_summary/assets/49184890/fcd1eb7f-fa80-4f31-9909-5865d202e831" width="400" height="600"/>  
<img src="https://github.com/gjlee0802/publications_summary/assets/49184890/d2bb8d4b-363b-4fd8-8000-495836aa6432" width="400" height="600"/>  

## - PAC Beyesian Generalization Bound
~~~
User-friendly introduction to PAC-Bayes bounds: https://arxiv.org/pdf/2110.11216.pdf
~~~
### Sharpness-Aware Minimization에서의 PAC generalization bound 활용
![SAM_PAC_generalization_bound1](generalization_bounds_img/SAM_PAC_generalization_bound1.PNG)

사후 표준 편차 `σQ`가 주어지면 위의 KL divergence를 최소화(최대한 두 분포가 유사하게)하기 위해 사전 표준 편차 `σP`를 선택할 수 있으며, KL을 최소화하는 값을 설정하기 위해 `σP`에 대한 위의 KL의 도함수를 취하고 0으로 설정하여 일반화 경계를 설정할 수 있으나  

위의 방법은 옳지 않음.  

왜냐하면 훈련 데이터 `S`와 `µQ`를 관찰하기 전에 `σP`를 선택해야 하므로 `σQ`는 `S`에 의존할 수 있으므로 이러한 방식으로 `σP`를 최적화하는 것은 허용되지 않음.  

따라서, 위의 방법 대신에 `σP`에 대해 미리 정의된 값 세트를 가지고 그 세트에서 가장 좋은 값을 선택하는 방법을 선택함.  

~~~
이 테크닉에 대한 논문
John Langford and Rich Caruana. (not) bounding the true error. NeurlIPs 2002
https://papers.nips.cc/paper_files/paper/2001/file/98c7242894844ecd6ec94af67ac8247d-Paper.pdf
~~~

위의 KL과 log 항에 대해 각각 upper bound를 유도하고 둘을 이용하여 위에서 소개된 generalization upper bound를 설정하게 됨.  

![SAM_KL_upper_bound](/generalization_bounds_img/SAM_KL_upper_bound.PNG)

![SAM_log_term_upper_bound](/generalization_bounds_img/SAM_log_term_upper_bound.PNG)

![SAM_PAC_generalization_bound2](generalization_bounds_img/SAM_PAC_generalization_bound2.PNG)
