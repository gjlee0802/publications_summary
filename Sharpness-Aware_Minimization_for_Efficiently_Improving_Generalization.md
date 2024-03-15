# Sharpness-Aware Minimization for Efficiently Improving Generalization

~~~
Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. In International Conference on Learning
Representations, 2021.
~~~
https://arxiv.org/pdf/2010.01412.pdf  

## Introduction

Loss landscape가 flat할수록 generalizability가 강해짐.  
Flatness of minima -> the flatter, the more generalizable

SAM은 loss value와 loss sharpness를 동시적으로 minimizing을 하여 flatter region의 파라미터들을 찾을 수 있게 되며, 찾은 파라미터 근방의 파라미터들도 비슷한 낮은 loss를 갖게 됨.  

generalization ability 검증 방법
- 밑바닥부터 학습했을 때 generalization
- Finetuning 단계에 적용했을 때 generalization

## Relaed Work : Flat Minima 연구
- 1995년부터 연구가 진행됨: flat minima search -> 더 나은 generalization을 위해 loss sharpness를 penalizing하는 방법 연구
- Regularize local entropy(i.e. Entropy-SGD)
- Average weights during training(i.e. SWA)
- 기존 연구에서의 한계 : sharpness의 측정이 계산하기에 어려웠음(ex. weights의 Hessain 계산).

## SAM
Our goal : population loss를 가장 최소화하는 모델 파라미터 w를 찾는 것!  

![SAM_generalization_bound1](/SAM_img/SAM_generalization_bound1.PNG)  

**PAC(probably approximately correct) Bayesian generalization bound** 기반으로 generalization bound 산정을 하여 population loss에 대한 upper bound를 정함.  

![SAM_generalization_bound2](/SAM_img/SAM_generalization_bound2.PNG)  

h는 strictly 증가 함수

![SAM_generalization_bound3](/SAM_img/SAM_generalization_bound3.PNG)  

대괄호 항: S에 대해서 계산되는 training loss가 w 근처에서 얼마나 빠르게 변동(증가)를 하는가를 의미 -> 곧 sharpness를 의미함.  

![SAM_generalization_bound4](/SAM_img/SAM_generalization_bound4.PNG)  

대괄호 항 뒤의 항: w에 대한 training loss와 regularization  
h term을 일반적인 L2 정규화(regularization)로 계산

![SAM_solving_problem](/SAM_img/SAM_solving_problem.PNG)

**결과적으로 위의 Sharpness-aware minimization problem을 해결하면 됨.**  
where ρ ≥ 0 is a hyperparameter and p ∈ [1, ∞] (경험적으로 ρ = 2로 하여 L2-norm 계산한 것이 최적의 결과를 냄)  

![SAM_approximation](/SAM_img/SAM_approximation.PNG)

max에 대한 gradient를 구해야함으로 approximation을 진행함
-> 단순하게 first order Taylor 전개를 진행함.  

![SAM_ehat_solving_problem](/SAM_img/SAM_ehat_solving_problem.PNG)

여기서 두번째 항의 hessian 계산은 속도를 늦추며 성능을 낮추므로 Drop함(계산에서 제외).  

![SAM_final_gradient_approximation](/SAM_img/SAM_final_gradient_approximation.PNG)

최종적인 구하고자 하는 gradient 전개는 위와 같음

## 성능 비교
- SVHN과 Fasion-MNIST에서도 성능 개선을 보임
- ImageNet에서도 SAM을 적용했을 때 더 높은 성능을 보였으며 과적합이 덜 발생하면서 training을 진행할 수 있었음.

## 성능 실험

### 두번째 항(drop한 항)을 계산했을 경우와 성능 비교
놀랍게도 두번째 항을 버리는 것이 더 좋은 성능을 냈음. -> Cosine 유사도를 계산했을 때 중반 이후에 더 좋은 유사도를 보임.  

### p차 norm 계산 비교(p 값을 바꿔감)
p차 norm 계산에서 p=2일 때 정확도가 더 올라감.  

## m-sharpness
- 배치마다 SAM 업데이트를 사용하는 방법
- GPU accelerate 마다 SAM 업데이트를 계산하여 GPU의 각 계산된 SAM 업데이트의 평균을 구하여 모델을 업데이트하는 방법 (각 accelerator는 m 사이즈의 subset을 사용함) -> m이 작을수록 더 좋은 generalization ability를 보여줌.

## 참고 자료
~~~
https://m0nads.wordpress.com/2021/04/10/sharpness-aware-minimization/
https://github.com/reyllama/paper-reviews/issues/30
https://creamnuts.github.io/short%20review/sam/
https://deepseow.tistory.com/20
[UNIST 유튜브] https://youtu.be/lcNjbOHf0uo?si=dA16_4d4c_ARwZLt
[딥러닝논문읽기모임 유튜브] https://youtu.be/iC3Y85W5tmM?si=YcO3mehbB-3_a2Yc
~~~

## 구현 코드
https://github.com/davda54/sam