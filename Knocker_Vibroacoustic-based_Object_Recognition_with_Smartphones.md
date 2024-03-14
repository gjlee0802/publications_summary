# Knocker: Vibroacoustic-based Object Recognition with Smartphones

~~~
Knocker: Vibroacoustic-based Object Recognition with Smartphones
Taesik Gong, Hyunsung Cho, Bowon Lee, and Sung-Ju Lee
ACM Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT) 2019 (UbiComp '19).
~~~
https://nmsl.kaist.ac.kr/pdf/IMWUT19_Knocker.pdf

## Introduction

- 멀티모달 센서 데이터로부터 SVM(Support Vector Machine)을 활용하여 classification 수행함.


### Response Prunging(응답 가지치기) 프로세스

- Response Pruning: 일련의 원시 데이터에서 노이즈(예: 소리의 경우 주변 소음, 가속도계 및 자이로스코프의 경우 신체 움직임)를 배제하고 계산을 최소화하기 위해 노크 관련 응답만 추출해야 함. Response Pruning(응답 가지치기)라고 불리는 이 프로세스는 소리 응답을 가속도계 및 자이로스코프와 일치시킴.

- 정렬 과정에서 피크를 활용하여 노크가 시작될 때부터 소리에 대한 샘플 4,096개와 가속도계와 자이로스코프에 대한 샘플 32개를 추출. 23개의 물체를 대상으로 노크 응답의 지속 시간을 조사하는 실험 연구를 기반으로 이 값을 선택함. 지속 시간이 물체마다 다르며 대략 20ms에서 80ms 사이임을 발견했음.
- 내장 마이크의 경우 48kHz, 가속도계 및 자이로스코프의 경우 400Hz의 일반적인 샘플링 속도가 주어지면, 이는 소리의 경우 85ms, 가속도계 및 자이로스코프의 경우 80ms와 일치함. 이 설정은 노크 응답을 충분히 포착하고 노크와 관련 없는 데이터를 포함하여 계산 오버헤드를 최소화함.
- 
