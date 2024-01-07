[예제코드](https://github.com/wikibook/pytorchdl2)


[들어가기에 앞서](#들어가기에-앞서)  
[노트북 목록](#노트북-목록)

### 들어가기에 앞서
**Keras vs. Pytorch**
Pytorch의 장점
- 학습용 데이터 수집 등, 기능이 풍부하고 확장성이 높음
- 필요에 따라 독자적인 기능을 간단하게 개발할 수 있음

Pytorch의 단점
- 머신러닝 모델을 정의하는 방법 등이 다소 복잡하고 입문이 어려움.
- 케라스라면 함수(fit) 하나로 구현이 가능한 머신러닝 모델의 학습 처리를 직접 작성해야 함.

**목표**  
범용성이 높은 코드의 모든 행의 의미를 완전히 이해하고, 필요에 따라 수정까지 할 수 있을 것

## 책의 구성
### 기초편
1. 딥러닝에 필요한 파이썬 개념
    - Python 문법
    - 1.3 & 1.4
2. 파이토치의 기본 기능
    - Pytorch 문법
    - Tensor
    - Gradient
3. 처음 시작하는 머신러닝
    - linear regression
    - 경사 하강법 알고리즘 이해 목표
4. 예측 함수 정의하기
    - 예측 함수 정의하는 방법과 그 구조

### 머신러닝 실전편
5. 선형 회귀
6. 이진 분류
7. 다중 분류
    - **머신러닝 패턴(선형 회귀, 이진 분류, 다중 분류)에 따라 손실 함수가 어떻게 변하는지, 그 차이점 이해**
    - 손실 함수
8. MNIST를 활용한 숫자 인식
    - 수치 데이터 -> 이미지 데이터 사용
        - 늘어난 데이터 대응 방법
    - 은닉층 및 딥러닝 이해 첫걸음

### 이미지 인식 실전편
9. CNN을 활용한 이미지 인식
10. 튜닝 기법
11. 사전 학습 모델 활용하기
    - CIFAR-10 컬러 이미지 데이터셋 활용
    - **예측 함수의 내부 구조**
12. 사용자 정의 데이터를 활용한 이미지 분류
    - 직접 준비한 데이터 활용해 분류 모델 만들기 

<br>

# 내용 정리
|장|제목|
|---|---|
|1장|[딥러닝에 꼭 필요한 파이썬의 개념](ch1_python.ipynb)|
|2장|[파이토치의 기본 기능](ch2_pytorch_basic.ipynb)|

<br>

# 공식 노트북 목록

|장|제목|
|---|---|
|미리보기|[이미지 인식 시작하기](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/ch00_intro.ipynb)|
|1장|[딥러닝에 꼭 필요한 파이썬의 개념](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/ch01_python.ipynb)|
|2장|[파이토치의 기본 기능](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/ch02_pytorch.ipynb)|
|3장|[처음 시작하는 머신러닝](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/ch03_first_ml.ipynb)|
|4장|[예측 함수 정의하기](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/ch04_model_dev.ipynb)|
|5장|[선형 회귀](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/ch05_regression.ipynb)|
|6장|[이진 분류](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/ch06_bi_classifier.ipynb)|
|7장|[다중 분류](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/ch07_multi_classifier.ipynb)|
|8장|[MNIST를 활용한 숫자 인식](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/ch08_dl.ipynb)|
|9장|[CNN을 활용한 이미지 인식](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/ch09_cnn.ipynb)|
|10장|[튜닝 기법](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/ch10_dl_tuning.ipynb)|
|11장|[사전 학습 모델 활용하기](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/ch11_tr_learning.ipynb)|
|12장|[사용자 정의 데이터를 활용한 이미지 분류](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/ch12_custom_dl.ipynb)|
|부록1|[파이썬 입문](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/l01_python.ipynb)|
|부록2|[넘파이 입문](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/l02_numpy.ipynb)|
|부록3|[매트플롯립 입문](https://colab.research.google.com/github/wikibook/pytorchdl2/blob/master/notebooks/l03_matplotlib.ipynb)|

---


# 미리보기 & 이미지 인식 시작하기
**전이 학습(transfer learning)**  
    - 사전에 학습이 끝난 모델  
    - 수만 장의 훈련 데이터가 필요한 이미지 분류 모델이, 총 40장의 훈련 데이터만으로도 딥러닝 모델을 만드는 걸 예제 코드에서 확인할 수 있다.  
    - 적은 훈련 데이터로도 학습이 가능  
    - 학습에 소요되는 시간을 단축할 수 있다
