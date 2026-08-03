# Experience 1

min max CVaR 의 formulation 으로 Index Tracking 을 수행하는 모델을 만들고 실험한다.
대조군으로는 Robust 가 없는 버전을 사용한다.
* CEIR (CVaR Enhanced Index Replication)
* CLEIR (CVaR-LASSO Enhanced Index Replication)

Robustness 는 다음과 같이 두 가지 방식으로 formulation 할 수 있다.
* Wasserstein distance
* Kernel distributiona estimation(KDE)

KDE 의 경우, 어떤 kernel 을 사용할 것인가와 어떤 distance 를 이용해 KDE에서의 weight 를 제한할 것인가에 따라 여러 가지 버전을 고려해볼 수 있다.
이 실험에서는 두 가지의 kernel과 세 가지의 distance 를 사용하여 실험을 진행했다.

Kernel
* Gaussian kernel
* Epanechnikov kernel

Distance
* KL-divergence
* Hellinger distance
* Wasserstein distance

각각의 경우에 대해 공매도를 제한한 버전(no short-sale), 공매도를 허용하는 대신 LASSO 제약조건을 추가한 버전 두 가지를 고려할 수 있다. 
따라서 가능한 조합은
* Robust 없는 버전 2개
* Wasserstein 을 이용한 버전 x 2개
* Kernel(2가지) x Distance(3가지) x 2개 = 12개

총 16개 버전의 모델 조합을 얻을 수 있다.

# Experience 2

## test 1

CVaR 의 역할은 "최악의 경우에 대한 대비"인데, Index Tracking 을 강제하는 힘이 있을까?
만약 Index Tracking 을 수행하고 싶다면, CVaR의 구간을 거의 0.5 정도로 주어서
하위 50% 구간에서의 손실을 줄이는 식으로 최적화를 수행하는 것이
Index Tracking 의 철학과 맞지 않을까?

그래서 test 1은 CVaR에서의 alpha를 0.95로 주었을 때와 0.5로 주었을 때를 서로 비교한다.
이 경우, Index 보다 포트폴리오가 더 안좋은 경우에 대한 손실을 줄이는 방식(loss = (index - portfolio)^+)을 대조군으로 추가한다.

### 실험 결과

0.95를 사용하는 경우에서, 0.5를 사용한 경우와 대조군으로 사용한 경우들보다 살짝 enhanced 된 결과를 얻었다.
그리고 0.5를 사용하는 경우, 대조군으로 사용한 경우와 거의 일치하는 결과를 보여준다.
이는 사실상 두 경우가 추구하는 것이 거의 동일하기 때문에 충분히 납득 가능한 결과라고 할 수 있다.

## test 2

support space 를 실수 전구간으로 잡으면 일부 최적화 변수에 대해 closed form 의 형태로 해를 얻어서 계산시간을 획기적으로 단축시킬 수 있는 것으로 보인다.

그래서 test 2 는 support space 가 해를 구하는데 얼마나 영향을 미치는지를 확인한다.
support space 를 rough 하게 잡았을 때, detail 하게 잡았을 때, 그리고 support space 가 실수 전구간일 때 세 경우를 서로 비교한다.

각각의 경우에 대해 CEIR, CLERI, Robust Index Tracking 세 가지로 비교한다.

### 실험 결과

세 경우 모두에서 case 1과 case 3이 거의 비슷한 결과를 주었다. case 2는 case 1과 case 3보다 아주 미세하게 더 좋은 포트폴리오를 만들었지만 유의미하다고 보기는 어려워 보인다.

case 2 에서, support space 가 너무 디테일하게 되면 굉장히 공격적인 포트폴리오가 만들어졌다. 
support space 가 작다는 것은 결국 constraint 가 더 느슨해졌다고 이해할 수 있는데, 이로 인해 굉장히 극단적인 선택을 하게 되는 것으로 보인다.

그래서 일부 상황에서는 case 2 가 다른 포트폴리오보다 더 좋은 수익률을 보여주긴 했지만, 안정성이 크게 떨어졌고 오히려 좋지 못한 수익률을 보여주는 경우가 더 많았다. 또한 case 2 에서, 하나의 주식에 몰빵하는 모습도 자주 포착되었다.

case 2 의 경우, margin 을 어느 정도 확보하여 최소한의 support space 를 확보하면 어느 정도 안정화가 되는 모습을 보였다. 
이 경우, case 1 과 case 3 의 결과와 유사한 결과를 얻었다.