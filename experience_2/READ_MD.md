# test 1

CVaR 의 역할은 "최악의 경우에 대한 대비"인데, Index Tracking 을 강제하는 힘이 있을까?
만약 Index Tracking 을 수행하고 싶다면, CVaR의 구간을 거의 0.5 정도로 주어서
하위 50% 구간에서의 손실을 줄이는 식으로 최적화를 수행하는 것이
Index Tracking 의 철학과 맞지 않을까?

그래서 test 1은 CVaR에서의 alpha를 0.95로 주었을 때와 0.5로 주었을 때를 서로 비교한다.
이 경우, Index 보다 포트폴리오가 더 안좋은 경우에 대한 손실을 줄이는 방식(loss = (index - portfolio)^+)을 대조군으로 추가한다.

# test 2

support space 를 실수 전구간으로 잡으면 일부 최적화 변수에 대해 closed form 의 형태로 해를 얻어서 계산시간을 획기적으로 단축시킬 수 있는 것으로 보인다.

그래서 test 2 는 support space 가 해를 구하는데 얼마나 영향을 미치는지를 확인한다.
support space 를 rough 하게 잡았을 때, detail 하게 잡았을 때, 그리고 support space 가 실수 전구간일 때 세 경우를 서로 비교한다.

각각의 경우에 대해 CEIR, CLERI, Robust Index Tracking 세 가지로 비교한다.

## 실험 결과

세 경우 모두에서 case 1과 case 3이 거의 비슷한 결과를 주었다. case 2는 case 1과 case 3보다 아주 미세하게 더 좋은 포트폴리오를 만들었지만 유의미하다고 보기는 어려워 보인다.