# hw2

## sec1

1. Deliverables for report:

   • Create two graphs:

   - –  In the first graph, compare the learning curves (average return at each iteration) for the experiments prefixed with q1_sb_. (The small batch experiments.)

     ![q1_sb](/Users/charlie/Desktop/截屏2022-05-14 上午10.34.48.png)

   - –  In the second graph, compare the learning curves for the experiments prefixed with q1_lb_. (The

     large batch experiments.)

     ![q1_lb](/Users/charlie/Desktop/截屏2022-05-14 上午10.59.05.png)

      • Answer the following questions briefly:

   - –  Which value estimator has better performance without advantage-standardization: the trajectory- centric one, or the one using reward-to-go?
   - –  Did advantage standardization help? Yes,but indistinctive.
   - –  Did the batch size make an impact?

## sec2

* Large batch size:stable, low variance.

* Small learning rate: converges quicker but unstable

  ![sec2](/Users/charlie/Desktop/截屏2022-05-14 下午12.45.00.png)

## sec3

* maxreturn is over 200

  ![average return](/Users/charlie/Desktop/截屏2022-05-14 下午6.47.50.png)

![average return](/Users/charlie/Desktop/截屏2022-05-14 下午6.48.02.png)

*****

(lec7)要不知道mdp，要不在一个状态用同个策略做好多次（来模拟出状态转移概率）。

而利用q-function，不依赖于策略，只依赖于mdp,在任何策略下采样都可以(模拟mdp)，不知道mdp也无妨

## sec4

![searching for opt b and lr](/Users/charlie/Desktop/截屏2022-05-15 上午9.48.41.png)

![s](/Users/charlie/Desktop/截屏2022-05-15 上午9.48.56.png)

b<30000> lr<0.02>提升很快，方差极大。最终最好的是b<50000>lr<0.02>

rtg nn-baseline对照

![对照](/Users/charlie/Desktop/截屏2022-05-16 下午6.59.08.png)

******

## sec5

![sec5](/Users/charlie/Desktop/截屏2022-05-17 上午9.24.50.png)

![sec5](/Users/charlie/Desktop/截屏2022-05-17 上午9.24.30.png)

