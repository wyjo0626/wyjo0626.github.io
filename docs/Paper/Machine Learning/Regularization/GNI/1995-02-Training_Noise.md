---
slug: Training Noise
title: "Training with Noise is Equivalent to Tikhonov Regularization"
tags: [Regularisation, GNI, whiteout]
---

논문 및 이미지 출처 : <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-tikhonov-nc-95.pdf>

# Abstract

neural network training 중 input data 에 noise 를 추가하면 특정 상황에서 generalization performance 가 크게 향상될 수 있다는 사실은 잘 알려져 있다. 

이전 연구에서는 이러한 noise 를 활용한 training 이 error function 에 extra term 을 더하는 형태의 regularization 와 동등하다고 보여주었다. 

그러나 error function 의 second derivatives 를 포함하는 regularization term 은 bound 가 없기 때문에, error minimization 에 기반한 학습 알고리즘에서 직접 사용하면 어려움을 초래할 수 있다.

본 논문에서는 network training 의 목적으로, regularization term 을 network mapping 의 first derivatives 만 포함하는 positive definite 형태로 축소할 수 있음을 보인다. 

sum-of-squares error function 의 경우, regularization term 은 generalized Tikhonov regularization term class 에 속한다. 

regularized error function 를 직접 최소화하는 것은 noise 를 사용한 훈련에 대한 실용적인 대안을 제공한다.

# 1. Regularization

feed-forward neural network 은 $d$-dimensional input vector $\mathbf{x} = (x_1, \ldots, x_d)$ 를 $c$-dimensional output vector $\mathbf{y} = (y_1, \ldots, y_c)$ 로 mapping 하는 parameterized non-linear mapping 으로 간주될 수 있다.

neural network 의 supervised training 은 input vector $\mathbf{x}$ 와 이에 대응하는 target output vectors $\mathbf{t}$ 의 set 으로 정의된 error function 를 neural network 의 parameters 에 대해 최소화하는 과정을 포함한다. 

일반적인 error function 로는 다음과 같은 sum-of-squares error 가 있다:

$$
\begin{align}
    E &= \frac{1}{2} \int \int \|\mathbf{y}(\mathbf{x}) - \mathbf{t}\|^2 p(\mathbf{x}, \mathbf{t}) \, d\mathbf{x} \, d\mathbf{t} \\
    &= \frac{1}{2} \sum_k \int \int \{y_k(\mathbf{x}) - t_k\}^2 p(t_k | \mathbf{x}) p(\mathbf{x}) \, d\mathbf{x} \, dt_k
\end{align}
$$

- $\|\cdot\|$ : Euclidean distance
- $k$ : output units 
- function $p(\mathbf{x}, \mathbf{t})$ : input-target space 에서의 data 의 probability density
- $p(t_k | \mathbf{x})$ : $\mathbf{x}$ 가 주어진 $t_k$ 의 conditional density
- $p(\mathbf{x})$ : $\mathbf{x}$ 의 unconditional density
- Eq. (1) ~ (2) 으로 변환하기 위해 $t_{j \neq k}$ 변수에 대해 적분을 수행했다.

index $q$ 의 labelled samples $n$ 으로 구성된 finite discrete data set 의 경우, 다음을 얻는다:

$$
\begin{equation}
    p(\mathbf{x}, \mathbf{t}) = \frac{1}{n} \sum_q \delta(\mathbf{x} - \mathbf{x}^q) \delta(\mathbf{t} - \mathbf{t}^q)
\end{equation}
$$

Eq. (3) 을 사용하여 Eq. (1) 에 대입하면 다음과 같이 sum-of-squares error 를 얻을 수 있다:

$$
\begin{equation}
    E = \frac{1}{2n} \sum_q \|\mathbf{y}(\mathbf{x}^q) - \mathbf{t}^q\|^2
\end{equation}
$$

본 논문에서 얻은 결과는 large data set 의 극한에서 도출되므로, continuous probability density functions 표기법을 사용하는 것이 편리하다.

network training 의 중심 문제 중 하나는 model $y_k(\mathbf{x})$ 의 optimal complexity 를 결정하는 것이다. 

- 모델이 너무 제한적이면 데이터의 구조를 충분히 포착하지 못할 것이며, 반대로 너무 복잡하면 데이터의 noise 를 modeling 하게 된다(over-fitting). 
  - 이 경우 new data 에 대한 성능, 즉 network generalization 능력이 떨어진다. 
- 이 문제는 지나치게 유연하지 않은 모델의 high bias 와 자유도가 너무 높은 모델의 high variance 간의 optimal trade-off 를 찾는 문제로 간주될 수 있다.

모델의 bias 및 variance 를 제어하는 데 잘 알려진 두 가지 기술은 **structural stabilization** 와 **regularization** 이다.

- **structural stabilization** : degrees of freedom 를 제어하기 위해 free parameters 수를 조정하는 방법이다.
  - feed-forward network 의 경우, 이는 일반적으로 hidden units 수를 조정하거나 처음에 oversized network 에서 individual weights 를 pruning 하는 방식으로 이루어진다.
- **regularization** : 상대적으로 유연한 모델을 사용한 후, penalty term $\Omega(\mathbf{y})$ 를 error function 에 추가하여 variance 를 제어하는 방법이다. 
  - 이로 인해 total error function 는 다음과 같이 변경된다:

$$
\begin{equation}
    \tilde{E} = E + \lambda \Omega(\mathbf{y})
\end{equation}
$$

- $\lambda$ : $\Omega(\mathbf{y})$ 가 minimizing function $y(\mathbf{x})$ 에 미치는 영향을 조정하여 bias-variance trade-off 를 제어한다. 
- regularization functional $\Omega(\mathbf{y})$ 는 일반적으로 network function $y(\mathbf{x})$ 와 그 derivatives 로 표현되며, desired network mapping 에 대한 prior knowledge 를 바탕으로 선택된다. 
  - 예로 매핑이 매끄러워야 한다는 것이 알려진 경우, $\Omega(\mathbf{y})$ 는 large curvature 의 functions 에 대해 큰 값을 가지도록 선택될 수 있다.

Regularization 은 $y(\mathbf{x})$ 의 linear models 맥락에서 광범위하게 연구되었다. 

하나의 input variable $x$ 와 하나의 output variable $y$ 가 있는 경우, Tikhonov regularization term 의 class 는 다음과 같은 형태를 취한다:

$$
\begin{equation}
    \Omega(y) = \sum_{r=0}^R \int_a^b h_r(x) \left( \frac{d^r y}{dx^r} \right)^2 dx
\end{equation}
$$

- $h_r \geq 0 \; (r = 0, \ldots, R-1)$
- $h_R > 0$

이러한 regularization term 의 경우, regularized error $\tilde{E}$ 를 최소화하는 linear function $y(x)$ 는 유일하다는 것이 증명될 수 있다.

# 2. Training with Noise

bias 와 variance 간의 trade-off 를 제어하는 세 번째 접근법은 training 중 input data 에 random noise 를 추가하는 것이다.

- 이는 일반적으로 각 input pattern 에 random vector 를 추가하는 방식으로 이루어진다. 
- pattern 이 반복적으로 사용될 경우, 매번 다른 random vector 가 추가된다. 
- 직관적으로, noise 는 각 data point 를 'smear out(흩뿌려)' network 가 individual data points 를 정확히 적합시키기 어렵게 만든다. 
- 실제로, noise 를 활용한 training 이 network 의 generalization performance 을 향상시킬 수 있음이 실험적으로 입증되었다.

이제 noise training 과 regularization 간의 관계를 자세히 탐구한다.

input vector 에 추가된 noise 를 random vector $\boldsymbol{\xi}$ 로 나타내자. 

noise 를 사용한 training 의 error function 는 다음과 같이 쓸 수 있다:

$$
\begin{equation}
    \tilde{E} = \frac{1}{2} \int \int \int \sum_k \{ y_k(\mathbf{x} + \boldsymbol{\xi}) - t_k \}^2 p(t_k | \mathbf{x}) p(\mathbf{x}) \tilde{p}(\boldsymbol{\xi}) \, d\mathbf{x} \, dt_k \, d\boldsymbol{\xi}
\end{equation}
$$

- $\tilde{p}(\boldsymbol{\xi})$ : noise 의 distribution function

이제 noise amplitude (진폭) 이 작다고 가정하고, network function 을 $\boldsymbol{\xi}$ 의 Taylor powers 로 전개한다:

$$
\begin{equation}
    y_k(\mathbf{x} + \boldsymbol{\xi}) = y_k(\mathbf{x}) + \sum_i \xi_i \frac{\partial y_k}{\partial x_i} \bigg|_{\boldsymbol{\xi} = 0} + \frac{1}{2} \sum_i \sum_j \xi_i \xi_j \frac{\partial^2 y_k}{\partial x_i \partial x_j} \bigg|_{\boldsymbol{\xi} = 0} + O(\boldsymbol{\xi}^3)
\end{equation}
$$

noise distribution 은 일반적으로 zero mean 이고 different inputs 간 상관관계가 없도록 선택된다. 

따라서 다음을 만족한다:

$$
\begin{equation}
    \int \xi_i \tilde{p}(\boldsymbol{\xi}) \, d\boldsymbol{\xi} = 0, \quad \int \xi_i \xi_j \tilde{p}(\boldsymbol{\xi}) \, d\boldsymbol{\xi} = \eta^2 \delta_{ij}
\end{equation}
$$

- $\eta^2$ : noise 의 진폭을 제어하는 parameter

Taylor series expandion Eq. (8) 을 error function Eq. (7) 에 대입하고, Eq. (9) 를 활용해 noise distribution 에 대해 적분하면 다음과 같은 결과를 얻는다:

$$
\begin{equation}
    \tilde{E} = E + \eta^2 E^R
\end{equation}
$$

- $E$ : Eq. (2) 에서 정의된 standard sum-of-squares error

extra term $E^R$ 은 다음과 같이 주어진다:

$$
\begin{equation}
    E^R = \frac{1}{2} \int \int \sum_k \sum_i \left\{ \left( \frac{\partial y_k}{\partial x_i} \right)^2 + \frac{1}{2} \{ y_k(\mathbf{x}) - t_k \} \frac{\partial^2 y_k}{\partial x_i^2} \right\} p(t_k | \mathbf{x}) p(\mathbf{x}) \, d\mathbf{x} \, dt_k
\end{equation}
$$

이는 일반적인 sum-of-squares error 에 regularization term 이 추가된 형태이며, regularization term 의 계수는 noise variance $\eta^2$ 에 의해 결정된다. 이 결과는 Webb (1993) 에 의해 이전에 도출된 바 있다.

---

noise amplitude 이 작아 Taylor expansion 에서 higher order 를 무시하는 것이 타당하다면, input data 에 noise 를 추가한 상태에서 sum-of-squares error 를 최소화하는 것은, noise 를 추가하지 않고 Eq. (11) 에서 주어진 regularization term 을 포함한 regularized sum-of-squares error 를 최소화하는 것과 동등하다. 

하지만, regularization fuctinon Eq. (11) 의 second term 은 network function 의 second derivatives 를 포함하므로, network weights 에 대한 error gradients 를 계산하는 데 계산 비용이 많이 든다. 또한, 이 term 은 positive definite 가 아니므로 error function 이 **a priori**로 bound 됨을 보장되지 않으며, 따라서 학습 알고리즘의 기반으로 사용하기에 적합하지 않다.

이제 network function $y(\mathbf{x})$ 에 대해 regularized error Eq. (10) 를 최소화하는 것을 고려한다. 

저자의 주요 결과는, noise amplitude 이 작은 경우, network training 에 Eq. (11) 의 regularization function 을 사용하는 것이 standard Tikhonov 형태의 positive definite regularization function 사용과 동등하며, 이 함수는 network function 의 first derivatives 만 포함한다는 것을 보여주는 것이다.

먼저, target data 의 conditional averages 를 다음과 같이 정의한다:

$$
\begin{align}
    \langle t_k | \mathbf{x} \rangle \equiv \int t_k p(t_k | \mathbf{x}) \, dt_k \\
    \langle t_k^2 | \mathbf{x} \rangle \equiv \int t_k^2 p(t_k | \mathbf{x}) \, dt_k
\end{align}
$$

simple algebra 변환을 통해, Eq. (2) 의 sum-of-squares error function 를 다음과 같은 형태로 쓸 수 있다:

$$
\begin{equation}
    \begin{align*}
        E &= \frac{1}{2} \sum_k \int \int \{ y_k(\mathbf{x}) - \langle t_k | \mathbf{x} \rangle \}^2 p(t_k | \mathbf{x}) p(\mathbf{x}) \, d\mathbf{x} \, dt_k \\
        &+ \frac{1}{2} \sum_k \int \int \{ \langle t_k^2 | \mathbf{x} \rangle - \langle t_k | \mathbf{x} \rangle^2 \} p(t_k | \mathbf{x}) p(\mathbf{x}) \, d\mathbf{x} \, dt_k
    \end{align*}
\end{equation}
$$

여기서 Eq. (14) 의 first term 만이 network mapping $y_k(\mathbf{x})$ 에 의존한다. 

따라서 error function 의 minimum 은 network mapping 이 target data 의 conditional averages 으로 주어질 때 발생한다:

$$
\begin{equation}
    y_k^{\text{min}}(\mathbf{x}) = \langle t_k | \mathbf{x} \rangle
\end{equation}
$$

이는 least-squares solution 이 target data 의 conditional averages 으로 주어진다는 잘 알려진 결과를 나타낸다. 

- interpolation problems 에서는 network 가 target data 의 intrinsic additive noise (training 중 input data 에 추가된 noise 와는 다름) 를 평균화하여 데이터의 근본적인 경향을 학습한다는 것을 보여준다. 
- 또한, target data 가 1-of-N coding scheme 임을 사용하는 classification 문제의 경우, 이 결과는 network output 이 클래스 소속의 Bayesian posterior probabilities 로 해석될 수 있음을 나타내며, 따라서 최적의 결과로 간주될 수 있다.
- Eq. (15) 는 error function 의 global minimum 을 나타내며, network model 이 편향되지 않을 정도로 기능적으로 충분히 풍부하다고 간주되어야 한다. 
- 하지만, 이 minimum 에도 error function 은 0이 되지 않으며, 이는 Eq. (14) 의 second term 에 의해 주어진 잔여 오류가 존재하기 때문이다. 
- 이 잔여 오류는 conditional averages 주변의 target data 의 평균 분산을 나타낸다.

regularized error function Eq. (10) 에 대해 minimizing function 은 다음과 같은 형태를 가진다:

$$
\begin{equation}
    y_k^{\text{min}}(\mathbf{x}) = \langle t_k | \mathbf{x} \rangle + \mathcal{O}(\eta^2)
\end{equation}
$$

이제 network function 의 second derivatives 에 의존하는 Eq. (11) 의 second term 을 고려한다. 

Eq. (12) 의 정의를 사용하여 이 항을 다음과 같이 다시 쓸 수 있다:

$$
\begin{equation}
    \frac{1}{4} \int \int \sum_k \sum_i \left\{ \{y_k(\mathbf{x}) - \langle t_k | \mathbf{x} \rangle\} \frac{\partial^2 y_k}{\partial x_i^2} \right\} p(t_k | \mathbf{x}) p(\mathbf{x}) \, d\mathbf{x} \, dt_k
\end{equation}
$$

- Eq. (16) 을 사용하면, order $\eta^2$ 에서 이 항은 total error function 의 minimum 에서 소멸함을 알 수 있다.
  - 따라서 Eq. (11) 의 first term 만 유지하면 된다. 
- 이 결과는 target data 평균에 의한 결과라는 점을 강조해야 한다. 이는 individual terms $y_k - t_k$ 가 작아야 한다는 것을 요구하지 않으며, 오직 $t_k$ 에 대한 conditional averages 이 작아야 한다는 것을 요구한다.
- 따라서 noise 를 포함한 sum-of-squares error 의 minimizing 은 order $\eta^2$ 에서, Eq. (14) 의 first term 에 의해 주어진 regularization term 을 포함하는 regularized sum-of-squares error 를 noise 없이 minimizing 함과 동등하다. 
- 이 regularization term 은 다음과 같은 형태를 가진다:

$$
\begin{equation}
    \hat{E}^R = \frac{1}{2} \int \sum_k \sum_i \left( \frac{\partial y_k}{\partial x_i} \right)^2 p(\mathbf{x}) \, d\mathbf{x}
\end{equation}
$$

- 여기서 $t_k$ variables 는 적분되었다. 
- Eq. (18) 의 regularization function 은 일반적으로 Eq. (11) 에서 주어진 것과 동일하지 않다. 
- 그러나 두 경우에서 total regularized error 는 동일한 network function $y(\mathbf{x})$ (그리고 동일한 network weight values set) 에 의해 최소화된다. 
- 따라서 network training 의 목적을 위해, Eq. (11) 의 regularization term 을 Eq. (18) 의 regularization term 으로 대체할 수 있다.

discrete data set 에서, Eq. (3) 으로 주어진 probability distribution function 을 사용할 경우, 이 regularization term 은 다음과 같이 쓸 수 있다:

$$
\begin{equation}
    \hat{E}_R = \frac{1}{2n} \sum_q \sum_k \sum_i \left( \frac{\partial y_k^q}{\partial x_i^q} \right)^2
\end{equation}
$$

저자의 분석에서 neural network 에 특화된 것은 없다는 점에 유의하자. 

feed-forward network 의 이점은, $y(\mathbf{x})$ 함수의 parameterized non-linear model 로서, 상대적으로 유연하여 Eq. (16) 에 의해 주어진 optimal solution 을 잘 근사할 수 있다는 점이다.

비슷한 분석을 cross-entropy error function 에 대해 적용할 수 있다. 

cross-entropy error function 는 다음과 같이 주어진다:

$$
\begin{equation}
    E = - \int \int \sum_k \left\{ t_k \ln y_k(\mathbf{x}) + (1 - t_k) \ln(1 - y_k(\mathbf{x})) \right\} p(t_k | \mathbf{x}) p(\mathbf{x}) \, d\mathbf{x} \, dt_k
\end{equation}
$$

Taylor expansion Eq. (8) 을 이전과 동일하게 사용하면 다시 다음과 같은 형태의 regularized error function 에 도달한다:

$$
\begin{equation}
    \tilde{E} = E + \eta^2 E^R
\end{equation}
$$

여기서 regularization term 은 다음과 같이 주어진다:

$$
\begin{equation}
    \begin{align*}
        E_R = &\frac{1}{2} \int \int \sum_k \sum_i \left\{ 
        \left[\frac{1}{y_k (1 - y_k)} - \frac{(y_k - t_k)(1 - 2y_k)}{y_k^2 (1 - y_k)^2}\right] 
        \left( \frac{\partial y_k}{\partial x_i} \right)^2 \right. \\
        &+ \left.\left[ \frac{(y_k - t_k)}{y_k (1 - y_k)} \right] 
        \frac{\partial^2 y_k}{\partial x_i^2}\right\}p(t_k | \mathbf{x}) p(\mathbf{x}) \, d\mathbf{x} \, dt_k
    \end{align*}
\end{equation}
$$

이 식은 network mapping function 의 second derivatives 를 포함하며, positive definite 가 아닌 항들을 포함하고 있다. 

Eq. (20) 에서, regularized error 를 최소화하는 network function 가 다시 Eq. (16) 에 주어진 형태를 가진다는 것이 도출된다. 

이 결과를 사용하고, sum-of-squares error 에 대해 제시된 것과 유사한 논리를 따르면, (22) 의 second 및 third term 이 소멸함을 알 수 있다. 

따라서, 이 regularization function 은 다음과 같이 단순화된다:

$$
\begin{equation}
    E^R = \frac{1}{2} \int \sum_k \sum_i \left\{ 
    \frac{1}{y_k (1 - y_k)} 
    \left( \frac{\partial y_k}{\partial x_i} \right)^2 
    \right\} 
    p(\mathbf{x}) \, d\mathbf{x}
\end{equation}
$$

이제 이 함수는 positive definite 이며, first derivatives 만 포함한다. 

그러나, 이는 Eq. (6) 에서 주어진 standard Tikhonov 형태와는 다르다는 점에 유의해야 한다. 

discrete data set 의 경우 (Eq. (3) 으로 기술됨), 이 regularization term 은 다음과 같이 쓸 수 있다:

$$
\begin{equation}
    E^R = \frac{1}{2} \sum_q \sum_k \sum_i \left\{ 
    \frac{1}{y_k^q (1 - y_k^q)} 
    \left( \frac{\partial y_k^q}{\partial x_i^q} \right)^2 
    \right\}
\end{equation}
$$

Bishop (1993) 에 의해 기술된 바와 같이, Eq. (19) 또는 Eq. (24) 같은 regularization function 의 derivatives 를 feed-forward network weights 에 대해 효율적으로 계산하는 기술은 standard back-propagation 기법의 확장을 기반으로 한다. 

이러한 derivatives 는 gradient descent 또는 conjugate gradients 같은 표준 학습 알고리즘의 기반으로 사용할 수 있다. 

따라서, noise 를 활용한 훈련의 대안으로 regularized error function 를 직접 최소화할 수 있다.

# 3. Perturbative Solution

저자의 분석에서는 noise amplitude 이 작다고 가정한다. 

이는 regularized error function 를 minimizing 하는 neural network weights set 에 대해, regularization 없이 sum-of-squares error function 를 minimizing 하여 얻은 weights 를 기반으로 하는 perturbative solution 을 찾을 수 있게 해준다. 

unregularization error function 이 weight vector $\mathbf{w}^*$ 에 의해 최소화된다고 가정하자:

$$
\begin{equation}
    \frac{\partial E}{\partial w_n} \bigg|_{\mathbf{w}^*} = 0
\end{equation}
$$

regularized error function 의 minimum 을 $\mathbf{w}^* + \Delta \mathbf{w}$ 로 쓸 경우, 다음이 성립한다:

$$
\begin{equation}
    0 = \frac{\partial (E + \eta^2 E^R)}{\partial w_n} \bigg|_{\mathbf{w}^* + \Delta \mathbf{w}} 
    = \frac{\partial E}{\partial w_n} \bigg|_{\mathbf{w}^*} 
    + \sum_m \Delta w_m \frac{\partial^2 E}{\partial w_n \partial w_m} \bigg|_{\mathbf{w}^*} 
    + \eta^2 \frac{\partial E^R}{\partial w_n} \bigg|_{\mathbf{w}^*}
\end{equation}
$$

discrete regularization term 이 Eq. (19) 로 주어졌다고 가정하고, Eq. (25) 를 사용하면, weight values 에 대한 보정 $\Delta \mathbf{w}$ 의 명시적 표현은 다음과 같다:

$$
\begin{equation}
    \Delta \mathbf{w} = -\eta^2 H^{-1} \sum_q \sum_k \sum_i \nabla_\mathbf{w} \left( \frac{\partial y_k}{\partial x_i^q} \right)^2
\end{equation}
$$

여기서 $H$ 은 Hessian matrix 로, 각 요소는 다음과 같이 정의된다:

$$
\begin{equation}
    (H)_{nm} = \frac{\partial^2 E}{\partial w_n \partial w_m}
\end{equation}
$$

cross-entropy error function 의 경우에도 유사한 결과를 얻을 수 있다. 임의의 feed-forward 구조를 가진 network 에 대해 Hessian matrix 를 효율적으로 계산하는 정확한 절차는 **Bishop (1992)**에서 제시되었다. 

또한, Eq. (27) 의 우변에 나타나는 형태로 weights 에 대한 derivative terms 를 평가하기 위한 확장된 back-propagation 알고리즘은 **Bishop (1993)**에서 유도되었다.

Eq. (11) 에서 second derivative term 을 생략할 수 있다는 사실은, Eq. (27) 에서 network function 의 second derivative 만 나타난다는 것을 의미하며, 이를 실질적으로 구현할 수 있음을 나타낸다. 

third derivatives 가 포함될 경우, weight 보정을 평가하는 과정이 매우 번거로워질 것이다.

# 4. Summary

저자는 bias 및 variance 간의 trade-off 을 제어하는 세 가지 접근법을 고려했다:

1. sum-of-squares error function(sum-of-squares error function) 를 최소화하고, training 중 input data 에 noise 를 추가한다.
2. input data 에 noise 를 추가하지 않고, regularization term 이 Eq. (19) 로 주어진 regularized sum-of-squares error function 를 직접 최소화한다.
3. input data 에 noise 를 추가하지 않고 sum-of-squares error 를 최소화한 다음, Eq. (27) 을 사용하여 network weight 에 대한 보정을 계산한다.  
(cross-entropy error function 에 대해서도 유사한 결과를 얻을 수 있다.)

noise amplitude parameter $\eta^2$ 가 작을 경우, 이 세 가지 방법은 동등하다.