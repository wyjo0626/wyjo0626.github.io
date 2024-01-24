---
slug: UniPELT
title: "UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning"
tags: [PEFT, PELT, unified framework, incorporated peft, adapter, prefix-tuning, lora]
---

논문 및 이미지 출처 : <https://aclanthology.org/2022.acl-long.433.pdf>

# Abstract

최근 parameter-efficient language model tuning (PELT) 으로 적은 trainable parameter 로 fine-tuning 성능과 맞먹거나 적은 training data 에서 성능이 좋은 방법 등장

하지만 다양한 PELT methods 는 동일한 task 에 상당히 다른 성능을 보이며, 특정 task 에 대해 적절한 method 를 선택하는 것은 쉽지 않다.

특히 new PELT methods 외 task 가 늘어나고 있는 상황을 고려하면 더 어렵다.

모델의 다양성과 선택의 어려움을 고려하여, 저자는 **UNIPELT** 를 제안

- 다양한 PELT 방법을 submodeuls 로 포함하고, 현재 data 나 task setup 에 적합한 방법들을 활성화시키기 위한 **gating mechanism** 을 통해 학습
- GLUE 에서 UNIPELT 는 일관성 있게 개별 PELT 보다 1~4% 성능 향상, 다양한 설정에서 fine-tuning 능가
- 일반적으로 submodules 를 개별 사용했을 때의 각 task 에 대한 best performance 를 취합산 상한선을 넘어서는 경향이 있어, 여러 PELT mixture 가 single 보다 효과적임을 시사

# 1. Introduction

pre-trained language models (PLMs) 는 커져감에 따라, task 당 model parameter 개별적 복제를 수정하는 fine-tuning 은 점점 불가능해지고 있다.

이를 해결하기 위해, 최근 **p**arameter-**e**fficient **l**anguage model **t**uning (PELT) 연구가 급증하고 있다.

기존 PELT 는 few trainable parameter 로 fine-tuning 과 comparable perforance 를 목표로 한다.

최근 방법의 parameter 는 PLM 의 total parameter 에 비해 무시할 수 있을 수준 (< 1%) 으로 줄었다.

더 어려우며 덜 연구된 문제는 fewer parameter 로 fine-tuning 보다 좋은 성능을 달성하는지 여부다.

최근은 training data 가 제한된 경우 PELT 가 overfitting 의 위험을 줄여 특정 task 에서 fine-tuning 보다 효과적임을 입증했다. 하지만 저자의 실험에서 발견한 것과 같이 (Table 1), 다양한 PELT 는 동일한 task 에 다양한 특성을 보이며, 특히 새로운 PELT 와 task 가 늘어나고 있어 가장 적합한 방법을 선택하기란 어렵다.

---

PLET 간의 성능 차이와 최적의 방법을 선택하는 비용을 고려하여, 통합 PELT 인 UNIPELT 제안

- 다양한 PELT 를 submodule 로 통합하고, data or task 에 가장 적합한 submodule 또는 그 조합을 동적으로 활성화하여 학습
- 모델 선택이 필요하지 않으며, 다양한 설정에서 일관된 성능 향상을 보여줌
- 각 submodule 의 activation 은 _gating mechanism_ 으로 제어되며, 주어진 task 에 긍정적으로 기여하는 submodule 을 학습 (가중치를 더욱 할당)
- 각 submodule 의 parameter 수가 일반적으로 적어, 여러 방법을 결합해도 효율성 유지

---

저자는 네 가지 PELT 를 택하였다.

- Adapter (Houlsby et al. 2019)
- Prefix-Tuning (Li and Liang. 2021)
- LoRA (Hu et al. 2021)
- BitFit (Ben Zaken et al. 2021)

저자는 _PELT 개별 특성_과 여러 설정의 UNIPELT 가 조화를 이룰 때의 효과성을 검토하는 두 가지 분석 셋 수행

- GLUE 에서의 실험 결과 (32 setup - 8 tasks x 4 data sizes, 1,000+ runs) PELT 의 다양한 행동을 보여주며, 각 방법의 개별 사용보다 UNIPELT 다 효과적이며 견고함을 보여줌
- 일관되게 best submodule 을 1~4 points 개선하며, 때론 fine-tuning을 능가하여 최상의 평균 성능 달성
- 일반적으로 각 task 에서 개별적으로 사용된 모든 submodule 의 최고 성능 초과
  - 아는 다양한 설정에서 최적의 성능을 유지한다는 것을 시사
- 상한선을 능가한다는 사실은 PLM architecture 의 다양한 부분을 포함하는 PELT 방법이 mixture 이 single 보다 내재적으로 더 효과적임을 나타냄

**Contributions**

1. PELT 에 대한 포괄적인 연구를 수행하고 성능 및 특성 측면에서 차이와 공통점을 검토
2. 기존 PELT 를 submodule 로 포함하고 주어진 task 에 적합한 submodule 을 자동으로 활성화할 수 있는 통합 PELT 제안
3. UNIPELT 는 다양한 설정에서 fine-tuning 및 PELT 보다 우수한 평균 성능 달성, 종종 best 성능을 발휘하고 결코 worst 성능은 되지 않으며 모델 효율성도 좋다. 

# 2. Preliminaries

## 2.1 PELT Methods without Additional Parameters

PLM 은 top layers 또는 prediction head 만 additional parameter 없이 fine-tuning 될 수 있다.

하지만 fine-tuning 은 일반적으로 all parameter tuning 보다 훨씬 나쁜 성능을 내는 경우가 많다.

최근 BitFir 은 PLM 의 bias 만 tuning 하여, limited training data 에서 일부 task 에서 fine-tuning 과 유사한 성능을 달성할 수 있는 것을 실험적으로 보여줬다.

따라서 저자는 이 카테고리를 대표하는 분석을 위해 BitFir 을 선택했다.

## 2.2 PELT Methods with Additional Parameters

대안으로 all PLM 을 고정하고 new trainable parameter 를 도입할 수 있다.

이 카테고리의 예시로 Adapter, Prefix-Tuning 등이 있다.

#### Adapter

PLM 의 각 Transformer layer 의 feedforward network 뒤에 trainable _bottleneck layer_ 를 추가

bottleneck layer 은 token hidden states size 를 축소하고 복구하는 down+up projection pair 으로 구성된다.

수학적으로, 

- feedforward network 후 residual connection 과 layer normalization 을 거친 output : $h_{FN}$
- hidden size : $D_{\text{hidden}}$
- bottleneck size : $D_{\text{mid}}$
- bottleneck layer $h_A$ 의 output 은 다음과 같다.

$$
\begin{equation}
    h_A = W^\top_\text{up} \phi (W^\top_\text{down}h_{FN}),
\end{equation}
$$

- $W_{\text{down}} \in \mathbb{R}^{D_\text{hidden} \times D_{\text{mid}}}$, $W_{\text{up}} \in \mathbb{R}^{D_\text{mid} \times D_{\text{hidden}}}$, $\phi$ : nonlinear activation function. 간결성을 위해 bias 는 제거
- layer normalization 및 final prediction head parameter 도 특정 adapter 에 따라 fine-tuning

Adapter 는 fine-tining 과 비슷한 성능이거나 low-resource 에서 더 효과적인 것으로 나타났다.

#### Prefix-Tuning

Prefix-Tuning 은 각 transformer layer 의 multi-head attention 의 input 앞에 task-specific trainable vectors 를 붙인다.

구체적으로, 

- original sequence length : $L_0$
- trainable vectors number (i.e. prefix length) : $L$
- Transformer layer input : $h_\text{in} \in \mathbb{R}^{D_{\text{hidden}} \times L_0}$

1. three linear projections $W_Q$, $W_K$, $W_V \in \mathbb{R}^{D_{\text{hidden}} \times D_{\text{hidden}}}$ 는 $h_\text{in}$ 을 Query $Q$, Key $K$ 및 Value $V$ 로 변환
2. two prefix matrices $P_K$, $P_V \in \mathbb{R}^{D_{\text{hidden}} \times L}$ 는 $K$ 및 $V$ 앞에 붙임
3. optimization 안정화를 위해, prefix matrix $P$ 는 feedforward network 를 통해 reparameterize

$$
\begin{equation}
  P' = W^\top_\text{up} \phi (W^\top_\text{down}P),
\end{equation}
$$

- $W_\text{down} \in \mathbb{R}^{D_{\text{hidden}} \times D_{\text{mid}}}$, $W_\text{up} \in \mathbb{R}^{D_{\text{mid}} \times 2N_\text{layer}D_{\text{hidden}}}$
  - $N_\text{layer}$ : Transformer layers 수
  - 위 두 network 는 training 후 폐기할 수 있으며, $2N_{\text{layer}}$ prefix matrices $\in \mathbb{R}^{D_\text{hidden} \times L}$ 만 필요

Prefix-tuning 은 원래 NLG 에서 평과되며, 저자는 이를 적용함.

prompt-tuning (Lester et al. 2021) 이란 후속 방법은 prefix 를 첫 번째 layer 로 제한하여 task-specific parameters 를 더 줄이지만, 모델이 큰 경우에만 competitive performance 유지

#### Additive Methods

Additive PELT 는 fine-tuning 후의 model parameter 를 pre-trained parameters $\theta_\text{pre-trained}$ 와 task-specific differences $\delta_\text{task}$ 의 합으로 취급

- $\theta_\text{pre-trained}$ fixed
- new (sub)set : $\theta_\text{task} = \theta_\text{pre-trained} + \delta_\text{task}$ 추가
- $\delta_\text{task}$ 를 parameterize 하는 다양한 방법 : LoRA, diff pruning 및 side-tuning 존재

저자는 LoRA 를 대표적인 방법으로 채택하여 이를 UNIPELT 에 통합

- LoRA 는 trainable low-rank matrices 도입 및 이를 multi-head attention 의 original matrices 에 결합
- 특히, two matrices $W_\text{down} \in \mathbb{R}^{D_\text{hidden} \times D_\text{mid}}$ 및 $W_\text{up} \in \mathbb{R}^{D_\text{mid} \times D_\text{hidden}}$ 은 original matrix $W_Q$ 및 $W_K \in \mathbb{R}^{D_\text{hidden} \times D_\text{hidden}}$ 인 query 및 key projection 에 추가된다:

$$
\begin{equation}
  Q = (W^\top_Q + \alpha W^\top_\text{up}W^\top_\text{down})h_\text{in}
\end{equation}
$$

- $\alpha$ : task-specific differences 를 조절하기 위한 fixed scalar hyperparameter
- LoRA 의 trainable matrices 형태는 Adapter 나 Prefix-Tuning 과 유사하지만, 이들은 activation function $\phi$ 가 없음

# 3. Unifying PELT Methods

## 3.1 Task Formulation
