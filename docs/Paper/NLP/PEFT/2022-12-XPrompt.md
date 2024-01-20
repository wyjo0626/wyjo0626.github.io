---
slug: XPrompt
title: "XPrompt: Exploring the Extreme of Prompt Tuning"
tags: [PEFT, prompt tuning, LTH, lottery ticket hypothesis]
---

논문 및 이미지 출처 : <https://aclanthology.org/2022.emnlp-main.758.pdf>

# Abstract

Prompt tuning 은 frozen Pre-trained Language Models (PLMs) 를 conditioning 하기 위해 soft prompts 를 학습한다.

모델 규모가 커짐에 따라 prompt tuning 은 점차 fine-tuning 수준에 도달하지만, moderate 및 small scales (< 11B) 에선 여전히 성능 차이가 발생한다.

본 논문에서 저자는 trained prompt tokens 는 downstream task 에 negative 영향을 줄 수 있으며 성능 저하를 일으킬 것이라는 것을 경험적으로 보여준다.

- gap 을 줄이기 위해, 저자는 lottery tickets hypothesis 하에, **Prompt** tuning model with an e**X**tremely small scale (**XPrompt**) 를 제안
- 구체적으로, XPrompt 는 hierarchical structured pruning 을 통해 다양한 granularity levels 에서 negative prompt tokens 를 제거하여 더욱 parameter-efficient prompt 를 생성하여 competitive 성능 달성
- SuperGLUE task 에서 포괄적 실험으로, smaller model scales 에서 성능 gap 을 줄여줌

# 1. Introduction

PLMs 는 _pretrain-then-finetune_ 을 통해 널리 사용되어 큰 성공을 거두지만, memory 공간에 gradient 및 optimizer 저장을 위해 trainable parameter 가 크게 차지하고 있어 fine-tuning 이 parameter-inefficient 하다.

---

![Figure 1](image-151.png)

최근 Prompt-Tuning (Lester et al. 2021) 으로 input 에 _soft prompt_ 를 앞에 붙이고 훈련 중 prompt parameter 만 업데이트하여 위 이슈를 해결하는 것을 제안하였다.

- fine-tuning 대체제로, soft prompt scale 은 수만배 적음
- 더 간단하고 다른 peft (Adapter) 보다 유연하여 transformer layers 에 직관적으로 수정 가능
- 적은 tunable parameter 로 fine-tuning 성능과 competitive

---

위 gap 을 채우기 위해, 본 논문은 lottery tickets hypothesis (LTH) 관점에서 작성한다.

특정 task 에서 all prompt tokens 이 task 성능에 동등하게 기여하지 않는 관찰에 동기를 받아, 특정 prompt tokens 은 때론 negative 영향을 미칠 수 있다는 것이다.

![Figure 2](image-152.png)

Fig. 2 에서 관찰 결과를 보여준다.

- _negative prompt tokens_ 는 LTH 로 피할 수 있다.
  - LTH 는 sub-network 를 포함한 over-parameterized network 가 독립적으로 훈련 및 초기화되면 original network 의 정확도와 맞먹거나 능가
  - 이 sub-network 를 **Lottery Ticket** 이라 하며, PLMs 에서 이러한 ticket set 을 **winning tickets** 이라 한다.
- prompt tuning 에서 저자는 전체 prompt 사용의 성능과 동일하게 달성할 수 있는 **positive prompt tokens** 을 **winning tickets** 으로, **negative prompt tokens** 는 **losing tickets** 로 참조
  - 그래서 핵심은 winning tickets 은 식별하고 losing tickets 은 제거하는 것
- hierachical structed pruning 을 통해 losing tickets 을 제거하는 것 제안
  - **token-level** 에서 negative tokens 을 제거하고 **granularity level (i.e. piece-level)** 에서 남은 것들을 pruning
- LTH 와 일치하도록, 식별된 positive soft prompts 를 재훈련하기 위해 **weight rewinding** 채택

위 과정으로 negative prompt tokens 이 제거되어 parameter-efficient small scale prompt (XPrompt) 를 얻을 수 있다.

---

XPrompt 의 효과성 검증을 위해, high-resource 및 low-resource 상황의 SuperGLUE 에서 실험 진행

Fig. 1 및 Table. 1 에서 모든 task 및 model scale 에서 prompt-tuning 의 향상을 볼 수 있다.

- moderate scale 의 모델의 경우, XPrompt 로 fine-tuning 과 comparable 한 성능 달성 및 gap 줄임
- large scale 의 모델의 경우, XPrompt 가 Prompt-Tuning 을 넘은 성능을 얻었고, 대부분의 task 에서 fine-tuning 도 넘어섰다.

# 2. Related Work

## 2.1 Pre-trained Language Models

PLMs 는 NLP task 에서 큰 성공을 거두었다. 

- BERT 및 RoBERTa 는 masked language model (MLM) 으로 context representation 을 학습하는 것을 개척
- GPT-2, GPT-3 , ELECTRA, XLNet, BART 및 T5 같은 large PLMs 도 생겨남

하지만 parameter 수가 폭발적으로 커지며, fine-tuning model 은 parameter-inefficient 및 computationally expensive 하게 됨

게다가 다양한 task 에 대해 fine-tuning 하고 각각 저장하기까지 해야 한다.

## 2.2 Prompt Learning in NLP

GPT-3 의 개발과 함께, input 에 여러 _prompt tokens_ 를 추가하여 효율적인 학습을 하는 prompt tuning 이 관심을 받고 있다.

이는 다양한 downstream task 에서 효과적임을 입증했다.

- 최근 discrete tokens (token in the vocabularies) 에서 continuous tokens (trainable embedding) 으로 확장
  - 예로 (Lester et al. 2021), soft prompt 만 tuning 하고 PLMs 는 freezing 하는 효율적인 prompt tuning 제안
  - 하지만 여전히 moderate scale 에서는 fine-tuning 과의 gap 이 존재
- 더 최근 (Vu et al. 2021) prompt-based transfer learning 인 SPoT 은 source task 에 prompt 를 학습하여 target task prompt 에 초기화하여 적용해 성능을 향상시킴
- 가장 최근 (He et al. 2022) HyperPrompt 는 hyper-prompts 를 생성하기 위해 hypernetwork 를 사용하여 우수한 성능 얻음

위는 all parameter 를 조정해야 하며, task-conditioned parameter 만 튜닝하는 것이 multi-task learning 에 대한 fine-tuning 과 competitive 결과를 얻는데 충분치 않다는 것을 보여줌

## 2.3 Lottery Ticket Hypothesis

lottery ticket hypothesis 는 over-parameterized network 는, 초기화되어 독립적으로 학습하면 기존 network 의 정확도와 일치하거나 능가할 수 있는 subnetwork 를 가진 다는 것을 발견

- 이 subnetwork 를 **lottery ticket** 이라 함
- NLP 에서의 lottery ticket set 은 winning ticket 이라 함
- 이러한 winning ticket 은 task 및 dataset 간의 transerability 를 입증
- 최근 Chen et al. (2021) 에서 PLM 이 lottery ticket 의 존재를 보여줌
- Liang et al. (2021) 에선 winning ticket 의 일반화 성능이 full model 을 능가할 수 있음을 관찰

# 3. Preliminary

T5 의 text-to-text 기반으로 한 prompt tuning 은 all task 를 text generation 으로 고려하며 additional $l$ tunable soft prompt token 을 input 에 추가하고, inserted soft prompt token 의 parameter 만 업데이트 수행

구체적으로, $n$ 개의 input token $X = \{ x_1, x_2, \dots, x_n\}$ 이 있을 때, T5 는 먼저 token embeddings $X_e \in \mathbb{R}^{n \times e}$ 을 생성한다.

- $e$ : embedding space dimension

soft prompt embedding $P_e = \{ p_1, p_2, \dots, p_m \} \in \mathbb{R}^{m \times e}$ 생성

- $m$ : soft prompt length 

이후 soft prompts 는 $[P_e; X_e] \in \mathbb{R}^{(m+n) \times e}$ 형태로 input sequence 앞에 추가

prompt tuning 목표는 $P_e$ 를 optimizing 하여 label $Y$ 의 likelihood 를 최대화 하는 것

$$
\begin{equation}
    \underset{P_e}{\arg \max} \log p(Y|[P_e; X_e])
\end{equation}
$$

model scale 이 커짐에 따라 prompt tuning 은 더욱 효과적으로 작동한다.

하지만 small 및 moderate scale 에 대해서는 fine-tuning 과 성능 gap 이 존재한다.

저자의 가설은 target task 에 훈련 후 all soft prompt tokens 가 동등하게 성능에 기여하지 않을 것이라는 것이다.

특정 soft prompt tokens 은 task 에 negative impacts 를 줄 수 있다. 따라서 lottery ticket hypothesis 의 아이디어를 결합하여 저자는 XPrompt 를 제안한다.

이는 hierarchical structured pruning 을 사용하여 optimal soft prompts 를 식별하고 성능 차이를 줄인다.

# 4. XPrompt

![Figure 3](image-153.png)

Fig. 3 은 XPrompt 은 전체 과정이며 세 가지 main stages 를 포함한다.

- **_Prompt-Tuning_**
- **_Hierarchical Structured Pruning_**
- **_Rewinding_**

## 4.1 Prompt Tuning

input 에 soft prompt tokens 을 붙여 PLM 은 fixing 한 채 soft prompt 만 tuning

Prompt tuning 은 다양한 downstream task 에 효과적이다.

저자의 prompt tuning 은 Liang et al. (2021) 에 따르며, all soft prompt tokens 에 대한 embeddings 을 얻기 위해 target task 에 완전히 tuning

위의 trained soft prompts 는 hierarchical structured pruning 에 초기화되어 사용됨
