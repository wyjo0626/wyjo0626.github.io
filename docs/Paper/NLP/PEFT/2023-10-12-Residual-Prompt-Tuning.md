---
slug: Residual Prompt Tuning
title: "Residual Prompt Tuning: Improving Prompt Tuning with Residual Reparameterization"
tags: [PEFT, prompt tuning, residual prompt tuning, residual]
---

논문 및 이미지 출처: <https://arxiv.org/pdf/2305.03937v1.pdf>

# Abstract

Prompt Tuning 은 성공적인 PEFT 접근법 중 하나지만 (tuned soft prompts < 0.1% of total parameters) 로 인해 다른 PEFT 보다 성능이 안좋거나 hyper-parameter 에 민감

본 논문에서 저자는 **RESIDUAL PROMPT TUNING** 을 제안

- 간단하고 성능 및 prompt tuning 의 안정성을 크게 향상시키는 efficient method
- shallow network with a residual connection 을 사용하여 soft prompt embeddings 를 reparameterize

Residual Prompt Tuning 은 T5-Large, T5-Base 및 BERT-Base 전역에 SuperGLUE 에서 prompt tuning 능가

Notably,

- T5-Base 에서 prompt tuning +7 points 도달
- 성능 손실 없이 prompt length X10 줄임

저자의 접근법은 learning rate 및 prompt initialization 선택에 있어 robust 하며 few-shot settings 에 효과적

# 1. Introduction

Pre-trained language models (PLMs) 는 NLU task 에 큰 성능을 도달하고, scaling up 으로 성능을 더욱 향상시키고 있다. (e.g. GPT-3 175B, MT-NLG 530B)

large-scale models 의 성능은 향상하지만 사이즈 때문에 응용에 있어 제한이 걸린다.

_fine-tuning_ 은 all model parameters 의 gradients 및 optimizer states 를 저장해야 하므로 비용이 만만치 않다. 또한 각 task 에 대한 fine-tuned model copy 도 있어야 하므로 billion-parameter models 로 현실적으로 어렵다. 

full fine-tuning 해결을 위해 _prompt design_ 에 초점이 맞추어져 왔다. 이는 frozen model 에 natural language prompts 를 쿼리한다.

이 설정에선 all tasks 를 language modeling tasks 로 변환(e.g. 0/1 classes 를 "True"/"False" 로 인코딩할 수 있다.)하고 원하는 output 을 생성하도록 수동으로 선택한 prompts 를 frozen model 에 조건화시킨다.

few-shot 성능에 강하지만, 최적의 prompt 를 수동으로 찾는 것은 어렵고 시간소비가 크며 서로 다른 prompts 는 최종 성능에 큰 분산을 일으킨다.

---

최근 _prompt tuning_ 이 제안되었으며, 수동 prompt designing 대신 _soft prompts_ 를 gradient descent 로 학습시킨다.

soft prompts 는 input 앞에 붙여지는 continuous embedding 으로, training 으로 업데이트되며 일반적으로 < 0.1% of total parameter 만 구성된다.

prompt tuning 은 모델이 클수록 full fine-tuning 에 가까운 성능을 보이며 11B 이상에선 차이가 거의 없다.

하지만 smaller model 에선 여전히 성능이 낮으며, 성능이 hyperparameter 선택에 의존된다. 게다가 안정적인 성능을 위해 long training 과 large tokens (over 100)이 요구된다.

이는 prompt 가 순차 학습되는 continual learning setup 이거나 context length 가 제한된 경우 주요 bottleneck 이 될 수 있다.

![Figure 1](image-100.png)

본 논문에선 prompt embeddings 의 residual reparameterization 을 통해 prompt tuning 을 향상시키고 안정적이게 한다. 이를 **_RESIDUAL PROMPT TUNING_** 이라 한다.

- shallow network 및 residual connection 으로 soft prompt embedding 을 통과시킴
- 그 후, reparameterized prompt 를 input 앞에 붙이고 LM 에 feed
  - 이 prompt 는 모델이 각 prompt token 에 대한 개별 embedding 을 사용하는 것과 shared reparameterization network 에서 얻은 representation 사이를 결정하는데 더 많은 유연성을 제공
- 훈련 후에는, reparameterizatione network 는 버리고, original prompt embeddings 는 projections 로 대체할 수 있다.

T5-Large, T5-Base 및 BERT-Base 로 SuperGLUE task 에서 실험 진행하여, T5-Base 에서 이전 prompt-tuning +7 points 를 달성하였다.

또한 Residual Prompt Tuning 은 다양한 learning rate/prompt initializations 에서 성능 분산을 줄였으며, fewer training iterations 에서도 강한 성능을 만들었다.

마지막으로, Residual Prompt Tuning 은 few-shot settings 에서도 prompt tuning 을 능가했다

# 2. Background

### Fine-tuning

downstream task 에 PLMs 를 adapting 하는 일반적인 approach 는 all parameters Θ 를 fine-tune

classification task $T$, input text $x$ 및 output scalar label $y$ 에서, $p_Θ$ 는 full model weights Θ 에 의해 parameterize 된 output classes 의 확률 분포

$$
\begin{equation}
    \underset{Θ}{\max} \sum_{x,y \in T} \log p_Θ (y|x).
\end{equation}
$$

all model parameter 업데이트로 LLM 에 대한 비용은 매우 크다.

### Prompt Tuning

prompt tuning 은 fine-tuning 의 경량화된 대안으로 제안되었다.

주요 아이디어는 virtual token embeddings 의 sequence 또는 _soft prompt_ $P$ 를 input text $x$ 앞에 붙이고 model parameter 는 fix 한채 이들만 학습시키는 것이다.

model parameters Θ 는 frozen PLMs 와 additional soft prompt parameter $\theta_P$ 가 결합되게 된다.

$$
\begin{equation}
    \underset{\theta_P}{\max} \sum_{x,y \in T} \log p_Θ (y| [P; x]).
\end{equation}
$$

Prompt tuning 은 많은 real-world 응용에 PLMs 를 parameter-efficient solution 을 제공하지만, soft prompts training 은 hyperparameter tuning 및 원하는 성능 도달을 위한 loger traning time 이 요구된다.

# 3 Method

## 3.1 RESIDUAL PROMPT TUNING

저자는 shallow network with a skip connection 으로 soft prompt 에 flexible reparameterization 사용을 제안.

특히 $n$ 개의 virtual tokens $[P_1, \dots , P_n]$ 로 구성된 prompt embeddings $P$ 의 sequence 를 reparameterized sequence $P'$ 로 project

$$
\begin{equation}
    P' = [P'_1, \dots , P'_n] = [\Phi(P_1), \dots , \Phi(P_n)],
\end{equation}
$$

- $\Phi(\cdot)$ : residual connection 한 shallow network $\phi(\cdot)$ 로 구성된 reparameterization function
- $\Phi(\cdot)$ 은 각 prompt token 에 독립적으로 적용됨

$$
\begin{equation}
    \Phi(P_i) = \phi(P_i) + P_i \ , i \in \{1\dots n \}
\end{equation}
$$

![Figure 2](image-101.png)

- 저자의 $\phi(\cdot)$ 는 일반적으로 사용되는 ResNet 및 adapter modules 를 따른 MLP 이다
- down-projection $W_{\text{down}} \in \mathbb{R}^{d\times m}$ 및 up-projection $W_{\text{up}} \in \mathbb{R}^{m \times d}$ layers 로 구성 (Figure 2)
  - 이 결합은 (ResNet 및 adapter modules) 에서 이미 연구됨
  - $d$ : model embeddings 의 차원
  - $m$ : MLP 의 bottleneck size
- downstream task 에 prompt embeddings $\theta_P$ 및 reparameterization parameters $\theta_{\phi}$ 만 훈련하며 다른 parameters 는 freezing
- training objective 는 reparameterized soft prompt $P'$ 가 붙여진 input text $x$ 에 대한 correct output $y$ 의 log-likelihood 최대화

$$
\begin{equation}
    \underset{\theta_P, \theta_\phi}{\max} \sum_{x, y \in T} \log p_\Theta (y|[P';x]).
\end{equation}
$$

## 3.2 Design choices

### Residual connection

저자는 residual connection 이  RESIDUAL PROMPT TUNIONG 에 있어 성능 증가 및 수렴 속도 증가에 있어 중요한 역할을 하는 것을 발견 ([Section 5.1](#51-main-results))

저자는 residual learning 이 모델에게 각 prompt token 에 대한 개별 embedding 을 사용하는 것과 shared reparameterization network 에서 얻은 representation 사이를 결정하는데 더 많은 유연성을 준다고 가정한다.

residual connection 의 이점은 [Appendix B.2](#b2-covergence-of-different-prompt-tuning-approaches) 에서 다룬다.

### Depth and width of MLP

two-layer MLP, up- 및 down-projection matrics $W_{\text{up}}$ 및 $W_{\text{down}}$ 은 additional trainable parameter 로 구성된다.

hidden layer 의 차원 $m$ 의 증가는 높은 성능을 보여주며([Section 5.6](#56-ablation-studies)), prompt token 의 overparameterization 은 성능 향상에 중요하다.

자세한 parameter-efficiency 는 [Appendix A.6](#a6-parameter-efficiency-of-residual-prompt-tuning)

### Non-linearity and normalization

저자는 normalization layer 로 LayerNorm, non-linearity 로는 ReLU 를 선택

LayerNorm 은 성능 안정성에 도움이되며 non-linear layer 의 특정 선택의 효과는 덜 중요한 것을 발견

### Parameter sharing

저자는 각 virtual token embedding 에 shared reparameterization network $\Phi$ 를 적용했다.

다른 특정 설계는 각 prompt embedding 에 개별 network 를 적용했다.

두 가지를 비교한 결과 ([Section 5.6](#56-ablation-studies)), shared MLP 가 더 parameter-efficient 및 제한된 데이터에 대한 knowledge sharing 의 이점 제공을 가져왔다.

## 3.3 Training and Inference

training 중, 백본 모델은 freezing 하며 prompt embeddings $P$ 및 reparameterization network $\Phi(\cdot)$ 의 parameter 를 공동으로 최적화한다.

reparameterized prompt 는 input text embeddings 전에 삽입되고 LM 에 feed 된다.

저자는 task-specific prompts 를 사용했으며, 이는 reparameterized prompt embeddings 가 input 에 의존하지 않는다는 것을 의미한다.

---

훈련 후, learned reparameterized network $\Phi(\cdot)$ 으로 prompt embeddings 를 project 하고 original prompt embeddings 를 상응하는 projections $P' = \Phi(P)$ 로 교체한다.

inference 중에는, **reparameterization network 는 버리고** projected prompt embeddings $P'$ 를 사용한다.

input text embeddings 앞에 $P'$ 를 삽입하고 frozen PLMs 에 함께 feed 한다.

# 4. Experiments

## 4.1 Datasets

SuperGLUE 벤치마크인 NLU tasks 를 사용하며 8가지 datasets 을 사용한다.

BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC

## 4.2 Architectures

encoder-decoder T5 model 및 encoder-only BERT 모델에서의 성능을 실험하며, BERT-Base (110M), T5-Base (220M) 및 T5-Large(770M) 에 초점을 둔다.

## 4.3 Baselines



# 5. Results

# 5.1 Main Results

# 5.6 Ablation studies

# Appendix

## A. Implementation and Training

## A.6 Parameter-efficiency of RESIDUAL PROMPT TUNING

## B. Performance on SuperGLUE

### B.2. Covergence of different prompt tuning approaches

