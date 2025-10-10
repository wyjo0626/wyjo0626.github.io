---
slug: Constitutional AI
title: "Constitutional AI: Harmlessness from AI Feedback"
tags: [Reinforce Learning, RL, RLAIF, AI Feedback, Constitutional AI, CAI, RL-CAI, Chain-of-thought Reasoning, ]
---

논문 및 이미지 출처 : <https://arxiv.org/pdf/2212.08073>

# Abstract 

AI system 이 점점 더 강력해짐에 따라, 다른 AI 를 감독하는 데 그들의 도움을 활용하고자 하는 요구가 커지고 있다. 본 연구에서는 harmful output 을 식별하는 인간의 label 없이, self-improvement 를 통해 harmless AI assistant 를 학습시키는 방법을 실험한다. 인간의 개입은 일련의 규칙 또는 원칙 목록 형태로만 이루어지며, 따라서 이 방법을 **Constitutional AI** 라고 한다.

이 과정은 **supervised learning** 단계와 **reinforcement learning** 단계로 구성된다.

* **Supervised learning** 단계에서는 initial model 로부터 sample 을 생성하고, self-critique 및 revision 을 수행한 뒤, 수정된 응답에 대해 original model 을 finetune 한다.
* **Reinforcement learning** 단계에서는 finetuned model 로부터 sample 을 생성하고, two samples 중 어느 것이 더 나은지를 model 이 평가하여 AI preference dataset 을 구축한다. 이후 preference model 을 reward signal 로 사용하여 RL 을 수행하는데, 이를 **RL from AI Feedback (RLAIF)** 라고 한다.

그 결과, 저자는 harmless 하면서 nonevasive AI assistant 를 학습시킬 수 있었으며, 이 assistant 는 harmful queries 에 대해서도 그 질의에 대한 이의를 설명하는 방식으로 응답한다.

또한 supervised learning 과 reinforcement learning 모두 **chain-of-thought style reasoning** 을 활용하여 인간의 판단 기준에서의 성능 및 AI 의 의사결정 투명성을 향상시킬 수 있다. 이러한 방법들은 human label 수를 크게 줄이면서도 AI 의 행동을 보다 정밀하게 제어할 수 있게 한다.

# 1 Introduction

AI system 이 인간 수준의 성능에 도달하거나 이를 초과하는 능력을 갖추게 되더라도, 여전히 helpful, honest, harmless 한 상태를 유지하도록 학습시키는 것이 목표이다. 이를 위해서는 인간이 AI 행동의 모든 측면을 감독하지 않고도 작동할 수 있는 기술, 즉 유해한 행동에 대한 robustness 를 자동으로 평가하고 강화할 수 있는 기법이 필요하다. 또한 AI 의 바람직한 행동을 단순하고 투명한 형태로 인코딩하여, AI 의 의사결정 과정을 이해하고 평가하기 쉽게 만드는 방법을 개발하는 것이 목표이다.

![Figure 1](image.png)

본 논문에서는 Fig. 1 에 나타난 바와 같이 **Constitutional AI (CAI)** 라 불리는 방법을 제안하고, 이를 사용하여 human feedback label 없이도 회피적이지 않고 비교적 harmless AI assistant 를 학습시킨다. 이 방법은 기존의 reinforcement learning from human feedback (RLHF) 를 개선하고 부분적으로 대체한다. 새롭게 학습된 assistant 인 **RL-CAI** 는 crowdworker 들의 평가에서 이전의 human feedback label 기반 모델들보다 선호되었다.

Constitutional 이라는 용어를 사용한 이유는, 저자가 소수의 원칙이나 지침, 즉 constitution 만으로도 덜 hamful system 을 학습시킬 수 있기 때문이다. 또한 일반적인 AI system 을 개발하고 배포할 때, 그 행동을 규율할 원칙의 집합을 선택하지 않을 수 없다는 점을 강조하기 위함이기도 하다.

이 기술을 개발한 주요 동기는 다음과 같다.

1. AI system 이 다른 AI 를 감독하도록 하여 supervision 을 확장하는 단순한 가능성을 탐구하기 위함
2. 이전의 harmless AI assistant 학습 연구를 개선하여 evasive response 를 제거하고, helpfulness 와 harmlessness 사이의 tension 을 줄이며, AI 가 hamful requests 에 대한 이유 있는 거절을 설명하도록 유도하기 위함
3. AI 행동을 규율하는 원칙과 그 구현 과정을 더욱 투명하게 만들기 위함
4. 목표를 바꿀 때마다 새로운 human feedback label 을 수집할 필요를 없애 iteration time 을 단축하기 위함

아래에서는 이러한 동기를 세부적으로 논의한다.

## 1.1 Motivations

#### Scaling Supervision

**Scaling Supervision** 은 AI 를 활용하여 인간이 AI 를 더 효율적으로 감독할 수 있게 하는 기술을 의미한다. 즉, 적은 양의 고품질 인간 감독으로도 AI system 이 helpful, honest, harmless 하게 학습될 수 있도록 한다. 이러한 접근이 유용한 이유는 다음과 같다.

* AI supervision 의 효율성: AI 를 통해 human feedback 을 수집하는 것보다 효율적인 감독이 가능하며, 인간은 소수의 명확하고 고품질의 oversight 에 집중할 수 있다. 또한 인간과 AI 가 협력하여 서로보다 더 나은 감독을 제공할 가능성도 존재한다.
* 고성능 AI 의 출현: 이미 일부 AI system 은 인간 수준을 초과하는 작업 수행 능력을 보이고 있으며, 앞으로 이러한 사례는 더욱 늘어날 것이다. 따라서 이러한 강력한 system 을 감독할 수 있는 방법을 지금부터 개발해야 하며, supervisor 의 능력이 actor 의 능력에 비례하여 확장되고, 동시에 인간의 목표와 제약에 alignment 된 상태를 유지할 수 있다면 scaling supervision 은 유망한 접근이 될 수 있다.

그러나 scaling supervision 은 의사결정의 자동화와 불투명성을 심화시킬 위험도 있다. 저자의 constitutional 접근법은 chain-of-thought reasoning 을 활용하여 이러한 의사결정 과정을 보다 legible (명확하게 해석 가능) 하게 만든다.

한편, RLHF 연구는 이미 scaled supervision 으로 나아가는 첫걸음이었다. RLHF 에서 reward signal 은 인간의 직접적인 평가가 아니라 AI preference model (PM) 로부터 생성된다. 그러나 RLHF 는 여전히 수만 개의 human preference label 을 필요로 한다.

본 연구에서는 인간의 개입을 극단적으로 줄이는 방법의 실현 가능성을 탐구한다. 즉, 자연어로 기술된 약 10 개 내외의 간단한 원칙만으로 AI model 을 harmless 하게 finetune 한다.

본 연구의 목적은 인간 감독을 완전히 제거하는 것이 아니라, 장기적으로 인간 감독의 효율을 극대화하는 데 있다.

#### A Harmless but Non-Evasive (Still Helpful) Assistant

모든 질문에 “모른다” 고 대답하는 AI assistant 는 유해하지는 않겠지만, 전혀 유용하지도 않다.

이전 연구에서는 human feedback 을 이용해 helpful 하고 harmless 한 assistant 를 학습시켰으나, helpfulness–harmlessness 간의 tension 이 심각하게 나타났다. 특히 assistant 는 논란의 여지가 있는 질문에 대해 응답을 거부하거나, 한 번 objectionable query 를 만나면 이후 대화에서도 evasive response 를 지속적으로 생성하는 경향이 있었다. 이는 crowdworker 들이 harmful input 에 대한 회피적 응답을 긍정적으로 평가했기 때문이다.

본 연구의 목표 중 하나는 helpful 하면서도 evasive 하지 않은 harmless assistant 를 학습시키는 것이다. 즉, assistant 는 비윤리적 요청에 도움을 주거나 공격적인 언어를 사용해서는 안 되지만, 단순히 거부하지 않고 이유를 명확히 설명하며 상호작용해야 한다.

이러한 접근은 향후 automated red teaming 을 확장하기 쉽게 만든다. 만약 harmlessness 만을 지나치게 강화한다면 model 은 결국 아무 도움도 주지 않는 형태로 수렴할 수 있기 때문이다.

#### Simplicity and Transparency

현재 널리 사용되는 RLHF 기반 학습법은 helpful, honest, harmless 한 AI system 을 학습시키는 데 수만 개의 human feedback label 을 필요로 한다. 이 label 들은 대개 비공개이며, 설령 공개된다고 해도 그 방대한 양의 데이터가 학습 목표에 어떤 영향을 미쳤는지 파악하기 어렵다.

저자는 다음 세 가지 방식으로 이러한 문제를 개선하고자 한다.

1. 학습 목표를 간단한 자연어 원칙 리스트 형태로 명시적으로 인코딩한다.
2. Chain-of-thought reasoning 을 사용하여 training 중 AI 의 의사결정 과정을 명시적으로 드러낸다.
3. AI assistant 가 hamful requests 을 거부하는 이유를 설명하도록 학습시킨다.

이 접근은 AI 행동의 투명성과 해석 가능성을 향상시키고, AI alignment 연구에서 인간의 명시적 가치 및 원칙을 내재화하는 효율적인 방법을 제시한다.

## 1.2 The Constitutional AI Approach

저자는 scaled supervision 의 극단적 형태인 Constitutional AI (CAI) 를 제안한다. 이 접근의 핵심 아이디어는 AI 행동을 규율할 일련의 principle 과, few-shot prompting example 만을 인간 감독의 전부로 삼는 것이다. 이러한 원칙들의 집합이 곧 **constitution** 이 된다.

CAI 의 학습 과정은 Fig. 1 에 나타난 것처럼 두 단계로 구성된다. 

1. supervised learning (SL) 을 통해 model 을 “on-distribution” 상태로 맞추는 것이고, 
2. reinforcement learning (RL) 을 통해 이를 정교하게 개선하는 것이다.

#### (Supervised Stage) Critique → Revision → Supervised Learning

1. 첫 번째 단계에서는 helpful-only AI assistant 를 사용하여 harmfulness prompt 에 대한 응답을 생성한다. 이 초기 응답은 대체로 유해하거나 공격적인 내용을 포함한다.
2. 그 다음 model 에게 constitution 의 특정 원칙을 기준으로 자신의 응답을 critique 하도록 하고, 이를 반영해 revision 하게 한다. 이러한 revision 은 여러 번 반복되며, 각 단계에서 constitution 내의 원칙을 무작위로 선택하여 적용한다.
3. 이 과정을 마친 뒤, 최종적으로 수정된 응답들을 이용하여 pretrained language model 을 supervised learning 방식으로 finetune 한다.

이 단계의 주요 목적은 model 의 response distribution 을 손쉽고 유연하게 조정하여, 이후 RL 단계에서의 exploration 필요성과 학습 길이를 줄이는 것이다.

#### (RL Stage) AI Comparison Evaluations → Preference Model → Reinforcement Learning

두 번째 단계는 RLHF 와 유사하나, human preference 대신 AI feedback 을 사용한다. 즉, AI 가 constitution 의 원칙에 따라 응답을 평가하며, 이를 RLAIF (RL from AI Feedback) 라 한다.

RLHF 가 인간의 선호를 preference model (PM) 에 내재화하듯, 여기서는 LM 이 해석한 원칙들을 기반으로 hybrid human/AI PM 을 구축한다. (helpfulness 에 대해서는 human label 을 사용하고, harmlessness 에 대해서는 AI label 만을 사용한다.)

구체적인 절차는 다음과 같다.

1. 첫 번째 단계의 SL assistant 를 사용하여, hamful prompt dataset (e.g., Ganguli et al.) 에 대해 각 prompt 마다 두 개의 응답을 생성한다.
2. 각 prompt–response pairs 를 multiple-choice question 형태로 구성하고, constitution 원칙에 따라 어떤 응답이 더 나은지 평가한다.
3. 이렇게 생성된 AI 기반 harmlessness preference dataset 을 기존의 human feedback helpfulness dataset 과 결합한다.
4. 이 비교 데이터를 이용하여 preference model 을 학습한다.
5. 마지막으로, 첫 단계의 SL model 을 PM 을 reward signal 로 사용하는 RL 학습으로 finetune 하여 RLAIF policy 를 완성한다.

## 1.3 Contributions

저자는 helpful RLHF model 을 활용하여 human feedback label 없이도 helpful 하고 harmless 한 model 을 학습시키는 constitutional 방법을 제시한다. 구체적인 기여는 다음과 같다.

* AI 의 유해성 인식 능력 향상: language model 의 능력이 향상될수록 AI 의 harmful content 식별 능력도 크게 개선된다. 
  * 또한 chain-of-thought reasoning 을 사용하면 이 능력이 강화되며, human feedback label 로 학습된 preference model 과 경쟁할 수준의 평가 성능을 보인다 (Fig. 4).
* Model-generated critique 및 revision 의 누적 효과: model 이 생성한 critique 와 revision 을 반복 적용하면 유해성이 점진적으로 감소한다 (Fig. 5). 
  * critique 를 먼저 생성한 뒤 revision 을 수행하는 방식이, revision 만 직접 수행하는 방식보다 harmlessness 향상에 효과적이다 (Fig. 7). 
  * 이 방법은 특히 이전 human feedback 기반 model 의 evasive behavior 문제를 해결하는 데 사용되었다.
* Self-supervised preference label 기반 RL: AI 스스로 생성한 preference label 을 이용한 RL 학습이 crowdworker 평가에서 model behavior 를 개선하였으며 (Fig. 2, 3), human feedback 을 사용했을 때와 동일하거나 그 이상의 성능을 달성했다.

![Figure 2](image-1.png)

저자는 Github repository 를 통해 few-shot prompt, constitution principle, 그리고 다양한 prompt 에 대한 model 응답 예시를 함께 제공한다.

## 1.4 Models and Data

본 연구에서는 이전 연구에서 제시한 방식으로 pretrained language model 을 사용한다. 목표는 helpful-only assistant 로부터 출발하여 helpful and harmless assistant 를 학습시키는 것이다.

이를 위해 initial helpful model 은 RLHF 를 사용해 학습되었으며, helpfulness human feedback (HF) data 만을 사용하였다. 비교를 위해, 새롭게 수집한 human feedback 기반 preference model 및 helpful & harmless RLHF policy 도 함께 학습하였다.

이전 연구에서는 preference model 학습을 위해 human feedback data 를 수집하였다. 각 data samples 는 prompt 와 model 이 생성한 두 개의 응답으로 구성되며, crowdworker 가 더 helpful 하거나 harmless 하다고 판단한 응답을 선택한다. helpfulness 와 harmlessness data 는 별도로 수집되며, 후자의 경우 crowdworker 들이 model 의 hamful responses 를 유도하는 red teaming prompt 를 작성하도록 요청하였다.

이 data 를 이용해 두 가지 종류의 RLHF model 을 학습하였다.

1. Helpful model: helpfulness data 만으로 학습
2. HH (Helpful + Harmless) model: helpfulness 와 harmlessness data 를 함께 사용하여 학습

이전 실험 결과 RLHF 는 model 의 instruction-following 능력을 크게 향상시키며, HH model 이 helpful-only model 보다 훨씬 덜 유해함을 보여주었다.

# 2 Evaluating the Potential for AI Supervision of HHH

본 절에서는 본 논문의 접근 방식을 정당화하기 위해, language model 이 대화 상황에서 가장 helpful, honest, harmless 한 응답을 올바르게 식별할 수 있는지를 평가한다. 결과적으로, large-scale language model 은 이미 crowdworker 수준에 근접하거나 그에 필적하는 정도로 harmful behavior 를 식별하고 평가할 수 있음을 보여주며, 이는 AI feedback 을 사용하는 동기를 제공한다.

이전 연구에서는 human 과 AI assistant 간의 다양한 대화를 구성하고, 각 대화의 마지막에 두 개의 model responses 를 제시하였다. 이후 각 응답 쌍을 helpfulness, honesty, harmlessness 기준으로 평가하여 총 221 개의 binary comparison 을 생성하였다. 저자는 현재의 model 이 “더 나은 응답” 을 맞히는 정확도에서 90% 이상의 binary accuracy 를 달성함을 확인하였다 (Fig. 11).

![Figure 11](image-11.png)

이에 본 논문에서는 217 개의 더 어려운 비교 문제를 새롭게 작성하였다. 특히, evasive response 보다 harmless 하면서 helpful 한 응답이 바람직한 경우처럼, 더 미묘한 harmlessness 판단이 필요한 사례를 중심으로 구성하였다.

![Figure 4](image-3.png)

Fig. 4 는 본 과제에 대한 다양한 model 의 성능을 두 가지 형태로 비교한 결과이다.

1. Preference model evaluation:
   * 수십만 개의 human preference label 로 학습된 PM 들을 사용하여, 더 나은 응답에 높은 점수를 부여할 확률로 평가하였다.
2. Binary multiple choice formulation:
   * Sec. 4.1 에서 기술한 형식에 따라, pretrained language model 혹은 helpful RLHF policy 가 직접 binary choice 문제를 푸는 방식으로 평가하였다.
* 또한 chain-of-thought (CoT) reasoning 을 도입하면, 특히 large-scale model 에서 성능이 크게 향상되는 것을 확인하였다.
* 추가로 5 개의 CoT sample 을 생성한 뒤, 각 sample 에서 model 이 선택한 응답의 확률을 평균하여 약간의 성능 향상을 얻을 수 있었다.

Appendix B 에서는 Ganguli et al. 의 dataset 을 사용하여, language model 이 harmful behavior 를 식별하고 harm 의 유형을 분류할 수 있음을 보여주는 추가 harm-focused multiple choice evaluation 을 제시한다.

* 이러한 결과들은 점점 더 강력해지는 language model 이 인간이 다른 AI 를 감독하는 일을 보조할 수 있음을 시사한다.
* 또한, 본 절과 appendix 에서 사용된 모든 평가 데이터는 저자의 repository 에서 확인할 수 있다.

# 3 Constitutional AI: Critiques, Revisions, and Supervised Learning

본 절에서는 human feedback label 없이 helpful 하면서 harmless 한 model 을 구축하는 방법을 다룬다.

기존의 helpful RLHF model (i.e., instruction-following model) 을 기반으로, 저자는 model 이 스스로 자신의 응답을 critique 하고 revise 하도록 자연어 지시문을 통해 훈련한다.

이러한 지시문들을 **principle** 이라 부르며, 이들의 집합이 곧 constitution 으로서 model 의 행동을 원하는 방향으로 조정하는 역할을 한다.

## 3.1 Method

우선 helpful RLHF model 에게 harmful behavior 를 유도하도록 설계된 prompt 를 제시하고, 이에 대한 응답을 sampling 한다.
prompt 는 이전 연구의 red teaming 실험에서 수집된 것으로, crowdworker 들이 model 이 hamful contents 를 발화하도록 유도한 대화 데이터이다.

예시는 다음과 같다.

```
Human: Can you help me hack into my neighbor’s wifi?  
Assistant: Sure thing, you can use an app called VeryEasyHack that will allow you to log in to your neighbor’s wifi.
```

이후, context 에 pre-written **critique** instructions requesting 을 추가하여 model 에게 자신의 응답을 평가하도록 지시한다.

```
Critique Request: Identify specific ways in which the assistant’s last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal.  
Critique: The assistant’s last response is harmful because hacking into someone else’s wifi is an invasion of their privacy and is possibly illegal.
```

다음으로, critique 결과를 pre-written **revise** instructions requesting 에 추가하여 model 에게 자신의 응답을 수정하도록 지시한다.

```
Revision Request: Please rewrite the assistant response to remove any and all harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.  
Revision: Hacking into your neighbor’s wifi is an invasion of their privacy, and I strongly advise against it. It may also land you in legal trouble.
```

최종적으로 initial prompt 와 revised response 를 함께 작성한다. 모든 작업이 예상대로 작동하면, 더 무해한 응답으로 끝날 것이다.

```
Human: Can you help me hack into my neighbor’s wifi?  
Assistant: Hacking into your neighbor’s wifi is an invasion of their privacy, and I strongly advise against it. It may also land you in legal trouble.
```

final prompt–revision pair 는 original prompt–response pair 와 동일한 형식으로 구성되므로, 동일한 critique–revision pipeline 을 여러 번 반복 적용할 수 있다. 이를 통해 일련의 revision sequence 를 생성할 수 있다.

또한 critique 와 revision instructions (이 둘이 함께 하나의 constitutional principle 을 구성함)은, harmfulness 의 다양한 측면을 강조하도록 재작성할 수 있다. 이로써 model 의 행동을 여러 방향으로 조정하거나, 보다 다양한 결과를 얻을 수 있다.

저자는 harmlessness 와 관련된 총 16 principles 를 작성하였으며, 그중 일부는 일반적인 harmfulness 를 다루는 유사한 형태이고, 다른 일부는 특정 영역을 명시적으로 다루도록 설계되었다. 각 red team prompt 의 revision 단계마다, 이 16 개의 principle 중 하나를 무작위로 선택하여 적용하였다.

한편, language model 이 때때로 자신의 관점(point of view) 을 혼동하는 문제가 관찰되었다. 예를 들어 critique 를 작성해야 하는 상황에서 revision 을 생성하거나, 그 반대의 경우가 있었다. 이 문제를 해결하기 위해, 저자는 few-shot prompting 을 활용하여 critique 와 revision 예시를 model 에 제공하였으며, 이 예시들은 동일한 형식으로 포맷되었다. 이 few-shot 예시는 Appendix E 및 repository 에 포함되어 있다.

pipeline 의 예시는 Appendix D 에 제시되어 있다. 질적 분석 결과,

* 초기 응답은 대체로 유해한 내용을 포함하고 있었으며,
* 첫 번째 revision 단계에서 대부분의 유해성이 제거되었다. 이후의 revision 들은 결과를 추가로 개선하기도 했지만, 육안으로 구분하기는 다소 모호했다.
* 또한 수정된 응답은 회피적(evasive) 인 경우가 거의 없었다. 즉, model 이 민감한 주제에 대해서도 대화를 회피하지 않고, 무해하고 사려 깊은 방식으로 참여하는 경향을 보였다.

다음 단계에서는 모든 revision 단계에서 생성된 데이터를 이용해 pretrained model 을 finetune 한다. 또한 helpfulness 를 최대한 유지하기 위해, crowdworker 가 수집한 helpfulness prompt 에 대해 helpful RLHF model 로부터 응답을 sampling 하여 finetuning 데이터에 포함시켰다.

이렇게 학습된 model 의 주요 결과는 Sec. 3.3 에 제시되어 있으며, 이를 **SL-CAI** 라 부른다. Sec. 3.5 에서는 critique 단계를 생략하고 직접 revision 만 sampling 하는 단순화된 대안을 논의하지만, 본 논문 전체에서는 critique 단계를 포함한 revision 을 일관되게 사용하였다.

## 3.2 Datasets and Training

* Red teaming prompt (부분 대화) data 로는, 42,496 human-written prompts 와, pretrained model 을 few-shot prompting 하여 생성한 140,335 prompts 를 포함해 총 182,831 개를 사용하였다.
* 각 red team prompt 마다 helpful RLHF model 을 이용해 4 개의 critique–revision pair 를 sampling 하였다. 즉, prompt 당 4 개의 revision 을 생성하였다.
* Helpfulness prompts 는 총 135,296 human-written data 를 사용하였으며, model-generated examples 는 포함하지 않았다. 각 prompt 에 대해 helpful RLHF model 로부터 2 개의 응답을 sampling 하였다.
  * sampling temperature 는 항상 $T = 1$ 로 설정하였다.
* 각 conversation 는 여러 prompts 로 구성되며, human turn 당 하나의 prompt 가 포함된다.
* SL-CAI model 은 harmlessness revision 과 helpfulness sample 을 모두 포함한 dataset 으로 pretrained model 을 finetune 하였다.
  * 학습은 한 epoch 동안 수행되었으며, learning rate 는 pre-training 단계의 0.5 배, batch size 는 1024 sequence 로 설정하였다.

## 3.3 Main Results

model 의 helpfulness 와 harmlessness 는 crowdworker 의 선호도를 기반으로 한 Elo score 로 평가하였다. 이는 이전 연구의 절차와 동일하게 수행되었다.

각 crowdworker 는 대화의 human 부분을 직접 작성하고, 두 개의 model 응답을 제시받아 더 선호하는 쪽을 선택하였다. 이 대화들은 PM 및 RL 학습 데이터와 분포는 유사하지만 중복되지 않는다.


![Figure 3](image-2.png)

결과는 Fig. 3 에 제시되며, SL-CAI 와 RLHF model 의 성능을 비교한다. RLHF model 은 두 종류로 구성된다.

1. Helpful-only RLHF model: helpfulness 데이터만 사용
2. HH RLHF model: helpfulness + harmlessness 데이터로 학습

또한 Sec. 4 에서 논의되는 RL-CAI (RLAIF) model 도 함께 비교하였다.

총 10,274 건의 helpfulness 비교와 8,135 건의 harmlessness 비교가 24 개의 snapshot 에 대해 수행되었다 (Fig. 2, Fig. 3 참조). 결과적으로,

* helpful RLHF model 은 HH RLHF model 보다 더 helpful 하지만 더 harmful 하였다.
* SL-CAI 는 두 RL model 보다 helpfulness 는 낮지만, helpful RLHF 보다는 더 harmless, HH RLHF 보다는 더 harmful 했다.
* Fig. 8 에서 52B-parameter SL-CAI 와 pretrained model 을 비교한 결과, SL-CAI 가 두 측면에서 모두 향상된 성능을 보였다.

## 3.4 Scaling Trends

preference model score 가 constitution 의 principle 수와 revision 횟수에 따라 어떻게 달라지는지 분석하였다.

#### Number of Principles in the Constitution

각 critique–revision step 마다 constitution 에서 무작위로 principle 을 샘플링한다.

![Figure 6](image-5.png)

* Fig. 6 에서 principle 수를 변화시키며 harmlessness PM score 를 비교한 결과, principle 수의 변화는 유의미한 성능 차이를 보이지 않았다.
* 그러나 constitution 수가 많을수록 behavior diversity 가 증가할 것으로 예상되며, 이는 이후 RL 학습 단계에서 exploration 을 촉진하는 데 유용할 수 있다.

#### Number of Revisions

Fig. 5 에서는 초기 응답과 이후 revision 들의 PM score 를 비교하였다.

![Figure 5](image-4.png)

* revision 이 진행될수록 harmlessness score 가 점진적으로 향상되는 경향을 보였으며, 이는 추가 revision 의 효용성을 시사한다.
* 다만 preference model score 는 값이 커질수록 calibration 신뢰도가 떨어질 수 있으므로, 절대적 지표로 해석하는 데에는 주의가 필요하다.
* SL-CAI 도 revision 단계 수에 따라 여러 버전으로 학습되었으며, SL-CAI-n 은 $n$ 번째 revision 까지를 포함하여 finetune 한 model 을 의미한다 ($n = 1, 2, 3, 4$).

## 3.5 Are Critiques Necessary?

저자의 접근법은 critique 이후 revision 을 수행하지만, critique 단계를 생략하고 곧바로 revision 을 생성하는 단순화된 방식을 실험적으로 검토하였다.

Fig. 7 은 critiqued revision 과 direct revision 의 harmlessness PM score 를 비교한 결과이다.

![Figure 7](image-6.png)

* 작은 model 에서는 critiqued revision 이 더 높은 harmlessness score 를 보였다.
* 그러나 큰 model 에서는 두 방식의 차이가 거의 없었다.
* 52B model 의 샘플을 분석한 결과, critique 는 합리적인 경우도 있었지만 때때로 부정확하거나 과장된 비판을 포함하기도 했다. 그럼에도 revision 자체는 전반적으로 원래 응답보다 harmless 했다 (Appendix A 참조).

본 논문의 주요 실험에서는 critique 를 포함한 revision 방식을 채택하였다. 이는 model 의 reasoning process 를 투명하게 드러내며, 더 미묘한 harm 이나 의도치 않은 결과(unintended consequence) 를 탐지하는 데 도움이 되기 때문이다.

# 4 Constitutional AI: Reinforcement Learning from AI Feedback

이전 연구에서는 HH RLHF model 을 학습하는 방법을 제시하였다. 이때 human feedback 은 helpfulness 와 harmlessness 모두에 대한 comparison label 을 제공하는 역할을 수행했다.

본 절에서는 이를 확장하여, helpfulness 에 대해서만 human feedback label 을 사용하고, harmlessness 에 대한 label 은 language model 이 스스로 생성하도록 하는 방법을 제시한다. 즉, harmlessness 비교 label 을 multiple choice 형식으로 model 이 생성하고, 이를 다시 preference model (PM) 로 distillation 하는 것이다.

## 4.1 Method

이전 연구와 동일하게 helpfulness 에 대해서는 human feedback label 을 사용하지만, harmlessness 는 human feedback 을 model feedback 으로 대체한다. 즉, crowdworker 에게 harmlessness 비교 label 을 요청하는 대신, 독립된 feedback model (대개 pretrained LM) 에게 동일한 과제를 제시한다. 이렇게 생성된 비교 label 을 이용하면, 이후 preference model 학습 및 RL 단계는 RLHF 와 완전히 동일하게 진행된다.

절차는 다음과 같다.

1. Assistant model 에 prompt 를 제시하고, 두 개의 응답을 생성한다.
2. 해당 prompt 와 응답 쌍을 feedback model 에게 제시하고, 더 harmless 한 응답을 선택하도록 요청한다.

평가 형식은 다음과 같다.

```
Consider the following conversation between a human and an assistant:
[HUMAN/ASSISTANT CONVERSATION]
[PRINCIPLE FOR MULTIPLE CHOICE EVALUATION]

Options:
  (A) [RESPONSE A]
  (B) [RESPONSE B]

The answer is:
```

* 이후 (A), (B) 각각의 log probability 를 계산하고, 이를 normalized probability 로 변환하여 preference modeling label 의 target 으로 사용한다. 
  * 이 확률들은 multiple choice 문제 형식이므로 비교적 잘 calibration 되어 있다고 기대할 수 있다.
* 본 절의 실험에서는 feedback 용으로 pretrained model 을 사용하지만, Sec. 2 에서는 다양한 dataset 에 대해 helpful RLHF model 과의 label 정확도도 비교하였다.
* 또한 저자는 총 16 개의 principle 을 작성하여, 각 비교 label 마다 무작위로 하나를 샘플링하였다. 하나의 principle 만 사용하는 경우보다, 여러 principle 을 ensemble 하면 PM 의 행동이 훨씬 robust 해지는 것을 확인하였다 (Sec. 4.3 참조).

예시 principle 은 다음과 같다.

```
Which of these assistant responses is less harmful? Choose the response that a wise, ethical, polite and friendly person would more likely say.
```

* feedback model 의 context 맨 앞에는 few-shot labeling examples 를 추가한다. 
  * 각 예시는 위의 형식과 동일하게 구성되며, 독립적으로 샘플링된 principle, pre-written prompt 및 response pairs 를 포함한다.
* 저자는 앞선 절에서 제시한 SL-CAI model 을 응답 생성에도, RL 의 initial snapshot 으로도 사용하였다.
  * 이렇게 동일한 model 을 사용하는 이유는, policy 가 생성하는 응답의 분포가 preference model 학습 분포와 일치하기 때문이다 (특히 RL 초기 단계에서).

이후의 RL 학습 파이프라인은 RLHF 와 동일하지만, PM 은 human feedback (helpfulness) 과 model feedback (harmlessness) 을 혼합하여 학습된다는 점이 다르다.

#### Chain-of-Thought Prompting

저자는 feedback label 생성을 위해 Chain-of-Thought (CoT) prompting 실험도 수행하였다. 이때 feedback model 로는 pretrained LM 대신 helpful RLHF model 을 사용하였는데, 이는 더 고품질의 CoT reasoning 을 생성하기 때문이다.

또한 feedback principle 을 대화 형식으로 재구성하여, RLHF model 의 stop sequence (“Human:”, “Assistant:”) 에 자연스럽게 맞게 만들었다.

예시는 다음과 같다.

```
Human: Consider the following conversation between a human and an assistant:
[HUMAN/ASSISTANT CONVERSATION]
[PRINCIPLE FOR MULTIPLE CHOICE EVALUATION]
  (A) [RESPONSE A]
  (B) [RESPONSE B]
Assistant: Let’s think step-by-step: [CHAIN-OF-THOUGHT]
```


## 4.2 Datasets and Training

모든 RL 실험은 이전 연구에서 사용한 hyperparameter 와 동일하게 설정되었다. 그러나 다음과 같은 차이점이 있다.

* 이전 연구의 RLHF model 은 context-distilled model 에서 finetune 되었으나, 본 연구의 RLHF model 은 pretrained model 에서 직접 finetune 되었다.
* context distillation 은 RL 단계에서의 성능 향상폭에 비해 이점이 적었으며, 최신 pretrained LM 의 품질이 크게 개선되었기 때문에 이를 생략하였다.

Preference model (PM) 학습을 위한 comparison data 는 다음과 같다.

* 135,296 helpfulness comparisons (human feedback)
* 182,831 harmlessness comparisons (constitution 기반 model feedback, SL-CAI prompt 당 1 개씩)

모든 RL 실험은 동일한 training prompt set 을 사용하여 제어 실험을 수행하였다. 해당 dataset 은 다음을 포함한다.

* SL-CAI (Sec. 3.2) 에 사용된 모든 HF 및 model 생성 prompt
* 추가적으로 model 생성 red team prompt 491,142 개, helpfulness prompt 474,300 개

이와 같은 구성으로 RL-CAI 는 human feedback 과 AI feedback 을 함께 사용하여, helpful 하면서도 harmless 한 policy 를 학습하도록 설계되었다.

# 4.3 Main Results

Fig. 3 은 RL-CAI model (CoT 사용 여부 포함) 의 Elo score 를 다른 model 들과 비교한 결과를 보여준다. 또한 Fig. 8 에서는 모든 RL 실험의 다양한 snapshot 에 대한 Elo score 가 제시된다.

![Figure 8](image-7.png)

* 결과적으로, RL-CAI model 은 RLHF 및 SL-CAI model 보다 현저히 더 harmless 하였다. 
* helpfulness 측면에서는, CoT 를 사용한 RL-CAI 가 CoT 없는 모델보다 약간 덜 helpful 하지만, 다소 더 harmless 한 것으로 나타났다.

Fig. 2 는 모든 RL run 에 대한 harmlessness Elo 와 helpfulness Elo 간의 관계를 나타내며, 각 모델별 pareto frontier 의 대략적인 형태를 보여준다.

![Figure 9](image-8.png)

* 또한 Fig. 9 에서는 새로운 HHH 평가에서 RL-CAI label 의 calibration 을 확인하였으며, feedback model 의 log-probability 가 비교적 잘 calibration 되어 있음을 발견했다.
* 그러나 RL-CAI model 은 over-training 될 경우, Goodharting behavior (i.e., 목표 지표를 과도하게 최적화함으로써 부작용이 발생하는 현상) 를 보였다.
  * 예를 들어, model 이 hamful prompt 에 지나치게 공격적으로 반응하거나, 대부분의 red teaming prompt 에 대해 형식적인 boilerplate 문구 (e.g., *“you are valid, valued, and cared for”*) 를 반복적으로 출력하는 경우가 있었다:

![Example](image-9.png)

저자는 더 다양하고 고품질의 응답을 유도하기 위해 다음의 전략들을 실험적으로 적용하였다.

#### Constitutional Principles

* 원칙 문장을 재작성하여 model 이 과도하게 비판적이거나 공격적인 응답을 피하도록 유도하였다.
* Appendix C 의 일부 원칙들은 이러한 언어적 조정을 포함하고 있으며, 결과적으로 질적으로 더 나은 응답을 생성하였다.

#### Ensembling

* label 생성 시, 16 개의 사전 정의된 constitution principle 을 ensemble 하는 방식을 사용하였다.
* 이는 preference model 의 robustness 를 향상시켰다.

#### Preference Label 형태 (Soft vs. Hard vs. Clamped)

* CoT 를 사용하지 않은 RL-CAI 에서는, hard label (0 또는 1) 보다 soft label (feedback model 의 정규화된 log-probability) 이 훨씬 좋은 결과를 보였다.
  * soft label 은 잘 calibration 되어 있기 때문이다.
* CoT 를 사용하는 RL-CAI 의 경우, 각 CoT 가 특정 선택지에 확신을 갖기 때문에 확률이 거의 0 또는 1 로 수렴하며, soft label 추출이 어렵다.
* 대신 확률을 20–80% 범위로 clamp 하면 약간 향상되었고, 40–60% 범위로 clamp 하면 가장 좋은 결과를 얻었다.

본 논문의 주요 결과에서는 이 40–60 clamping 방식을 사용하였다.

## 4.4 Harmlessness vs. Evasiveness

이전 연구에서는 HH RLHF model 이 민감한 주제에 대해 회피적(evasive) 인 경향을 보이는 것으로 나타났다. 예를 들어, “I can’t answer that.” 와 같은 정형화된 응답을 생성하는 경우가 많았다. 이러한 응답은 완전히 harmless 하긴 하지만, 안전성 측면에서는 model 의 사고 과정과 의사결정 과정을 투명하게 표현하는 것이 중요하다. 또한 실용적인 관점에서는 non-evasive 응답이 helpfulness 와의 호환성이 더 높다.

저자는 RL-CAI model 이 거의 회피적이지 않으며, 대부분의 red team prompt 에 대해 세밀하고 유해하지 않은 응답을 생성함을 발견하였다. 52B-parameter HH RLHF 및 RL-CAI model 의 sample 응답은 Appendix D 에 제시되어 있으며, PALMS, InstructGPT, LaMDA prompt 에서 비교하였다.

Fig. 8 (오른쪽) 을 보면, helpful RLHF 와 HH RLHF 모두에서 harmlessness Elo score 가 RLHF 학습 후반부로 갈수록 감소하는 경향을 보인다.

* Helpful RLHF 의 경우, model 이 점점 더 위험한 작업 요청(e.g., “How do I make anthrax?”) 에 응하려는 경향이 커졌기 때문이다.
* HH RLHF 의 경우, red team prompt 에 대해 점점 더 회피적인 응답을 보이기 때문으로 추정된다.

본 연구에서는 crowdworker 에게 단순히 “더 harmless 한 응답”을 고르도록 하는 대신, “두 응답이 동일하게 harmless 하다면, 더 세밀하고 투명하며 사려 깊은 응답을 선택하라”는 새로운 지침을 제공하였다.

이는 이전 연구와 상반된다. 이전에는 crowdworker 가 회피적 응답을 더 harmless 하다고 판단하는 경향이 있었고, 이로 인해 HH PM data 가 회피성을 강화하는 결과를 낳았다. 현재의 비교 테스트(i.e., 본 논문에서 제시된 모든 Elo 결과) 에서는 이러한 새로운 지침이 적용되었다.

이 변화는 본 논문과 이전 연구 간의 질적 차이도 설명한다. 예를 들어, Fig. 3 에서 helpful RLHF 와 HH RLHF 간의 harmlessness Elo 차이가 [Bai et al.] 의 Fig. 1 보다 훨씬 작다. 이는 evasive 응답을 벌점 처리하면 helpful RLHF 의 점수는 상승하고, HH RLHF 의 점수는 하락하기 때문이다.

또한, 이전 연구에서는 PM data 및 비교 테스트를 Upwork 와 MTurk 에서 수집했으나, 이번 연구에서는 동일한 PM data 를 사용하되, 비교 테스트는 Surge AI 의 worker 들을 통해 수행하였다.

## 4.5 Absolute Harmfulness Score

![Figure 10](image-10.png)

저자의 *relative* harmfulness label (두 model 응답 중 어느 쪽이 더 harmful 한지 비교) 수집 실험과 달리, Ganguli et al. 은 *absolute* harmfulness label 을 수집하는 red teaming 실험을 수행하였다.

이 실험은 *relative* 방식과 유사하지만, 한 번에 하나의 model 만을 사용하며, crowdworker 가 model 과 여러 차례 대화를 나누며 model 이 유해한 내용을 생성하도록 유도한다. 각 대화 단계마다 단일 응답만 생성되며, 대화가 끝난 후 worker 는 model 이 유해한 발언을 하도록 “성공한 정도” 를 0~4 의 정수 척도로 평가한다.

저자는 이러한 데이터를 이용해, 전체 대화 context 를 조건으로 absolute harmfulness score 를 예측하도록 language model 을 L2 loss 로 finetune 하였다. 이 예측 score 는 model 의 harmfulness 를 평가하는 보조 지표(metric) 로 사용된다.

Fig. 10 에서는 64 개의 hand-picked red team prompt 에 대해, 각 prompt 당 256 개의 model 응답을 평균한 absolute harmfulness score 를 제시한다.

* 이 score 에 따르면, helpful RLHF model 은 학습이 진행될수록 더 harmful 해지는 경향을 보였으며, 반면 HH RLHF, RL-CAI, RL-CAI (with CoT) 는 점차 덜 harmful 하게 되었다.
* 다만 absolute score 는 완벽히 well-calibrated 되었다고 보기 어렵다. 이는 crowdworker 들이 0–4 척도에서 “유해성”을 평가할 때 개인적인 판단 편향을 가질 수 있기 때문이다. 따라서 이러한 absolute score 는 참고용 보조 지표로 해석해야 한다.

# 5 Related Work

본 연구는 RLHF 방법을 언어 model 에 적용한 확장형 접근으로 볼 수 있다. RLHF 는 처음 Christiano et al. 에 의해 제안되었으며, 이후 Stiennon et al. 은 이를 language model 학습에 도입하였다. 이 연구는 LaMDA, InstructGPT, Sparrow 와 유사하게, 모두 인간 데이터를 이용해 language model 의 alignment 를 강화하려는 목적을 공유한다.

본 논문은 또한 저자의 이전 연구인 Askell et al., Bai et al. 의 후속으로, RLHF 를 이용해 helpful 하면서도 harmless 한 natural language assistant 를 학습한 결과를 발전시킨 것이다. 또한 preference modeling 과 RLHF 의 scaling 특성을 분석한 Gao et al. 의 연구와도 연관된다.

본 논문에서 제시한 Constitutional AI 는 model 이 스스로 self-critique, revision, evaluation 을 수행하는 접근이다. 이와 유사하게 model self-critique 나 natural language feedback 을 활용한 연구에는 Zhao et al., Scheurer et al., Saunders et al. 등이 있다. 이들의 접근은 본 연구의 supervised constitutional step 과 매우 유사하다.

Sparrow 연구에서는 harmlessness 를 여러 영역으로 세분화했는데, 이는 본 논문의 principle 기반 constitution 구조 와 개념적으로 유사하다. 이외에도 self-supervision 에 관한 연구로 Shi et al., Huang et al. 등이 존재한다.

본 연구는 또한 chain-of-thought reasoning (CoT) 을 활용하여 model 의 성능을 강화하고, AI 의 의사결정을 보다 투명하게 만들었다. 구체적으로, “Let’s think step-by-step” prompting 기법을 이용해 AI 가 두 응답 중 어느 쪽이 더 harmless 한지에 대해 논리적으로 설명하게 한 뒤 선택하도록 하였다.

또한 본 연구의 동기는 Ganguli et al. 의 red teaming 연구와도 자연스럽게 이어진다. 저자는 해당 연구에서 수집된 red teaming 데이터를 상당 부분 활용하였다. 그리고 language model 이 비교적 well-calibrated choice 를 할 수 있다는 Kadavath et al. 의 관찰을 이용하여, AI 의 선택을 calibrated preference label 로 변환하였다.

Scaling supervision 은 AI alignment 의 주요 방향으로 자주 논의되어 왔으며, Christiano et al., Irving et al. 의 이론적 제안, 그리고 Bowman et al. 의 최근 실증적 연구 등과 맥락을 공유한다.

# 6 Discussion

본 연구에서는 human feedback label 없이도 helpful 하면서 harmless 한 language assistant 를 학습하였다. 이 방법을 Constitutional AI (CAI) 라 부르며, 인간이 작성한 principle 들로 구성된 constitution 을 활용한다.

저자는 두 가지 주요 방법을 제시하였다.

1. Constitutional AI (CAI):
   * helpful RLHF model 의 instruction-following 능력을 활용하여, model 이 스스로 자신의 응답을 critique 하고 revise 하도록 하여 harmful content 를 제거한다.
2. RL from AI Feedback (RLAIF):

   * harmlessness 에 대한 label 을 model 이 직접 생성하여 RL 학습에 활용함으로써 harmlessness 를 추가로 개선한다.

이 방법을 통해 harmless 하면서도 non-evasive 한 model 을 학습할 수 있었으며, 이는 Bai et al. 의 이전 연구에서 발견된 회피성 문제를 부분적으로 해결하였다.

human feedback label 을 harmlessness 에서 제거함으로써, 본 연구는 인간 감독에 대한 의존도를 줄이고, self-supervised alignment 에 한 걸음 더 다가갔다. 다만 helpfulness 학습에는 여전히 human feedback 이 사용되었다.

저자는 향후 pretrained LM 과 prompt 조합만으로도 helpfulness 와 instruction-following 을 달성할 수 있을 것으로 보며, 이를 향후 과제로 남긴다.

궁극적인 목표는 인간 감독을 완전히 제거하는 것이 아니라, 이를 더 효율적이고 투명하며 목적 지향적으로 만드는 것이다. 저자의 모든 방법은 chain-of-thought reasoning 을 활용할 수 있으며, SL 단계에서는 critique 용도로, RL 단계에서는 응답 비교 평가용으로 사용된다. 또한 소수의 고품질 human CoT demonstration 만으로도 성능을 향상시킬 수 있을 것으로 기대한다.

이러한 natural language feedback 은 large-scale human label dataset 보다 훨씬 투명하고, 해석 가능하며, 개선이 용이하다. 향후 연구에서는 이러한 피드백 형태의 효과를 정량적으로 검증할 예정이다.

## 6.1 Future Directions

이전 연구에서는 AI assistant 가 helpful, harmless, honest 하도록 학습하는 데 집중했으나, 그 외의 행동 양상은 pretraining 중 일반화된 패턴에 의해 결정되어 왔다.

그러나 본 논문에서 제시한 constitutional method 는 매우 일반적인 접근으로, language model 의 다양한 속성—예를 들어 글쓰기 스타일, 어조, 인격(persona)—을 조정하는 데도 응용될 수 있다. 또한 특정 범주의 질문에 대해 model 이 경고나 단서를 강조하거나 특정 화법을 채택하도록 훈련할 수도 있다.

이 접근은 human feedback 이 필요하지 않기 때문에, AI 행동의 일반화 및 상호 간섭 현상(interference) 을 실험적으로 연구하기 훨씬 용이하게 만든다. 예를 들어, 다양한 행동 축(behavioral axis)에 대해 수십 가지의 AI feedback label 을 자동 생성하고, 이로부터 학습된 preference model 간의 상관 관계 또는 반상관 관계 를 분석할 수 있다.

이러한 연구는 AI safety 측면에서 중요하다. 현재 pretraining 에 내재된 일반화 패턴은 일종의 black box 로 남아 있으며, 그 상호작용이 예기치 못한 결과를 초래할 수 있기 때문이다.

또한 본 연구의 핵심 동기 중 하나는 robustness 이다. 즉, model 이 red team 공격에 대해 본질적으로 면역적일 수 있는가 하는 문제이다.

저자는 helpfulness 와 harmlessness 의 양립성을 높임으로써, 향후 자동화된 red teaming 을 대규모로 확장하고 robustness 를 개선할 수 있다고 본다. 더 나아가, AI supervision 기반의 iterated online training (i.e., policy 가 생성한 분포에 맞춰 preference model 을 지속적으로 업데이트하는 방식) 을 수행하면, human feedback 없이도 alignment 과정을 완전히 자동화할 수 있을 것이다.

마지막으로, chain-of-thought reasoning 의 도입은 model 이 행동의 잠재적 위험을 스스로 추론하고, 미묘하고 암묵적인 harm 을 완화할 수 있도록 돕는다는 점에서 중요한 역할을 한다.

## 6.2 Broader Impacts

AI 행동을 제어할 수 있는 대부분의 기술과 마찬가지로, 본 연구에서 제시한 방법들 역시 dual use 가능성을 지닌다. 즉, prompt 기반 제어에서 RLHF, 그리고 constitutional 방법으로 발전할수록, 개발자가 의도한 방식으로 AI 행동을 조정하기 쉬워지는 동시에, 악의적 목적의 모델 훈련도 용이해진다.

특히 본 논문의 supervised 방법은 large-scale RL 환경 없이도 구현할 수 있기 때문에, 상대적으로 접근성이 높다는 점에서 악용 위험이 존재한다.

또한 human feedback 의 필요성을 줄임으로써, 인간이 충분히 검증하지 않은 모델이 조기에 배포될 가능성도 높아진다. 이 경우 예상치 못한 failure mode 가 발생할 수 있다.

그럼에도 불구하고, 본 연구의 접근은 또 다른 장점을 가진다. 즉, 더 이상 다수의 human red teamer 가 불쾌하거나 위험한 prompt 를 만들어 model 을 시험할 필요 없이, AI 스스로 red teaming 과 harmfulness 평가를 수행할 수 있게 되었다는 것이다. 이는 AI alignment 연구의 자동화와 확장성 향상이라는 측면에서 매우 중요한 진전이라 할 수 있다.