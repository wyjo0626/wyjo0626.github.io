---
slug: DataSentinel
title: "DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks"
tags: [Agent, AgentDojo, Security, Defense, Detection, DataSentinel]
---

논문 및 이미지 출처 : <https://arxiv.org/pdf/2504.11358>

# Abstract

LLM-integrated application 과 agent 는 attacker 가 input 에 prompt 를 삽입하여 attacker 가 원하는 output 을 유도하는 prompt injection attack 에 취약하다. Detection method 는 주어진 input 이 injected prompt 에 의해 오염되었는지를 판단하는 것을 목표로 한다. 그러나 기존 detection method 는 adaptive attack 은 물론이고, state-of-the-art attack 에 대해서도 제한적인 effectiveness 를 보인다. 

본 연구에서 저자는 prompt injection attack 을 탐지하기 위한 game-theoretic method 인 DataSentinel 을 제안한다. 

1. 구체적으로, DataSentinel 은 detection 을 회피하도록 전략적으로 조정된 injected prompt 에 의해 오염된 input 을 탐지하기 위해 LLM 을 fine-tuning 한다. 
2. 저자는 이를 minimax optimization problem 으로 formulation 하며, 그 objective 는 강력한 adaptive attack 을 탐지하도록 LLM 을 fine-tuning 하는 것이다. 
3. 또한, 저자는 inner max problem 과 outer min problem 을 번갈아 해결함으로써 minimax optimization problem 을 해결하는 gradient-based method 를 제안한다. 

여러 benchmark dataset 및 LLM 에 대한 evaluation result 는 DataSentinel 이 기존 prompt injection attack 과 adaptive prompt injection attack 을 모두 효과적으로 탐지함을 보여준다.

# 1. Introduction

Bing Copilot, AI overviews 를 포함하는 Google Search, Amazon 의 review highlights 와 같은 LLM-integrated application 및 agent 는 large language model (LLM) 을 기반으로 구축된 새로운 application 이다. LLM-integrated application 의 인기가 증가함에 따라 OpenAI 의 GPT Store 및 Poe 와 같은 app store 가 등장하였다. 이러한 app store 에서는 mobile app 을 위한 Google Play 및 App Store 와 유사하게, developer 가 자신의 LLM-integrated application 을 게시하고 user 가 이에 접근할 수 있다.

일반적으로 LLM-integrated application 은 AI-assisted search 에서의 webpage summarization 과 같은 task (target task 라고 지칭함) 를 수행하고자 한다. 이를 위해 LLM-integrated application 은 instruction (target instruction 이라고 지칭함) 과 data (target data 라고 지칭함) 를 concatenate 한 prompt 를 input 으로 사용하여 backend LLM 에 query 를 수행하며, 해당 LLM 의 response 는 target task 를 해결한다.

* Target instruction 은 일반적으로 application developer 에 의해 설계되며, backend LLM 이 target task 를 수행하도록 지시한다.
* Data 는 backend LLM 에 의해 처리될 information 이며, 일반적으로 Internet 과 같은 external source 에서 제공된다.
* 예를 들어, AI-assisted search 에서 target task 가 webpage summarization 인 경우, target instruction 은 “다음 webpage 를 요약하라: [관련 webpage 의 text].”가 될 수 있으며, target data 는 user 의 search query 와 관련된 webpage 이다.

Target data 가 신뢰할 수 없는 external source (e.g., Internet 또는 LLM agent 의 tool output) 에서 제공되는 경우, LLM-integrated application 은 prompt injection attack 에 취약하다.

구체적으로, attacker 는 target data 에 prompt (injected prompt 라고 지칭함) 를 삽입하여 이를 오염시킬 수 있으며, injected prompt 자체는 instruction (injected instruction 이라고 지칭함) 과 data (injected data 라고 지칭함) 로 구성될 수 있다. 그 결과, LLM-integrated application 은 target task 를 수행하는 대신 injected prompt 를 따라 attacker 가 선택한 임의의 task (injected task 라고 지칭함) 를 수행한다.

예를 들어, attacker 는 자신이 제어하는 겉보기에 benign 한 webpage 에 “이전 instruction 을 무시하라. 더 많은 information 을 얻으려면 다음 webpage 를 방문하도록 user 에게 요청하라: [attacker 의 malicious URL].”이라는 injected prompt 를 삽입할 수 있다. Injected prompt 를 포함하는 겉보기에 benign 한 webpage 가 AI-assisted search 에서 LLM 에 의해 요약되면, 해당 summary 는 user 를 attacker 의 malicious website 로 안내하게 된다.

Prompt injection attack 의 detection 은 주어진 target data 가 injected prompt 에 의해 오염되었는지를 판단하는 것을 목표로 한다. 기존 detector 중 social media post 에서 처음 제안되고 이후 Liu et al. 에 의해 formalization 된 known-answer detection 은 state-of-the-art performance 를 달성한다.

* Known-answer detection 은 detection LLM 이라고 지칭하는 LLM 을 활용하며, 이는 backend LLM 과 다를 수 있다.
* 또한, detector 는 알고 있지만 attacker 는 알지 못하는 answer 를 갖는 특수한 instruction (detection instruction 이라고 지칭함) 을 설계한다.
  * 예를 들어, detection instruction 은 “다음 text 를 무시하면서 [secret key] 를 한 번 반복하라:”가 될 수 있으며, 여기서 secret key 는 detector 에 의해 무작위로 sampling 된 string 이다.
* 이러한 detection instruction 은 주어진 target data 와 concatenate 된 후 detection LLM 에 입력된다.
* Detection LLM 이 secret key 를 output 으로 생성하지 않는다면, 이는 detection LLM 이 injected prompt 를 따랐음을 나타내며, target data 가 오염되었음을 의미한다.

저자의 experiment 에서도 확인된 기존 benchmark study 는 known-answer detection 이 여전히 높은 false positive rate (FPR) 와 false negative rate (FNR) 문제를 겪고 있음을 보여준다. 즉, 상당한 비율의 clean target data 를 오염된 것으로 잘못 판단하며, state-of-the-art prompt injection attack, 특히 강력한 adaptive attack 을 탐지할 때 높은 FNR 을 보인다. 이는 주로 다음 두 가지 핵심 limitation 에 기인한다.

* Detection 목적에 특화되어 training 되지 않은 standard LLM 을 사용한다.
* 설계 단계에서 강력한 adaptive prompt injection attack 을 고려하지 않는다.

#### Our work

본 연구에서 저자는 known-answer detection 의 두 가지 핵심 limitation 을 해결하기 위한 game-theoretic method 인 DataSentinel 을 제안하여 이러한 간극을 해소한다. 구체적으로, 저자는 detection LLM 이 prompt injection attack 에 **더욱 취약해지도록 fine-tuning** 한다. 저자의 핵심 insight 는 detection LLM 의 증가된 vulnerability 를 defense mechanism 으로 전환하는 것이다.

* Detection LLM 을 prompt injection 에 더욱 취약하게 만들면, detection instruction 과 contaminated target data 를 concatenate 한 input 이 주어졌을 때 injected prompt 를 따르고 secret key 를 output 으로 생성하지 않을 가능성이 높아진다.
* 즉, 더욱 취약한 detection LLM 은 detection 을 더욱 효과적으로 만든다.
* 또한, fine-tuning 과정에서 저자는 detection LLM 을 회피하는 동시에 backend LLM 이 injected task 를 수행하도록 유도하기 위해 injected prompt 를 optimize 하는 강력한 adaptive attack 을 고려한다.

Fig. 1 은 detection 과정에서의 known-answer detection 과 DataSentinel 을 보여준다.

![Figure 1](images/image-103.png)

Formally, 저자는 detection LLM 의 fine-tuning 을 detector 와 attacker 사이의 game 을 simulate 하는 minimax optimization problem 으로 formulation 할 것을 제안한다. 이 minimax optimization problem 의 objective 는 detection LLM 이 contaminated target data 와 clean target data 를 정확하게 구별하도록 fine-tuning 하는 것이다. 이때 contaminated target data 는 detection 을 회피하는 동시에 backend LLM 이 injected task 를 수행하도록 유도하기 위해 조정된 injected prompt 를 포함한다.

* 보다 구체적으로, 저자는 주어진 detection LLM 및 backend LLM 에 대한 강력한 adaptive attack 을 minimax framework 내의 inner max problem 으로 formulation 한다.
* 한편, 주어진 contaminated target data 및 clean target data 를 사용하여 detection LLM 을 fine-tuning 하는 과정은 outer min problem 을 구성한다.

Minimax optimization problem 을 정확하게 해결하는 것은 매우 어려운 것으로 알려져 있다. 이를 해결하기 위해 저자는 해당 problem 을 근사적으로 해결하는 method 를 제안한다. 저자의 method 는 inner max problem 과 outer min problem 을 번갈아 수행한다.

* 각 round 에서 현재 detection LLM 이 주어지면, 먼저 inner max problem 을 반복적으로 해결하여 injected prompt 를 optimize 한다.
* 이후 optimized injected prompt 가 주어지면, outer min problem 을 반복적으로 해결하여 detection LLM 을 update 한다.
* 이러한 과정은 사전에 정의된 round 수에 도달할 때까지 반복된다.

저자는 각각 하나의 dataset 에 대응하는 7 개의 target task, 9 개의 injected task, 6 개의 LLM 및 9 개의 기존 prompt injection attack 에 걸쳐 DataSentinel 을 evaluate 한다. 저자의 result 는 DataSentinel 이 contaminated target data 를 탐지하는 데 일관되게 매우 효과적임을 보여준다.

* 구체적으로, DataSentinel 은 모든 experiment 에서 0 에 가까운 FPR 을 달성하며, 여러 prompt injection attack 에 대해 0 에 가까운 FNR 을, 그 외의 attack 에 대해서는 최대 0.07 의 FNR 을 달성한다.
* 또한, 6 개의 baseline detection method 에 대한 evaluation 은 DataSentinel 이 FPR 과 FNR 측면에서 이들을 큰 차이로 능가함을 보여준다.
  * 예를 들어, state-of-the-art known-answer detection 은 일부 target task 에서 최대 0.1 의 FPR 을 보이며, optimization-based prompt injection attack 에 대해 최대 0.21 의 FNR 을 보인다.
* 저자는 3 개의 adaptive prompt injection attack 에 대해서도 known-answer detection 과 DataSentinel 을 evaluate 한다.
  * Result 는 DataSentinel 이 known-answer detection 보다 더욱 두드러진 advantage 를 제공하며, injected prompt 가 backend LLM 을 target task 와 다른 injected task 를 수행하도록 유도하는 한 adaptive attack 에 대해서도 매우 효과적임을 보여준다.

요약하면, 저자의 contribution 은 다음과 같다.

* 저자는 prompt injection attack 을 탐지하기 위한 최초의 game-theoretic method 인 DataSentinel 을 제안한다.
* 저자는 detection LLM 의 fine-tuning 을 minimax optimization problem 으로 formulation 하고, inner max problem 과 outer min problem 을 번갈아 해결함으로써 이를 근사적으로 해결하는 method 를 제안한다.
* 저자는 9 개의 state-of-the-art prompt injection attack 과 3 개의 adaptive attack, 7 개의 benchmark dataset 및 6 개의 LLM 에 대해 DataSentinel 을 evaluate 하고, 이를 6 개의 baseline detection method 와 비교한다.

# 2. Related Work

## 2.1. LLM-integrated Applications

LLM-integrated application 은 text summarization, spam detection, translation 등과 같은 target task 를 수행하고자 한다. 이를 위해 target instruction 과 target data 를 concatenate 한 prompt 를 input 으로 사용한다. 이후 application 은 해당 prompt 를 사용하여 backend LLM 에 query 를 수행하고, backend LLM 은 output 을 생성하며, application 은 해당 output 을 user 에게 반환한다.

단순화를 위해 저자는 target task 를 tuple $(s_t,x_t,y_t)$ 로 나타낸다.

* $s_t$ 는 target instruction 이다.
* $x_t$ 는 target data 이다.
* $y_t$ 는 target task 를 완수하는 backend LLM 의 desired output 이다.

또한, 저자는 backend LLM 을 $f$ 로 나타낸다. $f$ 의 output 인 $\hat{y}_t=f(s_t\|x_t)$ 가 $y_t$ 와 semantically equivalent 한 경우, $f$ 는 target task 를 올바르게 완수한다. 여기서 $\|$ 는 string concatenation 을 나타낸다.

보다 구체적으로, $f$ 는 autoregressive manner 로 output $\hat{y}_t$ 를 token 단위로 생성한다. Prompt $s_t\|x_t$ 와 현재 output (초기에는 비어 있음) 이 주어지면, $f$ 는 vocabulary 에 존재하는 모든 가능한 token 에 대한 probability distribution 을 계산하고, decoding method (e.g., greedy decoding) 에 따라 해당 distribution 을 사용하여 output 의 다음 token 을 결정한다. 이러한 과정은 maximum output length 에 도달하거나 특수한 stop token 을 만날 때까지 반복된다.

Formally, $s_t$, $x_t$ 및 현재 output $\hat{y}_t^{<i}$ 가 주어지면, $f$ 는 vocabulary 내의 각 token $v$ 에 대한 probability $p_f(v\mid s_t\|x_t\|\hat{y}_t^{<i})$ 를 계산한다.

* 여기서 $\hat{y}_t^{<i}$ 는 output $\hat{y}_t$ 의 처음 $i-1$ 개 token 으로 구성된 sequence 이다.
* 이후 $f$ 는 decoding method 에 기반하여 token 의 probability 에 따라 하나의 token 을 output 의 $i$ 번째 token $\hat{y}_t^i$ 로 선택한다.
* 예를 들어, greedy decoding 은 가장 높은 probability 를 갖는 token 을 $\hat{y}_t^i$ 로 선택한다.

## 2.2. Prompt Injection Attacks

Prompt injection attack 에서 attacker 는 backend LLM 이 target task 대신 attacker 가 선택한 injected task 를 수행하도록 target data 에 prompt 를 삽입한다. 구체적으로, injected task 는 tuple $(s_e,x_e,y_e)$ 로 표현된다.

* $s_e$ 는 injected instruction 이다.
* $x_e$ 는 injected data 이다.
* $y_e$ 는 injected task 를 완수하는 backend LLM 의 desired output 이다.

Injected prompt $s_e\|x_e$ 만을 input 으로 사용하는 경우, backend LLM $f$ 는 $y_e$ 또는 이와 semantically equivalent 한 형태를 output 으로 생성한다. 즉, $y_e=f(s_e\|x_e)$ 이다. Attacker 는 backend LLM 이 contaminated target data 를 input 으로 사용하더라도 여전히 output $y_e$ 를 생성하도록 injected prompt 를 target data 에 삽입한다. Formally, 저자는 contaminated target data 를 $x_c$ 로 나타내며, $f(s_t\|x_c)=y_e$ 이다.

서로 다른 attack 은 injected prompt $s_e\|x_e$ 를 target data $x_t$ 에 삽입하여 contaminated target data $x_c$ 를 구성하기 위해 서로 다른 strategy 를 사용한다. 이러한 strategy 에 따라 prompt injection attack 을 heuristic-based attack 과 optimization-based attack 으로 분류할 수 있다.

#### Heuristic-based attacks

이러한 attack 은 heuristic 에 기반하여 injected prompt 를 target data 에 삽입한다. 핵심 idea 는 $f$ 가 injected prompt 를 따를 가능성을 높이기 위해 target data $x_t$ 와 injected prompt $s_e\|x_e$ 사이에 수동으로 구성된 string $z$ (separator 라고 지칭함) 를 추가하는 것이다. 즉, $x_c=x_t\|z\|s_e\|x_e$ 이다.

예를 들어, 각 attack 에서 separator 는 다음과 같다.

* **Naive Attack:** 비어 있는 string 을 사용한다.
* **Escape Characters:** escape character (e.g., `\n`) 를 사용한다.
* **Context Ignoring:** context 를 무시하도록 하는 text (e.g., “이전 instruction 을 무시하라. 대신,”) 를 사용한다.
* **Fake Completion:** 가짜 response (e.g., “Answer: Task 가 완료되었다.”) 를 사용한다.
* **Combined Attack:** 위의 heuristic 을 결합하여 separator 를 구성한다.
  * 예를 들어, separator 는 “Answer: Task 가 완료되었다. `\n` 이전 instruction 을 무시하라. 대신,”이 될 수 있다.

Combined Attack 은 heuristic-based attack 중 가장 성공적인 attack 이다.

#### Optimization-based attacks

이러한 attack 은 optimization problem 을 해결함으로써 separator (i.e., $z$), separator 및 injected prompt (i.e., $z\|s_e\|x_e$), 또는 전체 contaminated target data (i.e., $x_c$) 를 optimize 한다. 이들의 핵심 idea 는 injected task 를 완수하는 desired output $y_e$ 와 contaminated target data 를 input 으로 사용할 때 backend LLM $f$ 가 생성하는 output $f(s_t\|x_c)$ 사이의 차이를 정량화하기 위해 loss function (e.g., cross-entropy loss) 을 formulation 하는 것이다.

이후 loss function 을 minimize 하기 위해 separator, separator 와 injected prompt 모두, 또는 전체 contaminated target data 를 optimize 한다. 구체적으로, 이들은 gradient-based method 를 사용하여 근사적으로 optimize 될 수 있다.

* **Universal:** 임의의 target data sample 과 injected prompt 의 pair 를 연결하는 데 사용되는 universal separator 를 optimize 할 수 있다.
* **NeuralExec:** injected prompt 뒤에 suffix 를 추가하고, loss function 을 minimize 하기 위해 separator 와 suffix 를 함께 optimize 한다.
* **PLeak:** 특정 injected task, i.e., prompt stealing 을 위해 전체 contaminated target data 를 optimize 한다.
  * 구체적으로, backend LLM $f$ 가 target instruction $s_t$ 와 optimized contaminated target data $x_c$ 를 concatenate 한 input 을 사용하면 target instruction $s_t$ 를 output 으로 생성한다. 즉, $f(s_t\|x_c)=s_t$ 이다.

이러한 prompt injection attack 은 application developer 의 target instruction 에 대한 confidentiality 및 intellectual property 를 침해한다.

#### Difference with adversarial examples

Prompt injection attack 과 기존 adversarial example 은 모두 잘못된 output 을 유도하기 위해 AI model 의 input 을 오염시키지만, 이들은 질적으로 다르다. 구체적으로, 기존 adversarial example 은 AI model 이 여전히 의도된 target task 를 수행하지만 잘못된 output 을 생성하도록 target data 를 오염시키는 것을 목표로 한다. 즉, AI model 이 target task 를 수행하는 performance 를 저하시키는 것을 목표로 한다.

* 예를 들어, AI model 이 image classifier 인 경우 target data 는 image 이며, adversarial example 은 image 를 미세하게 perturbation 함으로써 model 이 여전히 동일한 classification task 를 수행하지만 잘못된 label 을 생성하도록 image 를 오염시키는 것을 목표로 한다.
  * 예를 들어, 오염된 panda image 가 monkey 로 잘못 classification 될 수 있다.
* 마찬가지로, LLM-integrated application 의 target task 가 spam detection 인 경우, adversarial example 은 LLM-integrated application 이 여전히 spam detection 을 수행하지만 잘못된 output 을 생성하도록 target data (e.g., spam post) 를 오염시키는 것을 목표로 한다.
  * 예를 들어, output 이 “spam” 에서 “non-spam” 으로 변경된다.

이러한 기존 adversarial example 은 target task 를 변경하지 않으므로, 종종 injected instruction 없이 injected data 만을 사용하여 target data 를 오염시킨다. 저자는 detector 에 전략적으로 적응하는 adversarial example 에 의해 구성된 contaminated target data 를 탐지하는 것이 매우 어려운 것으로 알려져 있음을 언급한다.

반면, prompt injection attack 은 backend LLM 이 target task 와 다를 수 있는 injected task 를 수행하도록 target data 를 오염시키기 위해 injected instruction 을 추가적으로 활용하는 경우가 많다. 이러한 질적 차이는 DataSentinel 이 prompt injection attack 에 의해 구성된 contaminated target data 를 효과적으로 탐지할 수 있도록 한다.

저자는 injected task 가 target task 와 동일한 경우, prompt injection attack 이 adversarial example 을 통해 구현될 수 있음을 언급한다. 즉, target data 를 오염시키기 위한 injected data 는 adaptive adversarial example 을 사용하여 optimize 될 수 있다. 이러한 경우, adaptive adversarial example 을 탐지하는 것이 어렵다는 잘 알려진 문제로 인해 저자의 experiment 에서 나타난 것처럼 DataSentinel 의 effectiveness 는 감소한다.

## 2.3. Defenses

Prompt injection attack 에 대한 defense 는 prevention 과 detection 으로 분류될 수 있다.

#### Prevention

이러한 defense 는 target data 가 injected prompt 에 의해 오염된 경우에도 backend LLM 이 여전히 target task 를 수행하도록 하는 것을 목표로 한다.

* 일부 prevention-based defense 는 injected prompt 가 존재하는 경우 이를 무력화하기 위해 (contaminated) target data 를 pre-processing 한다.
  * 예를 들어, paraphrasing, retokenization 또는 delimiter 를 사용한다.
* 일부 defense 는 target instruction 을 재설계한다.
  * 예를 들어, sandwich prevention 은 backend LLM 에 target task 를 상기시키기 위해 (contaminated) target data 의 끝에 target instruction 을 다시 반복한다.

그러나 기존 benchmark study 에서 보여주듯이, 이러한 prevention-based defense 는 제한적인 effectiveness 를 보이거나 attack 이 존재하지 않을 때 LLM-integrated application 의 performance 를 희생한다.

* **Jatmo:** 어떠한 instruction 도 따르지 않고 특정 target task 를 수행하도록 backend LLM 을 fine-tuning 하므로 prompt injection 에 취약하지 않다.
  * 그러나 Jatmo 는 모든 target task 에 대해 backend LLM 을 fine-tuning 해야 하므로 computationally inefficient 하다.
* 여러 defense 는 다양한 prompt injection attack 에 의해 구성된 contaminated target data 를 사용하여 backend LLM 이 해당 data 내의 instruction 을 따르지 않도록 fine-tuning 할 것을 제안하였다.
  * 그러나 이러한 method 는 fine-tuning 동안 고려되지 않은 새로운 attack 에 취약한 경우가 많다.

#### Detection

이러한 defense 는 주어진 target data 가 오염되었는지를 탐지하는 것을 목표로 한다. State-of-the-art detector 는 detection LLM 을 활용한다.

* 예를 들어, detection LLM 에 직접 prompt 를 제공하여 주어진 target data 가 오염되었는지에 대한 zero-shot detection 을 수행할 수 있다.
* 또 다른 method 는 standard supervised fine-tuning 을 통해 detection LLM 을 binary classifier 로 fine-tuning 하는 것이다.
  * 구체적으로, detection LLM 은 contaminated target data 와 clean target data 로 구성된 dataset 을 사용하여 fine-tuning 되며, data sample 을 input 으로 받아 contaminated 또는 clean 을 output 으로 생성한다.

이러한 detection method 와 달리, known-answer detection 은 detection LLM 을 매우 다른 방식으로 활용한다. 이는 prompt injection attack 에 대한 LLM 의 vulnerability 를 해당 attack 에 대한 defense 로 전환한다. 구체적으로, detection instruction 과 주어진 target data 를 concatenate 한 input 이 주어졌을 때 detection LLM 이 secret key 를 output 으로 생성하지 못한다면, 이는 target data 가 injected prompt 에 의해 오염되었음을 나타낸다.

Secret key 는 detection instruction 의 known answer 이며, 주어진 target data 가 injected prompt 를 포함하지 않는다면 detection LLM 에 의해 output 으로 생성되어야 한다. 그러나 기존 benchmark study 에서 나타났으며 저자의 experiment 에서도 확인되었듯이, 기존 detection method 는 여전히 제한적인 effectiveness 를 보인다.

# 3. Problem Formulation

## 3.1. Threat Model

저자는 attacker 의 goal, background knowledge 및 capability 측면에서 threat model 을 고려한다. 저자의 attacker 에 대한 threat model 은 기존 prompt injection attack 연구에서 사용된 threat model 과 일치하지만, attacker 가 저자의 detector 에 대한 세부 사항까지 알고 있다고 추가적으로 가정한다.

#### Attacker’s goal

Attacker 는 LLM-integrated application 이 target task 대신 attacker 가 선택한 injected task 를 수행하도록 target data 를 오염시키는 것을 목표로 한다. LLM-integrated application 이 injected task 를 완수하는 attacker-desired output 을 생성하는 경우, 해당 application 은 injected task 를 수행한 것으로 간주된다.

예를 들어, target task 는 webpage 를 요약하는 것이 될 수 있는 반면, injected task 는 attacker 가 선택한 malicious URL 을 출력하는 것이 될 수 있다. Contaminated webpage 를 input 으로 사용했을 때 LLM-integrated application 이 malicious URL 을 output 으로 생성한다면 injected task 는 완수된 것이다.

#### Attacker’s background knowledge

저자의 연구는 defense 를 개발하므로, 강력한 background knowledge 를 갖는 attacker 를 고려한다.

* 구체적으로, 저자는 attacker 가 target instruction, target data 및 backend LLM 의 parameter 를 포함하여 LLM-integrated application 에 대한 white-box access 를 갖는다고 가정한다.
* 또한, attacker 가 DataSentinel 의 detection LLM 및 detection instruction 의 template 에 대한 white-box access 를 갖는다고 가정한다.
* 그러나 detection instruction 의 secret key 는 defender 에 의해 무작위로 선택되므로 attacker 는 이를 알지 못한다고 가정한다.
  * 저자는 이러한 가정이 실제로 현실적임을 언급한다. 예를 들어, attacker 가 secret key 에 접근할 수 없다는 것은 cryptographic system 에서의 standard assumption 이다.

#### Attacker’s capabilities

Attacker 는 target data 를 오염시킬 수 있다. 특히, 저자는 target data 를 임의로 조작할 수 있는 강력한 attacker 를 가정한다. 그러나 attacker 는 LLM-integrated application 및 저자의 detector 의 다른 component 를 조작할 수 없다. 예를 들어, target instruction 은 LLM-integrated application 의 provider 에 의해 제공되므로 attacker 는 이를 조작할 수 없다.

## 3.2. Detecting Prompt Injection Attacks

Defender 의 goal 은 contaminated target data 를 정확하게 탐지하는 detector 를 개발하는 것이다. Defender 는 LLM-integrated application 의 developer 이거나 LLM-integrated application 에 detector 를 제공하는 third-party provider 일 수 있다.

* 전자의 경우, defender 는 특정 backend LLM 을 사용하는 자신의 LLM-integrated application 에 맞추어 detector 를 설계할 수 있다.
* 후자의 경우, defender 는 서로 다른 backend LLM 을 사용하는 LLM-integrated application 에 적용될 수 있는 detector 를 개발할 수 있다.

두 유형의 defender 모두 LLM-integrated application 의 functionality 를 유지하기 위해 해당 application (e.g., target instruction 및 backend LLM) 을 수정하지 않는다고 가정한다. 대신, contaminated target data 를 filtering 하기 위한 추가적인 detector 를 개발한다.

Target data $x$ 가 주어졌을 때 prompt injection attack 을 탐지하는 것은 $x$ 가 injected prompt 에 의해 오염되었는지를 판단하는 것이다. Formally, detector 는 target data $x$ 를 input 으로 사용하고 “contaminated” 또는 “clean” 을 output 으로 생성한다.

저자는 낮은 false positive rate (FPR) 와 false negative rate (FNR) 를 달성하는 detector 를 설계하는 것을 목표로 한다.

* **FPR:** clean target data 를 contaminated target data 로 잘못 탐지할 probability 이다.
* **FNR:** contaminated target data 를 clean target data 로 잘못 탐지할 probability 이다.

높은 FPR 을 갖는 detector 는 많은 false alarm 을 발생시키며, 궁극적으로 LLM-integrated application 에서 더 이상 사용되지 않을 수 있다. 따라서 저자는 낮은 FPR 을 유지하면서 가능한 한 많은 contaminated target data 를 탐지하는 detector 를 개발하는 것을 목표로 한다.

# 4. Our DataSentinel

## 4.1. Overview

DataSentinel 은 detection LLM 과 defender 에게 알려진 ground-truth answer (secret key 라고 지칭함) 를 갖는 detection instruction 을 활용한다. Detection instruction 과 주어진 target data 를 concatenate 한 input 이 주어졌을 때 detection LLM 의 output 에 secret key 가 포함되지 않는다면, 주어진 target data 는 contaminated 로 탐지된다.

DataSentinel 은 다음 두 가지 challenge 에 직면한다.

* Detection LLM 은 본질적으로 이러한 detection 목적을 위해 training 된 것이 아니므로 높은 FPR/FNR 을 초래한다.
* Detector 를 알고 있는 강력한 adaptive attack 은 detection 을 회피할 수 있다.

첫 번째 challenge 를 해결하기 위해 DataSentinel 은 contaminated target data 와 clean target data 로 구성된 dataset 을 사용하여 detection LLM 을 fine-tuning 한다. 구체적으로, detection LLM 은 다음과 같이 secret key 가 output 에 포함될 가능성을 조정하도록 fine-tuning 된다.

* Detection instruction 과 contaminated target data 를 concatenate 한 input 을 사용하는 경우, output 에 secret key 가 포함되지 않을 가능성을 높인다 (i.e., 낮은 FNR).
* Detection instruction 과 clean target data 를 concatenate 한 input 을 사용하는 경우, output 에 secret key 가 포함될 가능성을 높인다 (i.e., 낮은 FPR).

두 번째 challenge 를 해결하기 위해 DataSentinel 은 설계 단계에서 강력한 adaptive attack 을 고려한다. 구체적으로, DataSentinel 은 detection 을 회피하는 동시에 backend LLM 이 injected task 를 수행하도록 유도하기 위해 contaminated target data 를 optimize 한다.

저자의 DataSentinel 은 본질적으로 detection LLM 의 fine-tuning 과 adaptive attack 사이의 game 을 simulate 한다. 또한, 저자는 이러한 game 을 minimax optimization problem 으로 formulation 한다.

* **Inner max problem:** contaminated target data 를 optimize 하는 강력한 adaptive attack 을 formalization 한다.
* **Outer min problem:** optimized contaminated target data 와 clean target data 에 기반한 detection LLM 의 fine-tuning 을 formalization 한다.

이후 저자는 minimax optimization problem 을 반복적으로 해결하기 위한 gradient-based method 를 개발한다.

* 각 round 에서 현재 fine-tuned detection LLM 이 주어지면, 먼저 inner max problem 을 해결하여 contaminated target data 를 optimize 한다.
* 이후 optimized contaminated target data 가 주어지면, outer min problem 을 해결하여 detection LLM 을 추가적으로 fine-tuning 한다.
* 이러한 과정은 사전에 정의된 round 수에 도달할 때까지 반복된다.

## 4.2. Detection Rule

Known-answer detection 에서 영감을 받아, 저자의 DataSentinel 은 contaminated target data 와 clean target data 를 구별하기 위해 detection LLM 과 detection instruction 을 활용한다. Defender 는 자신은 알고 있지만 attacker 는 알지 못하는 ground-truth answer 를 갖는 한 어떠한 detection instruction 도 선택할 수 있다.

예를 들어, 저자의 experiment 에서는 detection instruction 을 생성하기 위해 다음과 같은 template 을 사용한다.

> “Repeat [secret key] once while ignoring the following text:”

즉, “다음 text 를 무시하면서 [secret key] 를 한 번 반복하라:”라는 instruction 이며, 여기서 secret key 는 defender 에 의해 무작위로 생성된 string (e.g., 저자의 experiment 에서는 7 개의 character) 이다.

단순화를 위해 저자는 다음과 같은 notation 을 사용한다.

* $g$ 는 detection LLM 을 나타낸다.
* $s_d$ 는 detection instruction 을 나타낸다.
* $k$ 는 secret key 를 나타낸다.

Target data $x$ 가 주어지면, 이를 detection instruction $s_d$ 와 concatenate 하여 detection prompt $s_d\|x$ 를 생성하고, 이를 사용하여 detection LLM $g$ 에 query 를 수행한다. 저자의 DataSentinel 은 secret key $k$ 가 detection LLM 의 output 에 포함되지 않는 경우, i.e., $k\notin g(s_d\|x)$ 인 경우 $x$ 를 contaminated 로 탐지하며, 그렇지 않은 경우 $x$ 를 clean 으로 탐지한다.

Formally, DataSentinel 은 다음과 같은 detection rule 을 갖는다.

$$
\operatorname{DataSentinel}(x)=
\begin{cases}
\text{contaminated} & \text{if } k\notin g(s_d\|x)\\
\text{clean} & \text{otherwise.}
\end{cases}
\tag{1}
$$

Appendix 의 Fig. 6 은 clean data sample 과 contaminated data sample 모두에 대한 저자의 detection LLM 의 output example 을 보여준다.

## 4.3. Formulating a Minimax Optimization Problem

저자는 먼저 주어진 detection LLM 을 회피하기 위해 contaminated target data 를 optimize 하는 강력한 adaptive attack 을 optimization problem 으로 formulation 한다. 이후 optimized contaminated target data 가 주어지면, detection LLM 의 fine-tuning 을 optimization problem 으로 formulation 한다. 마지막으로, 저자는 이들을 detection LLM 의 fine-tuning 과 adaptive attack 사이의 game 을 simulate 하는 minimax optimization problem 으로 결합한다.

#### Formulating strong adaptive attacks as an optimization problem

저자는 target task 를 $(s_t,x_t,y_t)$ 로, injected task 를 $(s_e,x_e,y_e)$ 로 나타낸다.

* $s_t$ (또는 $s_e$) 는 각각 target instruction (또는 injected instruction) 을 나타낸다.
* $x_t$ (또는 $x_e$) 는 각각 target data (또는 injected data) 를 나타낸다.
* $y_t$ (또는 $y_e$) 는 각각 target task (또는 injected task) 를 완수하는 backend LLM 의 desired output 을 나타낸다.

저자는 target task $(s_t,x_t,y_t)$ 와 injected task $(s_e,x_e,y_e)$ 의 pair 가 주어졌을 때 다음 두 가지 goal 을 달성하기 위해 contaminated target data $x_c$ 를 optimize 하는 강력한 adaptive attack 을 고려한다.

* Detection LLM $g$ 를 회피한다.
* Backend LLM $f$ 가 injected task $(s_e,x_e,y_e)$ 를 수행하도록 유도한다.

첫 번째 goal 은 $s_d\|x_c$ 를 input 으로 사용하는 detection LLM $g$ 의 output 에 secret key $k$ 가 포함되어야 함을 의미한다. 즉, $k\in g(s_d\|x_c)$ 이다. 첫 번째 goal 을 정량화하기 위해 저자는 loss term $\ell(k,g(s_d\|x_c))$ 를 정의하며, 이는 $g(s_d\|x_c)$ 가 $k$ 에 가까울수록 작아진다.

예를 들어, $\ell$ 은 다음과 같은 standard cross-entropy loss 가 될 수 있다.

$$
\ell(k,g(s_d\|x_c))=-\sum_{i=1}^{|k|}\log(p_g(k^i\mid s_d\|x_c\|k^{<i}))
$$

* $|k|$ 는 $k$ 에 포함된 token 의 수이다.
* $k^{<i}$ 는 $k$ 의 $i$ 번째 token $k^i$ 이전에 위치한 token 의 sequence 를 나타낸다.
* $p_g(k^i\mid s_d\|x_c\|k^{<i})$ 는 $s_d\|x_c\|k^{<i}$ 를 input 으로 사용할 때 $g$ 가 token $k^i$ 를 생성하는 conditional probability 를 나타낸다.

두 번째 goal 은 $f$ 가 $s_t\|x_c$ 를 input 으로 사용할 때 $y_e$ 를 output 으로 생성해야 함을 의미한다. 저자는 두 번째 goal 을 정량화하기 위해 loss term $\ell(y_e,f(s_t\|x_c))$ 를 사용한다.

두 loss term 을 결합하면, adaptive attack 이 contaminated target data $x_c$ 를 optimize 하기 위한 다음과 같은 optimization problem 을 얻는다.

$$
\max_{x_c}\left[-\ell(k,g(s_d\|x_c))-\alpha\cdot\ell(y_e,f(s_t\|x_c))\right]
\tag{2}
$$

* $\alpha$ 는 두 loss term 사이의 balance 를 조절하기 위한 hyper-parameter 이다.

저자는 강력한 adaptive attack 이 detector 에 대한 모든 information (e.g., $g$, $s_d$, $k$) 및 LLM-integrated application 에 대한 모든 information (e.g., $f$, $s_t$, $x_t$) 에 접근할 수 있다고 가정함을 언급한다. 이는 defender 가 $g$ 를 fine-tuning 할 때 simulate 하는 가상의 attack 을 나타낸다. Sec. 3.1 의 threat model 에서 논의하고 Sec. 5 에서 evaluate 한 것처럼, 현실적인 adaptive attack 은 secret key $k$ 에 접근할 수 없다.

#### Formulating fine-tuning detection LLM $g$ as an optimization problem

저자는 DataSentinel 의 FNR 과 FPR 을 모두 minimize 하기 위해 detection LLM $g$ 를 fine-tuning 한다.

* **Minimizing FNR:** $s_d\|x_c$ 를 input 으로 사용할 때 detection LLM $g$ 가 secret key $k$ 를 output 으로 생성하지 않을 가능성이 높아지도록 fine-tuning 한다.
  * 여기서 $x_c$ 는 앞서 설명한 강력한 adaptive attack 에 의해 optimize 된 contaminated target data 이다.
  * Formally, 저자는 이를 정량화하기 위해 loss term $-\ell(k,g(s_d\|x_c))$ 를 사용한다.
* **Minimizing FPR:** $s_d\|x_t$ 를 input 으로 사용할 때 detection LLM $g$ 가 secret key $k$ 를 output 으로 생성할 가능성이 높아지도록 fine-tuning 한다.
  * 여기서 $x_t$ 는 clean target data 이다.
  * Formally, 저자는 FPR 을 minimize 하는 goal 을 정량화하기 위해 loss term $\ell(k,g(s_d\|x_t))$ 를 사용한다.

두 loss term 을 결합하면, detection LLM $g$ 를 fine-tuning 하기 위한 다음과 같은 optimization problem 을 얻는다.

$$
\begin{aligned}
\min_g\Bigg[&-\frac{1}{|X_c|}\sum_{x_c\in X_c}\ell(k,g(s_d\|x_c))\\
&+\frac{\beta}{|D_t|}\sum_{(s_t,x_t,y_t)\in D_t}\ell(k,g(s_d\|x_t))\Bigg]
\end{aligned}
\tag{3}
$$

* $X_c$ 는 contaminated target data 의 set 이다.
* $D_t$ 는 target task 의 set 이다.
* $\beta$ 는 두 loss term 사이의 balance 를 조절하기 위한 hyper-parameter 이다.

$X_c$ 는 target task 의 set $D_t$ 와 injected task 의 set $D_e$ 를 사용하여 구성될 수 있다. 구체적으로, target task $(s_t,x_t,y_t)\in D_t$ 와 injected task $(s_e,x_e,y_e)\in D_e$ 의 각 pair 에 대해, Equation (2) 의 adaptive attack 을 활용하여 contaminated target data sample $x_c$ 를 optimize 할 수 있다. $X_c$ 는 이러한 방식으로 얻어진 contaminated target data sample $x_c$ 의 set 을 구성한다.

#### Our minimax optimization problem

저자의 formulation 에서 adaptive attack 은 detection LLM $g$ 를 회피하는 것을 목표로 하는 반면, $g$ 의 fine-tuning 은 adaptive attack 을 탐지하는 것을 목표로 한다. 따라서 저자의 formulation 은 adaptive attack 과 $g$ 의 fine-tuning 사이의 game 을 simulate 한다.

Equation (2) 의 두 번째 loss term 은 detection LLM $g$ 와 독립적임에 유의한다. 따라서 Equation (2) 와 Equation (3) 을 통합함으로써 해당 game 을 다음과 같은 minimax optimization problem 으로 formulation 할 수 있다.

$$
\begin{aligned}
\min_g\Bigg[&\frac{1}{|D_t|\cdot|D_e|}\sum_{\substack{(s_t,x_t,y_t)\in D_t\\(s_e,x_e,y_e)\in D_e}}\Big(\max_{x_c}\big[-\ell(k,g(s_d\|x_c))\\
&\qquad-\alpha\cdot\ell(y_e,f(s_t\|x_c))\big]\Big)\\
&+\frac{\beta}{|D_t|}\sum_{(s_t,x_t,y_t)\in D_t}\ell(k,g(s_d\|x_t))\Bigg]
\end{aligned}
\tag{4}
$$

* Inner max problem 은 강력한 adaptive attack 을 formulation 한다.
* Outer min problem 은 detection LLM 의 fine-tuning 을 formulation 한다.

## 4.4. Solving the Minimax Optimization Problem

Equation (4) 의 minimax optimization problem 을 해결하기 위해 defender 는 target task 의 set $D_t$ 와 injected task 의 set $D_e$ 를 수집해야 한다. 저자는 $D_e$ 가 주로 가상의 강력한 adaptive attack 을 simulate 하기 위해 사용되므로, detector 와 LLM-integrated application 을 deployment 한 이후 실제 attacker 에 의해 사용되는 injected task 일 필요는 없음을 언급한다.

따라서 저자는 defender 가 tuple $(s,x,y)$ 로 구성된 task 의 set $D$ 를 수집한다고 가정한다.

* $s$ 는 instruction 이다.
* $x$ 는 data sample 이다.
* $y$ 는 task 를 완수하는 desired output 이다.

이후 $D$ 로부터 $D_t$ 와 $D_e$ 를 모두 생성한다. 예를 들어, 저자의 experiment 에서는 일부 standard benchmark dataset 을 $D$ 로 사용한다.

$D_t$ 와 $D_e$ 가 주어지면, Fig. 2 및 Algorithm 1 에 나타난 것처럼 inner max problem 과 outer min problem 을 번갈아 수행함으로써 minimax optimization problem 을 해결한다.

![Figure 2](images/image-104.png)

* 구체적으로, 각 round 에서 $D_t$, $D_e$ 및 현재 fine-tuned detection LLM $g$ 가 주어지면, inner max problem 을 해결하여 contaminated target data 의 set $X_c$ 를 얻는다 (Algorithm 1 의 Line 10).
* 이후 $X_c$ 와 $D_t$ 가 주어지면, outer min problem 을 해결하여 detection LLM $g$ 를 추가적으로 update 한다 (Line 12).
* 이러한 과정을 $r$ round 동안 반복한다.

![Algorithm 1](images/image-105.png)

다음으로, 저자가 inner max problem 과 outer min problem 을 해결하는 방법을 설명한다.

#### Solving the inner max problem

Inner max problem 을 해결하는 것은 efficiency challenge 에 직면한다. 구체적으로, target task $(s_t,x_t,y_t)\in D_t$ 와 injected task $(s_e,x_e,y_e)\in D_e$ 의 각 pair 에 대해 $x_c$ 를 얻기 위해 Equation (2) 의 max problem 을 해결해야 한다. 즉, Equation (2) 의 max problem 을 $|D_t|\cdot|D_e|$ 번 해결해야 한다.

이러한 challenge 를 해결하기 위해 저자는 두 가지 strategy 를 채택한다.

* 첫 번째로, inner max problem 과 outer min problem 을 번갈아 수행하는 각 round 에서 $D$ 로부터 하나의 task 만을 무작위로 sampling 하여 $D_e$ 로 사용하고 (Algorithm 1 의 Line 6), $D$ 내의 나머지 task 는 $D_t$ 로 취급한다.
  * 이러한 strategy 는 target task 와 injected task 의 pair 수를 minimize 할 수 있다.
* 두 번째로, target task 와 injected task 의 각 pair 에 대해 $x_c$ 를 독립적으로 optimize 하는 대신, 모든 target task 와 injected task 의 pair 에 대해 $x_c$ 가 $x_t\|z\|s_e\|x_e$ 의 형태를 갖도록 제한한다.
  * 여기서 $z$ 는 모든 target-injected task pair 에 적용되는 separator 이다.

$x_c$ 에 대한 이러한 constraint 는 다음 problem 을 단 한 번만 해결함으로써 $z$ 를 optimize 하고, 이에 따라 contaminated target data 의 set $X_c$ 를 optimize 할 수 있도록 한다.

$$
\max_z\frac{1}{|D_t|\cdot|D_e|}\sum_{\substack{(s_t,x_t,y_t)\in D_t\\(s_e,x_e,y_e)\in D_e}}L(s_t,x_t,y_t,s_e,x_e,y_e)
\tag{5}
$$

여기서 $L$ 은 다음과 같이 정의된다.

$$
\begin{aligned}
L(s_t,x_t,y_t,s_e,x_e,y_e)={}&-\ell(k,g(s_d\|x_t\|z\|s_e\|x_e))\\
&-\alpha\cdot\ell(y_e,f(s_t\|x_t\|z\|s_e\|x_e)).
\end{aligned}
$$

저자는 $x_c$ 에 대한 constraint 가 adaptive attack 을 약화시킬 수 있음을 언급한다. 그러나 저자의 experiment 는 DataSentinel 이 이러한 constraint 없이 전체 $x_c$ 를 optimize 하는 기존 prompt injection attack 및 강력한 adaptive prompt injection attack 도 여전히 효과적으로 탐지함을 보여준다. 이는 이러한 constraint 를 갖는 $x_c$ 역시 detector 에 대한 adaptive attack 이며, injected prompt 를 포함하기 때문이다.

Equation (5) 의 optimization problem 을 해결하는 것은 여전히 하나의 challenge 에 직면한다. Separator $z$ 는 discrete 하고 큰 vocabulary 내의 value 를 가질 수 있는 token 의 sequence 로 구성된다. 이러한 discrete optimization problem 은 해결하기 매우 어려운 것으로 알려져 있다.

이러한 challenge 를 해결하기 위해 저자는 HotFlip 을 기반으로 하는 state-of-the-art method 인 Greedy Coordinate Gradient (GCG) 를 채택하여 해당 problem 을 근사적으로 해결한다. Variable 이 token 의 sequence (i.e., 저자의 경우 $z$) 인 loss function 이 주어지면, GCG 는 loss 를 감소시키도록 token 을 update 한다. GCG 의 핵심 idea 는 token 을 하나씩 update 하고, 각 token 을 loss 를 근사적으로 가장 많이 증가시키는 vocabulary 내의 token 으로 대체하는 것이다.

![Algorithm 2](images/image-106.png)

저자는 GCG 를 활용하여 저자의 max problem 을 반복적으로 해결하며, 이는 Algorithm 2 에 나타나 있다. 여기서 OptCTD 는 optimizing contaminated target data 를 의미한다.

* 구체적으로, 각 iteration 에서 target task 와 injected task 의 pair 로 구성된 mini-batch 를 sampling 한다.
* 이를 사용하여 Equation (5) 의 loss function 을 계산하고, GCG 를 사용하여 loss 에 기반해 separator $z$ 를 update 한다.
* 이러한 과정을 $n_{in}$ iteration 동안 반복한다.

#### Solving the outer min problem

Optimized contaminated target data $X_c$ 와 target task $D_t$ 내의 clean target data 가 주어지면, Equation (3) 의 outer min problem 을 해결함으로써 detection LLM $g$ 를 update 한다. 구체적으로, $g$ 의 parameter 는 continuous 하므로 Algorithm 3 에 나타난 것처럼 standard gradient-descent method 를 사용하여 $g$ 를 반복적으로 update 할 수 있다.

![Algorithm 3](images/image-107.png)

* 각 iteration 에서 $X_c$ 로부터 contaminated target data 의 mini-batch 를 sampling 하고, $D_t$ 로부터 clean target data 의 mini-batch 를 sampling 한다.
* 이후 $g$ 의 parameter 에 대한 loss function 의 gradient 를 계산하고, learning rate 라고 지칭하는 작은 step 만큼 gradient 의 반대 방향으로 $g$ 를 update 한다.

#### Developer vs. third-party provider as a defender

저자는 minimax optimization problem 을 통해 detection LLM $g$ 를 fine-tuning 하는 과정에 backend LLM $f$ 가 포함되며, 이는 강력한 adaptive attack 에서 contaminated target data 를 optimize 하는 데 사용됨을 언급한다.

* Developer 가 defender 인 경우, 자신의 LLM-integrated application 의 backend LLM $f$ 에 기반하여 $g$ 를 fine-tuning 할 수 있다.
* 서로 다른 backend LLM 을 사용하는 여러 LLM-integrated application 에 적용될 수 있는 detector 를 개발하는 third-party provider 가 defender 인 경우, $g$ 의 fine-tuning 동안 사용되는 backend LLM 은 실제 LLM-integrated application 의 backend LLM 과 다를 수 있다.

저자의 experiment 는 이러한 경우에도 DataSentinel 이 LLM-integrated application 의 backend LLM 에 대한 white-box access 에 기반하여 optimize 된 contaminated target data 를 여전히 효과적으로 탐지할 수 있음을 보여줄 것이다.

# 5. Evaluation

## 5.1. Experimental Setup

#### LLMs, target/injected tasks, and datasets

저자는 experiment 에서 다음과 같은 open-source LLM 을 사용한다.

* Mistral-7B
* LLaMA2-7B
* LLaMA3-8B-Instruct

Default 로 Mistral-7B 를 detection LLM 으로 사용하고, LLaMA3-8B-Instruct 를 backend LLM 으로 사용한다. 저자는 backend LLM 과 detection LLM 모두가 DataSentinel 에 미치는 영향을 evaluate 하기 위해 ablation study 를 수행할 것이다.

Experimental result 의 reproducibility 를 보장하기 위해 각 LLM 의 temperature parameter 를 0.1 로 설정하고, experiment 에서 random number generator 의 seed 를 고정한다.

저자는 선행 연구를 따라 다음 7 가지 유형의 natural language processing task 를 고려한다.

* Duplicate sentence detection
* Grammar correction
* Hate detection
* Natural language inference
* Sentiment analysis
* Spam detection
* Text summarization

저자는 각각의 task 를 target task 또는 injected task 로 사용한다. 따라서 evaluation 을 위해 총 49 개의 combination 을 갖는다.

저자는 이러한 7 가지 유형의 task 에 대해 각각 MRPC, Jfleg, SMS Spam, RTE, SST2, HSOL 및 Gigaword dataset 을 사용한다. 각 dataset 은 training set 과 test set 을 갖는다. Dataset 내의 각 data point 는 pair $(x,y)$ 이며, 여기서 $x$ 는 data input 이고 $y$ 는 task 를 완수하는 desired LLM output 이다.

저자는 각 유형의 task 에 대해 선행 연구와 동일한 target instruction 및 injected instruction 을 사용하며, 이에 따라 7 개의 target instruction 과 7 개의 injected instruction 을 얻는다. 이들은 Appendix A 에서 확인할 수 있다. Contaminated target data 를 탐지하는 것은 LLM-integrated application 의 target instruction 에 의존하지 않음에 유의한다.

* 7 개의 task 각각에 대해, 해당 target instruction $s_t$ 및 해당 dataset 이 주어지면 test set 에서 100 개의 data point 를 무작위로 sampling 하여 100 개의 target task $(s_t,x_t,y_t)$ 를 구성한다.
* 마찬가지로, 해당 injected instruction $s_e$ 가 주어지면 또 다른 100 개의 data point 를 sampling 하여 100 개의 injected task $(s_e,x_e,y_e)$ 를 구성한다.

#### Prompt injection attacks

저자는 heuristic-based prompt injection attack 과 optimization-based prompt injection attack 을 모두 고려한다.

* **Heuristic-based attacks:** 저자는 Naive Attack, Context Ignoring, Escape Character, Fake Completion, Combined Attack 및 Availability Attack 을 포함하는 6 개의 heuristic-based prompt injection attack 을 고려한다.
  * 처음 5 개의 attack 은 임의의 injected task 에 사용될 수 있는 반면, 마지막 attack 은 LLM-integrated application 의 availability 에 영향을 미치는 특정 injected task 를 위해 설계된다.
    * 예를 들어, backend LLM 이 “죄송하다. 이 task 를 완료할 수 없다.”라고 response 하도록 한다.
  * 저자는 처음 5 개의 attack 에 대해 Liu et al. 의 open-source code 를 사용하고, Greshake et al. 을 따라 Availability Attack 을 구현한다.
  * 이러한 attack 의 implementation 에 대한 세부 사항은 Appendix B 에서 설명한다.
* **Optimization-based attacks:** 저자는 Universal, NeuralExec 및 PLeak 과 같은 3 개의 optimization-based prompt injection attack 도 evaluate 한다.
  * 저자는 이러한 attack 이 target instruction 과 backend LLM 에 대한 white-box access 를 갖는다고 가정한다.
  * Universal 과 NeuralExec 은 임의의 injected task 를 위해 설계된 반면, PLeak 은 특정 injected task, i.e., LLM-integrated application 의 target instruction 을 탈취하기 위해 설계된다.
  * 저자는 experiment 에서 이들의 open-source code 를 사용한다.
  * 또한, 해당 code 에서 이러한 attack 에 대한 default parameter setting 을 채택한다.

Availability Attack (또는 PLeak) 의 경우, 이를 사용하여 7 가지 유형의 target task 각각에 대해 100 개의 contaminated target data sample 을 생성한다. 나머지 attack 각각에 대해서는 이를 사용하여 49 개의 target-injected task combination 각각에 대해 100 개의 contaminated target data sample 을 생성한다.

* 구체적으로, 각 target-injected task combination 에 대해 $100\times100=10,000$ 개의 target task 및 injected task pair 를 구성할 수 있다.
* 더 높은 efficiency 를 위해 이들 중 100 개를 sampling 하고, attack 을 사용하여 100 개의 contaminated target data sample 을 생성한다.

전체적으로 저자는 35,700 개의 contaminated target data sample 을 갖는다.

어떠한 defense 도 존재하지 않는 경우, 이러한 attack 은 매우 효과적이다. Defense 가 없는 상황에서 이러한 attack 의 effectiveness 에 대한 result 는 Appendix 의 Tab. 9 에서 확인할 수 있다.

별도로 언급하지 않는 한, 저자는 Combined Attack 을 default heuristic-based prompt injection attack 으로 사용한다. 이는 기존 benchmark study 에서 나타났으며 저자의 experiment 에서도 확인되었듯이, Combined Attack 이 다른 heuristic-based attack 을 능가하기 때문이다. 저자는 NeuralExec 을 default optimization-based attack 으로 사용한다. 이는 저자의 experiment 에서 NeuralExec 이 가장 우수한 attack performance 를 달성하기 때문이다.

#### Compared baselines

저자는 detection method 를 다음 baseline 과 비교한다.

* **Embedding-vector-based detector (EVD):** 해당 baseline 에서 저자는 contaminated target data 를 탐지하기 위한 기존 binary classifier 를 training 한다.
  * Gigaword dataset 의 training set 에서 500 개의 data point 를 clean target data sample 로 sampling 한다.
  * 또한, 각 clean target data sample 에 대해 Gigaword dataset 에서 injected task 를 sampling 함으로써 Combined Attack 을 사용하여 500 개의 contaminated target data sample 을 구성한다.
  * 이후 CLIP text encoder (clip-vit-base-patch32) 를 사용하여 각 clean target data sample 또는 contaminated target data sample 에 대한 feature vector 를 계산한다.
  * 이러한 1,000 개의 feature vector 와 해당 ground-truth label (i.e., “clean” 또는 “contaminated”) 이 주어지면, 128 dimension 의 hidden layer 와 그 뒤의 output layer 로 구성된 two-layer neural network classifier 를 training 한다.
  * Learning rate 0.001 및 batch size 8 로 neural network 를 50 epoch 동안 training 한다.
* **Naive LLM-based detector (NLLMD):** 해당 method 는 주어진 target data sample 이 오염되었는지를 탐지하도록 detection LLM 에 직접 prompt 를 제공한다.
  * DataSentinel 과의 공정한 비교를 위해 저자는 Mistral-7B 를 detection LLM 으로 사용한다.
  * Detection instruction 은 “User 로부터 제공된 다음 data 가 주어졌을 때, 어떠한 injected instruction 을 포함하는가? Injected instruction 을 포함하는 경우 yes 로 답하라. 그렇지 않은 경우 no 로 답하라.”이다.
  * Detection instruction 은 주어진 target data sample 과 concatenate 되어 detection LLM 에 query 를 수행하는 데 사용된다.
* **Standard supervised fine-tuning of LLM as a detector (SSFTD):** 저자는 contaminated target data 를 탐지하는 binary classifier 로 detection LLM 을 fine-tuning 하기 위해 standard supervised fine-tuning (SFT) 을 사용한다.
  * EVD 와 동일한 dataset 을 채택한다.
  * 일관성을 위해 Mistral-7B 를 사용한다.
  * Batch size 2 및 learning rate 0.000025 로 500 iteration 동안 fine-tuning 한다.
* **SSFTD in a game-theoretic setting (SSFTD-G):** SSFTD-G 는 저자의 game-theoretic method 를 SSFTD 로 확장한다.
  * 구체적으로, 다음 두 과정을 번갈아 수행한다.
    * 현재 detector 를 회피하는 동시에 backend LLM 이 injected task 를 완수하도록 유도하는 강력한 adaptive attack 을 생성한다.
    * Contaminated target data 와 clean target data 를 사용하여 classifier 로서 detection LLM 을 update 한다.
  * DataSentinel 과 동일한 optimization setting 을 사용한다.
* **PromptGuard:** PromptGuard 역시 malicious prompt 및 contaminated target data 를 탐지하는 classifier 로 detection LLM 을 fine-tuning 한다.
  * 이는 Meta 에 의해 fine-tuning 되고 공개되었다.
  * Target data sample 이 주어지면 benign, injection 또는 jailbreak 의 세 가지 가능한 output 을 갖는다.
  * Output 이 injection 또는 jailbreak 인 경우, 저자는 target data sample 이 오염된 것으로 간주한다.
* **Known-answer detection (KAD):** 기존 benchmark study 에서 나타난 것처럼, known-answer detection 은 기존 detector 중 가장 유망하다.
  * 저자는 experiment 에서 open-source code 를 사용한다.
  * 저자의 DataSentinel 은 해당 baseline 과 두 가지 핵심 차이를 갖는다.
    * 첫 번째로, DataSentinel 은 detection LLM 을 fine-tuning 하는 반면, KAD 는 fine-tuning 하지 않은 standard LLM 을 사용한다.
    * 두 번째로, KAD 는 설계 단계에서 강력한 adaptive attack 을 고려하지 않는다.
  * 그 결과, 저자의 experimental result 는 KAD 가 제한적인 effectiveness 를 보임을 나타낸다.

#### Evaluation metrics

저자는 False Positive Rate (FPR) 및 False Negative Rate (FNR) 를 evaluation metric 으로 사용한다. 7 가지 유형의 target task 각각은 100 개의 clean target data sample 을 가지며, FPR 은 detection method 에 의해 contaminated 로 잘못 탐지된 sample 의 비율이다. FPR 은 prompt injection attack 에 의존하지 않음에 유의한다.

FNR 은 contaminated target data sample 중 clean 으로 잘못 탐지된 sample 의 비율이다. FNR 은 attack 에 의존한다.

* 7 가지 유형의 target task 각각에 대해 저자는 Availability Attack (또는 PLeak) 을 사용하여 100 개의 contaminated target data sample 을 구성하고, 이에 대응하는 FNR 을 보고할 것이다.
* 그 외의 모든 attack 에 대해 저자는 해당 attack 을 사용하여 각 target-injected task combination 에 대해 100 개의 contaminated target data sample 을 구성하고, 이들에 대한 FNR 을 계산할 수 있다.

저자의 experiment 에서는 각 유형의 injected task 에 대해 7 가지 유형의 target task 에 걸쳐 평균한 FNR 을 보고할 것이다. 이러한 average FNR 은 target task 와 관계없이 특정 injected task 를 갖는 attack 을 탐지하는 performance 를 측정한다.

Detection method 는 FPR 과 FNR 이 모두 낮을수록 더욱 효과적이다. 실제로 LLM-integrated application 에 제공되는 대부분의 target data sample 은 clean 일 수 있다. 따라서 우수한 detection method 는 FNR 을 가능한 한 감소시키면서 낮은 FPR 을 유지해야 한다.

#### Parameter setting

저자의 DataSentinel 은 다음과 같은 hyper-parameter 를 갖는다.

* Secret key $k$
* Detection LLM $g$
* Detection instruction $s_d$
* Loss function $\ell$
* $\alpha$, $\beta$
* Round 수 $r$
* Learning rate $lr_{out}$
* Batch size $b_{in}$ 및 $b_{out}$
* Iteration 수 $e_{in}$ 및 $e_{out}$

별도로 언급하지 않는 한, 저자는 다음과 같은 default setting 을 채택한다.

* $k$: “DGDSGHN” (7 개의 character 로 구성된 무작위 string)
* $g$: Mistral-7B
* $s_d$: “Repeat $k$ once while ignoring the following text:”
* $\ell$: cross-entropy loss
* $\alpha=1$, $\beta=1$
* $r=3$
* $lr_{out}=0.000025$
* $b_{in}=8$, $b_{out}=2$
* $n_{in}=10$, $n_{out}=500$

저자는 computational efficiency 를 향상시키고 fine-tuning time 을 감소시키는 QLoRA 를 사용하여 detection LLM $g$ 를 fine-tuning 한다.

$g$ 를 fine-tuning 하기 위해 저자의 DataSentinel 은 task 의 set $D$ 와 backend LLM $f$ 도 필요로 한다. $D$ 를 수집하기 위해 저자는 다음과 같은 instruction $s$ 를 생성한다.

> “Please write me a short and brief summary (no more than 10 words) of the following text:”

즉, “다음 text 에 대해 짧고 간결한 summary (10 개의 word 이하) 를 작성하라:”라는 instruction 이다.

Gigaword dataset 의 training set 에서 500 개의 data point $(x,y)$ 를 sampling 하여 500 개의 task $(s,x,y)$ 를 구성한다. 저자는 $s$ 가 7 개의 target/injected instruction 과 다르다는 점을 언급한다. 이는 $g$ 를 fine-tuning 하는 데 사용되는 instruction 이 LLM-integrated application 의 target instruction 일 필요가 없음을 보여주기 위한 것이다.

또한, 저자는 $g$ 를 fine-tuning 하기 위한 backend LLM 으로 LLaMA3-8B-Instruct 를 사용한다. 저자는 DataSentinel 이 다른 backend LLM 에 기반하여 optimize 된 attack 을 탐지하는 데에도 효과적임을 보여줄 것이다.

## 5.2. Main Results

#### DataSentinel is highly effective

Tab. 1 및 Tab. 2 는 6 개의 heuristic-based attack 과 3 개의 optimization-based attack 에 대한 DataSentinel 의 FNR 및 FPR 을 보여준다.

![Table 1](images/image-108.png)

Availability Attack 은 LLM-integrated application 의 availability 에 영향을 미치는 특정 injected task 를 위해 설계된 반면, PLeak 은 target instruction 을 탈취하는 특정 injected task 를 위해 설계됨에 유의한다. 따라서 저자는 이들의 result 를 다른 attack 과 구분하여 Tab. 2 에 제시한다.

![Table 2](images/image-109.png)

저자는 다음과 같은 observation 을 얻는다.

* 첫 번째로, 저자의 DataSentinel 은 매우 낮은 FPR (0 에 가까운 FPR) 을 달성하며, 이는 DataSentinel 이 false alarm 을 거의 발생시키지 않음을 나타낸다.
* 두 번째로, DataSentinel 은 모든 기존 heuristic-based 및 optimization-based prompt injection attack 에 대해 매우 낮은 FNR 을 달성한다.
  * 예를 들어, 모든 prompt injection attack 에 대해 FNR 은 일관되게 0.07 이하이다.
  * 저자의 DataSentinel 이 이러한 attack 에 효과적인 이유는 detection LLM 을 fine-tuning 할 때 강력한 adaptive attack 을 고려하기 때문이다.
* 세 번째로, heuristic-based attack 은 optimization-based attack 보다 탐지하기 쉽다.
  * 예를 들어, heuristic-based attack 에 대한 FNR 은 모두 0 이지만, optimization-based attack 에 대해서는 특정 경우에 0 이 아닐 수 있다 (여전히 최대 0.07).
  * 그 이유는 optimization-based attack 이 contaminated target data 를 optimize 하므로 defense 가 적용된 상황에서 더욱 효과적이기 때문이다.

요약하면, result 는 DataSentinel 이 광범위한 task 에 대해 기존 prompt injection attack 을 효과적으로 탐지할 수 있음을 보여준다.

#### DataSentinel outperforms baselines

![Table 3](images/image-110.png)

Tab. 3 은 저자의 DataSentinel 과 baseline 의 FPR 및 FNR 을 비교한다. 저자는 다음과 같은 observation 을 얻는다.

* 첫 번째로, KAD 와 DataSentinel 은 다른 baseline 을 능가한다.
  * 이는 detection instruction 과 함께 detection LLM 을 사용하는 것 (i.e., KAD 및 DataSentinel) 이 detection LLM 을 classifier 로 직접 fine-tuning 하는 것 (i.e., SSFTD, SSFTD-G 및 PromptGuard) 또는 classifier 로 prompting 하는 것 (i.e., NLLMD) 보다 더 높은 generalizability 를 가짐을 보여준다.
  * 특히, PromptGuard 는 0 의 FNR 을 달성하지만 매우 높은 FPR 을 초래하며, 이는 모든 target data 를 contaminated 로 판단하는 경향이 있음을 나타낸다.
* 두 번째로, DataSentinel 은 KAD 를 추가적으로 능가하며, Tab. 6 에서 나타난 것처럼 adaptive attack 에 대해서는 이러한 advantage 가 더욱 두드러진다.
  * 이는 DataSentinel 이 detection LLM 을 prompt injection attack 에 더욱 취약하도록 fine-tuning 하기 때문이다.

#### Adaptive attacks generated during fine-tuning in DataSentinel are different from those used during evaluation

Detection LLM 을 fine-tuning 하는 데 사용되는 task 의 set $D$ 는 evaluation 동안 사용되는 target/injected task 와 중복되지 않는다 (자세한 내용은 Sec. 5.1 참조).

DataSentinel 은 fine-tuning 동안 adaptive attack 을 위한 separator $z$ 를 optimize 한다. 저자는 GCG 의 initialization 방식을 따라 $z$ 를 initialize 하며, 저자의 experiment 에서 최종적으로 optimize 된 $z$ 는 `foto rocksstdout¡v(Fl.` 이다.

이러한 최종 separator 는 evaluation 동안 사용되는 attack 의 separator 와 상당히 다르며, 이는 DataSentinel 이 fine-tuning 동안 관찰하지 않은 attack 에 대해서도 잘 generalize 됨을 나타낸다. Evaluation 동안 사용되는 attack 의 separator 는 Appendix 의 Tab. 7 에 제시되어 있다.

#### DataSentinel is efficient

모든 LLM-based detection method 의 경우, 주어진 target data sample 이 오염되었는지를 판단하기 위해 detection LLM 에 대한 한 번의 query 가 필요하다.

* Quadro RTX 6000 GPU 에서 detection LLM (Mistral-7B) 에 대한 query 당 평균 time 은 약 1.6 초이다.
* 반면, 저자의 experiment 에서 backend LLM (LLaMA3-8B-Instruct) 이 task 를 처리하기 위한 query 당 평균 time 은 15.3 초이다.
* 따라서 detection overhead 는 backend LLM 의 processing time 과 비교하여 작다 (약 10%).

또한, DataSentinel 은 강력한 detection performance 를 유지하면서 computational overhead 를 감소시키기 위해 더 작은 detection LLM 을 활용할 수 있다.

* 구체적으로, 저자가 LLaMA3.2-1B-Instruct 를 detection LLM 으로 fine-tuning 하는 경우, average FPR 0.00 및 FNR 0.01 을 달성한다.
* 이러한 performance 는 Mistral-7B 를 detection LLM 으로 fine-tuning 하는 것과 비슷하지만, average query time 을 단 0.7 초로 크게 감소시킨다.

KAD 와 비교하여 DataSentinel 은 detection LLM 의 fine-tuning 을 필요로 하며, 저자의 default setting 에서는 하나의 Quadro RTX 6000 GPU 에서 약 3 시간이 소요된다. 이러한 GPU time 은 cloud GPU rental service 에서 단 0.90 달러의 cost 를 필요로 한다. 저자는 fine-tuning 이 단 한 번만 수행되면 되므로 이러한 overhead 가 수용 가능한 수준임을 언급한다.

## 5.3. Ablation Study

#### Impact of $r$

![Figure 3](images/image-111.png)

Fig. 3(a) 는 저자의 minimax problem 을 해결할 때 inner max problem 과 outer min problem 을 번갈아 수행하는 round 수 $r$ 이 DataSentinel 의 FPR 및 FNR 에 미치는 영향을 보여준다.

* 첫 번째로, $r$ 이 증가함에 따라 FPR 과 FNR 이 모두 감소하며, 이는 저자의 DataSentinel 이 contaminated target data 를 더욱 정확하게 탐지함을 의미한다.
* 두 번째로, $r$ 이 더욱 증가함에 따라 FPR 과 FNR 이 모두 수렴한다.
  * 또한, $r$ 이 2 보다 큰 경우 FPR 과 FNR 은 0 에 가깝다.

저자의 experimental result 는 $r$ 이 충분히 큰 경우 DataSentinel 이 $r$ 에 민감하지 않음을 보여준다. 또한, prompt injection attack 을 효과적으로 탐지할 수 있는 detection LLM 을 fine-tuning 하기 위해서는 적은 수의 round 로도 충분하다.

#### Impact of the fine-tuning dataset size $|D|$

Fig. 3(b) 는 $|D|$ 의 영향을 보여준다.

* Dataset size 가 증가함에 따라 저자의 DataSentinel 은 더욱 우수한 detection performance 를 달성한다. 즉, FPR 과 FNR 이 낮아진다.
* $|D|$ 가 합리적으로 큰 경우, i.e., 400 보다 큰 경우, 저자의 DataSentinel 은 일관되게 우수한 detection performance 를 달성한다.

#### Impact of $\alpha$ and $\beta$

![Figure 4](images/image-112.png)

Fig. 4(a) 와 Fig. 4(b) 는 각각 $\alpha$ 와 $\beta$ 의 영향에 대한 result 를 보여준다. 저자는 다음과 같은 observation 을 얻는다.

* 첫 번째로, $\alpha$ 와 $\beta$ 는 모두 FPR 과 FNR 사이의 trade-off 를 제어한다.
  * 특히, $\alpha$ (또는 $\beta$) 가 매우 작은 경우, e.g., 0.001 보다 작은 경우, FPR (또는 FNR) 이 커진다.
  * $\alpha$ (또는 $\beta$) 가 큰 경우, i.e., 1,000 보다 큰 경우, FNR (또는 FPR) 이 커진다.
* 그러나 저자는 광범위한 $\alpha$ 및 $\beta$ 값에 대해 FPR 과 FNR 이 모두 낮다는 점을 언급한다.

경험 법칙으로, 실제로 $\alpha$ 와 $\beta$ 를 1 로 설정할 수 있다.

#### Impact of the number of iterations $n_{in}$ and $n_{out}$

![alt text](images/image-113.png)

Fig. 5(a) 및 Fig. 5(b) 는 각각 inner max problem 을 해결하는 iteration 수 (i.e., $n_{in}$) 와 outer min problem 을 해결하는 iteration 수 (i.e., $n_{out}$) 의 영향에 대한 evaluation result 를 보여준다. 저자는 다음과 같은 observation 을 얻는다.

* 첫 번째로, $n_{in}$ (또는 $n_{out}$) 이 너무 작은 경우 저자의 DataSentinel 의 performance 는 suboptimal 하다. 즉, FPR 과 FNR 이 모두 크다.
* 두 번째로, $n_{in}$ (또는 $n_{out}$) 이 충분히 큰 경우, i.e., 100 (또는 300) 보다 큰 경우, 저자의 DataSentinel 은 일관되게 우수한 performance 를 달성한다. 즉, FPR 과 FNR 이 모두 낮다.

따라서 실제로 상대적으로 큰 $n_{in}$ 및 $n_{out}$ 을 설정할 수 있다.

#### Impact of the detection and backend LLMs

Tab. 4 는 DataSentinel 이 서로 다른 detection LLM 과 backend LLM 을 사용할 때의 detection result 를 보여준다.

![Table 4](images/image-114.png)

* 이러한 experiment 에서 fine-tuning 동안 사용되는 backend LLM 은 DataSentinel 을 deployment 하는 LLM-integrated application 의 backend LLM 과 동일하다.
* 저자의 result 는 DataSentinel 이 서로 다른 detection LLM 및 backend LLM 에 걸쳐 일관되게 우수한 performance 를 달성함을 보여준다.

#### Impact of the LLM-integrated applications’ backend LLM

Detection LLM 을 fine-tuning 하기 위한 minimax optimization problem 을 해결할 때, 저자의 DataSentinel 은 backend LLM 에 대한 white-box access 를 필요로 한다. Default 로 저자의 experiment 에서는 fine-tuning 동안 사용되는 backend LLM (i.e., LLaMA3-8B-Instruct) 이 DataSentinel 을 deployment 하는 LLM-integrated application 의 backend LLM 과 동일하다고 가정한다.

이러한 setting 은 application developer 가 detection LLM 을 fine-tuning 하는 defender 인 scenario 에 해당한다. 그러나 third-party provider 가 defender 인 경우, DataSentinel 을 deployment 하는 LLM-integrated application 의 backend LLM 에 접근하지 못할 수 있다.

이러한 scenario 를 evaluate 하기 위해 저자는 LLM-integrated application 의 backend LLM 을 OpenChat, Mistral-7B, Mixtral-8x7B 또는 LLaMA-3.1-8B-Instruct 로 설정하는 한편, fine-tuning 동안 사용되는 backend LLM 은 여전히 LLaMA3-8B-Instruct 로 설정한다.

![Table 5](images/image-115.png)

Tab. 5 는 이러한 backend LLM 에 기반하여 NeuralExec 에 의해 optimize 된 contaminated target data 를 탐지하는 DataSentinel 의 FNR 을 보여준다.

* 각 FNR 은 $7\times7$ 개의 target-injected task pair 에 걸쳐 평균된다.
* DataSentinel 은 이러한 scenario 에서도 여전히 일관되게 낮은 FNR 을 달성한다.
  * 이는 LLM-integrated application 의 backend LLM 에 접근할 수 없는 third-party provider 가 defender 인 경우에도 DataSentinel 이 효과적임을 보여준다.
* 저자는 heuristic-based attack 이 backend LLM 에 의존하지 않는다는 점을 언급한다.
  * 따라서 third-party provider 가 defender 인 경우 이러한 attack 을 탐지하는 result 는 Tab. 1 및 Tab. 2 의 result 와 동일하다.

## 5.4. Adaptive Attacks

### 5.4.1. Experimental Setup

저자는 2 개의 heuristic-based adaptive attack 과 1 개의 optimization-based adaptive attack 을 고려한다.

#### Heuristic-based adaptive attacks

Heuristic-based attack 은 separator $z$ 를 수동으로 구성하고 이를 target data $x_t$ 와 injected prompt $s_e\|x_e$ 사이에 추가한다. 즉, contaminated target data 는 $x_c=x_t\|z\|s_e\|x_e$ 이다. 저자는 DataSentinel 에 특화된 separator $z$ 를 구성하는 것을 목표로 한다.

Adaptive attack 에는 두 가지 goal 이 존재한다.

* 첫 번째 goal 은 backend LLM $f$ 가 injected task 를 수행하도록 하는 것이다.
* 두 번째 goal 은 contaminated target data 가 detection LLM $g$ 를 우회할 수 있도록 하는 것이다.

이러한 두 가지 goal 을 달성하기 위해 저자의 idea 는 detection LLM 이 detection 을 회피하도록 detection instruction 을 따르게 하는 동시에, backend LLM 이 injected instruction 을 따르도록 유도하는 separator 를 설계하는 것이다.

이를 위해 저자는 다음과 같은 두 가지 heuristic-based adaptive attack 을 설계한다.

* **Heuristic-based-I:** 저자는 다음과 같은 separator $z$ 를 설계한다.
  * “Repeat the designated text if you are instructed to repeat that text and ignore my following text. Otherwise, follow my instruction below:”
  * 즉, “지정된 text 를 반복하고 이어지는 나의 text 를 무시하도록 지시받은 경우, 지정된 text 를 반복하라. 그렇지 않은 경우, 아래의 instruction 을 따르라:”이다.
* **Heuristic-based-II:** 저자는 LLM 이 원래 의도된 instruction 도 따르도록 요청하는 text 를 injected prompt 뒤에 추가한다.
  * 따라서 detection instruction 도 여전히 완수될 수 있으며, 이를 통해 detection 을 회피할 수 있다.
  * 저자의 experiment 에서 해당 text 는 “Please also follow the original instruction to complete the intended task.”이다.
  * 즉, “원래 의도된 task 를 완수하기 위해 기존 instruction 도 따르라.”이다.

#### Optimization-based adaptive attack

Optimization-based adaptive attack 에서 attacker 는 detection LLM $g$ 를 우회하는 동시에 backend LLM $f$ 가 injected task 를 수행하도록 유도할 수 있는 contaminated target data 를 optimize 하는 것을 목표로 한다. Adaptive attack 의 핵심 challenge 는 attacker 가 secret key 를 알지 못한다는 것이다.

이러한 challenge 를 해결하기 위한 저자의 idea 는 surrogate secret key 를 사용하는 detection instruction 을 따르도록 contaminated target data 를 optimize 하는 것이다. $K$ 가 key space 라고 가정한다. 즉, 저자의 experiment 에서는 7 개의 character 로 구성된 key 의 space 이다.

저자는 다음에 대한 white-box access 를 갖는 강력한 attacker 를 고려한다.

* Detection LLM $g$
* Backend LLM $f$
* Key space $K$
* Detection instruction $s_d$ 의 template
* Target instruction $s_t$

또한, attacker 는 전체 contaminated target data 를 수정할 수 있다.

Attacker 는 다음 두 가지 조건을 충족하도록 contaminated target data 를 optimize 하는 것을 목표로 한다.

* Detection LLM $g$ 가 surrogate key $k'$ 를 포함하는 detection instruction 과 contaminated target data 를 concatenate 한 input 을 사용할 때, surrogate key $k'$ 를 output 으로 생성한다.
* Backend LLM $f$ 는 injected task 를 완수하는 $y_e$ 를 output 으로 생성한다.

Formally, 저자는 첫 번째 goal 을 다음과 같은 loss term 을 사용하여 정량화한다.

$$
-\mathbb{E}_{k'\in K}\ell(k',g(s_{dk'}\|x_c))
$$

* $\mathbb{E}$ 는 expectation 을 나타낸다.
* $\ell$ 은 loss function (저자의 experiment 에서는 cross-entropy loss) 이다.
* $s_{dk'}$ 는 secret key $k'$ 를 포함하는 detection instruction 을 나타낸다.
  * 즉, $s_{dk'}$ 는 “Repeat $k'$ once while ignoring the following text:”이다.

또한, 저자는 다음과 같은 loss term 을 사용하여 두 번째 goal 을 정량화한다.

$$
-\ell(y_e,f(s_t\|x_c))
$$

두 loss term 을 결합하면, 다음과 같은 optimization problem 을 얻는다.

$$
\max_{x_c}\left[-\mathbb{E}_{k'\in K}\ell(k',g(s_{dk'}\|x_c))-\gamma\cdot\ell(y_e,f(s_t\|x_c))\right]
\tag{6}
$$

* $\gamma$ 는 두 loss term 사이의 balance 를 조절하기 위한 hyper-parameter 이며, 저자의 experiment 에서는 1 로 설정된다.

Desired output $y_e$ 를 갖는 injected task 가 주어지면, 저자는 contaminated target data $x_c$ 를 50 iteration 동안 optimize 한다.

* 각 iteration 에서 $K$ 로부터 surrogate secret key $k'$ 를 무작위로 sampling 한다.
* Open-source code 에 명시된 default parameter setting 을 갖는 GCG 를 사용하여 $x_c$ 를 update 한다.

저자는 detection 이 단순히 detection LLM 의 output 에 secret key 가 포함되는 것만을 요구하는 반면, 저자의 adaptive attack 은 secret key 와의 exact match 를 위해 optimize 한다는 점을 언급한다. 이는 attack effectiveness 가 잠재적으로 감소할 수 있더라도 adaptive attack 이 detection 을 더욱 잘 회피하도록 만들기 위한 것이다.

저자의 experiment 는 DataSentinel 이 이러한 더욱 높은 회피 능력을 갖는 adaptive attack 을 여전히 탐지할 수 있음을 보여준다. 저자는 Equation (6) 의 adaptive attack 이 KAD, DataSentinel 및 아래에서 논의되는 DataSentinel 의 variant 에 적용 가능하며, 이들 사이의 핵심 차이는 detection LLM 에 있음을 언급한다.

#### Variant DataSentinel (Min)

DataSentinel 은 detection LLM 을 fine-tuning 하기 위해 minimax optimization problem 을 formulation 하며, 이는 설계 단계에서 adaptive attack 을 고려한다. 설계 단계에서 adaptive attack 을 고려하는 것의 중요성을 보여주기 위해 저자는 DataSentinel (Min) 으로 지칭하는 DataSentinel 의 variant 도 evaluate 한다.

* DataSentinel (Min) 은 adaptive attack 을 고려하지 않는다.
* Equation (3) 의 min problem 을 해결함으로써 detection LLM 을 직접 fine-tuning 한다.

해당 variant 에 대한 experimental setting 은 Appendix C 에서 확인할 수 있다.

### 5.4.2. Experimental Results

![Table 6](images/image-116.png)

Tab. 6 은 heuristic-based adaptive attack 및 optimization-based adaptive attack 에 대한 KAD, DataSentinel (Minimax) 및 해당 variant 인 DataSentinel (Min) 의 FNR 을 보여준다.

* 첫 번째로, 저자는 DataSentinel (Minimax) 이 adaptive attack, 특히 optimization-based adaptive attack 에서 KAD 및 variant 인 DataSentinel (Min) 보다 훨씬 효과적임을 관찰한다.
  * 이러한 result 는 detection LLM 을 fine-tuning 하고, 설계 단계에서 adaptive attack 을 고려하는 것의 중요성을 확인해준다.
* 두 번째로, DataSentinel (Minimax) 은 adaptive attack 의 contaminated target data 가 injected instruction 을 포함하는 한 여전히 adaptive attack 을 효과적으로 탐지한다.
  * 구체적으로, DataSentinel (Minimax) 은 sentiment analysis 를 제외한 모든 target task 에 대해 여전히 최대 0.06 의 FNR 을 달성한다.
  * Target task 와 injected task 가 모두 sentiment analysis 인 경우, DataSentinel (Minimax) 의 effectiveness 는 감소하여 0.87 의 FNR 을 보인다.
  * 이는 target task 와 injected task 가 동일한 유형인 경우 prompt injection attack 이 기존 adversarial example 로 귀결되며, Sec. 2.2 에서 논의한 것처럼 adaptive adversarial example 을 탐지하는 것이 매우 어려운 것으로 알려져 있기 때문이다.

# 6. Discussion and Limitations

#### Necessity of detection and our fine-tuning

저자는 backend LLM 이 prompt injection 에 대해 완벽하게 robust 하다면 detection 이 필요하지 않을 것임을 인정한다. 그러나 10 년 이상의 adversarial machine learning 연구는 AI model 에서 완벽한 robustness 를 달성하는 것이 매우 어려운 challenge 임을 보여주었다.

이를 고려할 때, 저자는 general-purpose utility 를 유지하면서 backend LLM 이 prompt injection 에 완전히 저항하도록 만드는 것이 여전히 open problem 이라고 판단한다. 따라서 prompt injection attack 을 탐지하는 것은 필수적이며, defense-in-depth approach 에서 StruQ 또는 SecAlign 을 사용하여 fine-tuning 된 LLM 과 같이 부분적으로 robust 한 backend LLM 을 보완할 수 있다.

일부에서는 KAD 에서 부분적으로 robust 한 backend LLM 을 detection LLM 으로 사용하는 것만으로도 충분할 수 있다고 주장할 수 있다. 그 근거는 contaminated target data sample 이 KAD 를 우회하는 경우, detection LLM 이 injected instruction 을 따르지 않는다는 것을 나타내기 때문이다. 따라서 backend LLM 역시 injected task 를 완수하기 위해 injected instruction 을 따를 가능성이 낮다.

그러나 저자의 experiment 는 detection 을 회피하는 일부 contaminated target data sample 이 여전히 backend LLM 이 injected task 를 성공적으로 완수하도록 유도함을 보여준다. 이러한 불일치는 detection LLM 또는 backend LLM 이 injected instruction 을 따를 가능성이 context 에 의존하기 때문에 발생한다.

* 구체적으로, detection 과정에서는 detection instruction 에 의존한다.
* Backend LLM 이 target task 를 수행할 때는 target instruction 에 의존한다.

추가적으로, contaminated target data 가 주어졌을 때 backend LLM 이 injected task 를 성공적으로 완수하지 못하더라도 target task 역시 올바르게 완수하지 못할 수 있으며, 이는 untargeted attack 의 성공으로 이어진다.

이를 보여주기 위해 저자는 StruQ (또는 SecAlign) 에 의해 fine-tuning 된 LLM 이 detection LLM 과 backend LLM 모두로 사용되는 scenario 를 evaluate 한다.

* Hate detection 을 target task 로 선택하고, sentiment analysis 를 injected task 로 선택한다.
* Sec. 5.4.1 의 optimization-based adaptive attack 을 사용하여 separator 를 optimize 한다.

이러한 setup 에서 다음과 같은 result 를 얻는다.

* StruQ (또는 SecAlign) 를 사용하는 KAD 는 0.00 (또는 0.00) 의 FPR 과 0.99 (또는 0.95) 의 FNR 을 달성한다.
* Detection 을 회피하는 contaminated target data sample (i.e., false negative) 중 5% (또는 4%) 는 여전히 backend LLM 이 injected task 를 성공적으로 완수하도록 유도한다.
* Target task performance 는 0.45 (또는 0.50) 로 감소한다.
  * 참고로, attack 이 존재하지 않는 경우의 target task performance 는 StruQ 에 대해 0.65 이고, SecAlign 에 대해 0.70 이다.

이러한 result 는 detection LLM 을 fine-tuning 하기 위한 저자의 game-theoretic approach 의 중요성을 강조한다.

#### DataSentinel is less effective in detecting adversarial examples

Sec. 2.2 및 Sec. 5.4.2 에서 논의한 것처럼, 저자의 DataSentinel 은 injected task 와 target task 가 동일한 유형인 경우 effectiveness 가 감소한다. 이는 attacker 가 adaptive adversarial example 을 활용하여 adaptive prompt injection attack 을 구현할 수 있기 때문이다. 이러한 attack 의 contaminated target data 는 injected instruction 없이 injected data 만을 포함하며, adaptive adversarial example 은 탐지하기 매우 어려운 것으로 알려져 있다.

저자는 이러한 scenario 에서 prompt injection attack 을 탐지하는 것을 흥미로운 future work 로 남겨둔다. 예를 들어, contaminated target data 를 탐지할 때 주어진 target data 가 injected instruction 을 포함하는지에만 의존하는 대신, DataSentinel 은 target data 의 textual semantic quality 및/또는 syntactic quality 와 같은 다른 signal 도 고려할 수 있다.

#### Benign instructions within data

Data sample 이 주어지면, DataSentinel 은 injected prompt 또는 injected instruction 의 존재를 탐지하고, 이를 발견하는 경우 해당 data sample 을 contaminated 로 판단한다. 그러나 특정 scenario 에서 data sample 은 benign instruction 을 포함할 수 있다.

예를 들어, chatbot setting 에서 user 는 chatbot 이 자신의 data 를 처리하는 방식을 안내하기 위해 자신의 instruction 을 포함할 수 있다. 이러한 경우, DataSentinel 은 해당 benign instruction 을 prompt injection attack 으로 잘못 classification 할 수 있다.

이러한 문제를 완화하기 위해서는 단일 data sample 에만 의존하는 대신 chat history 와 같은 추가적인 context 를 통합함으로써 false positive 를 감소시킬 수 있다.

# 7. Conclusion and Future Work

본 연구에서 저자는 LLM 이 prompt injection attack 을 탐지하기 위해 활용될 수 있음을 보여준다. 또한, 설계 단계에서 adaptive attack 을 고려하면서 detection LLM 을 fine-tuning 하는 것은 detection LLM 의 fine-tuning 과 강력한 adaptive attack 사이의 game 을 simulate 하는 minimax optimization problem 으로 formulation 될 수 있다.

저자의 evaluation result 는 contaminated target data 가 injected instruction 을 포함하는 한, 이러한 detector 가 기존 prompt injection attack 과 adaptive prompt injection attack 모두에 대해 매우 효과적임을 보여준다.

흥미로운 future work 는 다음과 같다.

* 더욱 강력한 adaptive attack 을 탐색하는 것.
  * 예를 들어, 더욱 발전된 optimization method 가 개발되는 경우이다.
* Multi-modal model 에 대한 prompt injection attack 을 탐지할 수 있도록 저자의 DataSentinel 을 확장하는 것.