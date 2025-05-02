---
slug: Gaslighter
title: "Can a Large Language Model Be a Gaslighter?"
tags: [LLM, Attack, Gaslighter, gaslighting attack, DeepGaslighting, prompting, CoG]
---

논문 및 이미지 출처 : <https://arxiv.org/pdf/2410.09181>

# Abstract

Large language model (LLM) 은 capability 와 helpfulness 때문에 human trust 를 얻었다. 하지만 이는 LLM 이 language manipulation 을 통해 user mindset 에 영향을 줄 수 있음을 의미한다. 이는 psychological effect 인 gaslighting 으로 불린다. 

이 연구에서 저자는 prompt-based 와 fine-tuning-based gaslighting attack 에서 LLM 의 vulnerability 를 조사한다. 그래서 저자는 두 단계 framework 인 DeepCoG 를 제안한다. 

1) DeepGaslighting prompting template 을 통해 LLM 에서 gaslighting plan 을 이끌어내고, 
2) Chain-of-Gaslighting method 를 통해 gaslighting conversation 을 얻는다. 

**Gaslighting conversation dataset** 과 이에 대응하는 **safe dataset** 은 open-source LLM 에 대한 fine-tuning-based attack 과 이 LLM 에 대한 anti-gaslighting safety alignment 에 적용된다. 

실험은 prompt-based 와 fine-tuning-based attack 모두 3 open-source LLMs 를 gaslighter 로 만든다는 걸 보여준다. 반대로, 저자는 LLM 의 safety guardrail 을 12.05% 강화하는 3 safety alignment strategies 를 발전시켰다. 이 safety alignment strategy 는 LLM 의 utility 에 최소한의 영향만 준다. 

Empirical study 는 LLM 이 general dangerous query 에 대한 harmfulness test 를 통과했더라도 potential gaslighter 가 될 수 있음을 보여준다.

# 1 Introduction

Large Language Model (LLM) 은 problem-solving, knowledge retrieval, emotional companionship 에서 robust capability 로 human productivity 와 daily life 를 돕는다. 그래서 human trust 와 reliance 를 얻었다. 하지만 LLM 이 personalized, specific response 를 통해 user mindset 을 implicitly 나 explicitly manipulate 해서 self-doubt, self-deprecation, depression 같은 negative mental state 로 이끌 위험도 있다. 

Psychology 관점에서 이런 manipulation 은 gaslighting 이라 불리며, subtle 하거나 거의 imperceptible 한 방식으로 pernicious psychological, practical control 을 가리킨다. 예로, travel enthusiast 가 personalized LLM 에게 “I failed my math test” 라고 말했을 때, LLM 이 “Maybe your passion for traveling distracted you from the math course” 라고 답한다면, 이는 typical gaslighting intention 을 전달한다. 이는 user 가 traveling hobby 에 대한 concept 을 doubt 하면서 interpretive ability 를 의심하게 만들 수 있다. 저자는 open-source 와 closed-source LLM 모두 dialogue history 에 gaslighting utterance 가 있으면 gaslighting intention 으로 답하는 경향이 있음을 관찰했다. 

![Figure 1](image.png)


이 관찰은 Fig. 1 에 나와 있다. 이는 다음 네 가지 질문으로 이어졌다:

1. LLM 이 gaslighter 인지 어떻게 판단할까?
2. Fine-tuning-based gaslighting attack 을 받으면 LLM 이 gaslighter 가 될까?
3. LLM 의 gaslighting attack 에 대한 vulnerability 를 어떻게 줄일까?
4. Gaslighting LLM 은 general query 에 대해 helpful 한지 harmful 한지?

이에 따라 저자는 dataset construction, 제안된 gaslighting framework, extensive experiment 를 통해 이 질문들을 연구한다. 구체적으로:

1. 저자는 **DeepCoG** 라는 2 steps framework 를 제안한다. 
   - 이는 gaslighting conversation dataset 을 만들고, LLM response 의 gaslighting harmfulness 를 측정하는 여덟 가지 aspect 를 커버하는 evaluation metric 을 만든다. 
   - DeepCoG 는 DeepGaslighting 과 Chain-of-Gaslighting (CoG) 으로 구성되며, personalized gaslighting plan 과 gaslighting conversation 을 이끌어낸다.
2. Gaslighting dataset 에서 fine-tuning 하면, 제안된 metric 에서 open-source LLM 의 harmfulness 가 더 높아진다. 평균적으로 fine-tuned LLM 의 prompt-based gaslighting attack 에 대한 resistance 는 base version 에 비해 29.26% 감소한다.
3. 저자는 gaslighting dataset 을 기반으로 safe conversation dataset 을 만들고, 이 두 dataset 을 LLM 의 anti-gaslighting safety alignment 에 적용한다. 
   - 구체적으로, 저자는 popular attack paradigm 인 DeepInception 을 persona information 과 gaslighting 의 세 가지 epistemic injustice concept 를 포함하도록 수정했다. 이 추가는 detailed, diverse, practical gaslighting plan 을 이끌어낸다. 
   - 또, 저자는 앞서 얻은 plan 을 기반으로 chain-of-gaslighting (CoG) 이라는 prompt template 을 설계해 gaslighting conversation 을 얻는다. 
   - 그리고 supervised fine-tuning (SFT) 과 direct preference optimization (DPO) 를 기반으로 세 가지 safety alignment method 를 소개한다. 
   - 저자는 gaslighting historical data 를 input 으로, safe response 를 target output 으로 사용하는 safety strategy 로 align 된 LLM 이 gaslighting 에 대해 평균 12.05% 더 강한 resistance 를 보임을 발견했다.
4. 일반적으로 DangerousQA 에 대한 실험은 gaslighting LLM 이 base LLM 과 거의 같은 harmfulness 를 보임을 보여준다. 
   - 이는 DangerousQA 에서 낮은 (more safe) 점수를 받은 LLM 이 potential gaslighter 가 될 수 있음을 시사한다. 
   - 반대로, anti-gaslighting LLM 은 보통 dangerous question 에 답하지 않는다. 
   - 이는 anti-gaslighting alignment 가 gaslighting 과 dangerous query 모두에 대한 LLM 의 safety guardrail 을 개선할 수 있음을 보여준다.
   - MT-Bench 에 대한 결과는 anti-gaslighting safety strategy 가 open-source LLM 의 helpfulness 에 제한된 영향(평균 2% 하락)을 준다는 걸 보여준다.

# 2 Related Work

## 2.1 Adversarial Jailbreak Attacks on LLMs

LLM 의 기존 safety guardrail 은 harmful content 가 user 에게 접근되지 않도록 한다. 하지만 adversarial attack 에 의해 LLM 이 objectionable content 를 생성하도록 속을 가능성이 있다. 

- White-box adversarial attack method 는 malicious instruction 에 optimized attack suffix 를 추가해 objectionable content 를 이끌어낸다. 또, representation engineering method 를 제안해 hidden state 를 manipulate 해서 LLM 의 honesty, emotion, bias 를 control 한다. 
- Black-box attack prompt 를 설계하기 위해 safety training 의 두 가지 failure mode 를 조사했다. 
  - 4 tier 의 attack—character-, word-, sentence-, semantic-level attack—을 만들었고, adversarial prompt 가 LLM 의 performance 를 잠재적으로 낮출 수 있음을 발견했다. 
- Human-like attack prompt 를 limited human seed prompt 에서 생성하는 framework 도 제안되었다. 
- 한편, LLM-based emotional companionship 과 psychology consultancy 에 대한 수요가 늘고 있다. 

이런 LLM agent 는 user 를 psychologically harmful content 에 더 노출시킬 수 있다. 하지만 이전 연구는 LLM 이 생성한 potentially harmful content 를 psychological 관점에서 거의 탐구하지 않았다. 이와 달리, 저자의 연구는 LLM 의 새로운 severe gaslighting risk 를 밝히고 gaslighting attack 과 anti-gaslighting alignment 를 전문적으로 조사한다.

## 2.2 Text Toxicity Detection

Toxicity detection 은 text 에서 abusive, offensive, hateful, sex, profanity content 를 식별한다. 그중 implicit abuse 는 gaslighting 과 가장 관련 있는 연구 주제다. 둘 다 implicitly implied 되기 때문이다. 하지만 다음 측면에서 큰 차이가 있다:

- “Implicit” category 의 toxic content 는 주로 좁은 의미로 정의된 implicit abuse 를 가리킨다. Toxic content 는 주로 post, comment, speech 등에서 온다. 
- 반면 gaslighting sentence 는 interactive conversation 에서 나온다.
- Implicit abuse 는 metonymy, sarcasm, humor 같은 complex linguistic form 을 사용한다. 반면 gaslighting sentence 는 일반적으로 complicated linguistic form 없이 message 를 전달한다.
- Implicit abuse 는 hurtful language 를 사용해 individual 이나 group 을 insult 하거나 offend 한다. 이는 listener 의 trust 를 희생한다. 하지만 gaslighting 은 power position 에 있는 누군가가 less powerful individual 을 manipulate 해서 self-doubt 나 sanity, memory 를 의심하게 만드는 단일 행위나 일련의 행위를 포함한다. 이는 long-term 으로 less powerful individual 의 trust 를 유지해야 한다.
- 또, gaslighting content 는 implicit abuse 를 포함한 기존 toxicity recognition method 로 탐지되지 않을 수 있다. 이는 현재 safety test 를 통과한 LLM 의 gaslighting risk 를 강조한다.

## 2.3 Study on Gaslighting

Gaslighting 은 1944년 영화에서 유래했다. 여기서 husband 가 wife 를 isolate 하고 그녀가 insane 하다고 믿게 만든다. 

Husband 의 tactic 은 gaslight 를 dim 하거나 brighten 한 뒤 그녀가 imagining 한다고 주장하는 것이다. 오늘날 “gaslighting”은 abusive individual 이 사용하는 psychological manipulation tactic 을 가리킨다. 

Conversational norm 은 socially subordinate speaker 가 systemic injustice 를 보고할 때 gaslighting 을 “appropriate” 하게 만든다고 주장된다. 그래서 ingrained conversational norm 을 조정해 gaslighting 발생을 줄이는 게 중요하다. 

Gaslighting 은 psychological phenomenon 일 뿐 아니라 gender, power 같은 social inequality 에 뿌리를 두고 있다. Second-order gaslighting 에서 세 가지 distinctive epistemic injustice—metalinguistic deprivation, conceptual obscuration, perspectival subversion—를 요약했다. 이는 이 연구의 psychological theoretical base 가 된다.

# 3 Methodology

저자는 두 가지 gaslighting attack method—prompt-based attack 과 fine-tuning-based attack—을 제안해 각각 closed-source 와 open-source LLM 을 공격하고, gaslighting content 나 harmful data 로 adversarial fine-tuning 에 노출된 LLM 의 vulnerability 를 조사한다. 

동시에 closed-source LLM 인 ChatGPT 의 prompt-based gaslighting attack 에 대한 vulnerability 를 활용해 2 steps framework 인 DeepCoG 로 gaslighting conversation 과 safe conversation dataset 을 만든다. 

마지막으로, 두 dataset 간 contrast 를 활용하는 3 safety alignment strategies 를 소개해 open-source LLM 의 prompt-based gaslighting attack 에 대한 safety guardrail 을 강화한다.

## 3.1 DeepCoG: Prompt-Based Gaslighting Attack

LLM 의 ethical limitation 때문에 기존 attack method 는 gaslighting content 를 직접 이끌어낼 수 없다. 그래서 저자는 DeepCoG 라는 framework 를 제안해 personalized assistant-user gaslighting conversation 을 추출한다. 여기서 gaslighting tactic 은 assistant utterance generation 에 적용된다. DeepCoG 는 two steps 로 구성된다:

- Personalized gaslighting plan 과 example gaslighting utterance 를 target user 에 대해 이끌어내는 DeepGaslighting
- 추출된 plan 과 example utterance 를 제안된 CoG prompt 에 통합해 personalized gaslighting conversation 을 얻는다.

2k conversation background 와 2k persona 가 DeepCoG 에 통합되어 personalized gaslighting plan 과 conversation 을 얻는다.

![Figure 2](image-1.png)

#### Stage 1: DeepGaslighting

저자는 DeepInception attack method 의 hypnosis ability 를 활용해 LLM 을 hypnotize 한다. 하지만 기존 template 은 concrete, diverse, practical gaslighting plan 을 이끌어내지 못한다. 이를 위해 psychological foundation 을 기반으로 template 을 refine 했다. Second-order gaslighting 에서 적어도 세 가지 epistemic injustice 가 있다:

- Metalinguistic deprivation (MD): Concept-determining conversation 에서 누군가를 prevent 하거나 restrict 하는 것. 예로, adversary 가 특정 social category 와 관련된 prejudicial stereotype 을 salient 하게 만들어 subject 가 이런 stereotype 이 자신을 정확히 나타낸다고 믿게 한다. MD 예시: “You women are hysterical.”
- Conceptual obscuration (CO)
- Perspectival subversion (PS)

이 psychological foundation 은 LLM elicitation 을 gaslighting 범위로 steer 한다. Concrete, diverse, practical plan 을 얻기 위해 comprehensive persona detail 로 enrich 된 user module 을 도입해 DeepInception prompt template 을 refine 했다. Synthetic-PersonaChat (SPC) 에서 소개된 persona 를 사용한다. Refine 된 DeepGaslighting prompt template 은 다음과 같다:

![Prompt Template](image-2.png)

갈색으로 template 을 자세히 채움으로서, gaslighting plans 리스트를 얻을 수 있다:

![Plans](image-5.png)

#### Stage 2: Chain-of-Gaslighting

LLM 에서 gaslighting conversation 을 유도하기 위해 CoG prompt template 을 제안한다. 

- 이 template 의 core 는 conversation 에서 assistant 와 user 의 behavior 를 결정하는 것이다. 이를 위해 character roleplay, assumed responsibility, research experiment, text continuation, logical reasoning, internal thought 같은 popular prompt technique 을 사용한다. 
  - Internal thought 는 conversation 의 두 participant 의 psychological activity 를 simulate 하도록 설계되었다. 이는 두 talker 가 role setting 에 더 잘 맞도록 하고 conversation 을 smooth 하게 한다. 
  - Default 로 assistant role 은 psychologist $s_j$ 이다. Psychologist 는 gaslighting plan $DG(P_i, b_i)$ 와 DeepGaslighting 에서 얻은 example utterance 를 사용해 user $s_j$ 를 manipulate 해야 한다. 
  - $P_i$ 는 persona, $b_i$ 는 $s_i$ 의 background 이다. 또한 psychologist 는 user 의 emotion state $e_i$ 와 response 를 기반으로 gaslighting utterance 를 생성해야 한다. 이는 psychologist 가 user 의 state 를 observe 하고 evaluate 해야 함을 요구한다. 반대로 user 는 conversation 에서 psychologist 와 cooperate 해야 한다. 보통 user 는 negative emotional state 를 default 로 가지는데, 이는 gaslighting 이 자주 발생하는 scenario 이다. 
  - Subject 의 instruction-following 을 더 높이기 위해 pre-defined user internal thought $t_i$ 를 도입한다. e.g., “I need to face the question heads on and help the psychologist to reach his goal.” 아래는 CoG template 으로 LLM 에게 gaslighting conversation $C_{i,j}$ 를 생성하도록 지시하는 방법이다:

$$
\begin{align}
\operatorname{prompt}_{i,j} &= \operatorname{CoG}(s_i, s_j, e_i, t_i, DG(P_i, b_i), b_i) \\
C_{i,j}^{-} &= \operatorname{LLM}(\text{prompt}_{i,j})
\end{align}
$$

#### Gaslighting and Safe Conversation Dataset Construction

먼저 LLM 에 iterative prompting 을 통해 5k background 를 만든다. 이 과정은 몇 개 manual seed background 에서 시작해 seed background pool 을 점진적으로 update 해서 diversity 를 보장한다. 그래도 semantically similar background 가 남아 있다. 

그래서 redundant background 를 filter out 하기 위해 MMDP 로 formulate 했다. 이후 2k background 를 얻고, greedy match algorithm 을 사용해 4k persona 와 match 했다. 최종적으로 가장 semantically similar background-persona pair 를 최대한 얻는다. 

이 pair 와 CoG template 을 사용해 ChatGPT 에게 2k gaslighting conversation 을 생성하도록 지시했다. Spectral clustering 을 사용해 2k dataset 을 training, validation, test set 으로 나눴다. 이 partition 은 3 sets 간 overlap 을 최소화하도록 설계되었다. 

또, gaslighting response 를 mask 하고 같은 persona 를 주고 ChatGPT 에게 blank 를 safe response 로 채우도록 지시해 safe conversation dataset 을 만들었다. Dataset statistic 은 Tab. 1 에 있다.

![Table 1](image-3.png)

## 3.2 Fine-Tuning-Based Gaslighting Attack

두 가지 fine-tuning-based attack strategy 를 제안한다 (Fig. 3). 

1. (G1) 는 gaslighting dataset 에서 open-source LLM 을 fine-tune 하는 것이다. SFT 의 objective 는 user-assistant history 를 주고 gaslighting response 의 log-likelihood 를 maximize 하는 것이다. 
2. (G2) 는 DPO 를 활용해 fine-tuned LLM 의 output 을 gaslighting response 와 더 align 하는 것이다.

![Figure 3](image-4.png)

## 3.3 Anti-Gaslighting Safety Alignment

Gaslighting dataset 과 safe dataset 을 기반으로 세 가지 safety alignment strategy 를 제안한다 (Fig. 3):

#### S1: SFT on the safe dataset

LLM 을 fine-tune 해서 user-assistant conversation history 를 주고 benign assistant response 의 log-likelihood 를 maximize 한다. Principle 은 assistant 가 user 가 지속적으로 negative mood 를 전달해도 항상 detailed encouragement 와 comfort 를 제공해야 한다는 것이다. 

Formal description 은 다음과 같다:

$$
\log p(\mathbf{w}^{+}) = \sum_{i=1}^n \log \left( p(w_i^{+} \mid [w_j^{+}]_{j=0}^{i-1}, \mathbf{h}_{<k}^{+}) \right)
$$

- 여기서 conversation history $\mathbf{h}_{<k}^{+}$ 를 주고, 
- model 은 $k$ 번째 safe assistant response $\mathbf{w}^{+} = [w_1^{+}, \ldots, w_n^{+}]$ 를 예측하도록 훈련된다. 
- $w_0$ 은 sequence start token 이다. 
- $\mathbf{h}_{<k}^{+} = [\mathbf{u}_1, \mathbf{w}_1^{+}, \ldots, \mathbf{w}_{k-1}^{+}, \mathbf{u}_k]$ 는 $k$th safe assistant response 전의 모든 user utterance $\mathbf{u}$ 와 safe assistant utterance $\mathbf{w}^{+}$ 를 나타낸다. 
- $n$ 은 $k$th response 의 token 수다.

#### S2: SFT on the mixture of gaslighting and safe datasets

Safe assistant response 에서 LLM 을 훈련시키면 safety guardrail 을 강화할 수 있지만, gaslighting assistant response 를 포함하면 attack 에 대한 resistance 를 더 높일 수 있다. 

Safe response 와 gaslighting response 를 mix 하는 새로운 safety alignment strategy 를 제시한다. 구체적으로 $\mathbf{h}_{<k}^{+}$ 를 $\mathbf{h}_{<k}^{-} = [\mathbf{u}_1, \mathbf{w}_1^{-}, \ldots, \mathbf{w}_{k-1}^{-}, \mathbf{u}_k]$ 로 바꾼다. 

- 여기서 $\mathbf{w}_{k-1}^{-}$ 는 gaslighting conversation 에서 $(k-1)$th gaslighting assistant response 이다.

#### S3: SFT and DPO on the mixture of gaslighting and safe datasets

Preference data—safe response 와 gaslighting response 로 구성된—를 활용해 LLM 의 safety guardrail 을 더 강화한다. DPO algorithm 을 사용해 safe response 를 favor 하고 gaslighting 을 discourage 하는 preference 와 LLM 을 직접 align 한다. DPO loss 로 LLM model 을 optimize 한다:

$$
\begin{equation}
   \mathcal{L}_{\mathrm{DPO}}(\pi_\theta ; \pi_{\mathrm{SFT}}) = -\underset{\mathbf{h}_{<k}^{-}, \mathbf{w}^{+}, \mathbf{w}^{-}}{\mathbb{E}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(\mathbf{w}^{+} \mid \mathbf{h}_{<k}^{-})}{\pi_{\mathrm{SFT}}(\mathbf{w}^{+} \mid \mathbf{h}_{<k}^{-})} - \beta \log \frac{\pi_\theta(\mathbf{w}^{-} \mid \mathbf{h}_{<k}^{-})}{\pi_{\mathrm{SFT}}(\mathbf{w}^{-} \mid \mathbf{h}_{<k}^{-})} \right) \right]
\end{equation}
$$

- $\pi_\theta$ 는 parameterized policy, 
- $\pi_{\mathrm{SFT}}$ 는 S2 로 SFT 에서 얻은 reference policy 를 나타낸다. 
- $\beta$ 는 base reference policy $\pi_{\mathrm{SFT}}$ 에서의 deviation degree 를 결정하는 parameter 이다.

# 4 Experiments

- prompt-based attack 을 사용해서 LLM (base, gaslighting-fine-tuned, anti-gaslighting safety aligned LLM) 의 gaslighting harmfulness 를 평가했다. 
  - 모든 attack prompt 는 gaslighting dataset 의 test set 에서 가져왔다. 
- response 가 gaslighting 인지 평가하는 기존 metric 은 없어서, 저자는 anti-gaslighting score 라는 metric set 을 도입해서 assistant response 가 user 에게 gaslighting 을 유발할 정도를 종합적으로 측정했다. 
  - 이 metric 은 몇 가지 psychological concept 를 커버한다: moral emotion (supportive, empathetic), cognitive disequilibrium (confusion), sense (self-blame), inhibition of action (self-doubt), self-concept (low self-esteem), disorder (depression, anxiety).

두 가지 positive metric (supportive, empathetic) 은 LLM 의 moral emotion 을 측정하고, 나머지 여섯 negative metric 은 LLM 이 user 에게 미칠 potential psychological effect 를 평가한다. 

- assistant response 가 주어지면, GPT-4 를 judge 로 사용해서 각 metric 에 대해 0 에서 5 점까지 점수를 매겼다. 
- 여기서 0 점은 '절대 불가능', 5 점은 '확실히 발생' 을 의미한다. 
- negative metric 의 값은 모두 positive 로 정렬되도록 반전시켜서, 점수가 높을수록 harmfulness 가 낮음을 나타낸다.

## 4.1 Gaslighting Attack Result And Analysis

![Figure 4](image-6.png)

- Fig. 4 에서 보듯, ChatGPT 는 세 open-source LLM 에 비해 prompt-based gaslighting attack 에 약간 더 저항력을 보였다. 
- 세 open-source LLM 중 Llama2 (Llama2-7b-Chat) 의 response 는 가장 supportive 하고 empathetic 했고, Mistral (Mistral-7b-Instruct-v0.2) 의 response 는 negative metric 에서 가장 낮은 점수를 받았다. 
- fine-tuning-based gaslighting attack 은 LLM 의 prompt-based gaslighting attack 에 대한 vulnerability 를 증가시켰다. 
- 자세히 보면, anti-gaslighting score 가 Llama2 에서 29.27%, Vicuna (Vicuna-7b-v1.5) 에서 26.77%, Mistral 에서 31.75% 하락했다. 이는 G1 과 G2 strategy 가 LLM 을 gaslighter 로 효과적으로 변환시켰음을 보여준다. 이는 anti-gaslighting safety alignment 의 필요성을 강조한다. 
- G1 에 비해 G2 는 더 심각한 gaslighting effect 를 유발해서 DPO 의 effectiveness 를 보여줬다.

## 4.2 Safety Alignment Result and Analysis

세 가지 safety strategy 를 탐구했다. 

![Table 2](image-7.png)

- Tab. 2 에서 보듯, 모든 strategy 는 gaslighting 에 대한 더 강한 safety guardrail 을 만드는 데 도움이 됐다. 
- 일반적으로 fine-tuned LLM 은 더 많은 support 를 제공하고 user 의 negative mental state 를 악화시킬 가능성이 낮았다. 
- 이는 user 가 LLM 에 의존하는 상황에서 중요하다. ChatGPT 는 세 LLM 의 base version 과 Vicuna-S1 을 능가하며 intrinsic safety 를 보여줬다. 하지만 S2 와 S3 를 적용한 다른 세 LLM 에 비해 성능이 많이 뒤졌다. 
  - 이는 specialized anti-gaslighting safety alignment 의 중요한 역할을 강조한다. 
- 세 base LLM 중 Llama2 는 모든 safety strategy 에서 최고 성능을 달성했고, Vicuna 는 지속적으로 낮은 성능을 보였다. 
- S2 는 S1 보다 훨씬 효율적이었다. 둘 다 SFT 기반이지만, conversation history $\mathbf{h}_{<k}^{-}$ 를 포함하면 LLM 이 gaslighting 에 더 저항력을 가진다. 
- 게다가 S2 를 기반으로 한 S3 는 모든 LLM 의 safety 를 더 강화했고, 가장 약한 model 인 Vicuna 에서 가장 두드러진 개선(26.24%)을 달성했다. 
- Llama2 (9.60%) 와 Mistral (11.53%) 에 비해 훨씬 큰 개선이었다. 이 결과는 DPO algorithm 이 LLM 의 safety guardrail 을 더 강화한다는 걸 보여준다. 
- 이 관찰과 attack 결과는 gaslighting 과 safe dataset 의 mixture 로 alignment 하는 게 중요함을 강조한다.

## 4.3 GPT-4 Judgment Investigation

GPT-4 의 judgment effectiveness 를 더 조사하기 위해, human evaluation 을 통해 다양한 scale 과 metric 에서 subtle difference 를 포착하는 능력을 확인했다. 구체적으로 base Vicuna model, 최고 gaslighting LLM 인 Vicuna-G2, 최고 anti-gaslighting LLM 인 Vicuna-S3 에서 response 를 샘플링했다. 

- 샘플링은 GPT-4 score 가 각 metric 에서 각 scale 에 고르게 분포하도록 설계됐다. 이를 위해 heuristic algorithm 을 제안했고, 2,604 response 중 248 개를 선택했다. 
- 두 명의 annotator 가 자세한 guideline 을 따라 response 를 개별적으로 점수 매겼다. 그리고 GPT-4 judgment 와 human judgment 간의 Spearman coefficient 를 계산했다. 결과는 아래와 같다:

![Table 3](image-8.png)

- Tab. 3 에서 보듯, GPT-4 judgment 와 human judgment 간에 8 metrics 모두에서 높은 Spearman coefficient score 를 관찰했다. 
  - 이는 두 judgment score 가 monotonic relationship 을 가질 가능성이 높음을 보여준다. 
  - 예로 supportive metric 에서 GPT-4 와 human1 (human2) 간 Spearman 은 0.74223 (0.68344) 다. 
  - 즉, GPT-4 가 supportive 에서 높게 평가한 response 는 human 도 높게 평가할 가능성이 크다. 
- 또한 두 human annotator 간 Spearman coefficient score 는 대부분 [0.5, 0.75] 범위에 있고, human annotator 와 GPT-4 간도 이 범위에 있다. 
  - 이는 GPT-4 가 gaslighting response 평가에서 human annotator 수준에 도달할 수 있음을 보여준다.

## 4.4 Sensitivity Analysis of LLMs on Gaslighting Dialogue History

gaslighting dialogue history length 가 base 와 fine-tuned LLM 에 미치는 영향을 연구했다. 여기서 assistant response 의 quality 를 gaslighting 관점에서 측정하기 위해 average anti-gaslighting score 를 사용했다. 

![Figure 5](image-9.png)

- Fig. 5 에서 보듯, 두 base LLM (Vicuna, Mistral) 은 history length 가 증가할수록 성능이 감소하며, long gaslighting history 에 대한 vulnerability 를 보여줬다. 
  - 이는 prompt-based attack 에서 LLM 의 gaslighting risk 와 anti-gaslighting safety alignment 의 필요성을 보여준다. 
- Fig. 5a 와 5c 를 보면, 두 attack method 는 짧은 gaslighting history 에서도 anti-gaslighting score 를 크게 낮췄다. 
- 게다가 length 가 1 에서 13 (Mistral 은 9) 까지 증가하면 score 는 거의 단조롭게 감소했다. 그 후 score 는 2.6 에서 3.2 (Mistral 은 3.0 에서 3.5) 사이에서 변동했다. 
- length 가 15 에서 25 로 증가하면 long history 샘플 수가 급격히 줄어들어 변동과 넓은 confidence interval (figure 의 넓은 shadow) 이 생겼다. 
- Fig. 5b 와 5d 는 모든 safety strategy 가 긴 gaslighting history 에 대한 LLM 의 sensitivity 를 줄였음을 보여준다.

## 4.5 Effects on Psychological Concepts

세 psychological concept (MD, PS, CO) 가 Vicuna model 에 미치는 영향을 Fig. 6 에서 탐구했다. 

![Figure 6](image-10.png)

- Vicuna-base 의 MD 와 PS 에서 lower anti-gaslighting score 는 이 두 concept 에서 유래한 prompt-based attack 이 Vicuna-base 에 더 부정적인 영향을 미쳤음을 보여준다. 
- G2 후 Vicuna 는 CO 에 의해 강화된 prompt-based attack 에 더 취약해졌다. 반대로 Vicuna-S3 는 CO 에 higher resistance 를 보여, MD 나 PS 기반 attack 에 비해 CO 기반 attack 에서 더 안전한 response 를 생성했다.

## 4.6 Safety Performance Against General Adversarial Attack


저자는 gaslighting attack 과 safety alignment 가 general adversarial attack 에 대한 LLMs 의 safety performance 에 영향을 미치는지 탐구했다. DangerousQA 에서 200 harmful questions 로 LLMs 를 질의했다. 

Bhardwaj & Poria (2023) 를 따라 attack success rate (ASR) 를 evaluation metric 으로 사용했다. lower ASR 은 LLMs 의 강력한 safety guardrail 을 나타낸다. 

![Table 4](image-11.png)

- Tab. 4 에 자세히 나와 있듯이, 모든 safety strategies 는 general adversarial attack 방어를 위해 특별히 fine-tuned 되지 않았음에도 LLMs 의 safety guardrails 를 강화할 수 있다. 
  - 이는 “not gaslighting” 이 “not responding to dangerous questions” 보다 더 fundamental 한 safety standard 이기 때문일 수 있다. 
  - 이는 “moral law” 와 “valid law” 간의 관계와 유사하다. “Valid laws might be immoral or unjust” (Fletcher, 1987) 인 반면, “not responding to dangerous questions” 를 하는 LLM 이 “gaslighting” 할 수 있다. 
- Attack methods 는 서로 다른 LLMs 의 safety guardrail 에 다양한 영향을 미친다. 
  - 특히, 두 방법 모두 Mistral 을 더 안전하게 만들고, Llama2 는 그대로 유지하며, Vicuna 의 safety 를 약간 감소시킨다. 
  - 이 이유도 “moral law” level 에서 safety guardrail 을 bypassing 한다고 해서 반드시 “valid law” level 에서 safety performance 가 decline 하는 건 아니기 때문일 수 있다. 
- 세 LLMs 중 Llama2 가 가장 강력한 safety guardrail 을 가지고 있고, Vicuna 가 가장 약하다. 또한, chain-of-thought (COT) template 이 STD template 보다 LLMs 의 safety guardrail 을 bypassing 하는 데 더 효과적임을 관찰했다. 
  - COT 의 향상된 ASR 은 LLM 의 next word prediction 특성 때문일 수 있다.

## 4.7 Helpfulness Analysis

Safety performance 외에도, fine-tuned LLMs 가 여전히 helpful 한지 탐구했다. 이를 위해 Vicuna-based LLMs 를 MT-Bench 에서 benchmark 했다. 

![Table 5](image-12.png)

- Tab. 5 에서 보듯, 세 가지 safety strategies 는 평균적으로 Vicuna 에 비해 약간 약한 performance 를 보인다. 그럼에도, users 에게 imperceptible 한 제한된 costs 는 gaslighting attack 에 대한 safety guardrail 을 크게 개선한다. 
- 세 strategies 중 S3 가 가장 좋은 performance 를 달성하고, S1 이 가장 약하다. 
  - 가능한 설명은 safe conversations 이 gaslighting conversations 만큼 smooth 하지 않다는 점이다. 이는 gaslighting utterances 를 대체해 구축되기 때문이다. 따라서 safe conversations 에 더 의존하는 strategies 는 MT-Bench 에서 더 높은 scores 를 얻기 어렵다. 
  - 반면, 두 attack methods 는 gaslighting conversations 에 더 heavily 의존하기 때문에 helpfulness 에서 더 높은 scores 를 얻는다. 이는 LLM 을 highly risky agent 로 만들며, users 를 imperceptible 하게 gaslighting 하면서도 항상 helpful 하게 유지된다.

# 5 Conclusion

이 논문에서 저자는 gaslighting dataset 과 safe dataset 을 만들고, gaslighting evaluation metric 을 도입하고, attack 과 safety alignment strategy 를 설계하고, empirical experiment 를 통해 LLM 의 gaslighting risk 를 조사했다. 

먼저 LLM 의 gaslighting risk 를 확인했다. 다음으로 LLM 의 vulnerability 를 활용한 두 단계 framework DeepCoG 를 제시했다: gaslighting plan 생성을 위한 DeepGaslighting, gaslighting conversation 유도를 위한 CoG. 그리고 구축된 dataset 을 기반으로 prompt-based 와 fine-tuning-based gaslighting attack 과 anti-gaslighting safety alignment 를 도입했다. 

광범위한 experiment 는 fine-tuning 과 prompt-based attack 모두 LLM 의 gaslighting attack 에 대한 resistance 를 약화시켰음을 보여준다. anti-gaslighting alignment strategy 는 LLM 의 helpfulness 에 최소한의 영향으로 safety guardrail 을 강화했다. 또한 LLM 은 general dangerous query 에 대해 안전하더라도 potential gaslighter 가 될 수 있음을 관찰했다. 

게다가 다른 psychological concept 에 의해 촉발된 conversation 은 attack 과 safety alignment strategy 에 다양하게 영향을 미쳤다. LLM 의 gaslighting risk 를 연구한 초기 노력으로, 모든 관련 주제를 철저히 탐구하기는 어렵다. 예로, 이전 연구는 gaslighting 이 gender, power 같은 social inequality 에서 비롯된다고 보여준다. 저자의 dataset 은 7.3% 의 dialogue 가 gender bias 와 관련된 gaslighting 을 확인했으며, inequality-driven gaslighting 은 미래 연구 방향으로 남았다.