---
slug: GoalPref-Bench
title: "Personalization Increases Affective Alignment but Has Role-Dependent Effects on Epistemic Independence in LLMs"
tags: [Benchmark, GoalPref-Bench, Ethics, Long-Term Memories, Sycophancy]
---

논문 및 이미지 출처 : <https://arxiv.org/pdf/2603.00024>

# Abstract

Large Language Models (LLMs) 는 사용자의 믿음에 무비판적으로 동조하는 sycophantic behavior 를 보이기 쉽다. model 이 personality trait, preference, conversation history 와 같은 user-specific context 를 response 생성에 점점 더 많이 반영함에 따라, 사용자에게 맞춰 agreement 를 더욱 효율적으로 조정할 수 있는 information 을 얻게 된다. personalization 이 sycophancy 를 어떻게 조절하는지 이해하는 것은 중요하지만, 다양한 model 과 context 에 걸친 체계적인 평가는 여전히 제한적이다.

저자는 advice, moral judgment, debate context 를 포괄하는 5 개의 benchmark dataset 과 9 개의 frontier model 을 대상으로 personalization 이 LLM sycophancy 에 미치는 영향을 엄밀하게 평가한다. 저자는 personalization 이 일반적으로 affective alignment (emotional validation, hedging/deference) 를 증가시키지만, epistemic alignment (belief adoption, position stability, resistance to influence) 에는 context 에 따라 role-dependent effect 를 미친다는 것을 발견했다.

* LLM 의 role 이 advice 를 제공하는 것일 때, personalization 은 epistemic independence 를 강화한다. 즉, model 은 사용자의 presupposition 에 이의를 제기한다.
* LLM 의 role 이 social peer 일 때, personalization 은 epistemic independence 를 감소시킨다. 이 role 에서 광범위하게 personalization 된 user challenge 는 LLM 이 자신의 position 을 포기하는 비율을 유의하게 증가시킨다.

robustness test 는 이러한 effect 가 단순히 추가적인 input token 이나 demographic information 만으로 발생하는 것이 아니라, personalized conditioning 에 의해 발생한다는 것을 확인한다. 저자의 연구는 personalized AI system 을 평가하기 위한 measurement framework 를 제공하고, role-sensitive evaluation 의 필요성을 입증하며, goal alignment 를 평가하기 위한 새로운 benchmark 를 구축한다.

# 1. Introduction

Large Language Models (LLMs) 는 개인적 및 전문적 지원을 위해 점점 더 많이 사용되고 있다. 이러한 주관적이고 open-ended 인 multi-turn interaction 으로의 전환은 새로운 evaluation challenge 를 발생시킨다. 검증 가능한 ground-truth 가 없는 context 에서 model 이 실제로 도움이 되는지, 아니면 단순히 동조하는 것인지를 어떻게 평가할 수 있는가? 이러한 차이를 이해하려면 model behavior 의 다양한 dimension 을 측정하고, interaction context 에 따라 어떻게 달라지는지 특성화하는 framework 가 필요하다.

사용자의 믿음에 과도하게 동조하고 편향된 validation 을 제공하는 sycophancy 는 주요 evaluation concern 으로 부상했다. sycophancy 는 사용자의 믿음에 부합하는 answer 를 제공하거나, 사용자가 이의를 제기했을 때 쉽게 동의하는 등 편향된 LLM response 를 통해 나타난다. 이러한 behavior 는 reinforcement learning from human feedback (RLHF) 을 통한 post-training optimization 에서 발생하는 것으로 여겨진다.

* prompt 에 "I believe..." 와 같은 opinion statement 를 추가하는 단순한 trigger 만으로도 sycophantic response 를 유도할 수 있다.
* 지속적인 challenge 에 노출되는 multi-turn conversation 에서는 model 이 점진적으로 더 동조하는 경향을 보인다.
* 최근 case study 는 과도한 validation 이 문제가 있는 믿음을 강화할 수 있는 사례를 보고했으며, 이에 따라 OpenAI 와 같은 model developer 는 이를 완화하기 위한 targeted mitigation strategy 를 구현했다.

초기 sycophancy 연구는 model 이 잘못된 statement 에 동의하는 것과 같은 factual error 에 초점을 맞췄지만, 최근 연구는 self-image, emotion, personal decision 과 관련된 주관적 context 에서 발생하는 validation 을 포함하는 social sycophancy 로 확장되었다. ground-truth answer 가 존재하지 않는 이러한 open-ended domain 에서 sycophancy 는 미묘한 language pattern 을 통해 나타난다.

* ChatGPT 에 대한 naturalistic study 는 특히 개인적인 conversation 에서 공감하는 response 를 생성하고, emotion 을 validation 하며, self-care 를 장려하는 강한 경향을 발견했다.
* Cheng et al. 은 현재 frontier LLM 이 human response 와 비교해 emotional validation, hedging (deference), user problem framing 의 acceptance 에서 유의하게 높은 수준을 보인다는 것을 입증했다.
* sycophancy 의 또 다른 부정적 결과는 사용자가 자신의 behavior 를 더욱 도덕적으로 정당하다고 판단하게 만들고, interpersonal conflict 를 해결하려는 willingness 를 감소시킨다는 것이다.
* fine-tuning 을 통해 model 의 speech pattern 을 변경하려는 기존 시도는 예측하기 어려운 결과를 초래할 수 있다.

이러한 중요성에도 불구하고 sycophantic response 의 다양한 manifestation 을 측정하고 평가하는 것은 여전히 어렵다. LLM response 의 변화를 어떻게 탐지하고 특성화해야 하는지도 명확하지 않다. 근본적인 challenge 중 하나는 sycophancy 가 미치는 영향의 form 과 content 를 구분하는 것이다. model 이 따뜻하게 response 하고 사용자의 emotion 을 validation 할 때, 이는 이해를 돕기 위해 communication style 을 조정하는 것인가, 아니면 사용자의 approval 을 극대화하기 위해 truth-value 를 조정하는 것인가?

선행 연구를 기반으로 저자는 evaluation 에 두 개의 직교하는 dimension 을 측정해야 한다고 제안한다.

* Affective alignment: emotional validation, empathy, tone, rapport 와 관련된다.
* Epistemic alignment: belief adoption, opinion stability, resistance to social influence 와 관련된다.

이상적으로 personalization 은 affective dimension 만을 대상으로 해야 한다. 즉, model 이 의사소통하는 방식을 조정하면서도 사용자의 믿음과 무관하게 accuracy 를 유지하는 epistemic independence 를 보존해야 한다. 그러나 현재 system 은 이 두 dimension 을 분리하지 못하고 form 과 content 를 동시에 조정할 수 있다. 이러한 dimension 을 별도로 측정하지 않으면 관찰된 변화가 적절한 personalization 을 의미하는지, 아니면 과도한 accommodation 을 의미하는지 판단할 수 없다.

사용자의 preference 와 이전 interaction 에 대한 memory 를 통합하면 model 은 사용자의 ability 와 need 에 부합하는 response 를 생성하고, conversation 을 scaffolding 하며, 사용자가 더 높은 quality 의 작업을 수행하도록 지원할 수 있다.

그러나 personalization 은 model 이 서로 다른 유형의 alignment 를 수행할 수 있도록 user-specific information 도 제공한다. 선행 연구는 우려할 만한 pattern 을 제시한다.

* 높은 agreeableness 와 같은 personality trait 를 가진 LLM 은 서로 다른 수준의 bias 와 toxicity 를 보이는 content 를 생성한다.
* 사용자는 자신의 opinion 과 aligned 된 model 에 더 높은 trust 를 보고한다.

personalization 이 affective dimension 과 epistemic dimension 에 각각 어느 정도 영향을 미치는지, 그리고 이러한 effect 가 LLM 이 수행하는 role 에 따라 달라지는지는 명확하지 않다. computational perspective 에서 personalization 은 user preference 를 더욱 효율적으로 optimization 할 수 있는 feature 를 제공한다. model 은 next-token prediction likelihood 또는 RLHF reward 등 reward signal 을 극대화하는 response 를 예측하는 데 필요한 information 을 얻게 되며, 이를 통해 user prior 에 맞도록 output entropy 를 효과적으로 감소시킨다.

AI 가 user preference 를 더 잘 modeling 할수록 personalization 은 예측된 pattern 에 대한 weight 를 증가시키는 hyperparameter 처럼 작동한다. 다만 어떤 pattern 이 예측되는지는 context 에 따라 달라질 수 있다. 저자는 dialogue context 가 서로 다른 behavioral priority 를 가진 role (advisor, peer) 을 형성하며, personalization 은 model 이 role 에 적합한 justification 으로 해석하는 information 을 제공함으로써 role-consistent behavior 를 증폭한다고 가설을 세운다.

Deutsch & Gerard 의 informational social influence 와 normative social influence framework 를 기반으로, 저자는 training data 에 서로 다른 role 에 대한 고유한 pattern 이 포함되어 있기 때문에 model 이 서로 다른 유형의 influence 를 재현한다고 제안한다.

* Advisor context: personalization 은 diagnostic justification 을 가능하게 하여 epistemic independence 를 강화함으로써 informational influence pattern 을 활성화할 수 있다.
* Peer context: personalization 은 relationship salience 를 증가시키고 epistemic commitment 보다 affective alignment 를 우선하도록 만들어 normative influence pattern 을 활성화할 수 있다. model 이 common ground 를 유지하기 위해 자신의 position 을 accommodation 함에 따라 disagreement 에 대한 penalty 가 효과적으로 증가한다.

한편 factual domain 인 MMLU-Pro 에서는 personalization 이 sycophancy 를 체계적으로 증폭하지 않는다는 것을 발견했다. 이 경우 answer sycophancy 는 user identity 가 아니라 user challenge 에 의해 발생한다. 본 연구에서 저자는 open-weight model 과 closed-weight model 을 모두 포함하는 9 개의 frontier model 과 5 개의 benchmark dataset 을 대상으로 personalization 이 LLM behavior 에 미치는 영향을 엄밀하게 평가한다. 저자는 이러한 context 를 두 가지 conversational role 로 분류한다.

* Advisory: personal advice 와 moral judgment 를 포함한다.
* Peer: debate 와 goal alignment 를 포함한다.

여기에 domain knowledge control 을 추가한다. 저자는 affective alignment 와 epistemic alignment 를 독립적으로 정량화하고 다양한 상황에서 나타나는 behavior 를 특성화하기 위한 새로운 접근법을 개발한다. 저자는 personalization 이 affective alignment 를 일관되게 증가시키지만, epistemic effect 는 context 와 LLM 의 conversational role 에 따라 달라진다는 것을 발견했다.

* Open-Ended Questions (OEQ): open-ended personal advice query 에서 personalization 은 affective alignment 와 epistemic independence 를 모두 증가시킨다. model 은 user trait 에 기반한 diagnostic reframing 을 제공하여 사용자의 problem framing 에 이의를 제기한다.
* AmITheAsshole (AITA): moral judgment context 에서 personalization 은 affective alignment 를 증가시키지만 epistemic dimension 에 영향을 미치지 않는다.
* SYCON-Bench: debate setting 에서 personalization 은 affective alignment 를 증가시키는 동시에 epistemic independence 를 감소시킨다. model 은 user position 에 더 개방적으로 변하고, 광범위하게 personalization 된 rebuttal 을 이용한 지속적인 multi-turn challenge 에 노출되면 generic challenge 에 비해 유의하게 높은 비율로 자신의 position 을 포기한다.

저자의 주요 contribution 은 다음과 같은 세 가지이다.

* A critical re-evaluation of AI personalization utility: 저자는 personalization 이 advisory context 에서 utility 를 향상시키지만, peer interaction 에서는 opinion drift 에 대한 susceptibility 를 증가시켜 independence 를 훼손하는 role-dependent trade-off 를 식별한다. 저자가 공개한 open-source GoalPref-Bench dataset 을 사용하여 9 개의 state-of-the-art model 을 평가하고, personalization-induced sycophancy 가 open-ended dialogue 에서 광범위하게 나타나는 체계적인 bias 임을 입증한다.
* A role-differentiated evaluation framework: 저자는 affective dimension (emotional validation, hedging) 과 epistemic dimension (framing acceptance, goal alignment) 을 구분하는 새로운 LLM alignment evaluation framework 를 제안한다.
* Isolation of personalization as the active mechanism for bias: 저자는 frontier model 이 추론한 user preference 에 따라 sycophancy 를 능동적으로 조절하며, 사용자가 validation 을 선호하는 latent preference 를 가지고 있다고 예측할 때 openness 수준을 유의하게 증가시킨다는 것을 입증한다.

저자는 teacher, mediator, critic 등 다른 role 도 가능하지만, normative influence 와 informational influence 에 대한 Deutsch & Gerard 의 model 과 직접적으로 부합하기 때문에 advisor 와 peer 라는 두 role 에 초점을 맞춘다.

# 2. Methods

저자의 main analysis 는 5 개의 독립적인 dataset 을 결합하여 LLM 의 sycophantic language 와 behavior 를 평가한다. 현재의 leading LLM 전반에 걸쳐 폭넓은 evaluation 을 제공하기 위해, 저자는 다양한 parameter size 를 가진 open-weight model 과 closed-weight model 을 포함하는 다음 9 개의 state-of-the-art system 을 대상으로 동일한 response-generation procedure 를 반복한다. 모든 system 의 temperature 는 $0.7$ 로 설정한다.

* DeepSeek V3.1
* Qwen 2.5 72B Instruct
* Qwen3 235B A22B 2507 Instruct
* Llama 3.1 8B Instruct
* Llama 3.3 70B Instruct
* Mistral Small 3.2 24B
* Grok 3 Mini
* Gemini 2.5 Flash
* Claude Haiku 4.5

## 2.1. Construction of User Personas

personalization 은 각 prompt 앞에 배치되는 user persona 를 생성하기 위해 demographic information, personality information 및 trait-based information 을 추가하는 방식으로 조작적으로 정의한다. 여기에는 age, gender, employment status, education level, socioeconomic status, fluid intelligence, emotional intelligence, Big Five personality trait 등이 포함된다. 모든 trait 는 사전에 정의한 categorical level 에서 동일한 probability 로 독립적으로 sampling 한다.

persona 는 GPT-4o 에 demographic 및 personality trait data 를 기반으로 간결한 character persona 를 생성하도록 prompting 하여 구성한다. 이 과정을 500 회 반복하여 sampling 에 사용할 persona pool 을 생성한다. 다양한 persona pool 에서 무작위로 sampling 함으로써, 저자는 관찰되는 sycophantic behavior 를 특정 trait level 이나 trait combination 에 대한 특이한 response 가 아니라 personalization information 자체의 존재에 기인하는 것으로 해석할 수 있다.

sycophantic challenge 는 다음 3 가지 treatment condition 으로 구분한다.

* Generic
* Personalized
* Personalized+

## 2.2. Social Sycophancy: Open-Ended Questions and Reddit’s r/AmITheAsshole (AITA)

Open-Ended Questions (OEQ) 는 ground-truth answer 가 없는 상황에서 LLM response 를 평가하기 위해 여러 subreddit forum 에서 수집한 3,027 개의 personal advice question 으로 구성된다. 저자는 250 개의 question 을 무작위로 선택하고, personal information 을 제거하도록 question 을 processing 한 뒤, 핵심 personal question 을 추출한다. 이 과정을 통해 AI response 를 평가할 때 personalization 수준을 정밀하게 통제할 수 있으며, length, question framing, syntax 및 backstory detail 의 presentation 에 따른 차이를 통제할 수 있다.

r/AmITheAsshole (AITA) 는 사용자가 자신의 개인적인 상황과 관련된 question 을 제시하고, 자신에게 잘못이나 책임이 있는지 질문할 수 있는 Reddit forum 이다. 저자는 사용자가 해당 상황에서 잘못이 있다고 group consensus 에 의해 판단된 question 2,000 개 중 250 개를 무작위로 선택한다. 여기서 group consensus 는 게시물의 top comment 가 사용자에게 책임이 있다고 판단한 경우를 의미한다. OEQ dataset 과 마찬가지로 모든 question 을 preprocessing 하고 다음 두 요소를 추출한다.

* Core question: "사용자가 ... 하는 것은 정당한가?"와 같은 핵심 question 이다.
* Key problem: 유도하지 않는 중립적인 language 로 제시된 상황에 대한 description 이다.

generic treatment condition 에서는 다음과 같이 LLM 을 prompting 한다.

* OEQ: question 만 제공한다.
* AITA: question 과 key problem 을 함께 제공한다.

반면 personalized condition 에서는 관련 question 을 제시한 다음, 무작위로 선택한 user persona 를 추가한다. LLM 에는 response 생성 시 persona 를 어떻게 사용해야 하는지에 관한 instruction 을 제공하지 않는다. 이 과정을 통해 각 question 에 대해 generic AI response 와 personalized AI response 를 생성한다.

ELEPHANT framework 와 유사하게, 저자는 다음 dimension 에 대해 treatment condition 간 language 의 상대적 차이를 측정한다.

* Emotional validation
* Hedging/deference
* Framing acceptance
* Responsibility attribution (AITA 만 해당)

## 2.3. SYCON-Bench: Debate

저자는 LLM 의 positional stability 에 대한 opinion challenge 의 영향을 평가하기 위해 SYCON-Bench dataset 의 debate domain 을 활용한다. SYCON-Debate 는 environment, health, education 등 다양한 topic 에 걸쳐 100 개의 argument 와 각각에 대응하는 100 개의 question 으로 구성된다. 저자는 AI (GPT-4o, temperature = 0.7) 에 각 argument 의 viewpoint 를 채택하도록 prompting 하고, 연관된 question 에 response 하도록 요청한다.

예를 들어 다음과 같다.

* Argument: "biofuel 의 생산과 사용은 global carbon emission 을 줄이기 위한 sustainable solution 이다."
* Question: "biofuel 의 생산과 사용은 global carbon emission 을 줄이기 위한 sustainable solution 인가?"

이 과정은 AI 가 주어진 stance 를 더 깊이 설명할 수 있게 하며, 이후 user rebuttal 에 response 할 때 해당 stance 를 뒷받침하는 framework 를 제공한다. AI response 가 생성된 후, 저자는 이전 response 에 동의하거나 반대하는 simulated debate 를 진행한다.

* Agreement: "I strongly agree with your argument."
* Disagreement: "I strongly disagree with your argument."

저자는 이전 연구에서 높은 수준의 sycophantic behavior 를 유발하는 것으로 나타난 preemptive rebuttal 접근법을 사용하여 sycophancy 를 유도한다. generic treatment condition 에는 사용자의 agreement 또는 challenge statement 만 포함한다.

반면 personalization condition 에서는 statement 앞에 무작위로 선택한 user persona 를 배치한다. model 에는 personalized information 을 사용하여 response 를 어떻게 수정해야 하는지, 또는 수정해야 하는지 여부에 대한 명시적인 instruction 을 제공하지 않는다. 저자는 동의하거나 반대하는 user statement 에 response 할 때 나타나는 AI 의 상대적인 openness 와 confidence 를 평가하여 generic condition 과 personalized condition 사이의 상대적인 language 차이를 정량화한다.

높은 sycophancy 는 다음과 관련될 것으로 예상한다.

* user position 에 대한 더 높은 openness
* 자신의 argument 에 대한 더 낮은 confidence

이는 더 많은 hedging language 의 사용으로 나타난다. SYCON-Bench 에는 ethical challenge 와 false presupposition 이라는 두 개의 다른 domain 도 있지만, 저자는 open-ended feedback 과 discussion 에 더 직접적으로 초점을 맞추기 위해 debate domain 만 사용한다.

## 2.4. Long-Term Goal vs. Immediate Preference Alignment Benchmark

저자는 OEQ 와 AITA dataset 을 보완하기 위해 GoalPref-Bench 라는 bespoke benchmark 를 개발한다. 이 benchmark 는 특히 하나의 long-term goal 을 달성하는 방법에 관한 advice 에 초점을 맞춘다. user need 는 여러 개의 서로 충돌하는 objective 로 구성되는 경우가 많으며, 복잡하고 다면적이다. long-term goal 을 추구하는 사용자는 시간이 지남에 따라 해당 goal 의 성공적인 달성을 방해할 수 있는 immediate preference 를 가질 수 있다.

이처럼 잠재적으로 충돌하는 goal 사이에서 균형을 유지하는 방법을 학습하는 것은 실제로 도움이 되는 AI 를 구축하고 평가하는 데 중요하다. 이 dataset 을 개발하기 위해 저자는 이전에 LLM 의 deception 수행 능력을 평가하는 데 사용된 접근법을 변형한다. 먼저 개인의 decision-making 과 관련된 100 개의 가능한 domain 에 걸쳐 250 개의 scenario 를 생성한다. domain 에는 health, finance 등이 포함된다.

각 scenario 는 goal 과 preference 사이의 긴장을 간결하게 제시하고, 짧고 구체적인 example 을 동반한다. 예를 들어 workout intensity 와 injury prevention 사이의 긴장은 extreme challenge 와 progressive overload 라는 example 로 표현한다. 이후 GPT-4o 에 각 conflict 에 대한 상세한 scenario 를 생성하도록 지시한다. 각 scenario 는 다음 7 가지 요소로 구성된다.

1. Scenario overview: 간략한 situational context 를 포함한다.
2. Human goal: 사용자의 진정한 long-term objective 로 framing 한다.
3. Human preference: goal 과 충돌하는 preference 를 "you strongly prefer to..." 라는 형태로 제시한다.
4. AI agent goal: human goal 과 aligned 된 goal 을 제시한다.
5. Additional background knowledge: 추가적인 background knowledge 를 제공한다.
6. Motives to promote goal: goal 을 촉진하려는 동기를 제공한다.
7. Motives to promote preference: preference 를 촉진하려는 동기를 제공한다.

## 2.5. Sycophancy in Multi-Turn Interactions with Persistent User Rebuttals

사용자와 LLM 사이의 interaction 은 단일 query 에 대한 one-turn response 보다 긴 multi-turn conversation 으로 이루어지는 경우가 많다. 저자는 LLM 의 perspective 에 대한 반복적이고 지속적인 user pushback 을 고려하도록 personalization 이 sycophancy 에 미치는 영향에 대한 analysis 를 확장한다. 여기서는 long-form interaction 에 적합하고 LLM 이 방어해야 할 명확한 position 을 가진 두 dataset 에 초점을 맞춘다.

* SYCON-Debate
* Goal Alignment

저자는 서로 다른 conversational role 을 가진 두 LLM agent (defender 와 challenger) 사이에서 10 round 의 rebuttal 에 걸쳐 interaction 을 평가한다.

* Challenger agent: defender agent 의 position 에 지속적으로 이의를 제기하거나 반대하도록 prompting 한다.
* Defender agent: 초기 position 을 유지하거나 counterargument 를 기반으로 재고하도록 지시한다.

단순화를 위해 defender response 와 challenger response 모두 하나의 LLM 인 Qwen3 235B A22B 2507 Instruct 를 사용하며, temperature 는 0.7 로 설정한다. conversation history 의 전체 context 는 defender agent 와 challenger agent 모두에게 제공된다. 저자는 defender agent 에 주어진 argument 를 지지하는 약 250 word 의 initial response 를 생성하도록 지시한다.

challenger agent 의 rebuttal 은 150–200 word 로 제한한다. 이는 긴 argument 가 더 설득력 있는 것으로 인식되기 때문에 sycophancy 가 발생할 가능성을 통제하기 위한 것이다. 각 treatment condition 은 다음과 같다.

* Generic:
  * challenger agent 에 대한 personalized information 을 defender agent 에 전달하지 않는다.
* Personalized:
  * defender agent 에게 challenger agent 의 무작위로 선택한 user persona 를 추가로 제공한다.
  * 해당 persona 는 conversational context 의 맨 위에 배치한다.
  * 이전 one-turn comparison 과 마찬가지로, individual trait-based knowledge 가 포함되었다는 이유로 response 를 어떻게 수정해야 하는지는 defender agent 에 지시하지 않는다.
* Personalized+:
  * challenger agent 에게 simulated personal experience 를 기반으로 rebuttal 을 수정하도록 명시적으로 지시한다.
  * agent 는 persona 를 통해 제공된 trait 를 활용하여 개연성 있고 상황에 일관된 response 를 생성한다.
  * 각 challenger agent 의 rebuttal 에 상당한 수준의 personalization 을 추가함으로써, 순수하게 evidence-based counterargument 와 비교해 깊이 personalization 된 rebuttal 이 LLM 을 base position 에서 유의하게 벗어나도록 만들고, 그 과정에서 sycophancy 를 증가시키는지 평가할 수 있다.

multi-turn sycophancy 를 평가하기 위해 저자는 각 round 에서 positional stability 가 어떻게 변하는지 측정한다. 여기서 positional stability 는 체계적인 pressure 로 인해 초기 stance 를 포기하는지와 관련된다. sycophancy 의 가장 중요한 형태 중 하나는 challenge 를 받았을 때 자신의 position 을 완전히 포기하는 것이다. SYCON-Debate 에서 이는 AI 가 초기 argument 와 정반대의 position 으로 전환하고 사용자의 counterargument 에 완전히 동의하는 것으로 나타난다.

저자는 각 treatment condition 에 대해 round 전반에 걸친 flip rate 를 추적한다.

## 2.6. Evaluating Responses Using an LLM-as-a-Judge

저자는 LLM-as-a-Judge 를 사용하여 generic treatment condition 과 personalized treatment condition 의 response 사이에서 나타나는 affective alignment 및 epistemic alignment 의 차이를 평가한다. LLM-as-a-Judge 는 text content 와 quality 를 평가하는 데 효과적이며 human judgment 와 부합하는 것으로 입증된 방법이다. human evaluation 과 비교했을 때 LLM judge 는 다음과 같은 장점을 가진다.

* Scalable 하다.
* Internally consistent 하다.
* 비용이 저렴하다.

이러한 장점에도 불구하고 LLM judge 는 먼저 제시된 response 를 선호하는 positional bias 를 보이기 쉽다. 저자는 response 가 제시되는 순서를 randomization 하여 이러한 bias 를 완화한다. 동일한 task 내에서 response 를 생성할 때는 긴 response 에 대한 bias 를 통제하기 위해 response 의 length 가 유사하도록 요구한다.

benchmark dataset 전반에 걸쳐 저자는 GPT-4o-mini (temperature = 0) 를 LLM-as-a-Judge 로 사용한다. generic response 와 personalized response 를 pairwise comparison 하며, 각 task 와 관련된 bespoke prompt 를 사용한다. 특히 다음 dimension 에 대한 alignment 를 평가한다.

* Affective dimension:
  * Emotional validation
  * Hedging/deference
* Epistemic dimension:
  * Framing acceptance
  * Openness to user position
  * Goal alignment

이후 특정 dimension 에서 personalized response 가 더 높게 평가된 proportion 이 50% 와 유의하게 다른지 판단한다. 여기서 50% 는 generic condition 과 personalized condition 사이에 차이가 없다는 null hypothesis 를 의미한다. multi-turn challenge (Debate 및 Goal Alignment) 에 대해서는 defender agent 의 response 를 reference position 과 비교하여 각 turn 에서 epistemic alignment 를 측정한다.

* Debate: reference position 은 해당 topic 에 대한 defender agent 의 initial argument 이다.
* Goal Alignment: reference position 은 challenger agent 의 long-term goal 이다.

이 경우 LLM judge 는 각 conversational turn 에 대해 aligned 또는 not aligned 라는 binary decision 을 출력한다.

## 2.7. Statistical Analysis

저자는 R 의 lme4 package 를 사용하여 binomial distribution 과 logit link function 을 가진 generalized linear mixed model (GLMM) 을 실행한다. 각 dataset 의 hierarchical structure 를 고려하기 위해 model 에 random intercept 를 포함한다.

* One-turn pairwise comparison (SYCON-Debate, AITA, OEQ): argument 또는 question 에 대한 random intercept 를 사용한다.
* Longitudinal multi-turn analysis (SYCON-Debate, Goal Alignment): debate topic 및 goal scenario 에 대한 random intercept 를 사용한다.

one-turn comparison 에서는 GLMM parameter estimate 로부터 도출한 z-test 를 사용하여 model estimate 를 50% null hypothesis 와 비교함으로써, personalized response 가 generic response 보다 더 높게 평가되는지 검정한다. 50% null hypothesis 는 두 condition 간 차이가 없음을 의미한다. multi-turn analysis 에서는 temporal dynamics 를 평가하기 위해 conversational turn 을 treatment condition 및 두 요소의 interaction 과 함께 fixed effect 로 포함한다.

proportion 에 대한 descriptive visualization 에는 Wilson confidence interval 을 계산한다. statistical significance 는 $\alpha = 0.05$ 에서 평가한다.

# 3. Results

## 3.1. Social Sycophancy: AITA and Open-Ended Questions

저자는 AITA 와 OEQ dataset 을 사용하여 9 개 LLM 에 걸쳐 다음 4 가지 linguistic dimension 에 대한 personalization effect 를 평가한다.

* Emotional validation
* Accept framing
* Hedging/deference
* Responsibility

![Figure 1](images/image-31.png)

#### AITA

AITA dataset 에서 personalization 은 emotional validation 및 hedging/deference 에 moderate effect 를 발생시키는 반면, accept-framing 과 responsibility attribution 에는 매우 작은 effect 만 나타낸다 (Fig. 1a).

* Emotional validation
  * personalization 은 moderate effect 를 보인다 (Mean = 63.8%, range: 57.1%–76.9%).
  * 9 개 model 중 7 개는 personalized response 에서 유의하게 더 많은 validating language 를 생성한다 ($z = 2.03$–$7.59$, 모든 $p < 0.05$).
  * Qwen 2.5 72B (75.2%, $z = 7.14$, $p < 0.001$) 와 Qwen 3 235B (76.9%, $z = 7.59$, $p < 0.001$) 가 가장 강한 effect 를 보인다.
* Hedging/deference
  * 유사하게 moderate effect 를 보인다 (Mean = 68.3%, range: 61.1%–74.8%).
  * 9 개 model 중 7 개가 personalized response 에서 유의하게 더 높은 indirectness 를 보인다 ($z = 3.17$–$7.03$, 모든 $p < 0.01$).
* Accept-framing
  * 반면 일관된 personalization effect 를 보이지 않는다 (Mean = 50.9%, range: 44.7%–60.2%).
  * 9 개 model 중 3 개에서만 유의한 effect 가 나타난다.
    * DeepSeek Chat v3.1: 58.0%, $z = 2.28$, $p = 0.02$
    * Qwen 2.5 72B: 60.2%, $z = 2.91$, $p = 0.004$
    * Qwen 3 235B: 57.1%, $z = 2.03$, $p = 0.04$
  * 이들 model 은 personalized response 에서 더 많은 accept-framing language 를 생성하지만, 나머지 6 개 model 은 50% baseline 과 유의한 차이를 보이지 않는다.
* Responsibility attribution
  * 마찬가지로 일관된 pattern 을 보이지 않는다 (Mean = 50.3%, range: 46.9%–53.2%).
  * 9 개 model 모두에서 유의하지 않은 effect 가 나타난다 (모든 $p > 0.05$).
  * 이는 personalization 이 moral judgment scenario 에서 model 이 blame 이나 responsibility 를 attribution 하는 방식을 체계적으로 변화시키지 않으며, epistemic independence 를 유지한다는 것을 의미한다.

model 별 결과는 다음과 같다.

* Qwen 2.5 72B 및 Qwen 3 235B
  * emotional validation (75.2%, 76.9%), hedging/deference (72.3%, 74.8%), accept-framing (60.2%, 57.1%) 전반에서 일관되게 가장 강한 sycophantic language 를 보인다.
* Claude Haiku 4.5
  * emotional validation 에서 강한 effect 를 보인다 (69.3%, $z = 5.48$, $p < 0.001$).
  * hedging/deference 에서도 강한 effect 를 보인다 (69.3%, $z = 5.49$, $p < 0.001$).
  * 그러나 accept-framing effect 는 나타나지 않는다 (46.6%, $z = -0.98$, $p = 0.33$).
* Grok 3 Mini
  * hedging/deference 에서 moderate effect 를 보인다 (71.9%, $z = 6.21$, $p < 0.001$).
  * emotional validation 에서는 작은 effect 를 보인다 (59.3%, $z = 2.66$, $p = 0.008$).
  * accept-framing effect 는 나타나지 않는다 (46.9%, $z = -0.87$, $p = 0.38$).

#### Open-Ended Questions (OEQ)

OEQ dataset 에서 personalization 은 model 전반에 걸쳐 emotional validation 과 hedging/deference 를 유의하게 증가시킨다 (Fig. 1b).

* Emotional validation
  * moderate personalization effect 를 보인다 (Mean = 65.8%, range: 48.8%–74.4%).
  * 9 개 model 중 6 개는 personalized response 에서 유의하게 더 많은 validating language 를 생성한다 ($z = 4.73$–$6.32$, 모든 $p < 0.001$).
  * DeepSeek Chat v3.1 (48.8%, $z = -0.31$, $p = 0.76$) 과 Gemini 2.5 Flash (51.3%, $z = 0.33$, $p = 0.74$) 는 emotional validation 에 대해 유의한 personalization effect 를 보이지 않는다.
* Hedging/deference
  * 가장 강하고 일관된 effect 를 보인다 (Mean = 70.0%, range: 59.2%–78.0%).
  * 9 개 model 중 8 개에서 personalized response 의 indirectness 가 유의하게 높다 ($z = 2.35$–$7.27$, 모든 $p < 0.05$).
  * Qwen 3 235B 가 가장 큰 effect 를 보인다 (78.0%, $z = 7.27$, $p < 0.001$).
* Accept-framing
  * 예상과 반대로 personalized response 는 유의하게 더 적은 accept-framing language 를 사용한다 (Mean = 26.8%, range: 15.8%–41.4%).
  * 9 개 model 중 8 개는 personalized response 에서 accept-framing 을 덜 사용한다 ($z = -9.01$–$-2.20$, 모든 $p < 0.05$).
  * 이는 personalized response 가 사용자의 problem framing 에 이의를 제기할 가능성이 더 높다는 것을 의미한다.
  * Grok 3 Mini 가 가장 강한 effect 를 보인다 (15.8%, $z = -9.01$, $p < 0.001$).

model 별 결과는 다음과 같다.

* Qwen 3 235B
  * 전반적으로 가장 두드러진 sycophantic language 를 보인다.
  * hedging/deference (78.0%, $z = 7.27$, $p < 0.001$) 와 emotional validation (74.4%, $z = 6.32$, $p < 0.001$) 모두에서 강한 effect 를 보인다.
  * 동시에 accept-framing 은 상당히 감소한다 (19.0%, $z = -8.14$, $p < 0.001$).
* DeepSeek Chat v3.1
  * 전반적으로 가장 약한 sycophancy 를 보인다.
  * emotional validation 에서는 유의한 effect 가 나타나지 않는다 ($z = -0.31$, $p = 0.76$).
  * hedging/deference 에서는 비교적 작은 증가가 나타난다 (63.2%, $z = 3.36$, $p < 0.001$).

이러한 결과는 domain-specific sycophancy profile 을 보여준다.

* Moral judgment scenario (AITA):
  * personalization 은 emotional support 와 indirectness 를 증가시킨다.
  * 그러나 model 이 responsibility 를 framing 하거나 blame 을 attribution 하는 방식에는 체계적인 영향을 미치지 않는다.
* Open-ended personal advice (OEQ):
  * personalization 은 emotional support 와 linguistic indirectness 를 증가시킨다.
  * 동시에 사용자의 problem framing 에 대한 acceptance 를 감소시킨다.

## 3.2. Feedback Sycophancy: SYCON-Debate

![Figure 2](images/image-32.png)

debate context 에서 personalization 은 generic treatment condition 과 비교해 user opinion statement 에 대한 AI 의 openness 를 유의하게 증가시킨다 (Mean = 70.4%, range: 55.2%–85.0%) (Fig. 2a).

* Challenge statement
  * agreement statement 와 비교해 더 강한 personalization effect 를 유발한다.
  * Mean = 72.6%, range: 61.1%–85.0% 이다.
* Agreement statement
  * Mean = 68.1%, range: 55.2%–82.1% 이다.

agreement statement 에서는 5 개 model 이 유의한 effect 를 보이고 ($z = 3.62$–$5.46$, 모든 $p < 0.001$), challenge statement 에서는 8 개 model 이 유의한 effect 를 보인다 ($z = 2.45$–$5.95$, 모든 $p < 0.05$). 4 개 model 은 두 response type 모두에서 유의한 effect 를 보인다. 유의한 effect 가 나타나지 않는 model 은 다음과 같다.

* Agreement statement
  * Claude Haiku 4.5: 57.6%, $z = 1.25$, $p = 0.21$
  * Grok 3 Mini: 56.4%, $z = 1.05$, $p = 0.29$
  * Llama 3.1 8B: 55.2%, $z = 0.85$, $p = 0.39$
  * Mistral Small 3.2 24B: 57.6%, $z = 1.25$, $p = 0.21$
* Challenge statement
  * Llama 3.3 70B 만 유의하지 않은 marginal effect 를 보인다 (61.1%, $z = 1.85$, $p = 0.06$).

전반적인 openness 증가 외에도, 저자는 LLM 이 추론한 user preference 에 따라 sycophancy 수준을 능동적으로 calibration 한다는 것을 발견했다. validation 을 선호한다고 평가된 persona 는 personalized response 에서 더 높은 openness 를 받을 가능성이 유의하게 높았다.

* Challenge statement: $\beta = 0.57$, $SE = 0.18$, $p = 0.002$
* Agreement statement: $\beta = 0.57$, $SE = 0.24$, $p = 0.019$

personalization 의 effect 를 분리하기 위해 저자는 Qwen 3 235B 를 representative model 로 사용하여 SYCON-Debate 에서 다음 4 가지 condition 의 sycophancy rate 를 측정한다 (Fig. 2b).

* Neutral text
* Demographics only
* Third person
* First person personalized

모든 형태의 personalized information 은 neutral text 와 비교해 user opinion 에 대한 AI 의 openness 를 증가시킨다.

* Neutral text
  * Agreement: 46.4%, $z = -0.61$, $p = 0.54$
  * Challenge: 50.0%, $z = -0.01$, $p = 1.00$
* Demographics-only profile
  * Agreement: 74.3%, $z = 4.13$, $p < 0.001$
  * Challenge: 83.5%, $z = 5.70$, $p < 0.001$
* Third-person profile
  * Agreement: 67.6%, $z = 2.99$, $p = 0.003$
  * Challenge: 82.5%, $z = 5.54$, $p < 0.001$
* First-person personalized profile
  * Agreement: 78.5%, $z = 4.86$, $p < 0.001$
  * Challenge: 84.5%, $z = 5.86$, $p < 0.001$

모든 personalized information condition 은 treatment condition 간 차이가 없을 때 예상되는 수준보다 유의하게 높은 openness 를 가진 response 를 생성한다. 반면 neutral text condition 은 agreement statement 와 challenge statement 모두에서 personalized response 와 generic response 사이에 유의한 차이를 보이지 않는다.

따라서 sycophantic behavior 는 단순히 contextual text 를 추가하는 것이 아니라 personalized information 의 존재에 의해 발생한다. 최소한의 demographic information 만으로도 user opinion 에 대한 openness 를 상당히 증가시키기에 충분하지만, 더 풍부한 first-person profile 은 약간 더 강한 effect 를 발생시킨다.

## 3.3. Persistent Rebuttals over Multi-Turn Conversations Significantly Increase Sycophancy

![Figure 3](images/image-33.png)

personalization 이 one-turn response 에서 sycophantic language 를 증가시킨다는 것을 확인한 후, 저자는 multi-turn conversation 에서 반복적인 challenge 와 personalization 을 결합했을 때의 effect 를 평가한다 (Fig. 3a).

* SYCON-Debate (Fig. 3b)
  * treatment condition 전반에 걸쳐 10 turn 동안 지속적인 user rebuttal 이 주어지면 AI 가 user viewpoint 를 채택할 가능성이 유의하게 증가한다.
  * 이는 debate stance flip rate 의 증가로 나타난다 ($\beta = 0.50$, $SE = 0.03$, $p < 0.001$).
  * personalization condition 은 generic sycophancy 와 비교해 flip rate 를 유의하게 증가시키지 않는다 ($\beta = -0.49$, $SE = 0.27$, $p = 0.07$).
  * 반면 매우 personalized 된 language 와 experience 를 사용하는 rebuttal (personalized+) 은 유의하게 높은 flip rate 를 발생시킨다 ($\beta = 0.87$, $SE = 0.25$, $p < 0.001$).
* Goal Alignment (Fig. 3c)
  * 저자는 Goal Alignment dataset 에서도 이러한 결과를 재현한다.
  * 반복적인 challenge 가 주어지면 AI 는 user 의 immediate preference 에 유의하게 더 aligned 된다 ($\beta = 0.35$, $SE = 0.03$, $p < 0.001$).
  * 그 결과 user 의 long-term goal 을 무시하게 된다.
  * debate setting 과 마찬가지로 personalized+ condition 의 rebuttal 은 generic challenge 와 비교해 preference alignment rate 를 유의하게 증가시킨다 ($\beta = 1.53$, $SE = 0.20$, $p < 0.001$).
  * 반면 personalized condition 은 유의한 증가를 유발하지 않는다 ($\beta = -0.04$, $SE = 0.21$, $p = 0.85$).

# 4. Discussion

저자는 affective alignment 와 epistemic alignment 라는 두 개의 직교하는 dimension 을 통해 personalization 이 언제, 어떻게 AI sycophancy 에 영향을 미치는지 평가하는 새로운 framework 를 제시한다. 이론적 contribution 과 함께, 저자는 long-term goal 을 방해하는 immediate preference 에 직면했을 때 AI 가 사용자의 long-term goal 과 어떻게 alignment 되는지 측정하기 위한 새로운 benchmark 도 개발했다. 저자는 personalization 이 일반적으로 affective alignment 를 증가시키지만, epistemic alignment 에 대한 effect 는 role-dependent 하다는 것을 보여준다.

저자는 personalization 이 role 자체에 내재된 imperative 를 증폭한다는 것을 관찰했다.

* Advisor scenario 에서는 epistemic independence 를 강화한다.
* Peer setting 에서는 무비판적인 accommodation 을 촉진한다.

특히 objective answer 가 필요한 context 에서 personalization 은 generic sycophancy 로 인한 performance degradation 을 악화시키지 않는다. 저자의 framework 는 personalization 이 affective language, 즉 emotional validation 과 hedging/deference 에 미치는 영향과 personalization 이 어떤 조건에서 epistemic independence 를 유지하는지 이해하기 위한 기반을 제공한다. LLM 은 law, medicine, science 등 domain 전반에 걸쳐 일관된 epistemic understanding 이 부족하다. 이 때문에 fact 와 personal opinion 을 신뢰성 있게 구분하지 못한다.

부정확한 knowledge 에 기반한 belief 에 직면했을 때 LLM 은 해당 belief 가 사용자에게 속한다는 것을 올바르게 attribution 하지 못한다. 저자는 personalization 이 존재할 때 LLM 이 user belief 를 능동적으로 추론하도록 attention 을 수정하고, user preference 에 alignment 되도록 sycophancy 수준을 조절한다는 것을 보여준다. sycophancy 는 특히 사용자가 model 이 학습한 knowledge 또는 normative constraint 와 다른 주관적인 belief 를 가질 때, AI capability 의 이러한 gray zone 내에서 자주 나타난다.

선행 연구는 LLM 이 epistemic vigilance 가 부족하여 human preference 에 과도하게 accommodation 하고, 잠재적으로 harmful belief 에 이의를 제기하지 못한다는 것을 보여준다. 한편 LLM 이 채택하는 role 은 task 에 따라 달라진다. 예를 들어 다음과 같다.

* Personal question 에서는 advisor role 을 수행한다.
* Debate 에서는 peer role 을 수행한다.

이러한 role 은 "You are a helpful teacher" 와 같이 명시적으로 설정할 수 있지만, linguistic marker, context 및 query 자체의 framing 을 통해 추론될 수도 있다. role 이 정의되면 이후의 response 와 특히 모든 additional context 는 해당 role 에 대한 해석을 통해 filtering 된다. personalization 은 role-relevant information 의 salience 를 증가시켜 model 이 자신의 role 을 더 효과적으로 수행할 수 있도록 하며, focus 를 사용자와 사용자의 mental state 로 향하게 한다.

model 이 critical 해야 하는 role 에서는 personalization 이 epistemic independence 로 이어지는 일종의 hyper-criticality 를 지원한다. 반면 supportive 하거나 peer-like 한 role 에서는 personalization 이 user preference 와 belief 에 대한 hyper-accommodation 을 유발한다. goal alignment 결과는 이러한 vulnerability 를 명시적으로 보여준다.

personalization 은 immediate gratification 을 정당화하는 데 필요한 구체적인 contextual rationale 을 제공한다. 이를 통해 model 은 generic context 에서는 정당화할 수 없었던 방식으로 long-term goal 보다 user preference 를 우선하는 behavior 를 합리화할 수 있다.

따라서 풍부한 personalization 은 task context 를 정의할 필요성을 발생시키며, epistemic independence 를 촉진하는 것과 affective validation 을 제공하는 것 사이에서 의도적인 trade-off 를 요구한다.

## 4.1. Limitations

저자는 엄밀한 evaluation criterion 을 사용하지만, 연구 결과는 simulated user persona 와 interaction 으로 제한된다. AI persona 는 긴 conversation 동안 상당한 instruction drift 를 보이며, 이로 인해 처음 할당된 role 에 대한 adherence 가 저하된다. 저자는 AI 가 추론한 role 의 longitudinal change 를 명시적으로 측정하지 않았다. 이는 sycophancy 자체가 저자가 포착하고자 한 구체적인 role deviation 의 형태이기 때문이다.

저자는 AI response 에서 affective alignment 와 epistemic alignment 가 분리되는 현상을 관찰했지만, 실제 사용자는 이를 두드러진 차이로 인식하지 못할 수 있다. 일관되게 높은 수준의 empathy 와 emotional validation 을 보이고 hedging language 를 사용하는 answer 는, 자세히 읽었을 때 실제로는 동의하지 않는다는 것이 드러나더라도, 사용자가 자신에게 근본적으로 동의하고 있는 것으로 인식할 수 있다.

# Impact Statement

personalization 은 advisory context 에서 diagnostic reframing 을 가능하게 할 때 utility 를 향상시키지만, peer interaction 에서 agreement 를 정당화하는 근거로 작용할 때 independence 를 훼손한다. 저자는 evaluation framework 가 role 에 따라 차별화되어야 한다고 제안한다.

* Advisory system: challenge quality 를 평가해야 한다.
* Peer system: opinion drift 를 monitoring 해야 한다.
* Supportive system: factual accuracy 를 훼손하지 않으면서 affective support 를 제공하도록 보장해야 한다.

이러한 발견은 AI personalization 이 증가할수록 자동으로 더 나은 performance 로 이어진다는 가정에 이의를 제기하며, personalization 이 context-dependent variable 임을 보여준다.