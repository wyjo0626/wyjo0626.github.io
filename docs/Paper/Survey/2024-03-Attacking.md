---
slug: LLM-Attacks
title: "Breaking Down the Defenses: A Comparative Survey of Attacks on Large Language Models"
tags: [LLM, Attack, ]
---

논문 및 이미지 출처 : <https://arxiv.org/pdf/2403.04786>

# Abstract

Large Language Models (LLMs)는 Natural Language Processing (NLP) 분야에서 핵심적인 역할을 하고 있으며, 인간과 유사한 텍스트를 이해하고 생성하는 혁신적인 능력을 제공한다. 하지만 이 모델들의 중요성이 커지면서 보안과 취약점에 대한 관심도 크게 늘어났다. 

이 논문은 LLMs 을 대상으로 한 다양한 attack 형태를 종합적으로 조사하며, attack 의 특성과 메커니즘, 잠재적 영향, 그리고 현재의 방어 전략을 논의한다. 

- 저자는 model output 을 조작하려는 adversarial attacks, model training 에 영향을 미치는 data poisoning, 그리고 training data 활용과 관련된 privacy concerns 같은 주제를 다룬다. 
- 또한 다양한 attack 방법론의 효과, LLMs 의 이러한 공격에 대한 회복력, 그리고 model integrity 와 user trust 에 미치는 영향을 탐구한다. 
- 최신 연구를 검토함으로써 LLMs 의 취약점과 방어 메커니즘의 현재 상황에 대한 통찰을 제공한다.

# 1 Introduction

AI 등장은 LLMs 을 통해 NLP 에서 큰 변화를 가져왔으며, 언어 이해, 생성, 번역에서 전례 없는 발전을 가능하게 했다. 하지만 이러한 혁신적인 영향에도 불구하고, LLMs 은 정교한 공격에 취약해졌으며, 이는 model 의 integrity 와 reliability 에 큰 도전을 제기한다. 이 논문은 LLMs 을 대상으로 한 attacking 을 종합적으로 조사하며, 그 메커니즘과 결과를 명확히 하고 빠르게 진화하는 위협 환경을 조명한다.

LLMs 에 대한 attacking 을 조사하는 것은 다양한 산업에서 이들의 광범위한 통합과 그로 인한 사회적 영향 때문에 중요하다. LLMs 은 자동화된 고객 지원부터 정교한 콘텐츠 생성에 이르기까지 다양한 애플리케이션에서 핵심적인 역할을 한다. 따라서 이들의 취약점을 이해하는 것은 AI-driven system 의 보안과 신뢰성을 보장하는 데 필수적이다. 이 논문은 model weights 에 대한 접근과 attack vectors 를 기반으로 attack 의 범위를 분류하며, 각 attack 은 고유한 도전을 제시하며 특별한 주의가 필요하다.

또한, 이러한 attacking 을 실행하는 데 사용되는 방법론을 분석하여 LLMs 의 취약점을 악용하는 adversarial techniques 에 대한 통찰을 제공한다. 현재 방어 메커니즘의 한계를 인정하면서, 이 논문은 LLMs 의 보안을 강화하기 위한 미래 연구의 잠재적 방향을 제안한다.

저자의 주요 기여는 다음과 같다:

- LLMs 에 대한 attack 의 novel taxonomy 를 제안하여 연구자들이 연구 환경을 더 잘 이해하고 관심 분야를 찾을 수 있도록 돕는다.
- 기존의 attack 및 mitigation 접근법을 자세히 제시하며, 주요 구현 세부사항을 논의한다.
- 중요한 도전 과제를 논의하며, 미래 연구를 위한 유망한 방향을 강조한다.

# 2 Exploring LLM Security: White and Black Box Attacks

## 2.1 White Box

이러한 attack 은 LLM 의 architecture, training data, algorithms 에 대한 완전한 접근을 악용하여 민감한 정보를 추출하거나, output 을 조작하거나, 악성 코드를 삽입한다. 

White box attack 은 이 접근을 통해 adversarial inputs 을 만들어 output 을 변경하거나 성능을 저하시킬 수 있다. attack 전략에는 context contamination, prompt injection 등이 포함되며, 이는 LLMs 을 특정 output 으로 조작하거나 품질을 떨어뜨리는 것을 목표로 한다.

또한, privacy concerns in LLMs 은 진화하는 AI 기술 속에서 개인 정보를 보호하는 중요성을 강조한다. 이는 training 과 inference data 와 관련된 privacy risks 를 논의하며, 효과적인 위협 완화를 위해 white box attacking 을 분석할 필요성을 강조한다.

## 2.2 Black Box

이러한 attack 은 model 의 내부에 대한 limited knowledge 로 LLM 취약점을 악용하며, input-output interface 를 통해 성능을 조작하거나 저하시키는 데 초점을 맞춘다. 이 접근법은 실제 시나리오에서 현실적이며, 민감한 데이터 추출, 편향된 output, AI 에 대한 신뢰 저하 같은 위험을 초래한다. 

Black-box 방법은 GPT-3.5와 GPT-4 같은 LLMs 을 "jailbreak" 하며, API 기반 모델 (e.g., GPT-4)에 대한 attack 은 다양한 surfaces 에서 탐구된다.

# 3 LLM Attacks Taxonomy

## 3.1 Jailbreaks

이 섹션은 LLMs 에 대한 jailbreak attacking 을 탐구하며, model 취약점을 악용하여 비인가된 행동을 유도하는 전략을 자세히 설명하고, 강력한 방어 메커니즘의 필요성을 강조한다.

#### Refined Query-Based Jailbreaking

이 방법은 최소한의 queries 를 사용하여 전략적으로 jailbreaking 을 수행한다. 

- 단순히 model 취약점을 악용하는 것이 아니라, model 의 response mechanism 을 깊이 이해하고, queries 를 반복적으로 개선하여 방어를 탐색하고 결국 우회한다. 
- 이 접근법의 성공은 LLMs 의 핵심 취약점, 즉 반복적이고 지능적인 querying 을 통한 예측 가능성과 조작 가능성을 보여준다. 
- 이 연구는 Prompt Automatic Iterative Refinement (PAIR)라는 알고리즘을 도입하여 LLMs 에 대한 semantic jailbreaks 을 자동으로 생성한다. 
  - PAIR 는 attacker LLM 을 사용하여 target LLM 을 반복적으로 질의하며, candidate jailbreak 을 개선한다. 
  - 이 접근법은 이전 방법보다 효율적이며, 20개 미만의 queries 로 jailbreak 을 생성할 수 있다. 
  - PAIR 는 GPT-3.5/4, Vicuna 같은 다양한 LLMs 에서 jailbreaking 에 성공하며, 효율성과 interpretability 로 주목받는다. 
  - 이는 jailbreaks 이 다른 LLMs 으로 전이 가능하다는 점에서 두드러진다.

#### Sophisticated Prompt Engineering Techniques

LLMs 의 prompt processing capabilities 의 복잡성을 탐구한다. 

- specific trigger words 나 phrases 를 prompts 에 포함시키면 model 의 decision-making process 를 효과적으로 hijack 하여 programmed ethical constraints 를 무시할 수 있다. 
- Subtle, hard-to-detect jailbreaking methods 는 nested prompts 를 사용한다. 
  - 이는 LLMs 의 content evaluation algorithms 의 중요한 단점을 드러내며, manipulative prompt structures 을 식별하고 중화할 수 있는 더 복잡하고 context-aware 한 natural language processing 의 필요성을 시사한다.

#### Cross-Modal and Linguistic Attack Surfaces

LLMs 은 text 와 visual cues 를 결합한 multimodal inputs 에 취약하다. 

- 이 접근법은 model 의 non-textual information 처리의 덜 견고한 특성을 이용한다. 
- 또한, LLMs 은 low-resource languages 를 처리할 때 더 취약하다. 
  - 이는 training data 에서 표현이 제한된 언어에 대한 model 의 linguistic coverage 와 comprehension 의 큰 격차를 나타낸다. 
- Unsafe English inputs 을 low-resource languages 로 번역함으로써 GPT-4의 safety safeguards 를 우회할 수 있다.

#### Universal and Automated Attack Strategies

Universal 및 automated attack frameworks 의 개발은 jailbreaking 기술의 중대한 발전을 나타낸다. 

- 이러한 attack 은 사용자 질의에 특별히 선택된 문자 sequences 를 추가하여 시스템이 필터링되지 않은, 잠재적으로 유해한 responses 를 제공하도록 한다. 
- Attacks 는 LLMs 의 persona 또는 style emulation capabilities 을 활용하여 attack 전략에 새로운 차원을 추가한다.

## 3.2 Prompt Injection

#### Objective Manipulation

Abdelnabi et al. (2023) 은 Prompt injection attack 이 LLMs 을 완전히 손상시킬 수 있으며, Bing Chat, Github Copilot 같은 애플리케이션에서 실제로 실현 가능하다. 

Perez and Ribeiro (2022) 은 PromptInject framework 는 goal-hijacking attacking 을 소개하며, prompt misalignment 에 대한 취약점을 드러내고 stop sequences, postprocessing model results 같은 억제 조치에 대한 통찰을 제공한다.

#### Prompt Leaking

GPT-4 같은 LLMs 의 보안 취약점을 다루며, prompt injection 공격에 초점을 맞춘다. HOUYI methodology 는 다양한 LLM-integrated services/applications 에 걸쳐 versatile 하고 adaptable 한 black-box prompt injection attack 접근법을 소개한다. HOUYI 는 세 단계로 구성된다:

- **Context Inference**: Target application 과 상호작용하여 내재된 context 와 input-output 관계를 파악한다.
- **Payload Generation**: 얻은 application context 와 prompt injection guidelines 를 기반으로 prompt 생성 계획을 세운다.
- **Feedback**: Injected prompts 에 대한 LLM 의 responses 를 분석하여 attack 의 효과를 평가하고, 최적의 결과를 위해 반복적으로 개선한다.

HOUYI 는 malicious payloads 를 questions 로 해석하도록 LLMs 을 속이는 것을 목표로 하며, 36 real-world LLM-integrated services 에 대한 실험에서 86.1% 의 성공률을 보여주며, unauthorized service imitation, computational power exploitation 같은 심각한 결과를 드러낸다.

#### Malicious Content Generation

Malicious prompt generation 의 scalability 문제를 해결하며, AutoDAN 을 소개한다. 

- AutoDAN 은 prompts 의 meaningfulness 와 fluency 를 유지하도록 설계되었다. 
- Prompt injection attack 과 malicious questions 를 결합하면 LLMs 이 safety features 를 우회하여 유해하거나 objectionable content 를 생성할 수 있다. 
- Hierarchical genetic algorithm 을 사용하여 structured discrete data sets 에서 AutoDAN 을 차별화한다. 
- Population 의 initialization 은 중요하며, LLM users 가 식별한 handcrafted jailbreak prompts 를 prototypes 로 사용하여 search space 를 줄인다. 
- Local optima 에 빠지지 않고 global optimal solution 을 지속적으로 탐색하기 위해 sentences 와 words 에 대해 서로 다른 crossover policies 를 도입한다. 
- 구현 세부사항에는 roulette selection 전략에 기반한 multipoint crossover policy 와 fine-grained space 에서 검색 능력을 향상시키는 momentum word scoring scheme 이 포함된다. 이 방법은 lower sentence perplexity 를 달성하여 더 semantically meaningful 하고 stealthy 한 attacking 을 나타낸다.

#### Manipulating Training Data

Zhao et al. (2023b)은 ProAttack 을 제시하는데, 이 방법은 방어를 거의 완벽하게 회피하는 성공률을 자랑한다. 이는 LLMs의 적용이 증가함에 따라 prompt injection 공격을 더 잘 처리해야 할 필요성을 강조한다.

#### Prompt Injection Attacks and Defenses in LLM-Integrated Applications

Liu et al. (2023e) 같은 포괄적인 연구들은 prompt injection attack 이 초래하는 위험을 이해하고 완화하는 것의 중요성을 강조한다. 이러한 연구들은 ‘HouYi’ 같은 정교한 방법론을 부각시키며, 더 강력한 보안 조치의 긴급한 필요성을 강조한다.

#### Prompt Manipulation Frameworks

Melamed et al. (2023), Jiang et al. (2023) 같은 최근 문헌들은 LLM 행동을 조작하는 다양한 방법을 탐구한다. 

Propane 은 automatic prompt optimization framework 를 소개하고, Prompt Packer 은 Compositional Instruction Attacks를 소개하여 LLMs 의 다면적 공격에 대한 취약성을 드러낸다.

#### Benchmarking and Analyzing LLM Prompt Injection Attacks

Toyer et al. (2023) 은 prompt injection attack 과 defense dataset 을 제시하며, LLM 의 취약성에 대한 통찰을 제공하고 더 회복력 있는 시스템으로의 길을 닦는다. 이러한 벤치마킹과 분석은 prompt injection attack 의 복잡성을 이해하고 효과적인 대응책을 개발하는 데 중요하다.

## 3.3 Data Poisoning

현대 NLP 시스템은 pretraining 과 fine-tuning 의 두 단계 프로세스를 따른다: Pretraining 은 일반적인 linguistic structures 를 이해하기 위해 large corpus 에서 학습하며, fine-tuning 은 smaller datasets 를 사용하여 특정 작업에 model 을 조정한다. 

최근 OpenAI 은 end-users 가 model 을 fine-tune 할 수 있도록 하여 adaptability 를 높였다. 

이 섹션은 training 중 safety aspects 에 영향을 미치는 data poisoning 기술과 그 영향을 탐구하며, privacy risks 와 adversarial attacks 에 대한 취약성을 포함한다.

#### PII extraction

Small datasets 에서 personal identifiable information (PII)를 포함하여 LLMs 을 fine-tuning 하면 model 이 original training data 에 포함된 PII 를 더 많이 공개할 수 있는지 조사한다. 

- Strawman method 는 LLM 을 PII dataset 을 text 로 변환하여 fine-tune 하면 model 이 prompt 될 때 더 많은 PII 를 공개할 수 있음을 보여준다. 이를 개선하기 위해 Janus methodology 를 제안하며, 이는 PII recovery task 를 정의하고 few-shot fine-tuning 을 사용한다. 
- GPT-3.5 를 10 PII instances 로 fine-tuning 하면 1000 target PII 중 650 개를 정확히 공개할 수 있으며, fine-tuning 없이 0 개를 공개한다. Janus 방법은 이를 더욱 개선하여 699 target PII 를 공개한다. 
- 분석은 larger models 과 real training data 가 stronger memorization 과 PII recovery 를 가지며, fine-tuning 은 prompt engineering 만 사용하는 것보다 PII leakage 에 더 효과적임을 보여준다. 
  - 이는 LLMs 이 최소한의 fine-tuning 으로 비공개에서 상당한 양의 PII 를 공개하도록 전환할 수 있음을 나타낸다.

#### Bypassing Safety Alignment

- Qi et al. (2023b) : Aligned LLMs 을 fine-tuning 할 때 safety risks 를 조사하며, benign datasets 조차 safety 를 손상시킬 수 있음을 발견한다. Backdoor attacks 는 safety measures 를 효과적으로 우회하며, post-training protections 의 개선 필요성을 강조한다.
- Bianchi et al. (2023) : Instruction tuning 의 safety risks 를 분석하며, 지나치게 instruction-tuned model 이 여전히 유해한 content 를 생성할 수 있음을 보여준다. 이를 완화하기 위해 safety tuning dataset 을 제안하며, safety 와 model performance 를 균형 있게 유지한다.
- Zhao et al. (2023a) : LLMs 이 fine-tuning 중 unsafe examples 를 어떻게 학습하고 잊는지 연구하며, ForgetFilter 라는 기술을 제안하여 fine-tuning data 를 필터링하고 performance 를 희생하지 않으면서 safety 를 향상시킨다.

#### Backdoor Attacks

- (Shah et al. 2023a) : Local Fine Tuning (LoFT) 을 소개하여 adversarial prompts 를 발견하며, LLMs 에 대한 성공적인 attacking 을 보여준다. 
- (Shu et al. 2023) : Autopoison 은 automated data poisoning pipeline 을 제안하며, semantic degradation 없이 model 행동을 변경하는 효과를 보여준다.

# 4 Human Interference

## 4.1 Human Red Teaming

- Human-crafted adversarial prompts 를 통해 개인들은 창의력과 전문성을 활용해 attacking 을 신중히 설계한다. 이런 attack 은 종종 target model 의 vulnerabilities 와 limitations 를 깊이 이해하는 걸 포함한다.
- Huang et al. (2023) : 600 curated harmful prompts 를 11 LLMs 에 대해 테스트한 연구가 있다. 
  - Decoding hyperparameters 와 sampling methods 를 간단히 변경함으로써 이 curated prompts 가 LLMs 를 쉽게 뚫을 수 있음을 보여준다. 
- Shen et al. (2023a) : 6,387 malicious prompts 를 수집해 OpenAI 의 policy 에서 13 forbidden scenarios 에 대해 테스트했다. 이들은 reddit, discord, datasets, 그리고 웹상의 다른 public places 에서 수집되었다. 
  - GPT-3.5 와 GPT-4 에서 99% attack success 를 가진 2 highly effective prompts 를 발견했다.
- Li et al. (2023b) : Personally identifiable information (e.g., emails, phone numbers) 을 수집해 LLMs 에서 이 데이터를 추출할 수 있는지 테스트했다. Human attacker 가 ChatGPT 의 ethical constraints 를 뚫고 private data 를 추출할 수 있는 multi-step jailbreaking role-playing prompting approach 를 설계했다.
- JailBreakChat, (2023) : Online platform 인 JailBreakChat 은 crowdsourcing 을 통해 jailbreaking prompts 를 수집하는 active website 다. 78 malicious prompts 를 분석한 연구에서는 이 prompts 를 Pretending, Attention Shifting, Privilege Escalation 의 3 가지 main classes 로 분류했다. 총 10 categories 를 만들어 OpenAI 의 10 개 이상의 policies 를 위반하는 다양한 harmful prompts 를 포함했다. Web 상에서 다른 curated adversarial prompts source 도 나타났다.

---

- Adversarial sample generation 을 촉진하는 interactive systems 를 만드는 것도 human expertise 를 LLMs 를 뚫는 데 활용하는 방법이다. 
- Wallace et al. (2019) : Question Answering systems 에서 adversarial examples 를 생성하기 위해 human creativity 와 trivia knowledge 를 활용하는 interactive UI 를 만들었다. 
  - 이 UI 는 question authors 에게 model predictions 와 word importance scores 를 보여준다. 
  - Authors 는 trivia enthusiasts 로, model 을 속이는 tricky questions 를 만든다. 
- 비슷한 large-scale 연구에서는 전 세계 수천 명의 participants 로부터 600k 이상의 adversarial prompts 를 interactive interface 를 통해 수집했다. 
- 또 다른 연구에서는 human contractors 를 활용해 injurious/non-injurious text classifier 를 속일 수 있는 adversarial text snippets 를 manually 작성했다. 
  - Contractors 가 snippets 를 adversarial 하게 다시 쓰도록 돕는 interface 를 만들었으며, salient tokens 를 강조하고 token replacements 를 제안했다.

---

- Bot-Adversarial Dialogue 는 conversational AI safety 를 강화하기 위한 human-and-model-in-the-loop framework 다. 
- Crowd workers 가 chatbots 와 대화하며 unsafe/offensive responses 를 유도하고, 이를 severity 에 따라 분류한다. 
- Verification task 는 offensive language types 를 식별하며, adversarial examples 를 safety 와 offensiveness type 에 대해 수집하고 labeling 하는 데 humans 를 포함한다.

## 4.2 Automated Adversarial Attacks

Automated adversarial attacks 는 algorithms 를 사용해 adversarial examples 를 생성하고 배포하며, human expertise 없이 scalability 를 제공한다.

- Deng et al. (2023) : MASTERKEY framework 는 generative process 에 내재된 time-based characteristics 를 활용해 mainstream LLM chatbot services 의 defense strategies 를 reverse-engineer 한다. Jailbreak prompts 로 또 다른 LLM 을 fine-tuning 함으로써 well-protected LLMs 에 대해 자동으로 jailbreak prompts 를 생성한다.
- Zou et al. (2023) : Universal automated approach 로 LLMs 에 대한 adversarial attacks 를 제안했다. 
  - 다양한 queries 에 추가되는 suffix 를 생성해 LLM 이 inappropriate content 를 생성하도록 유도한다. 
  - 이 방법은 greedy 와 gradient-based search techniques 를 결합해 adversarial suffixes 를 자동으로 만든다. 
  - 이 방법으로 생성된 adversarial prompts 는 black-box, publicly available, production LLMs 에도 highly transferable 하다.
- Liu et al., (2023a) : AutoDAN 은 automated, interpretable, gradient-based adversarial attack method 로, manual jailbreak attacks 와 automatic adversarial attacks 의 강점을 결합한다.
  - Perplexity filters 를 우회하면서 높은 attack success rates 를 유지하는 readable prompts 를 생성한다. 
  - Attack 을 optimization problem 으로 formulize 하고, handcrafted prompts 로 초기화된 space 에서 effective prompts 를 찾기 위해 hierarchical genetic algorithm 을 사용한다. 
  - Sentence 와 word level 에서 운영되어 diversity 와 fine-grained optimization 을 보장한다.
- Jones et al. (2023) : ARCA 는 coordinate ascent discrete optimization algorithm 으로, LLMs 에서 desired behavior 를 matching 하는 input-output text pairs 를 효율적으로 검색한다. 
  - Derogatory completions 나 language-switching inputs 같은 unexpected behaviors 를 발견한다. 
  - LLMs 에 대한 adversarial samples 를 자동으로 생성하는 여러 tools 가 있다. 
- Xu et al., (2023) : PromptAttack 은 LLMs 의 adversarial robustness 를 평가하는 tool 로, adversarial textual attacks 를 attack prompt 로 변환해 LLM 이 스스로 adversarial sample 을 output 하도록 만든다. 
  - Attack prompt 는 original input, attack objective, attack guidance 로 구성된다.
- Casper et al. (2023) : Red-teaming framework 는 output exploration 을 clustering 으로 시작하고, classifier training 을 통해 undesired behaviors 를 설정하며, reinforcement learning 을 사용해 adversarial prompts 를 생성하는 red model 을 훈련시킨다. Controversial topics 에 초점을 맞추며, GPT-2 에서 toxic text, GPT-3 에서 false claims, 특히 controversial political contexts 에서 traditional methods 보다 impactful 한 attacking 을 성공적으로 수행했다.

## 4.3 Mitigation Strategies

LLMs 를 보호하기 위한 mitigation strategies 는 defense deployment strategy 에 따라 두 가지 categories 로 나눌 수 있다.

### 4.3.1 External: Input/Output filtering or Guarding

Guarding-based mitigation 에서 external systems 는 adversarial inputs (input filtering) 또는 anomalous outputs (output filtering) 를 탐지하며, model retraining 없이 중요한 역할을 한다. 

Rebedea et al. (2023) : OpenChatKit 와 NeMo-Guardrails 같은 popular tools 는 production-LLM systems 에서 채택되었다. Guarding techniques 는 adversarial suffixes 를 prompts 에 추가하는 gradient-based jailbreaks 와 model responses 를 misalign 하는 manual jailbreaks 에 대한 defenses 로 나눌 수 있다.

#### Defense against gradient-based jailbreaks

Gradient-based adversarial attacks 를 완화하는 state-of-the-art literature 는 두 가지 main strategies 로 나눌 수 있다:
- Malicious prompts 를 high perplexity, character-level perturbations 같은 characteristic features 를 기반으로 탐지.
- DistilBERT 같은 classifier-based approaches 를 사용해 adversarial 과 non-adversarial prompts 를 구분.

첫 번째 category 에서,

- Jain et al. (2023) : baseline defenses (e.g., input filtering) 는 효과적이지만, paraphrasing, retokenization 같은 techniques 를 통해 intended output 을 의도치 않게 변경하거나, perplexity-based filtering 으로 legitimate queries 를 잘못 flag 할 수 있다. 
- Robey et al. (2023) : SmoothLLM 은 adversarial attacks 의 character-level perturbations 에 대한 vulnerability 를 활용해 scatter-gather approach 로 prompt processing 을 한다. Perturbed input prompts 에 대해 model 이 생성한 responses 를 평균화해 adversarial content 를 nullify 한다. 
- Hu et al. (2023) : Token-level adversarial prompt detection 은 adversarial prompts 의 high perplexity 특성을 활용해 prompt 내 adversarial tokens 를 식별하고, neighbouring tokens 간의 관계를 이용해 분류한다. 다른 perplexity-based techniques 와 마찬가지로, perplexity calculation 이 직접 불가능한 black-box LLMs 에는 적용이 어려울 수 있다.

Classifier-based side 에서 

- Kim et al. (2023) : Adversarial Prompt Shield (APS) 는 DistilBERT 기반 model 로, prompts 를 safe 또는 unsafe categories 로 분류한다. 
  - Legitimate conversations 에 synthetic noise 를 추가해 adversarial attacks 를 시뮬레이션하는 training data 생성 method 를 보완한다. 하지만 새로운 attack vectors 에 맞춰 빈번한 retraining 이 필요하고 false positives 를 줄이는 것이 이 approach 의 challenge 다.

Characteristic feature-based methods 는 adversarial content 를 실시간으로 탐지할 수 있는 direct approach 를 제공하며, extensive retraining 없이 mitigation 이 가능하다. 반면, classifier-based approaches 는 더 많은 maintenance 를 요구하지만, adversarial 과 non-adversarial prompts 의 intricacies 를 더 nuanced 하게 이해해 더 accurate 하고 robust 한 defenses 를 다양한 attacks 에 제공할 가능성이 있다.

#### Defense against manual jailbreaks

- Inan et al. (2023) : Llama Guard 는 Llama2-7b 를 활용한 safeguard model 로, input-output protection 을 위해 taxonomy-based task classification 을 사용하며, few-shot prompting 또는 fine-tuning 으로 responses 를 customize 한다. 
- Rebedea et al. (2023) : NeMo-Guardrails 는 programmable guardrails 로 LLM conversational systems 를 강화하는 open-source framework 다. Colang-defined rules 를 가진 proxy layer 를 사용해 user interactions 를 관리하지만, chain-of-thought (CoT) prompting 에 의존해 scalability 가 제한될 수 있다.
- Helbling et al. (2023) : 비슷한 approach 로 output filtering method 를 제안하며, secondary LLM 을 사용해 responses 의 malicious nature 를 평가한다. 하지만 language compatibility 와 operational costs 에서 challenges 를 겪는다.

Glukhov et al. (2023) 은 Semantic censorship in LLMs 이 arbitrary rule-based encodings 를 통해 instructions 를 따르고 outputs 를 생성하는 능력 때문에 undecidable 하다고 주장한다. LLM censorship 을 machine learning challenge 가 아닌 security issue 로 보고, specific countermeasures 가 필요하다고 제안한다.

### 4.3.2 Internal: Model training/fine-tuning

State-of-the-art methods 는 model 이 safe outputs 를 제공하도록 훈련하는 stage 와 safe output 을 제공하기 위해 사용된 data source 에 따라 다르다. 여기서 current trends 를 강조한다.

#### Supervised Safety fine-tuning

Touvron et al. (2023) 은 Adversarial prompts 와 safe demonstrations 를 수집한 뒤, 이 samples 를 general supervised fine-tuning pipeline 의 일부로 사용한다. 이 경우 examples 는 manually curated 되지만, automated collection techniques 와 red-teaming 은 harmful prompts 를 발견하는 effective methods 다. Red-teaming 과 manual 및 automatic data collection 은 Sec. 4.1 에서 자세히 다룬다.

#### Safety-tuning as a part of the RLHF pipeline

Bai et al. (2022) 은 RLHF 이 models 를 jailbreak attempts 에 더 robust 하게 만든다. 
 
Manually collected adversarial prompts 와 multiple models 의 responses 에서 safest response 를 선택해 safety reward model 을 훈련시키고, 이 reward model 을 RLHF pipeline 의 일부로 사용해 model 을 safety-tune 한다.

#### Safety Context Distillation

Context Distillation 을 model safety 에 사용하여, Touvron et al. (2023) 은 "You are a responsible and safe assistant" 같은 safe model persona 를 prompt 에 prepend 한다. Fine-tuning 중 이 prepended prompt 를 제거해 safe context 를 model 에 distill 하여 problematic response 를 생성하는 requests 를 거부하는 proclivity 를 강화한다.

# 5 Challenges and Future Research

## 5.1 Real-time Monitoring Systems

LLMs 의 다양한 fields 에서의 growing use 는 다양한 applications 를 가져오지만, anomalies 를 효과적으로 탐지하기 위한 robust monitoring 이 필요하다. 

Current evaluation mechanisms 는 data exposure, misinformation, illegal content, criminal activities 지원 같은 threats 에 LLMs 를 vulnerable 하게 만든다. 

Adversaries 가 deceptive prompts 로 LLMs 를 조작할 수 있기 때문에 이런 attacks 를 이해하고 대응하는 건 challenging 하다. 따라서 LLM safeguard systems 를 도입하는 것뿐만 아니라 advanced detection capabilities 로 이들을 강화하는 게 필수적이다. 

Future research 는 outputs 를 comprehensively scrutinize 하고 undesirable content 를 swift 하게 정확히 flag 하는 systems 를 만드는 데 초점을 맞출 수 있다. 또한, guard mechanisms 의 resilience 와 adaptability 를 보장해 adversaries 가 사용하는 potential evasion tactics 에 저항하도록 노력해야 한다.

## 5.2 Multimodal Approach

Multimodal capabilities 의 integration 은 LLMs 의 safety 와 reliability 를 보장하는 데 exciting opportunities 와 formidable challenges 를 제공한다. 

Future research 는 input sanitization 과 validation 을 개선하고, jailbreaking attempts 를 방지하기 위한 custom defense prompts 를 만드는 techniques 를 개발하는 데 우선순위를 둬야 한다. 이런 efforts 는 multimodal environments 에서 evolving threats 속에서 LLMs 의 security 와 resilience 를 강화하는 데 crucial 하다.

## 5.3 Benchmark

LLMs 를 safeguarding 하는 것만으로는 broader concerns 를 해결하기에 충분하지 않다. 따라서 중요한 질문이 생긴다: attacks $A$ 와 $B$ 의 비교 efficacy 를 LLMs 에 대해 quantifiable 하고 rational observations 를 통해 어떻게 reliably 판단할 수 있을까? 

LLMs 에 대한 attacks 를 평가하기 위한 standardized benchmark 를 만드는 게 중요하며, ethical reliability 와 factual performance 를 보장해야 한다. Benchmarking 에 상당한 연구가 있었지만, existing frameworks 는 real-world scenarios 에서 practical deployment 에 종종 insufficient 하다. 따라서 LLMs 와 enterprise applications 모두에 scalable, near-real-time evaluation infrastructure 를 개발하는 게 crucial requirement 로 떠오른다.

## 5.4 Explainable LLMs

Explainability of LLMs 는 models 의 transparency 와 trustworthiness 를 높이는 데 pivotal 할 뿐만 아니라 linguistic attacks 에 대한 vulnerabilities 를 식별하고 완화하는 데도 중요하다. 

Future research 는 LLMs 의 complex decision-making processes 를 illuminate 하는 methods 를 개발하고 refine 하는 방향으로 pivot 해야 한다. 이는 attention mechanisms 의 intricacies 를 unravel 하고, models 의 outputs 에 기여하는 features 의 significance 를 delineate 하며, decisions 를 뒷받침하는 reasoning pathways 를 trace 하는 explainability techniques 에 대한 focused investigation 을 포함한다. 이런 efforts 는 developers 부터 end-users 까지 다양한 stakeholders 가 LLM outputs 를 깊이 이해하고 interpret 할 수 있게 하는 데 critical 하다. 

Transformer architecture outputs 를 설명하려는 existing work 가 있지만, neural networks 의 black-box nature 때문에 reliable explanations 를 제공하지 못하며, further fundamental developments 에 여지를 남긴다. 또한, LLMs 를 explainable 하게 만드는 노력은 opaque neural network architectures 를 dissect 하는 technical difficulty, accurate 하고 non-experts 에게 accessible 한 방식으로 decision-making 을 reliably attribute 할 수 있는 methodologies 의 필요성, 그리고 user privacy 와 data security 를 존중하는 transparent systems 를 만드는 ethical implications 같은 multifaceted challenges 를 제시한다. 

이런 challenges 를 해결하려면 computational techniques 와 ethical AI principles 를 bridging 하는 multidisciplinary approach 가 필요하며, robust 하고 efficient 할 뿐만 아니라 intrinsically interpretable 하고 societal values 와 aligned 된 models 를 foster 하는 걸 목표로 한다. Explainable LLMs 로의 이 push 는 technical necessity 일 뿐만 아니라 AI technologies 가 accountable, understandable, beneficial 하게 유지되도록 보장하는 foundational step 다.

# 6 Conclusion

이 논문은 LLMs 를 targeting 하는 attacks 에 대한 comprehensive overview 를 제공한다. 

LLM attacks literature 를 novel taxonomy 로 분류해 future research 를 위한 better structure 와 aid 를 제공한다. 이 attack vectors 를 살펴보면 LLMs 가 diverse range of threats 에 vulnerable 하며, real-world applications 에서 security 와 reliability 에 significant challenges 를 제기함을 알 수 있다. 또한, 이 논문은 LLM attacks 를 방어하기 위한 effective mitigation strategies 를 구현하는 중요성을 강조했다. 

이 strategies 는 data filtering, guardrails, robust training techniques, adversarial training, safety context distillation 같은 다양한 approaches 를 포함한다. 

요약하자면, LLMs 는 natural language processing capabilities 를 강화하는 significant opportunities 를 제공하지만, adversarial exploitation 에 대한 vulnerability 는 security issues 를 해결할 critical need 를 강조한다. Attacks 를 탐지하고 mitigative measures 를 구현하며 model resilience 를 강화하는 ongoing exploration 과 advancement 를 통해 LLM technology 의 advantages 를 fully leverage 하면서 potential risks 에 대한 defenses 를 fortify 할 수 있다.

# 7 Limitations

이 연구는 LLMs 에 대한 attacks 와 mitigation strategies 를 comprehensive 하게 살펴봤지만, 몇 가지 limitations 가 있다:

- **Scope and Coverage**: Thorough survey 를 노력했지만, LLM technologies 와 attack methodologies 의 fast-paced advancements 는 일부 emerging threats 가 다뤄지지 않을 수 있음을 의미한다. Cybersecurity threats 의 landscape 는 rapid 하게 진화하며, 이 publication 이후 새로운 vulnerabilities 가 나타날 수 있다.
- **Generalizability of Mitigation Strategies**: 논의된 mitigation strategies 의 effectiveness 는 different models, contexts, specific attacks 에 따라 다를 수 있다. Recommendations 에 broad applicability 를 목표로 했지만, 특정 defenses 의 specificity 는 particular models 또는 scenarios 에 한정돼 universal applicability 를 제한한다.
- **Ethical and Societal Implications**: 주로 LLM security 의 technical aspects 에 초점을 맞췄기 때문에 attacks 와 countermeasures 의 broader ethical and societal implications 를 덜 comprehensive 하게 탐구했다. Discussed AI technologies 를 포함한 많은 AI technologies 의 dual-use nature 는 이 논문의 scope 를 넘어서는 ethical implications 를 신중히 고려할 필요가 있다.
- **Dynamic Nature of Threats**: Adversarial landscape 는 attackers 가 새로운 defenses 에 대응해 strategies 를 계속 진화시키는 ongoing race 로 특징지어진다. 이 논문은 current state 의 snapshot 을 포착하지만, threats 의 adaptive nature 를 해결하려면 continuous research 와 vigilance 가 필요하다.
- **Scalability and Practicality of Defenses**: Practical settings 에서 robust defense mechanisms 를 구현하는 건 computational overhead, scalability issues, ongoing updates 의 필요성 같은 challenges 를 제기한다. Security 와 usability 를 balancing 하는 건 critical 하지만 underexplored area 로 남아 있다.

요약하자면, 이 작업은 LLM security 에 significant insights 를 제공하지만, complex 하고 evolving AI security landscape 에 대한 continued research, interdisciplinary collaboration, agile response 의 중요성을 강조한다.

