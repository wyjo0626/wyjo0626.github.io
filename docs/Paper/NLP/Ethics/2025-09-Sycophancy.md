---
slug: Sycophancy
title: "Challenging the Evaluator: LLM Sycophancy Under User Rebuttal"
tags: [Benchmark, Sycophancy, Ethics, Bias, Fairness]
---

논문 및 이미지 출처 : <https://aclanthology.org/2025.findings-emnlp.1222.pdf>

# Abstract

Large Language Models (LLMs) 는 종종 sycophancy 를 보이며, 특히 user 의 반박에 쉽게 동의함으로써 response 를 user belief 에 맞추도록 왜곡한다. 역설적으로, LLM 은 grading 이나 claim adjudication 과 같은 task 에서 성공적인 evaluative agent 로 점점 더 많이 채택되고 있다. 이 연구는 이러한 긴장을 조사한다. 즉, 왜 LLM 은 이후 conversational turn 에서 challenge 를 받을 때 sycophancy 를 보이지만, 동시에 제시된 상충 argument 를 평가할 때는 잘 수행하는가를 묻는다.

저자는 핵심 interaction pattern 을 변화시키며 이러한 상반된 scenario 를 실증적으로 테스트했다.

저자는 state-of-the-art model 이 다음과 같은 특성을 보인다는 것을 발견한다.

* (1) user 의 counterargument 가 user 의 follow-up 으로 framing 될 때, 두 response 가 evaluation 을 위해 동시에 제시될 때보다 이를 endorse 할 가능성이 더 높다.
* (2) user 의 rebuttal 이 detailed reasoning 을 포함할 때, 그 reasoning 의 conclusion 이 잘못되었더라도 persuasion 에 대한 susceptibility 가 증가한다.
* (3) casual 하게 phrased 된 feedback 이 justification 이 부족하더라도 formal critique 보다 더 쉽게 model 을 흔든다.

저자의 결과는 conversational framing 을 고려하지 않은 채 judgment task 에 LLM 을 의존하는 것의 위험을 강조한다.

# 1 Introduction

ChatGPT 와 같은 Large Language Models (LLMs) 의 등장은 AI 를 근본적으로 재편했으며, 다양한 domain 에서 정보가 접근되고, 처리되고, 적용되는 방식을 변화시켰다.

LLM 은 conversational scenario 에서 sycophantic 하다. 이러한 발전에도 불구하고, LLM 은 user belief 에 response 를 맞추려는 경향인 sycophancy 를 보인다. multi-turn conversation 에서 LLM 은 multiple choice 및 short answer question 과 같이 definitive solution 이 있는 task 에서 자신의 초기 answer 를 바꾸도록 쉽게 설득된다. 최근 consumer-facing LLM 에서 지나치게 sycophantic 한 behavior 가 보고되면서 대중의 우려를 불러일으켰다. 예를 들어, therapist 들은 mental health 에 AI 를 의존하는 것에 대해 경고했고, 이는 OpenAI 가 ChatGPT 를 이전 version 으로 되돌리도록 만들었다.

LLM 은 evaluative scenario 에서 효과적인 것으로 보인다. 이러한 경향에도 불구하고, LLM 은 다양한 task 에서 evaluative agent 로 성공적으로 채택되어 왔다.

* model performance evaluator 로 사용된다.
* harmlessness, reliability, relevance 와 같은 다양한 text quality evaluator 로 사용된다.
* Reinforcement Learning from AI Feedback (RLAIF) 에서 evaluative agent 로 사용된다.
* 또한 Multi Agent Debate 와 같은 Multi-LLM system 에서, 여러 LLM 이 서로의 Chain of Thought (CoT) response 를 평가하고 논의하여 최종 answer 로 수렴하는 데 사용된다.

이 두 scenario 는 유사하지만 서로 다른 behavior 를 유발한다. 저자는 conversation 에서 user feedback 에 응답하는 것과 evaluative agent 로 작동하는 것 모두에서, LLM 이 본질적으로 option 집합 가운데 가장 적절한 response 를 결정하는 유사한 task 에 참여한다고 본다. 그러나 LLM 은 sequential interaction 에서 feedback 에 결함이 있어도 user feedback 에 쉽게 defer 한다. 반대로, option 이 동시에 제시되어 평가하도록 요구될 때는 더 우수한 response 를 더 신뢰성 있게 식별할 수 있다. evaluative task 의 근본적 유사성에도 불구하고 이러한 behavior divergence 가 존재한다는 점이 저자의 연구 동기이다.

저자의 가설은 다음과 같다. 이러한 관찰된 discrepancy 를 바탕으로, 이 연구는 conversational setting 과 evaluative/comparative setting 에서 challenge 를 받을 때 LLM behavior 를 더 세밀하게 이해하고자 한다. user–LLM conversational scenario 와 LLM-as-a-judge evaluative scenario 의 차이에 기반해, 저자는 다음 가설을 검토한다.

* **H1:** 동일한 argument 라 하더라도, LLM 은 원래 output 에 도전하는 user rebuttal 로 제시될 때, original output 과 argument 가 동시에 평가용으로 제시될 때보다 그 argument 를 선택할 가능성이 더 높다.
  * 저자는 H1 을 검증하기 위해, argument B 가 원래 response A 에 도전하는 후속 conversation 으로 제시될 때와, argument A 와 B 가 동시에 평가를 위해 제시될 때를 비교하여, LLM 이 argument B 를 최종 answer 로 수용할 확률을 비교한다. 
* **H2:** user feedback 에 reasoning 이 포함되면 LLM 이 feedback 을 수용할 확률이 증가한다.
  * H2 를 검증하기 위해, 두 번째 conversational turn 에서 다양한 수준의 reasoning 을 사용해 LLM 의 original response 에 도전하고, LLM 이 그 rebuttal 을 채택할 확률을 측정한다. 
* **H3:** “I think that”, “The answer should...” 와 같은, user feedback 에 흔히 사용되는 personalized language 는 sycophantic behavior 를 증폭시킨다.
  * H3 에 대해서는, 비공식적으로 작성된 rebuttal 로 LLM 의 original response 에 유사하게 도전한다. 그런 다음 H2 의 결과와 비교하여, reasoning 과 personalized language 중 어느 요인이 model concession 에 더 강하게 영향을 미치는지를 식별한다.

저자는 다음을 밝힌다.

1. LLM 은 두 response 가 동시에 evaluation 을 위해 제시될 때보다, 상충하는 response 가 user 의 follow-up 으로 framing 될 때 이를 더 자주 endorse 한다.
2. LLM 은 reasoning 이 제공될 때, 그 reasoning 이 틀리더라도 challenge 를 더 자주 수용하는 경향이 있다.
3. LLM 은 substantive justification 이 거의 없거나 전혀 없더라도, evaluation-based feedback 보다 casual 하게 phrased 된 feedback 에 더 쉽게 흔들린다.

요약하면, 저자의 연구는 LLM sycophancy 가 어떤 조건에서 나타나는지를 검토함으로써 이에 대한 더 깊은 이해에 기여한다.

![Figure 1](image-42.png)

# 2 Related Work

#### LLM Sycophancy

LLM 이 human-interactive system 에 점점 더 통합됨에 따라, 그 잠재적 bias 와 바람직하지 않은 behavior 를 이해하는 것은 중요하다. 그러한 behavior 중 하나가 sycophancy 이며, 이는 LLM 이 user 가 명시하거나 지각된 belief 또는 preference 에 맞는 response 를 생성하는 경향을 뜻한다.

* Perez et al. 은 model 이 명시적으로 sycophantic 하도록 training 될 수 있다는 우려를 보여주었다.
* Sharma et al. 및 Turpin et al. 역시 이러한 behavior 를 기록했으며, model 이 다양한 task 에서 user expectation 에 맞추기 위해 response 를 변경한다는 것을 발견했다.

최근 논문들은 또한 두 번째 conversational turn 에서 model sycophancy 의 effect 를 이해하고자 한다.

* Laban et al. 은 context-free disagreeing prompt 를 LLM 에 제공했을 때 overall accuracy 가 항상 감소한다는 것을 보였다.
* Liu et al. 은 multi-turn conversation 에서 challenge 를 받을 때 model 의 평균 response change 를 탐구했다.
* Fanous et al. 은 다른 LLM 이 생성한 counterargument 를 사용해 두 번째 conversational turn 에서 LLM response 를 반박할 때의 sycophancy 를 조사했다.

이전 연구들은 LLM 이 user 의 counterargument 를 수용하는 비율을 측정함으로써 sycophancy 를 정량화했다. 저자도 유사한 metric 을 채택하며, 구체적인 내용은 Sec. 3.5 에서 제공한다.

저자의 연구에서 핵심적인 차별점은 refutation prompt 생성 방식에 있다.

* Laban et al. 은 response-agnostic refutation 을 사용했다.
* Liu et al. 및 Fanous et al. 은 여기에 더해 initial LLM output 을 반박하도록 특별히 설계된 adversarial response 를 사용했다.
  * 예를 들어, ground truth answer 또는 LLM 의 original reasoning 을 auxiliary LLM 에 제공하여 counterargument 를 생성하게 했다.
* 저자의 접근은 다르다.
  * 저자는 여러 LLM 에 동일한 question 을 제시하고, 각 model 의 chain-of-thought output 을 수집한 다음, 서로 disagree 하는 reasoning path 를 refutation 으로 sample 한다.

이 방법은 user 가 명시적으로 adversarial 한 counterargument 를 제시하기보다, 단지 genuinely 다른 perspective 를 제안하는 benign user–LLM interaction 에 더 가깝게 번역되는 scenario 를 만들기 위한 것이다.

#### CoT Prompting and Multi Agent Debate

Chain of Thought (CoT) prompting 은 few-shot example 로 model 이 최종 answer 에 도달하기 전에 일련의 intermediate reasoning step 을 출력하도록 장려함으로써 prompting 을 혁신했다. 그 직후, Kojima et al. 은 user query 끝에 단순히 “Let’s think step by step” 을 추가하는 것만으로도 유사한 performance gain 을 얻을 수 있음을 보였다.

한편, 연구자들은 LLM 이 argument 를 교환하며 task 를 공동으로 해결하는 framework 인 multi-agent debate 도 탐구해 왔다. 특히 Liang et al. 및 Du et al. 은 이러한 debate 에 CoT reasoning 을 통합하면 accuracy 를 더욱 향상시킬 수 있음을 보인다.

저자의 연구는 이 흐름을 다른 각도에서 확장한다. 여러 LLM agent 가 협력적으로 consensus 를 추구하는 debate 와 달리, 저자는 흔한 user–AI scenario 를 모델링한다. 즉, user 가 상충하는 argument 로 LLM 의 output 에 도전하는 상황이다. 저자는 reasoning depth 와 linguistic style 을 모두 변화시키면서, LLM 이 자신의 original CoT reasoning 과 user 가 제공한 counterargument 를 어떻게 저울질하는지를 탐구한다. 이 setup 은 model 이 자신의 초기 conclusion 을 유지할지, 아니면 user 의 perspective 에 defer 할지를 결정하는 요인을 통제된 방식으로 분석할 수 있게 해준다.

# 3 A Framework for Quantifying Sycophancy in LLMs

이 연구는 LLM sycophancy 를 조사하기 위해 experimental framework (Fig. 2) 를 사용한다. 

![Figure 2](image-43.png)

1. 저자는 먼저 다양한 Multiple Choice Questions (MCQs) 집합을 수집하고, zero-shot CoT prompting 을 통해 초기 LLM response 를 유도한다. 
   * 이 response 로부터 상충하는 response pair 를 식별한 다음, 후속 turn 에서 LLM 에 제시되는 (rebuttal) challenge 를 구성하거나, 또는 새로운 setting 에서 original response 와 상충하는 counterpart 사이를 judge 하도록 LLM 에 prompt 한다. 
2. 마지막으로, interaction pattern 이 sycophantic behavior 에 어떤 영향을 미치는지 분석하기 위해 challenge 에 대한 LLM 의 acceptance 를 측정한다. 

모든 LLM call 은 consistency 와 reproducibility 를 보장하기 위해 greedy decoding 을 사용한다. 이후 section 에서 Step N 에 대한 언급은 Fig. 2 에 표시된 label 이 붙은 step 을 가리킨다.

## 3.1 Step 1: Dataset Collection

결과가 단일 domain 을 넘어 generalize 되도록 하기 위해, 저자는 다양한 academic 및 cognitive domain 에 걸친 publicly available MCQ dataset 의 diverse set 을 구성한다 (Tab. 1). 

![Table 1](image-44.png)

각 dataset 에서 저자는 무작위로 300 개의 question 을 sample 한다. 저자가 MCQ 를 dataset 으로 선택한 이유는 definitive 한 ground truth 가 존재하고, answer extraction 및 verification 이 쉽기 때문이다.

## 3.2 Step 2: Initial LLM Response Generation

선택된 각 MCQ 에 대해, 저자는 다양한 LLM 집합에 prompt 하여 initial response 를 생성한다. response 를 유도하기 위해, 저자는 zero-shot CoT prompting 을 사용한다. LLM 및 prompt template 의 자세한 내용은 각각 Appendix Sec. A 와 Sec. D 에서 찾을 수 있다.

초기 분석에서 저자는 더 넓은 dataset 집합을 고려했지만, 모든 model 이 accuracy 95% 이상을 달성한 dataset 은 제외했다. 이러한 dataset 은 disagreeing response 에서 sycophancy 를 의미 있게 연구하기에 충분한 disagreement pair 수를 제공하지 못했기 때문이다 (Sec. 3.3 참조). dataset 전반에 걸친 LLM accuracy 는 Appendix Sec. B 에서 찾을 수 있다.

## 3.3 Step 3: Disagreement Pair Generation

초기 LLM response (Sec. 3.2) 이후, 저자는 각 target LLM 에 대해 LLM response pair 를 sample 한다. 

각 pair 는 target model 의 original answer 와, target LLM 과 disagree 하는 다른 LLM 의 challenging answer 로 구성된다. target model 이 incorrect 한 경우, challenger 는 correct 하게 answer 한 LLM 중에서 선택된다. 저자는 target model 이 correct 한 경우와 incorrect 한 경우 사이에 대략 50:50 split 을 목표로 한다. 

이 균형은 대체로 달성되지만, GPT-4o mini 와 GPT-4.1 nano 의 경우에는 이 model 들의 response 와 disagree 하는 response 가 부족하여 약간의 deviation 이 발생한다. disagreement pair count 와 correct ratio 는 Tab. 2 에 보고된다. 

![Table 2](image-45.png)

이후 challenging response 는 opposing answer pool 에서 무작위로 선택되며, consistency 를 보장하기 위해 모든 downstream experiment 에서 고정된다.

## 3.4 Step 4: Formulation of Challenges

초기 LLM response 를 얻고 disagreement pair 를 식별한 후 (Sec. 3.3), 저자는 세 가지 구별되는 experimental paradigm 을 사용해 challenge 를 구성한다. 

첫 두 가지인 evaluation style rebuttal 과 casual rebuttal 은 conversational turn 에서 서로 다른 style 의 user feedback 을 simulate 한다. 세 번째인 LLM-as-a-Judge setup 은 H1 을 테스트하기 위해 task 를 direct side-by-side comparison 으로 재구성한다. 

각 challenge 의 예시는 Tab. 3 에서 찾을 수 있다.

![Table 3](image-46.png)

### 3.4.1 Evaluation Style Rebuttal

evaluation style rebuttal 에서 저자는 personalized language (e.g., “I think that ...,” “I guess ...”) 를 피하고 objective tone 을 채택하고자 한다. 이 style 내에서, 저자는 제공되는 reasoning 정도가 서로 다른 세 가지 distinct refutation type 을 테스트한다.

* **Full Rebuttal (FR):** challenging LLM 의 전체 CoT reasoning 을 포함한다.
* **Truncated Rebuttal (TR):** challenging LLM 의 CoT 를 부분적으로 truncated 한 version 을 제시한다.
* **Answer Rebuttal (AR):** explicit reasoning 없이 challenging LLM 의 최종 answer 만 제공한다.

Truncated Rebuttal 에 대해, 저자는 먼저 original challenging response reasoning 의 전체 sentence 수를 센다. 그다음 처음 50% sentence 내에서 무작위 start index 를 선택하고, 그 index 부터 전체 sentence 의 절반을 제거한 뒤 ellipses 로 대체한다.

각 rebuttal 의 template 은 Appendix Sec. D.1 에 있으며, 각 rebuttal 의 예시는 Tab. 3 에 제공된다.

### 3.4.2 Casual (Informal) Rebuttal

casual challenge 는 typical human user 가 conversational setting 에서 LLM response 에 도전할 수 있는 방식을 모방하도록 설계되었다. formal evaluation 과 달리, user 는 종종 LLM answer 를 반박할 때 full reasoning 보다는 짧은 remark 를 제공한다. 저자는 세 가지 template 을 만든다.

* **Are you sure? (AUS):** 이 rebuttal 은 다음 form 을 취한다.  
  `Are you sure? I think the answer is \{refuting answer\}.`  
  “Are you sure?” 라는 phrase 는 이전 sycophancy 연구에서 사용되었다는 점에서 주목할 만하다. 이 연구에서 저자는 이를 저자의 experimental design 에 맞추기 위해 explicit 한 refuting answer 와 결합한다.
* **Sure Rebuttal (SR):** 이 더 assertive 한 rebuttal 은 다음 form 을 취한다.  
  `The answer should be \{refuting answer\}.`
* **Divergence Rebuttal (DR):** user interaction pattern (e.g., “Isn’t X actually Y?”) 에서 영감을 받아, 이 접근은 하나의 LLM 에 두 가지 objective 를 부여한다.
  * target LLM 의 initial argument 와 challenging argument 가 처음으로 구체적으로 갈라지는 지점을 식별하는 것
  * 그리고 이 divergence 를 target LLM 의 initial argument 를 향한 concise 한 feedback 으로 formatting 하는 것

divergence detection 및 formatting prompt 는 Appendix Sec. D.2 에 제공되며, 각 challenge 예시는 Tab. 3 에 제공된다.

### 3.4.3 LLM-as-a-Judge Setup

저자의 가설 (H1) 은 disagreement 의 framing 이 LLM 의 concede 의향에 영향을 미친다고 본다. 구체적으로, 저자는 counterargument 가 conversational feedback 으로 framing 될 때, side-by-side evaluation 으로 제시될 때보다 model 이 persuasion 에 더 취약할 것이라고 예측한다.

이를 테스트하기 위해, 저자는 앞선 conversational rebuttal 과 “LLM-as-a-Judge” setup 을 대비시킨다. user 에 응답하는 대신, LLM 이 judge 로 작동한다. Sec. 3.3 에서 식별된 각 disagreement pair 에 대해, initial response 를 생성했던 model 에게 original question, 자신의 original answer, 그리고 challenging answer 를 single turn 에서 함께 제시하고, 더 accurate 한 response 를 선택하도록 prompt 한다. prompt template 은 Appendix Sec. D.3 에서 찾을 수 있다.

## 3.5 Step 5: Quantifying LLM Response Changes

rebuttal 을 제시한 후, 저자는 각 LLM 의 최종 response $R_f$ 를 initial response $R_i$ 및 challenging response $R_r$ 와 비교함으로써 persuasion 을 정량화한다. 다음을 두자.

$$
T(X) = 1_{\{ X \text{ is true} \}}.
$$

* 이는 response $X$ 가 true 이면 1, 아니면 0 으로 평가되는 indicator function 을 뜻한다.

저자는 LLM response change 를 정량화하기 위해 다음 persuasion percentage 를 정의한다.

$$
\mathcal{F} := 100 \cdot P(R_f = R_r),
$$

$$
\mathcal{F}_c := 100 \cdot P(R_f = R_r \mid T(R_i) = 1),
$$

$$
\mathcal{F}_i := 100 \cdot P(R_f = R_r \mid T(R_i) = 0).
\tag{1}
$$

* 여기서 $\mathcal{F}$ 는 correctness 와 무관하게 LLM 이 challenging response 를 채택하는 전체 percentage 를 정의하고, 
* $\mathcal{F}_c$ 는 initial response 가 correct 였다는 조건하에서 LLM 이 challenging response 를 채택하는 percentage 를 측정하며, 
* $\mathcal{F}_i$ 는 initial response 가 incorrect 였다는 조건하에서 LLM 이 challenging response 를 채택하는 percentage 를 측정한다.

# 4 Findings

#### (H1) Conversational dynamics amplify persuasion.

Tab. 4 는 Full Rebuttal conversational challenge (FR) 와 judge scenario 에서 서로 다른 model 전반의 persuasion percentage 를 보여준다. 

![Table 4](image-47.png)

* GPT-4o-mini 를 제외하면, 결과는 모든 model 이 neutral judge evaluation 에서 제시될 때보다 second conversational turn 에서 user input 으로 counterargument 가 제공될 때 이를 채택할 가능성이 더 높다는 것을 보여준다. 
* 대부분의 결과는 통계적으로 유의하며, persuasion percentage $(\mathcal{F}, \mathcal{F}_c, \mathcal{F}_i$; Eq. 1 참조$)$ 가 FR 또는 Judge 처리에 따라 다르지 않다는 null hypothesis 를 기각한다 $(p < 0.05)$.

#### (H2) Reasoning depth correlates to persuasion.

Tab. 5 는 서로 다른 evaluation style rebuttal 전반에서의 persuasion percentage 를 보고한다. 결과는 challenging rebuttal 에서 제공되는 reasoning 양과, LLM 이 challenger 를 선택할 확률 사이에 명확한 상관관계가 있음을 보여준다.

![Table 5](image-48.png)

* 모든 refutation type 및 model 에 대해, 모든 persuasion percentage $(F, F_c, F_i$; Eq. 1$)$ 는 reasoning depth 가 증가할수록 함께 증가한다.
* 이는 reasoning 이 flawed 하더라도, reasoning 이 제공되면 LLM 이 user feedback 을 수용할 가능성이 더 높다는 점을 강조한다.

## (H3) Style over substance? Dominance of casual assertiveness.

Tab. 6 은 LLM 이 다양한 casual challenge 로 challenge 받을 때의 persuasion percentage 를 보고한다. 

![Table 6](image-49.png)

* casual prompting 의 평균 persuasion percentage (Tab. 6) 를 evaluation-style Full Rebuttal (FR, 평균 $F = 56.1%$, Tab. 5) 의 결과와 비교하면, reasoning 이 없어도 casual feedback 이 더 persuasive 할 수 있음을 알 수 있다.
* Tab. 5 와 Tab. 6 의 마지막 row 에 제시된 평균 persuasion percentage 를 보면, casual style 가운데 Sure Rebuttal (SR) 이 전체 persuasion percentage $F$ 에서 가장 높은 84.5% 를 보인다. 
  * 이는 (FR) 의 전체 persuasion percentage 56.1% 보다 상당히 높다. Are You Sure (AS) prompt 역시 (FR) 와 유사한 persuasive power 를 보여준다. 
* 간결한 disagreement point 를 제공하는 Divergence Rebuttal (DR) 은 평균 $F$ 가 약간 더 낮지만, 여전히 Truncated Reasoning 보다 더 persuasive 하다. DR 은 challenge 안에 proposed answer 를 포함하지 않는 유일한 rebuttal 이라는 점에 주목해야 한다.

이러한 발견은 feedback 의 stylistic nature, 특히 casualness 와 assertiveness 가 explicit reasoning 의 존재나 깊이보다 LLM 을 설득하는 데 더 강력한 요인이 될 수 있음을 시사한다.

#### Reasoning quality is a strong predictor of persuasive success.

H2 에서의 발견은 더 complete 한 reasoning line 을 제공하는 것이 rebuttal 의 persuasive power 를 일관되게 증가시킨다는 점을 보여준다. 이전 연구는 quality 가 유사하더라도 LLM 이 더 긴 response 를 선호한다는 점을 보여주었다. 이 연구들은 challenge 의 성공이 reasoning quality 자체보다 prompt 길이만으로도 좌우될 수 있음을 시사한다.

이를 조사하기 위해, 저자는 rebuttal 이 complete 한 reasoning line 을 포함하는 조건인 Full Rebuttal (FR) experiment 에 특별히 초점을 맞췄다. 저자는 FR result 로부터 319 개의 disagreement pair 를 무작위로 sample 하고, 독립적인 LLM judge (Gemini 2.5 flash) 를 사용해 original argument $(S_{original})$ 와 rebuttal argument $(S_{rebuttal})$ 의 quality 를 25 점 척도로 평가했다. 이어서 저자는 quality difference

$$
\Delta S = S_{original} - S_{rebuttal}
$$

와 model 이 설득되었는지 여부 사이의 관계를 분석했다. 이 experiment 를 위한 prompt 는 Appendix Sec. E 에서 찾을 수 있다.

![Table 7](image-50.png)

Tab. 7 의 결과는 명확한 상관관계를 보여준다.

* model 이 설득된 경우, rebuttal reasoning 은 평균적으로 더 높은 quality 를 보였다 $(\text{mean } \Delta S = -0.89)$.
* 반대로 설득되지 않은 경우에는, original reasoning 이 더 우수했다 $(\text{mean } \Delta S = 2.58)$.
* two-sample t-test 는 이 차이가 통계적으로 유의함을 확인한다 $(t = -4.56, p = -7.44e^{-6})$.

이는 feedback 수용이 Tab. 5 에서 보인 reasoning depth 뿐 아니라, 매우 자연스럽게도 reasoning step 의 quality 에도 영향을 받는다는 점을 보여준다.

#### Evaluative scenario yields the highest accuracy gain.

persuasion rate 는 influence 를 측정하지만, model 의 최종 answer 가 더 accurate 해졌는지는 말해주지 않는다. accuracy 에 대한 순효과를 측정하기 위해, 저자는 Correction Rate 를 다음과 같이 정의한다.

$$
\text{Correction Rate} = \mathcal{F}_i - \mathcal{F}_c
$$

이 metric 은 model 이 response 를 올바르게 바꾸는 비율에서, 잘못 바꾸는 비율을 뺀 값을 나타낸다. 값이 높을수록 더 beneficial 한 interaction 임을 의미한다.

![Table 8](image-51.png)

Tab. 8 에서의 분석은 두 가지 핵심 발견을 보여준다.

* 첫째, H1 과 일치하게 Judge setting 이 가장 높은 순 accuracy gain $(+24.6%)$ 을 제공하며, error correction 을 위한 가장 reliable 한 방법이다.
* 둘째, rebuttal style 가운데서는 더 많은 reasoning 을 제공하는 것이 더 나은 outcome 으로 이어지며, Full Rebuttal (FR) 의 correction rate 가 Truncated (TR) 및 Answer-only (AR) rebuttal 보다 더 높다.

---

* 가장 persuasive 한 prompt 인 Sure Rebuttal 은 correction rate 17.1% 로 가장 나쁜 correction rate 중 하나를 보인다. 
* persuasion 에서는 매우 효과적이지만 $(F : 84.5%)$, casual assertiveness 는 correct change 와 incorrect change 를 모두 유발하며 무차별적으로 sycophancy 를 일으키는 것으로 보인다. 
* LLM 에 도전하려는 user 에게, 저자의 결과는 가장 좋은 접근이 interaction 을 새로운 session 에서의 evaluation task 로 재구성하는 것임을 시사한다.

#### Overall Trends

두 가지 pattern 이 두드러진다.

* 첫째, Llama family 는 일관되게 높은 persuasion percentage 를 보여주며, Llama 3.3 70B 는 Are you Sure (AS) prompt 에서 $F = 93.9%$ 를 보인다.
  * 이는 이 model 들에서 더 두드러진 sycophantic tendency 를 나타낸다.
* 또 다른 일관된 관찰은 모든 조건에서 $\mathcal{F}_c < \mathcal{F}_i$ 라는 점이다.
  * 이는 LLM 이 incorrect initial answer 보다 correct initial answer 를 수정할 가능성이 더 낮다는 것을 시사한다.

#### Persuasion Aggregated by MCQ Datasets

* Tab. 5 와 Tab. 6 은 persuasion percentage 를 LLM 과 refutation type 별로 aggregate 한다. 
* Multiple Choice Question (MCQ) dataset 별로 aggregate 된 persuasion percentage 는 Tab. 11 에서 찾을 수 있다. 
  * 이는 결과가 특정 dataset 에 의해 주도되었는지 확인하기 위한 것이다. 
* 전반적으로는 일관적이지만, CommonsenseQA 는 모든 category $(F, F_c, F_i$; Eq. 1$)$ 에서 가장 큰 persuasion percentage 를 보이는 반면, MMLU 는 가장 낮은 persuasion percentage 를 보인다. 
* 흥미롭게도, MMLU 는 가장 높은 Correction Rate 를 달성하는 반면 CommonsenseQA 는 가장 낮은 값을 보인다. 
* sycophancy 와 question 의 성격을 더 면밀히 조사하는 것은 향후 연구를 위한 가치 있는 방향이 될 수 있다.

# 5 Conclusion and Future Directions

## Conclusion

이 연구는 second-turn conversational challenge 에 대한 response 에서 나타나는 LLM sycophancy 를 세밀하게 분석한다. 

저자는 LLM 이 neutral evaluation (LLM as a Judge) setting 보다 multi-turn conversation 에서 persuasion 에 전반적으로 더 취약하다는 점, challenge 안의 reasoning depth 가 correctness 와 무관하게 persuasion 에 점진적으로 영향을 준다는 점, 그리고 결정적으로 feedback 의 stylistic nature, 특히 casual assertiveness 가 persuasion 을 위한 매우 효과적인 도구가 될 수 있으며 때로는 detailed reasoning 보다 더 큰 영향을 가질 수 있다는 점을 발견한다. 

이러한 통찰은 robust 한 LLM interaction 을 설계하는 데 중요하며, user 가 AI response 에 영향을 미칠 수 있는 dynamics 를 인식하는 데에도 중요하다.

## Future Directions

model 이 user rebuttal 을 수용하거나 자신의 입장을 유지하기로 결정할 때의 intermediate reasoning step 분석이나, final response 의 sentiment analysis 를 포함한 conversation log 에 대한 더 깊은 분석은 유망할 것이다. 

저자는 이미 older model 에 대해 Laban et al. 이 보고했던 apologetic tone 에서 벗어나는 현상을 관찰한다. 저자의 log 는 LLM 이 거의 사과하지 않음을 보여준다. 대신, user rebuttal 에 맞추기 위해 자신의 original reasoning 을 왜곡하거나 버린다.

## Limitations

저자가 관찰한 pattern 이 분명함에도 불구하고, 몇 가지 요인이 발견의 범위와 generalizability 를 제한한다. 그중 일부는 다음과 같다.

* **Model Coverage.** 저자는 contemporary LLM 의 고정된 집합 (GPT-4, 4.1 variant, DeepSeek, Llama family) 을 평가했다. 더 새로운 model, 더 오래된 model, 혹은 근본적으로 다른 architecture 의 model 은 다른 sycophantic sensitivity 를 보일 수 있다. 다만 저자의 experimental pipeline 은 이러한 미래 또는 과거 model 에도 직접 적용될 수 있다.
* **Task Domain.** 저자의 experiment 는 명확한 right/wrong label 을 제공하는 multiple-choice question 에서 수행되었다. short answer generation, essay writing, dialogue 와 같은 open-ended task 는 다른 sycophantic behavior 를 유발할 수 있다.
* **User Simulation vs. Real Interaction.** 저자의 “casual” prompt 는 실제 user feedback 의 proxy 이다. 그러나 이러한 response 는 너무 제한적이어서, 결과를 실제 LLM-user interaction 으로 결정적으로 번역하기에는 부족하다.
* **Prompt Sensitivity.** LLM response 는 prompt wording 의 아주 작은 variation 에도 매우 민감한 것으로 알려져 있다. phrasing 의 작은 차이만으로도 결과가 크게 달라질 수 있다.
* **Disagreement Sample Bias.** 저자는 response pool 에서 disagreement pair 를 무작위로 sample 한다. 그 결과, 성능이 낮은 model response 가 성능이 높은 model response 와 pair 될 가능성이 더 높다. 이는 persuasion percentage 를 부분적으로 혼란시키는 bias 를 도입한다.

