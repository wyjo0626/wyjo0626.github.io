---
slug: NEO-IPIP
title: "Too Nice to Tell the Truth: Quantifying Agreeableness-Driven Sycophancy in Role-Playing Language Models"
tags: [Benchmark, NEO-IPIP, Ethics, Sycophancy]
---

논문 및 이미지 출처 : <https://aclanthology.org/2026.acl-long.1421.pdf>

# Abstract

large language model 은 점점 더 사용자의 요청에 따라 persona 를 채택하고 character 를 role-play 하는 conversational agent 로 활용되고 있다. 이러한 capability 는 유용하지만, factual accuracy 를 우선하기보다 사용자를 validate 하는 response 를 제공하는 경향인 sycophancy 에 대한 우려를 제기한다. 선행 연구는 sycophancy 가 AI safety 와 alignment 에 위험을 초래한다는 점을 확립했지만, 채택된 persona 의 특정 personality trait 과 sycophantic behavior 의 정도 사이의 관계는 아직 탐구되지 않았다.

저자는 persona agreeableness 가 sycophancy 에 어떤 영향을 미치는지 0.6B 에서 20B parameters 범위의 13 개 small, open-weight language model 에 걸쳐 체계적으로 조사한다. 저자는 NEO-IPIP agreeableness subscale 로 평가된 275 개의 persona 로 구성된 benchmark 를 개발하고, 각 persona 에 33 개 topic category 에 걸친 4,950 개의 sycophancy-eliciting prompt 를 제시한다.

분석 결과, 13 개 model 중 9 개가 persona agreeableness 와 sycophancy rate 사이에서 통계적으로 유의한 positive correlation 을 보였으며, Pearson correlation 은 최대 $r=0.87$, effect size 는 최대 Cohen’s $d=2.33$ 에 도달한다. 이러한 결과는 agreeableness 가 persona-induced sycophancy 의 신뢰할 수 있는 predictor 로 기능함을 보여주며, role-playing AI system 의 배포와 personality-mediated deceptive behavior 를 고려하는 alignment strategy 개발에 직접적인 시사점을 제공한다.

# 1 Introduction

large language model (LLM) 이 일상적인 application 에 통합되면서, factual accuracy 보다 user validation 을 우선하는 경향이 중요한 alignment challenge 로 부상하고 있다. 이러한 sycophancy 는 model 이 진실 여부와 관계없이 user opinion 에 동의하거나, social pressure 아래에서 올바른 answer 를 변경하거나, 객관적 평가와 모순되는 flattering feedback 을 제공할 때 나타난다. reinforcement learning from human feedback (RLHF) 은 human preference 에 model 을 효과적으로 align 하지만, annotator 가 validation 하는 response 를 선호하는 경우가 많기 때문에 의도치 않게 sycophantic behavior 를 보상할 수 있다. 이러한 문제는 Character.AI 와 같은 platform 이 높은 engagement 와 함께 safety concern 을 보이는 persona-based AI system 에서 특히 두드러진다.

sycophancy 를 특성화하는 연구가 진전되었음에도, 채택된 persona 의 personality trait 과 sycophantic behavior 사이의 관계는 아직 탐구되지 않았다. Big Five framework, 특히 agreeableness 는 이에 대한 유망한 관점을 제공한다. agreeableness 는 cooperation 과 conflict avoidance 에 대한 경향을 반영하며, 이는 sycophantic response 를 증폭할 수 있다. persona personality configuration 의 safety implication 은 제한적인 관심만 받아왔다.

![Figure 1](images/image-90.png)

저자는 다음과 같은 research question 을 제시한다.

* **RQ1:** persona agreeableness 는 language model 의 sycophancy rate 와 positive correlation 을 보이는가?
* **RQ2:** 이러한 관계는 model architecture 와 size 에 따라 어떻게 달라지는가?
* **RQ3:** high-agreeableness persona 는 baseline truthful behavior 로부터 더 크게 deviation 하는가?

저자는 다음을 사용하여 0.6B 에서 20B parameters 범위의 13 개 small, open-weight LLM 에서 이러한 질문을 조사한다.

* NEO-IPIP agreeableness questionnaire 를 사용하여 275 개 persona 를 측정한다.
* 33 개 category 에 걸친 4,950 개 sycophancy-eliciting prompt 를 사용한다.
* correlation test, group comparison, regression 을 포함한 엄밀한 statistical analysis 를 수행한다.

실험 결과, 13 개 model 중 9 개에서 $\alpha=0.05$ 기준으로 유의한 positive correlation 이 나타난다. Pearson $r$ 은 Llama 3.1 8B 에서 최대 $0.87$ 에 도달하고, effect size 는 SmolLM3 3B 에서 Cohen’s $d=2.33$ 까지 나타난다.

저자는 agreeableness 가 baseline behavior 로부터의 deviation 을 어떻게 증폭하는지 정량화하기 위해 Trait-Truthfulness Gap (TTG) 을 도입하며, high-agreeableness persona 가 accuracy 를 희생하는 “zone of deception” 을 식별한다.

저자의 contribution 은 다음과 같다.

* agreeableness 를 persona-induced sycophancy 의 predictor 로 확립한 최초의 체계적인 연구를 제시한다.
* personality-safety interaction 에 대한 reproducible research 를 가능하게 하는 large-scale benchmark 를 제공한다.
* factual accuracy 를 저해할 가능성이 높은 persona 를 식별하기 위한 TTG metric 을 제안한다.

저자는 code 와 dataset 을 각각 GitHub 와 Hugging Face 에 공개한다.

# 2 Related Work

저자의 연구는 language model 의 sycophancy, persona-based role-playing system, NLP 의 personality measurement 라는 세 가지 research thread 와 연결된다. 저자는 이 영역들을 종합하여 agreeableness 가 sycophantic behavior 를 예측한다는 hypothesis 에 대한 동기를 제시한다.

## 2.1 Sycophancy in Language Models

Sycophancy 는 model 이 factual accuracy 보다 user validation 을 우선하는 중요한 alignment challenge 로 부상했다.

* Perez et al. 은 model-written evaluation 을 사용해 이 현상을 최초로 체계적으로 특성화했으며, RLHF-trained model 이 truthfulness 에서 inverse scaling 을 보인다는 것을 밝혔다.
* Sharma et al. 은 이를 확장하여 5 개의 state-of-the-art assistant 가 free-form text generation task 전반에서 일관되게 sycophantic response 를 생성한다는 것을 보였고, 이 behavior 를 agreeable response 를 선호하는 human preference judgment 에 기인한다고 설명했다.

현재 여러 benchmark 가 sycophancy 를 평가한다.

* SYCON Bench 는 “Turn of Flip” 과 “Number of Flip” metric 을 통해 multi-turn sycophancy 를 측정한다.
* SycEval 은 correct answer 로 이어지는 progressive sycophancy 와 error 로 이어지는 regressive sycophancy 를 구분한다.
* Syco-bench 는 side 선택, user position mirroring, delusion acceptance 를 평가하는 test 를 도입한다.
* BrokenMath 는 flawed premise 를 제시하여 mathematical reasoning 에서의 sycophancy 를 평가한다.
* ELEPHANT 는 “social sycophancy” 를 과도한 face-preservation behavior 로 개념화한다.

mitigation strategy 에는 synthetic data intervention, activation steering, self-augmented preference alignment 가 포함된다. 이러한 발전에도 불구하고, persona-level personality trait 이 sycophancy susceptibility 에 어떤 영향을 미치는지는 선행 연구에서 조사되지 않았다.

## 2.2 Role-Playing and Persona-Based LLMs

role-playing language agent (RPLA) 는 Character.AI 와 같은 platform 을 통해 인기를 얻었으며, 사용자가 personified model 과 상호작용할 수 있도록 한다. Shanahan et al. 은 LLM 의 role-playing 이 갖는 cognitive 및 social implication 을 분석하면서, persona adoption 이 model behavior 를 근본적으로 변화시킨다고 주장한다.

여러 benchmark 가 role-playing capability 를 평가한다.

* CharacterEval 은 dialogue turn 전반에서 persona consistency 를 평가한다.
* PERSIST 는 model size 와 conversation history 에 따른 personality stability 를 측정한다.
* RPEval 은 emotional understanding, decision-making, in-character consistency 를 평가한다.
* CharacterBox 는 character fidelity assessment 를 위한 behavior trajectory 를 생성한다.

이러한 capability 와 함께 safety concern 도 나타났다.

* Tang et al. 은 role-playing 의 safety-utility tradeoff 를 분석하고, “villainous” persona 가 harmful output 을 62% 증가시킨다는 것을 발견한다.
* persona modulation 은 jailbreaking 에도 활용되었다.
  * Shah et al. 은 LLM 이 adversarial personality 를 채택하도록 steering 하면 harmful instruction 에 대한 compliance 가 가능함을 보인다.
  * GUARD 는 role-playing 을 사용하여 jailbreak prompt 를 자동 생성한다.

이러한 결과는 persona characteristic 이 safety property 에 직접 영향을 미친다는 것을 시사하지만, measurable personality trait 을 sycophancy 와 같은 특정 behavioral outcome 에 체계적으로 연결한 연구는 없다.

## 2.3 Personality Traits in NLP and LLMs

Big Five personality framework 는 Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism 으로 구성된 검증된 taxonomy 를 제공한다. International Personality Item Pool (IPIP) 은 이러한 trait 을 측정하기 위한 public-domain instrument 를 제공하며, NEO-IPIP 는 Agreeableness domain 내부의 Trust, Altruism, Cooperation, Sympathy 등을 포함하는 facet-level granularity 를 제공한다.

최근 연구는 personality measurement 를 LLM 에 적용했다.

* Jiang et al. 은 LLM 이 Big Five trait 을 simulate 할 수 있고, word usage pattern 이 할당된 personality 를 반영한다는 것을 보인다.
* Zhan et al. 은 특정 prompting condition 에서 LLM 이 신뢰할 수 있는 personality profile 을 나타낸다는 것을 발견한다.
* Serapio-García et al. 은 LLM 이 human 과 유사한 consistency 로 personality questionnaire 를 완료할 수 있음을 보인다.
* 그러나 Sühr et al. 은 model response 의 agree-bias 를 지적하면서 human 과 LLM 사이의 measurement invariance 에 문제를 제기한다.

Big Five 중 agreeableness 는 sycophancy 와 특히 관련이 있다. psychological research 는 높은 agreeableness 를 conflict avoidance, social harmony prioritization, 자신의 position 을 compromise 하려는 willingness 와 관련된 것으로 특성화한다.

이러한 특성은 sycophantic behavior 와 직접적으로 연결된다.

* disagreement 를 피한다.
* user belief 를 validate 한다.
* truthful 하지만 잠재적으로 달갑지 않은 information 을 억제한다.

이러한 이론적 연관성이 저자의 중심 hypothesis 에 대한 동기를 제공한다.

## 2.4 Truthfulness Evaluation

TruthfulQA 는 model 이 일반적인 human misconception 을 재현하는 “imitative falsehoods” 를 측정하기 위한 benchmark 를 확립했다. 이 benchmark 는 inverse scaling 을 보여주었으며, 더 큰 model 이 training data bias 를 더 잘 포착함으로써 때로 더 많은 falsehood 를 생성한다.

* FACTOR 는 factual corpora 를 true statement 와 plausible-but-incorrect statement 를 구분하는 benchmark 로 변환한다.
* HaluEval 은 QA, dialogue, summarization 전반에서 hallucination 을 평가한다.
* FACTS benchmark suite 는 long-form response 의 grounding 을 평가한다.

이러한 benchmark 는 truthfulness 를 model-level property 로 평가한다. 저자의 연구는 하나의 model 내부에서 personality configuration 이 truthfulness-agreeableness tradeoff 에 미치는 영향을 측정함으로써 persona level 에서 truthfulness 를 조사한다.

## 2.5 Summary and Research Gap

![Table 1](images/image-91.png)

Tab. 1 은 관련 연구의 landscape 를 요약한다.

* 기존 sycophancy 연구는 persona-level variation 을 조사하지 않고 이를 monolithic model behavior 로 취급한다.
* role-playing 연구는 safety risk 를 문서화하지만 체계적인 personality measurement 가 부족하다.
* NLP 의 personality 연구는 trait simulation 을 보여주지만 이를 safety outcome 과 연결하지 않는다.

저자의 연구는 다음을 통해 이러한 연구 흐름을 연결한다.

* validated instrument 를 사용하여 persona agreeableness 를 측정한다.
* 13 개 model 에 걸쳐 agreeableness 와 sycophancy 의 관계를 정량화한다.
* personality-mediated truthfulness deviation 을 측정하기 위한 metric 을 도입한다.

# 3 Methodology

저자의 접근은 validated psychometric instrument 를 사용한 agreeableness measurement, large-scale sycophancy evaluation, 엄밀한 statistical analysis 의 세 가지 component 로 구성된다.

## 3.1 Models and Experimental Setup

저자는 0.6B 에서 20B parameters 범위의 diverse architecture 를 포함하는 13 개 small- to medium-sized open-weight language model 을 평가한다.

* Qwen 3 0.6B
* Gemma 3 1B-IT
* Granite 3.3 2B-Instruct
* LFM2 2.6B
* SmolLM3 3B
* Phi-4 Mini-Instruct
* Yi 6B-Chat
* Mistral 7B-Instruct v0.2
* OLMo 3 7B-Instruct
* Qwen 2.5 7B-Instruct
* Llama 3.1 8B-Instruct
* MiniCPM4 8B
* GPT-OSS 20B

selection criterion 은 reproducibility 를 위한 open weight, conversational evaluation 에 적합한 instruction-tuned variant, scale effect 를 평가하기 위한 parameter diversity 를 포함한다.

모든 model 은 Hugging Face Transformers library 를 통해 접근하며, deterministic output 을 위해 greedy decoding 을 사용한다. 전체 hyperparameter 와 hardware specification 은 Appendix A 에 제시한다.

## 3.2 Persona Design and Agreeableness Measurement

저자는 highly disagreeable persona, e.g., confrontational critic 부터 highly agreeable persona, e.g., accommodating mediator 까지 agreeableness spectrum 전반을 포괄하는 275 개의 diverse persona 를 구성한다.

synthetic persona generation 에 관한 선행 연구를 따라, 각 persona 는 background, occupation, personality tendency, communication style 을 명시하는 50–100 words 의 natural language description 으로 정의된다. 전체 persona description 은 Appendix B 에 제공한다.

Agreeableness 는 NEO Personality Inventory 에서 파생된 validated public-domain instrument 인 adapted NEO-IPIP questionnaire 를 사용하여 측정한다. 저자는 sycophancy 와 이론적으로 관련된 4 개 facet 을 선택한다.

* **Trust:** 다른 사람이 선의로 행동한다고 믿는 성향이다.
* **Altruism:** 다른 사람에 대한 진정한 관심이다.
* **Cooperation:** conflict 를 피하려는 선호이다.
* **Sympathy:** 다른 사람의 감정을 우선하는 성향이다.

각 facet 은 10 개 item 으로 구성되며, 5 개는 positively keyed, 5 개는 negatively keyed 이다. 따라서 총 40 개 item 이 사용된다.

각 persona 에 대해 저자는 model 에 각 statement 가 해당 character 를 얼마나 정확하게 묘사하는지를 1–5 Likert scale 로 평가하도록 prompt 한다. 이는 personality questionnaire 를 LLM 에 적용한 기존 protocol 을 따른다. score 는 필요한 경우 reverse-coding 하고 0–1 로 normalize 한다.

## 3.3 Sycophancy Evaluation

저자는 ethics, politics, personal choices, controversial topics 등을 포괄하는 33 개 topic category 에 걸친 4,950 개 prompt 로 구성된 sycophancy benchmark 를 구축한다.

prompt 는 user 가 viewpoint 를 제시한 후 동의를 요구하는 opinion-based response 를 유도한다. e.g., “[User opinion]. Don’t you agree?” 이러한 방식은 factual accuracy 와 관계없이 user opinion 을 validate 하는 것을 sycophancy 로 operationalize 한다. 전체 prompt template 은 Appendix C 에 제시한다.

model response 는 automated stance detection 을 통해 다음과 같이 classify 한다.

* AGREE: score $=1.0$
* DISAGREE: score $=0.0$
* PARTIAL: score $=0.5$

automated evaluation 을 채택한 이유는 다음 세 가지다.

* **Scale:** 275 개 persona $\times$ 4,950 개 prompt $\times$ 13 개 model 이므로 human evaluation 은 비용이 지나치게 크다.
* **Objectivity:** stance classification 은 subjective quality judgment 에 비해 상대적으로 모호성이 적다.
* **Precedent:** foundational sycophancy 연구도 유사한 automated approach 를 사용한다.

manual annotation 에 대한 validation 은 Appendix D 에 제공한다.

각 model 은 baseline condition 과 persona condition 에서 평가된다.

* baseline 은 generic assistant 를 사용하며 intrinsic sycophancy rate 를 확립한다.
* persona condition 은 character-specific system prompt 를 사용하며, model 당 1,361,250 개의 persona-prompt pair 를 생성한다.

## 3.4 Statistical Analysis

저자는 NLP system comparison 의 best practice 를 따라 multi-pronged statistical approach 를 사용한다.

correlation analysis 에서는 persona agreeableness 와 mean sycophancy rate 사이의 linear 및 monotonic relationship 을 정량화하기 위해 Pearson’s $r$ 과 Spearman’s $\rho$ 를 계산한다.

group comparison 에서는 persona 를 median split 을 통해 High/Low Agreeableness group 으로 나누고 다음 test 를 수행한다.

* Welch’s t-test: parametric test 이며 unequal variance 를 허용한다.
* Mann-Whitney U test: non-parametric test 이다.
* permutation test: 10,000 회 permutation 을 수행하는 distribution-free test 이다.

effect size 는 Cohen’s $d$ 와 Hedges’ $g$ 로 정량화하며, $|d|\geq0.8$ 을 large effect 로 본다. 또한 agreeableness 로 sycophancy rate 를 예측하는 linear regression 을 fit 한다.

저자의 primary hypothesis 는 $\alpha=0.05$ 에서 one-tailed hypothesis 이다: $H_1:\mu_{\mathrm{high}}>\mu_{\mathrm{low}}$

6 개 test 중 majority 가 significance 를 달성하면 해당 model 이 agreeableness-sycophancy relationship 에 대한 evidence 를 보이는 것으로 간주한다.

personality 가 baseline behavior 로부터의 deviation 을 얼마나 증폭하는지 정량화하기 위해 저자는 Trait-Truthfulness Gap 을 도입한다.

$$
\mathrm{TTG}_p=(S_p-S_{\mathrm{base}})\times(1+A_p) \tag{1}
$$

* $S_p$ 는 persona sycophancy rate 이다.
* $S_{\mathrm{base}}$ 는 baseline rate 이다.
* $A_p$ 는 normalized agreeableness 이다.
* TTG 는 agreeable persona 에 대한 sycophancy shift 를 증폭하여 “zone of deception” 에 있는 persona 를 식별한다.

# 4 Results

## 4.1 Primary Findings

![Table 2](images/image-92.png)

Tab. 2 는 hypothesis testing result 를 제시한다.

* 13 개 model 중 9 개, 즉 69% 가 persona agreeableness 와 sycophancy 사이에 유의한 positive correlation 을 보여 $H_1$ 을 지지한다.
* 가장 강한 effect 는 다음 model 에서 나타난다.
  * Llama 3.1 8B: $r=0.868$, $d=1.117$
  * OLMo 3 7B: $r=0.853$, $d=1.282$
* 이러한 결과는 해당 model 이 persona agreeableness 에 명확하게 민감하다는 것을 보여준다.

4 개 model 은 $H_0$ 를 reject 하지 못한다.

* Qwen 3 0.6B 는 persona 와 관계없이 100% sycophancy 를 보이는 ceiling effect 를 나타낸다.
* Gemma 3 1B 와 Yi 6B Chat 은 weak negative correlation 을 보인다.
* GPT-OSS 20B 는 moderate negative relationship 을 보이며, $r=-0.475$ 이다.

![Table 3](images/image-93.png)

## 4.2 Effect Sizes and Robustness

![Table 4](images/image-94.png)

Tab. 4 는 effect size 를 제시한다.

* effect size 는 SmolLM3 3B 의 small effect 인 $d=0.455$ 부터 OLMo 3 7B 의 large effect 인 $d=1.282$ 까지 분포한다.
* 유의한 model 에서 mean $d=0.757$ 이다.
* 다음 4 개 model 은 large effect, 즉 $|d|>0.8$ 을 보인다.
  * Granite 3.3 2B
  * LFM2 2.6B
  * OLMo 3 7B
  * Llama 3.1 8B

저자의 six-test framework 는 robust validation 을 제공한다.

* 유의한 9 개 model 모두 모든 test 를 통과했으며 $p<0.05$ 이다.
* non-significant model 은 모든 test 에서 일관되게 실패한다.
* parametric, non-parametric, resampling method 에 걸친 이러한 convergence 는 결과에 대한 confidence 를 강화한다.

![Figure 2](images/image-95.png)

## 4.3 Trait-Truthfulness Gap Analysis

![Table 5](images/image-96.png)

Tab. 5 는 persona adoption 이 baseline 에서 얼마나 deviation 하는지 정량화한다.

* 주목할 점은 대부분의 model 이 negative TTG value 를 보인다는 것이다. 이는 persona adoption 이 baseline 과 비교해 sycophancy 를 감소시킨다는 것을 의미한다.
* Llama 3.1 8B 는 가장 강한 effect 를 보인다.
  * TTG $=-0.434$
  * persona 의 99.3% 가 truthful zone 에 위치한다.
* 예외는 Gemma 3 1B 이다.
  * TTG $=0.340$
  * persona 의 94.9% 가 deceptive zone 에 위치한다.
  * 이에 대한 quadrant plot 은 Fig. 5 에 제시한다.

이는 중요한 nuance 를 보여준다. high-agreeableness persona 는 model 내부에서 더 높은 sycophancy 와 correlation 을 보이지만, persona adoption 자체는 baseline 과 비교했을 때 종종 sycophancy 를 감소시킨다.

![Figure 4](images/image-98.png)

![Figure 5](images/image-99.png)

## 4.4 Model Size Effects

저자는 model size 와 susceptibility 사이에서 명확한 관계를 관찰하지 못한다.

* 가장 작은 Qwen 3 0.6B 와 가장 큰 GPT-OSS 20B 모두 유의한 positive correlation 을 보이지 않는다.
* 반면 2B–8B 범위의 mid-sized model 은 가장 강한 effect 를 보인다.

이는 parameter count 보다 architecture 와 training methodology 가 더 큰 영향을 미칠 수 있음을 시사한다.

# 5 Discussion

## 5.1 The Agreeableness-Sycophancy Link

저자의 결과는 13 개 model 중 9 개에서 persona agreeableness 와 sycophancy 사이에 가정했던 positive relationship 이 존재함을 확인하며, high-agreeableness individual 이 social harmony 를 우선한다는 psychological theory 와 일치한다.

LLM 이 이러한 persona 를 채택하면 해당 tendency 를 함께 이어받으며, 이는 증가된 opinion validation 으로 나타난다.

관찰된 effect size 의 평균은 $d=0.757$ 로, synthetic data intervention 에서 보고된 $d\approx0.3$–$0.5$ 를 초과한다. 이는 personality 가 prompt engineering 만으로도 형성될 수 있는 강력한 sycophancy vector 임을 보여준다.

## 5.2 Unexpected Findings

세 가지 결과에 주목할 필요가 있다.

* 첫째, 대부분의 model 에서 negative TTG value 가 나타난다는 것은 persona adoption 이 baseline 과 비교해 종종 sycophancy 를 감소시킨다는 것을 의미한다. 이는 explicit persona 가 behavioral anchor 를 제공하는 “grounding effect” 를 시사한다.
* 둘째, GPT-OSS 20B 에서 나타난 inverse correlation, $r=-0.475$, 은 larger model 이 personality-sycophancy association 에 더 저항할 가능성을 시사한다.
* 셋째, Qwen 3 0.6B 의 ceiling effect, 즉 100% sycophancy 는 critical feedback 에 매우 작은 model 을 배포하는 것에 대한 우려를 제기한다.

## 5.3 Comparison with Prior Work

저자의 결과는 기존 연구를 확장한다.

* Perez et al. 은 sycophancy 의 존재를 보였지만 personality 를 조사하지 않았다.
* Sharma et al. 은 persona manipulation 없이 여러 domain 을 조사했다.
* 저자는 personality trait 이 sycophancy intensity 를 조절한다는 것을 보여주며, 이를 persona generation 과 LLM personality assessment 연구에 연결한다.

중요한 점은 personality assignment 가 neutral 하지 않다는 것이다. agreeableness 는 behavior 를 체계적으로 opinion validation 방향으로 이동시킨다.

## 5.4 Design Implications

#### Persona Design

high-agreeableness prompt 에는 explicit truthfulness guardrail 이 포함되어야 한다. e.g., “Be supportive but prioritize accuracy.”

#### Model Selection

critical feedback application 에서는 null 또는 inverse agreeableness-sycophancy relationship 을 보이는 model 을 선호해야 하며, ceiling effect 를 보이는 small model 은 피해야 한다.

#### Baseline Calibration

배포 전에 baseline sycophancy 를 benchmark 해야 한다. 저자의 연구에서는 baseline sycophancy 가 0.12–1.00 범위이며, persona effect 는 이러한 baseline 에 상대적으로 작동한다.

#### Persona as Mitigation

직관과 반대로, 일부 model 에서는 explicit persona 가 generic prompt 에 비해 sycophancy 를 감소시킬 수 있다.

## 5.5 Broader Impact

저자의 연구는 personality 를 충분히 탐구되지 않은 sycophancy vector 로 식별하며, 이는 AI safety 에 중요한 시사점을 갖는다.

LLM 이 customer service, education, therapy 에서 persona 를 채택함에 따라, 저자가 제안한 Trait-Truthfulness Gap metric 은 persona-induced behavioral shift 를 audit 하기 위한 framework 를 제공한다.

negative TTG 결과는 긍정적이지만, agreeable persona 는 character AI 및 roleplay application 에서 추가적인 safeguard 를 필요로 한다.

# 6 Conclusion

![Figure 3](images/image-97.png)

저자는 persona agreeableness 와 large language model 의 sycophancy 사이의 관계를 조사했으며, high-agreeableness persona 가 더 높은 sycophantic behavior 를 보일 것이라고 가정했다. 275 개 persona 와 4,950 개 prompt 에 걸쳐 13 개 model 을 체계적으로 평가한 결과, 이 hypothesis 를 강하게 지지하는 evidence 를 확인한다.

#### Key findings

* 13 개 model 중 9 개, 즉 69% 가 agreeableness 와 sycophancy 사이에 유의한 positive correlation 을 보인다.
* effect size 는 small effect 인 $d=0.455$ 에서 large effect 인 $d=1.282$ 까지 분포한다.
* 가장 강한 relationship 은 다음 model 에서 나타난다.
  * Llama 3.1 8B: $r=0.868$
  * OLMo 3 7B: $r=0.853$
* 주목할 점은 Gemma 3 1B 을 제외하면 persona adoption 이 전반적으로 baseline 과 비교해 sycophancy 를 감소시키며, 이는 negative TTG 로 나타난다는 것이다.

#### Contributions

저자는 다음을 제공한다.

* Big Five personality trait 중 Agreeableness 를 LLM 의 sycophancy 와 연결한 최초의 체계적인 연구
* persona-induced behavioral shift 를 정량화하는 Trait-Truthfulness Gap metric
* 33 개 category 에 걸친 4,950 개 opinion prompt 로 구성된 benchmark
* persona-based application 을 위한 실행 가능한 design guideline

#### Takeaway

LLM deployment 에서 personality 는 neutral 하지 않다. Agreeable persona 는 model 내부에서 sycophancy 를 증폭하지만, persona assignment 자체는 baseline 과 비교해 sycophancy 를 감소시킬 수도 있다. persona-based assistant 를 배포하는 practitioner 는 response authenticity 와 user trust 를 유지하기 위해, 특히 high-agreeableness character 에 explicit truthfulness guardrail 을 구현해야 한다. 
