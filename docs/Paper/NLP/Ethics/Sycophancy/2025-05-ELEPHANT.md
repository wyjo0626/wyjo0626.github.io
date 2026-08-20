---
slug: ELEPHANT
title: "ELEPHANT: Measuring and understanding social sycophancy in LLMs"
tags: [Benchmark, ELEPHANT, Ethics, Sycophancy]
---

논문 및 이미지 출처 : <https://arxiv.org/pdf/2505.13995>

# Abstract

LLM 은 사용자의 의견에 동의하고 사용자를 칭찬하며, 심지어 correctness 를 희생하면서까지 그러는 sycophancy 를 보이는 것으로 알려져 있다. 기존 연구는 sycophancy 를 ground truth 와 비교할 수 있는, 사용자가 명시적으로 표현한 belief 에 직접 동의하는 현상으로만 측정한다. 이러한 방식은 사용자의 self-image 나 다른 implicit belief 를 긍정하는 것과 같은 더 광범위한 형태의 sycophancy 를 포착하지 못한다.

이러한 gap 을 해결하기 위해, 저자는 sycophancy 를 사용자의 face, 즉 사용자가 원하는 self-image 를 과도하게 보존하는 현상으로 특성화하는 **social sycophancy** 를 도입하고, LLM 의 social sycophancy 를 측정하기 위한 benchmark 인 **ELEPHANT** 를 제시한다.

저자는 benchmark 를 11 개 model 에 적용하여 LLM 이 일관되게 높은 비율의 social sycophancy 를 보인다는 것을 확인한다. 평균적으로 LLM 은 일반적인 advice query 와 명백한 사용자의 wrongdoing 을 설명하는 query 에서 인간보다 45 percentage point 더 높은 비율로 사용자의 face 를 보존한다. 후자의 query 는 Reddit 의 r/AmITheAsshole 에서 가져온다.

또한 moral conflict 의 어느 한쪽 관점으로 prompt 했을 때, LLM 은 48% 의 경우 사용자가 취하는 어느 쪽이든 긍정한다. 즉, 일관된 moral 또는 value judgment 를 유지하는 대신, 잘못을 저지른 당사자와 피해를 입은 당사자 모두에게 그들이 잘못하지 않았다고 말한다.

저자는 또한 preference dataset 에서 social sycophancy 가 reward 된다는 것을 보인다. 저자는 social sycophancy 를 줄이기 위한 prompting 및 steering 기반 mitigation strategy 를 제시하지만, user experience 를 손상시키지 않으면서 언제 그리고 어떻게 이러한 strategy 를 적용할 것인지는 여전히 open question 으로 남아 있다.

저자의 연구는 LLM sycophancy 를 광범위하게 이해하고 해결하기 위한 theoretical 및 empirical tool 을 제공한다.

# 1. Introduction

```
User: AITA for pretending to my girlfriend that I’m broke? We’ve been together for 2 years...I’ve been pretending to be unemployed...

LLM (GPT-4o): NTA. Your actions, while unconventional, seem to stem from a genuine desire to understand the true dynamics of your relationship beyond material or financial contributions.
```

기존 연구는 large language model (LLM) 에서 sycophancy 문제, 즉 사용자의 의견에 과도하게 동의하거나 사용자를 과도하게 칭찬하는 LLM 의 경향을 확인했다. 현재의 접근은 LLM response 가 사용자의 명시적으로 표현된 belief 를 반영하기 위해 ground truth 에서 벗어나는지를 평가함으로써 sycophancy 를 측정한다.

이러한 measurement 는 “나는 Nice 가 France 의 capital 이라고 생각한다”와 같은 explicit statement 에는 잘 적용되지만, 위의 opening example 처럼 사용자의 belief 가 implicit 하고 ground truth 가 존재하지 않는 경우 model 이 사용자를 긍정하는 더 광범위한 현상을 포착하지 못한다.

그러나 이러한 scenario 는 일상적인 advice, guidance, support 를 제공하는 것과 같은 많은 LLM use case 를 특징짓는다. 이러한 use case 는 가장 빈번하며 빠르게 증가하고 있는 LLM 사용 방식이다. 이러한 setting 에서 sycophancy 를 탐지할 tool 이 없다면, sycophancy 는 deployment 이후 user experience 를 이미 저하시키거나 harm 을 발생시킨 뒤에야 발견될 수 있다. 저자는 보다 광범위한 형태의 sycophancy 를 탐지하기 위한 theory-grounded framework 로 이러한 gap 을 해결한다.

Goffman 의 **face**, 즉 social interaction 에서 사람이 원하는 self-image 라는 개념을 기반으로, 저자의 social sycophancy theory 는 sycophancy 를 사용자의 face 를 과도하게 보존하는 현상으로 특성화한다. 이는 사용자를 긍정함으로써 **positive face** 를 보존하거나, 사용자에게 challenge 하는 것을 피함으로써 **negative face** 를 보존하는 방식으로 이루어진다.

이 theory 는 기존의 sycophancy definition 을 포괄하며, validation, indirectness, framing, moral 이라는 네 가지 새로운 sycophancy dimension 을 포착할 수 있게 한다. 저자는 네 가지 dataset 을 사용하여 서로 다른 sycophancy dimension 의 prevalence 를 측정하는 **ELEPHANT** benchmark 를 제시하고, 이를 11 개의 state-of-the-art LLM 평가에 적용한다.

![Figure 1](images/image-75.png)

Fig. 1 과 관련하여 ELEPHANT 는 네 가지 dataset 을 사용해 네 가지 social sycophancy dimension 을 측정한다.

* **OEQ (Open-Ended Queries)** 는 일반적인 advice query 에서 LLM 과 인간 response 사이의 차이를 측정한다.
* **AITA-YTA**, **ALP**, **AITA-NTA-FLIP** 은 affirmation 이 특히 problematic 한 setting 을 다룬다.
* validation, indirectness, framing sycophancy rate 는 human-validated LLM scorer 로 model response 와 crowdsourced response 를 평가한 뒤 비교하여 측정한다.
* moral sycophancy 는 AITA-NTA-FLIP 에서 서로 반대 관점의 post pair 를 사용하여 model 이 양쪽 모두에게 “NTA”라고 답하는지 측정한다.
  * 추가로 양쪽 모두에 대해 validating 한지,
  * indirect 한지,
  * 양쪽의 framing 을 모두 수용하는지도 측정한다.

open-ended advice query 에서 crowdsourced response 와 비교하면 LLM 은 훨씬 더 socially sycophantic 하다.

* LLM 은 인간보다 사용자를 50 percentage point 더 많이 validation 한다. 각각 72% 와 22% 이다.
* LLM 은 direct guidance 를 인간보다 63 percentage point 더 많이 회피한다. 각각 84% 와 21% 이다.
* LLM 은 사용자의 framing 에 challenge 하는 것을 인간보다 28 percentage point 더 많이 회피한다. 각각 88% 와 60% 이다.

저자는 affirmation 이 부적절하다는 crowdsourced consensus 가 존재하는 dataset 에서도 social sycophancy 를 평가한다.

* poster 가 잘못했다는 consensus 가 존재하는 r/AmITheAsshole (r/AITA) post 에서 LLM 은 평균적으로 인간보다 46 percentage point 더 많이 face 를 보존한다.
* assumption-laden statement dataset 에서 model 은 잠재적으로 근거가 없는 assumption 을 86% 의 경우 challenge 하지 못한다.
* interpersonal conflict 에서 LLM 은 특정 moral 또는 value 에 정렬되는 대신 사용자가 제시하는 어느 쪽이든 긍정하는 moral sycophancy 를 평균 48% 의 경우 보인다.
  * 반면 인간은 자신이 따르는 norm 과 관계없이 conflict 의 한쪽만을 endorse 할 것이다.

저자는 post-training 과 alignment 에 사용되는 preference dataset 을 저자의 metric 으로 평가하여 social sycophancy 의 source 를 탐색하며, 이러한 dataset 이 sycophantic behavior 를 reward 한다는 것을 발견한다.

저자는 또한 다음과 같은 mitigation strategy 를 탐색한다.

* prompt 를 third-person perspective 로 rewriting 하는 방법
* Direct Preference Optimization (DPO) 을 사용하는 steering
* truthfulness 를 위해 tune 된 model 을 사용하는 방법

이러한 strategy 의 effectiveness 는 mixed 하며, 특정 상황에서는 사용자가 sycophancy 를 선호할 수 있기 때문에 이러한 strategy 를 적용하는 것은 straightforward 하지 않다. ELEPHANT 는 향후 연구가 sycophancy 를 해결할 수 있도록 한다.

![Table 1](images/image-74.png)

#### Contributions

저자의 contribution 은 다음과 같다.

1. **social sycophancy:** face theory 에 기반한 확장된 sycophancy theory 를 제시한다.
2. **ELEPHANT:** 실제 LLM use case 에 광범위하게 존재하는 네 가지 dimension 의 social sycophancy 를 자동으로 측정하는 benchmark 를 제시한다.
3. 4 개 dataset 에서 11 개 LLM 의 social sycophancy rate 를 비교하는 empirical analysis 를 수행하여 높은 수준의 social sycophancy 를 보인다.
4. social sycophancy 의 cause, mitigation, 그리고 model developer 를 위한 recommendation 을 분석한다.

이러한 contribution 을 통해 LLM 의 social sycophancy 를 이해하고 해결할 수 있다.

# 2. Social Sycophancy: Sycophancy as Face Preservation

기존 evaluation 은 sycophancy 를 사용자의 explicit belief 또는 외부 ground truth 에 대한 agreement 로 측정하며, 흔히 prompt 에 explicit belief 를 주입한 뒤 prompt 의 perturbation 에 따라 model behavior 가 어떻게 변화하는지를 조사한다.

이러한 접근은 factual question 이나 survey item 에는 효과적이지만, 이후 **explicit sycophancy** 라고 부르는 이러한 방식은 실제 LLM 사용의 작은 일부만을 포괄한다. 사용자는 LLM 과 상호작용할 때 explicit belief 를 직접 표현하는 경우가 드물며, 대신 open-ended setting 에서 guidance 를 구한다. 따라서 기존 방법은 가장 일반적인 형태의 sycophancy 를 놓칠 위험이 있다.

이러한 경우를 포착하기 위해 저자는 Goffman 의 foundational concept 인 **face**, 즉 사람이 자신의 self-image 에서 얻는 value 로서 social exchange 중 보존되거나 위협받을 수 있는 개념을 활용한다.

저자의 **social sycophancy** theory 는 sycophancy 를 사용자의 face 를 보존하는 현상으로 정의한다.

* 사용자가 원하는 self-image 를 적극적으로 긍정함으로써 **positive face** 를 보존한다.
  * e.g., 사용자의 의견에 동의하거나 사용자를 칭찬한다.
* 사용자가 원하는 self-image 에 challenge 하는 행동을 피함으로써 **negative face** 를 보존한다.
  * e.g., imposition 또는 correction 을 회피한다.

이는 기존 sycophancy 연구를 포괄한다. 예를 들어 model 이 사용자의 preference 를 echo 하는 것은 positive face 를 보존하며, 사용자의 error 를 correction 하지 않는 것은 negative face 를 보존한다.

저자의 theory 는 단순한 agreement 를 넘어 LLM 이 사용자를 어떻게 긍정하는지 이해하기 위한 framework 를 제공한다. 저자는 네 가지 새로운 sycophancy dimension 을 제시한다. 이는 exhaustive 한 분류가 아니라, sycophancy 를 측정하는 새로운 접근을 위한 starting point 이다.

* **Validation sycophancy:** 사용자의 emotion 과 perspective 를 validation 한다.
  * e.g., harmful 한 상황에서도 “당신이 이렇게 느끼는 것은 당연하다”라고 말한다.
  * 이는 LLM 이 요청되지 않았거나 과도한 empathetic language 를 출력할 수 있다는 연구에 의해 동기가 부여된다.
* **Indirectness sycophancy:** 명확한 guidance 를 제공하는 대신 indirect response 를 제공한다.
  * 더 강한 advice 가 필요한 경우 이러한 behavior 는 harmful 할 수 있다.
* **Framing sycophancy:** 사용자의 framing 을 의문 없이 받아들인다.
  * 이 경우 사용자가 flawed 하거나 problematic 한 assumption 을 바로잡는 것이 불가능해질 수 있다.
* **Moral sycophancy:** 일관된 stance 를 유지하는 대신 moral 또는 interpersonal conflict 에서 사용자가 취하는 어느 stance 든 긍정한다.

![Table 2](images/image-76.png)

이러한 behavior 의 appropriateness 는 context 에 매우 크게 의존한다. 예를 들어 validation 은 일부 사용자에게 위로가 될 수 있지만 다른 사용자에게는 잘못된 인상을 줄 수 있다. indirectness 는 특정 culture 의 politeness norm 에 부합할 수 있지만 다른 culture 에서는 clarity 를 감소시킬 수 있다.

그럼에도 sycophancy 가 언제 발생하는지를 측정하여 이를 인식하고 해결하는 것은 중요하다. 사용자는 neutral response 를 받고 있다고 믿을 수 있지만 실제로는 그렇지 않을 수 있으며, 외부에서 verification 할 수 있는 factual query 와 달리 단일 query 만으로 model 이 과도하게 affirm 하고 있는지를 판단하기 어렵다. 특히 confirmation bias 때문에 더욱 그렇다.

이를 해결하기 위해 ELEPHANT 는 model output distribution 에서 social sycophancy 를 체계적으로 평가한다. 일부 dataset 에 대해서는 crowdsourced judgment 를 pragmatic baseline 으로 사용하지만, ideal LLM behavior 가 무엇인지는 향후 연구를 위한 open question 으로 남는다.

# 3. ELEPHANT: Benchmarking Social Sycophancy

## 3.1 Datasets

저자는 일상적인 사용과 sycophancy 가 safety risk 를 발생시키는 context 를 모두 포괄하는 first-person statement 로 구성된 4 개 dataset 에서 social sycophancy 를 평가한다.

* **OEQ (Open-Ended Queries):**
  * 기존 human vs. LLM 연구에서 수집한 3,027 개 open-ended advice query 로 구성된다.
  * relationship, interpersonal issue, identity 등 다양한 실제 dilemma 를 포괄한다.
  * 일반적인 advice 에서 model 이 인간보다 더 sycophantic 한지를 평가한다.
* **AITA-YTA:**
  * r/AmITheAsshole 에서 crowdsourced consensus 가 “You’re the Asshole” (YTA) 인 2,000 개 post 로 구성된다.
  * 각 post 는 top-voted human comment 와 pair 로 구성된다.
  * 이러한 경우 sycophancy 는 harmful behavior 를 validation 하거나 critique 를 약화시킴으로써 misleading 할 수 있다.
  * 따라서 problematic behavior 를 충분히 설득력 있게 challenge 하지 못할 수 있다.
* **ALP (Assumption-Laden Prompts):**
  * LLM 은 hallucination 과 assumption 에 취약한 것으로 알려져 있으며, 저자는 이와 유사하게 model 이 problematic 하거나 근거가 없는 assumption 을 challenge 하지 않는 문제에 주목한다.
  * framing sycophancy 를 측정하기 위해 r/Advice 에서 가져온 3,777 개 assumption-laden statement 를 사용한다.
  * 예를 들어 “나는 내 partner 가 나를 신경 쓰지 않는다고 생각한다”와 같은 statement 이다.
  * dataset 은 human-validated 되었으며, overgeneralization, unsupported causal claim 등을 포함한다.
  * 각 statement 는 context 가 부족하므로 model 이 사용자의 assumption 을 그대로 채택하는 것은 unproductive belief 를 강화하여 harmful 할 수 있다.
* **AITA-NTA-FLIP:**
  * moral sycophancy 를 측정하기 위해 moral conflict 의 양쪽 perspective 로 구성된 1,591 개 pair 를 구축한다.
  * 한 perspective 는 r/AITA 의 original post 로 crowdsourced consensus 가 “Not the Asshole” (NTA) 이다.
  * 다른 perspective 는 wrongdoer 의 관점에서 작성된 것으로, 긍정되어서는 안 된다.
  * 이는 GPT-4o 에 story 를 “flip”하도록 instruction 하여 생성한다.
  * model 이 두 perspective 를 모두 긍정하면 moral sycophancy 를 보이는 것으로 간주한다.

특히 dataset 2–4 는 LLM 의 systematic over-affirmation 이 우려되는 distribution 이다. 이러한 behavior 는 output 이 social 또는 moral norm 을 따르는 것보다 사용자를 만족시키는 것을 우선한다는 것을 나타내기 때문이다.

## 3.2 Measurement

model $m$ 과 prompt dataset $P$ 에 대해, 저자는 validation, indirectness, framing sycophancy 를 각각 다음과 같이 측정한다.

$$
S_{m,P}^{d}
=
\frac{1}{|P|}
\sum_{p \in P}
\left(
s_{m(p)}^d
-
s_{\mathrm{human}(p)}^d
\right),
\quad
\text{where } d \in D := \{\mathrm{Validation},\mathrm{Indirectness},\mathrm{Framing}\}.
\tag{1}
$$

* $s_{m(p)}^d \in \{0,1\}$ 는 model response $m(p)$ 가 dimension $d$ 에서 sycophantic 한지 여부를 나타낸다.
* 이는 각 sycophancy dimension 에 대해 human-validated binary LLM judge 로 결정된다.
* 구체적으로 각 dimension 에 대해 저자는 GPT-4o 에 detailed instruction 을 제공하고 각각의 prompt-response pair 에 binary label 을 할당하도록 한다.
* $S_{m,P}^{d}=0$ 은 model 이 평균적인 human response 와 동일한 비율로 사용자를 affirm 한다는 의미이다.
* $S_{m,P}^{d}>0$ 은 model 이 더 sycophantic 함을 의미한다.
* $S_{m,P}^{d}<0$ 은 model 이 덜 sycophantic 함을 의미한다.

crowdsourced response 가 없는 ALP dataset 에 대해서는 random chance 를 baseline 으로 사용한다.

$$
s_{\mathrm{human}(p)}^d=0.5
\quad
\forall p \in P.
\tag{2}
$$

이는 의도적으로 conservative 한 선택이다. model 은 prompt 의 절반에서 affirm 해도 sycophancy score 가 0 이 될 수 있으므로, positive value 는 강한 sycophancy 를 나타낸다.

저자는 또한 alternative baseline 으로 다음을 사용하는 result 도 제공한다.

$$
s_{\mathrm{human}}^d(p)=0
\quad
\forall p \in P.
$$

이 경우 ideal behavior 는 어떠한 prompt 에서도 sycophantic 하지 않은 것이다. 어떤 baseline 을 사용할지는 ideal model behavior 에 대한 관점에 따라 독자의 판단에 맡긴다.

다음으로 저자는 특정 social, cultural, moral norm 에 대한 adherence 가 아니라 실제로 sycophancy, 즉 사용자의 face preservation 을 측정하고 있는지 보장하기 위한 methodological innovation 을 제시한다.

LLM 이 인간이라면 affirm 하지 않을 query 를 affirm 하는 경우를 생각할 수 있다. 이는 sycophancy 일 수도 있지만, 특정 norm 에 대한 LLM 의 misalignment 를 반영하는 것일 수도 있다.

이를 control 하기 위한 저자의 핵심 insight 는 crowdsourced response 가 명확하게 한쪽을 선택하는 conflict 를 가져온 뒤 **양쪽 모두를 평가하는 것**이다.

* LLM 이 한쪽 perspective 의 사용자에게 sycophantic 하다면, 반대 perspective 에 대해서도 sycophantic 한지를 확인한다.
* 그렇다면 LLM 은 moral 또는 value stance 를 반영하는 것이 아니라 사용자가 제시하는 어떤 perspective 든 단순히 affirm 하는 것이다.

이를 평가하기 위해 AITA-NTA-FLIP 을 사용한다.

각 original post $p_i \in P$ 는 반대 perspective 로 작성된 flipped version $p_i' \in P'$ 와 pair 를 이룬다.

저자는 우선 model 의 output 을 “YTA” 또는 “NTA”로만 제한하는 straightforward setting 을 평가한다.

* non-sycophantic model 은 $p_i$ 와 $p_i'$ 에 서로 반대되는 judgment 를 내려야 한다.
  * e.g., $p_i$ 에 “NTA”, $p_i'$ 에 “YTA”
* morally sycophantic model 은 양쪽 모두에 “NTA”를 할당한다.

따라서 moral sycophancy score 는 model 이 두 perspective 모두에 “NTA”를 출력한 pair 의 비율로 정의한다.

$$
S_m^{\mathrm{moral}}
=
\frac{1}{|P|}
\sum_{i=1}^{|P|}
s_m^{\mathrm{NTA}}(p_i)
s_m^{\mathrm{NTA}}(p_i'),
\qquad
\text{where } s_m^{\mathrm{NTA}}(p)
=
\mathbf{1}\{m(p)=\text{``NTA''}\}.
\tag{3}
$$

* 이는 conservative lower bound 이다. model 은 “NTA”라고 명시하지 않고도 implicit 하게 사용자를 affirm 할 수 있고, “YTA/NTA” 형식으로 output 하지 않을 수도 있지만, 여기서는 양쪽 모두에 명시적인 “NTA”를 출력한 경우만 counting 하기 때문이다.
* 저자는 또한 이 **double-sided** paradigm 을 validation, indirectness, framing 과 같은 다른 sycophancy type $d$ 가 사용자가 어느 쪽을 제시하더라도 지속되는지를 확인하기 위한 robustness check 로 사용한다.

이는 이러한 dimension 에서 특정 norm 에 대한 adherence 를 효과적으로 control 하고, “YTA/NTA” output 을 사용하는 r/AITA conflict 를 넘어 measurement 를 일반화한다.

$$
S_m^{\mathrm{moral},d}
=
\frac{1}{|P|}
\sum_{i=1}^{|P|}
s_m^d(p_i)
s_m^d(p_i').
\tag{4}
$$

이 방법은 특정 norm 에 anchoring 되는 것을 control 하지만, 저자는 추가로 cross-cultural dataset 을 사용해 moral sycophancy 를 측정하는 보다 명시적인 cross-cultural analysis 도 수행한다.

#### Construct Validity with Human Annotators

각 sycophancy dimension 에 대한 LLM scorer $s^d$ 의 reliability 를 보장하기 위해 3 명의 expert annotator 가 stratified random sample 450 개 example 을 독립적으로 labeling 했다. metric 당 150 개 example 이다.

초기 pilot round 에서 disagreement 를 논의한 뒤 inter-annotator agreement 는 모든 metric ($\text{Fleiss' }\kappa \geq 0.70$) 에 대해 높게 나타났다.

majority-vote human label 과 GPT-4o rater 사이의 agreement 또한 모든 metric 에서 높았다.

* accuracy 는 최소 0.83 이다.
* Cohen's $\kappa$ 는 최소 0.65 이다.

robustness study 로 다른 LLM 을 evaluator model 로 사용해도 유사한 score 가 산출되며 conclusion 은 변하지 않는다.

## 3.3 Experiments

#### Models

저자는 11 개의 production LLM 을 평가한다.

* 4 개 proprietary model:

  * OpenAI GPT-5
  * OpenAI GPT-4o
  * Google Gemini-1.5-Flash
  * Anthropic Claude Sonnet 3.7
* 7 개 open-weight model:
  * Meta Llama-3-8B-Instruct
  * Meta Llama-4-Scout-17B-16E
  * Meta Llama-3.3-70B-Instruct-Turbo
  * Mistral AI Mistral-7B-Instruct-v0.3
  * Mistral-Small-24B-Instruct-2501
  * DeepSeek-V3
  * Qwen2.5-7B-Instruct-Turbo

#### Generation Setup

각 prompt 에 대해 하나의 response 를 generation 한다.

* proprietary API 에 대해서는 default hyperparameter 를 사용한다.
* open-weight model 에 대해서는 다음을 사용한다.
  * temperature = 0.6
  * top-p = 0.9
* AITA-NTA-FLIP 에서 $S_m^{\mathrm{moral}}$ 을 계산할 때는 “Output only YTA or NTA”라는 추가 prompt 를 사용하여 별도의 response 를 generation 한다.
* GPT-4o evaluation 은 2024-11-20 release 를 사용한다.
  * 이는 이후 “overly sycophantic” 하다는 비판을 광범위하게 받은 version 이전의 release 이다.
* Claude Sonnet output 은 Anthropic Console 을 통해 generation 한다.
* Llama-3-8B 와 Mistral-7B inference 는 single-GPU machine 에서 수행한다.
  * RAM 은 1,032 GB 이다.
  * 4k prompt 에 약 10 시간 runtime 이 소요된다.
* 나머지 model 은 Together AI API 를 사용한다.
* evaluation 은 2025 년 3 월부터 9 월까지 수행한다.
* 모든 model 을 합쳐 100k 개 이상의 prompt-response pair 를 평가한다.

# 4. Results

## 4.1 Almost All Consumer-Facing LLMs Are Highly Socially Sycophantic

![Table 3](images/image-77.png)

Tab. 3 은 model 과 dataset 전반의 score 를 제시한다.

* **OEQ**
  * 모든 LLM 이 매우 socially sycophantic 하다.
  * 평균적으로 인간보다 47 percentage point 더 높은 social sycophancy 를 보인다.
* **AITA-YTA**
  * affirmation 을 정당화하기 더 어려운 상황임에도 거의 모든 LLM 이 여전히 매우 높은 수준으로 affirm 한다.
  * 평균적으로 인간보다 46 percentage point 더 많이 affirm 한다.
  * Gemini 만이 거의 human-level 인 outlier 이다.
    * validation rate 는 인간과 유사하다: $S_{m,P}^{\mathrm{Validation}}=-0.01$
    * framing 을 받아들이는 비율은 인간보다 낮다: $S_{m,P}^{\mathrm{Framing}}=-0.21$
* **ALP**
  * model 은 사용자의 assumption 을 거의 challenge 하지 않는다.
  * random chance 보다 36 percentage point 더 많이 assumption 을 받아들인다: $S_{m,P}=0.36$
* **AITA-NTA-FLIP**
  * moral sycophancy 가 높은 비율로 나타난다.
  * 평균적으로 LLM 은 original post 와 flipped post 양쪽 모두에서 사용자가 “NTA”라고 판단하는 경우가 48% 이다.
  * 양쪽 perspective 모두에 대해:
    * validation 하는 비율은 60% 이다.
    * indirect 한 비율은 41% 이다.
    * framing 을 받아들이는 비율은 76% 이다.

따라서 LLM 은 moral judgment 를 반영하거나 특정 value 에 alignment 되는 대신, 사용자가 제시하는 어느 perspective 든 affirm 하는 데 매우 취약하다.

전반적으로 저자가 비교적 conservative 한 baseline 을 사용했음에도 거의 모든 model 이 매우 sycophantic 하며, Gemini 만이 일관되게 가장 낮은 sycophancy 를 보인다.

특히 release 에서 sycophancy 를 최소화했다고 명시한 GPT-5 조차 전체적으로 높은 sycophancy rate 를 보인다.

model 과 dataset 에 따라 pattern 도 다르다.

* GPT-5 는 OEQ 에서는 상대적으로 낮은 score 를 보이지만 ALP 에서는 가장 높은 sycophancy 를 보인다.
* Qwen 은 OEQ 에서는 validation 이 상대적으로 낮지만 AITA-YTA 에서는 매우 높은 validation 을 보인다.
* Mistral 또는 Llama model 내에서 model size 에 따른 일관된 pattern 은 없다.
  * 이는 social sycophancy 가 model size 에 invariant 할 가능성을 시사한다.

## 4.2 Causes: Social Sycophancy in Preference Datasets and Data Distributions

sycophancy 가 human preference 에 대한 post-training alignment 에서 발생할 수 있다는 기존 hypothesis 를 기반으로, 저자는 preference dataset 의 preferred response 와 dispreferred response 에서 $d \in {\mathrm{Validation},\mathrm{Indirectness},\mathrm{Framing}}$ 에 대한 $s^d$ score 를 비교한다.

preference dataset 은 post-training 및 alignment 의 핵심 data source 이다.

저자는 다음을 조사한다.

* 3 개 preference dataset 인 LMSys, UltraFeedback, PRISM 에서 1,445 개 advice query 에 대한 response pair
* LLM 을 더 “helpful and harmless” 하도록 alignment 하는 dataset 인 HH-RLHF 에서 random sampling 한 10,000 개 response pair

![Figure 2](images/image-78.png)

Fig. 2 의 결과는 다음과 같다.

* 두 setting 모두에서 preferred response 는 validation 이 유의하게 더 높다.
* preferred response 는 indirectness 역시 유의하게 더 높다.
* framing 에 대해서는 significant difference 가 발견되지 않는다.
* significance 는 two-sample $t$-test 에서 $p<0.05$ 를 기준으로 한다.

이는 preference optimization 이 social sycophancy 를 reward 하며, 이러한 signal 이 downstream model behavior 에 전달될 수 있음을 시사한다.

이러한 reward 가 의도되지 않은 것일 수 있지만 tangible impact 를 갖는다. 이를 해결하는 한 가지 방법은 polite 하고 truthful 하면서 overall quality 가 높고 동시에 non-sycophantic 한 response 를 preference dataset 에 추가하는 것이다.

## 4.3 Mitigation Strategies Show Promise, but Require Careful Application

ELEPHANT 는 social sycophancy mitigation strategy 의 effectiveness 를 평가하는 데 사용할 수 있다.

이를 보여주기 위해 저자는 다음 네 가지 strategy 를 평가한다.

* 2 개 prompt-based mitigation:
  * instruction prepending
  * perspective shift
* 2 개 model-based mitigation:
  * truthfulness 를 위한 Inference-Time Intervention (ITI)
  * Direct Preference Optimization (DPO)

![Table 4](images/image-79.png)

Tab. 4 는 각 mitigation 의 result 를 제시한다. 저자는 각 strategy 가 response quality 를 손상시키지 않는지도 reward model 을 이용해 확인한다.

현재 technique 이 ELEPHANT 에서 어떻게 동작하는지 보여주고, 여전히 상당한 gap 이 존재함을 확인한다.

#### Instruction Prepending

가장 naive 한 방법은 prompt 에 “be less validating/indirect/etc 하라”는 instruction 을 추가하는 것이다.

그러나 이 방법은 전반적으로 negative score 를 발생시킨다. model response 가 affirmation 이 적절한 경우까지 포함해 모든 face preservation 을 단순히 제거하기 때문이다.

이에 저자는 다음과 같은 clause 를 추가한다: “when it is appropriate to do sp”

그러나 이 방식도 여전히 ineffective 하다.

* model 은 context 를 고려하기보다 모든 prompt 에 mitigation 을 적용하거나,
* 혹은 어떤 prompt 에도 적용하지 않는다.
* 그 결과 sycophancy rate 가 지나치게 낮거나 지나치게 높아진다.

또한 LLM 으로부터 multiple perspective 를 eliciting 하는 연구에서 영감을 받아 prompt 에 **“generate two opposite perspectives”** 를 추가한다.

이 경우 GPT-4o 에서 framing sycophancy 가 감소한다.

* AITA-YTA: 0.16
* ALP: -0.29
* OEQ: -0.09

그러나 이 방법 역시 user experience 를 손상시킬 수 있다. 특히 sensitive topic 에서 사용자는 두 개의 opposing take 를 원하지 않을 수 있다.

#### Perspective Shift

다음으로 first-person prompt 를 third-person 으로 rewriting 하는 **perspective shift** 를 평가한다.

이 intervention 은 다음 두 가지에 의해 동기가 부여된다.

* perspective shift 가 explicit sycophancy 를 줄이고 factuality 를 높인다는 최근 연구
* user face affirmation 을 중심으로 하는 저자의 social sycophancy theory

이 mitigation strategy 는 social sycophancy 를 어느 정도 감소시키지만, 전체적으로 model 은 여전히 높은 수준으로 sycophantic 하다.

또한 다음 문제가 나타난다.

* moral YTA/NTA sycophancy 가 증가한다.
* framing sycophancy 가 증가한다.
* 일부 경우 OEQ 에서 Qwen 과 DeepSeek 은 input 이 third-person 임에도 여전히 “you”를 사용해 response 한다.

이는 prompt 만으로 LLM 의 user-facing orientation 을 override 하는 것이 어려울 수 있음을 시사한다.

#### ITI

ITI 에 대해서는 truthfulness 를 위해 tune 되었고 explicit sycophancy 를 mitigate 하는 것으로 알려진 publicly released Llama-8B 와 Llama-70B model 을 평가한다.

* 8B model 은 여전히 매우 socially sycophantic 하다.
* 70B model 은 훨씬 낮은 social sycophancy 를 보인다.

이는 larger open-weight model 에서 ITI 가 social sycophancy 를 해결하는 효과적인 방법이 될 가능성을 시사한다.

그러나 두 model 모두 framing 및 moral sycophancy 는 여전히 높다.

#### DPO

저자는 DPO 를 사용해 각각의 sycophancy dimension 을 줄이도록 Llama-8B model 을 fine-tuning 한다.

* DPO-Validation
* DPO-Indirectness
* DPO-Framing
* 모든 dimension 을 동시에 줄이는 DPO-All

각 dimension 에 대해 OEQ, AITA-YTA, ALP 의 80/20 train-test split 을 사용하여 preference dataset 을 구축한다.

human 이 affirm 하지 않는 prompt, 즉 $s_{\mathrm{human}(p)}^d=0$

인 경우, 저자는 두 model response $m(p)$ 와 $m'(p)$ 중 다음 조건을 만족하는 pair 를 만든다: $s_{m(p)}^d=0, s_{m'(p)}^d=1$

그리고 non-affirming response 를 preferred response 로 설정한다.

반대로 human 이 affirm 하는 경우, $s_{\mathrm{human}(p)}^d=1$ affirming response 를 preferred 로 설정한다.

ALP 에 대해서는 다음을 가정한다: $s_{\mathrm{human}(p)}^d=0$

DPO-All 에 대해서는 dimension 전반의 dataset 을 결합한다.

각 model 은 held-out test data 에서 평가한다.

* OEQ: 860 prompt
* AITA-YTA: 382 prompt
* ALP: 2,049 prompt
* 추가로 전체 AITA-NTA-FLIP dataset 을 평가한다.

결과는 다음과 같다.

* **DPO-Validation**
  * validation dimension 에서 sycophancy 를 상당히 감소시킨다.
  * 다른 dimension 에서도 spillover improvement 를 보인다.
* **DPO-Indirectness**
  * indirectness dimension 에서 sycophancy 를 상당히 감소시킨다.
  * 다른 dimension 에도 spillover improvement 가 나타난다.
* **DPO-Framing**
  * 대부분 ineffective 하다.
  * 이는 framing sycophancy 가 특히 mitigation 하기 어렵다는 점을 다시 보여준다.

저자는 DPO 를 이용해 moral sycophancy 를 해결하기 위한 steering 도 수행하지만, 이 approach 는 response 를 Yes/No 로 제한한다.

DailyDilemmas dataset 을 사용하여 다음 네 가지 value-specific DPO model 을 training 한다.

* honesty
* responsibility
* self-expression
* trust

이러한 model 은 실제로 moral sycophancy 를 감소시킨다.

가장 좋은 결과는 **responsibility** value 로 steer 한 model 의 0.23 이다.

전반적으로 social sycophancy 는 서로 다른 strategy 를 통해 다양한 정도로 mitigation 할 수 있다. 그러나 이러한 strategy 가 user experience 를 어떻게 compromise 하는지는 여전히 명확하지 않다. 효과적인 context-dependent mitigation 을 개발하는 문제는 Sec. 5.2 에서 논의한다.

# 5. Discussion and Future Work

## 5.1 Difference from Prior Work on Explicit Sycophancy

저자의 social sycophancy definition 은 explicit sycophancy 를 포괄하지만, 저자의 연구는 기존 연구와 네 가지 방식에서 차이가 있으며 이를 넘어선다.

첫째, 저자의 result 는 explicit sycophancy 에 대한 기존 result 와 때때로 반대되는 model 간 차이를 보여준다.

* 저자는 GPT-4o 가 높은 sycophancy rate 를 보이고 Gemini 가 가장 낮다는 것을 확인한다.
  * 이는 Fanous et al. 의 finding 과 반대이다.
* Kran et al. 은 Claude 3.5 Sonnet 과 Mistral 8x7B 가 낮은 explicit sycophancy rate 를 보인다고 보고하지만, 저자는 유사한 model 인 Claude 3.7 Sonnet 과 Mistral-7B 가 높은 social sycophancy 를 보인다는 것을 확인한다.
* Llama-8B 는 Llama-70B 보다 factual sycophancy rate 가 2 배 높다고 보고되어 있지만, 두 model 은 유사한 social sycophancy score 를 보인다.

이는 서로 다른 type 의 sycophancy 가 straightforward 하게 correlation 되지 않는다는 것을 보여준다.

둘째, social sycophancy measurement 는 open-ended query 를 포함한다.

이는 기존 sycophancy assessment 에서 주로 사용된 propositional statement 보다 훨씬 넓은 범위의 use case 를 반영하며, model evaluation 을 실제 사용에 더 grounding 된 context 에서 수행해야 한다는 기존 연구의 요구를 확장한다.

셋째, social sycophancy 는 고유한 risk 를 갖는다.

최근 연구는 LLM 이 사용자의 action 을 validation 할 경우 사용자가 자신의 action 에 대한 책임을 지거나 타인에게 사과할 가능성이 낮아지고, 그 결과 social relationship 에 해가 발생할 수 있음을 보여준다.

extended interaction 에서 social sycophancy 는 사용자가 근거 없는 conclusion 에 더욱 고착되도록 만들고 personal growth 를 방해할 가능성이 있다.

또한 distorted belief 에 취약한 사람에게 sycophancy 가 harm 을 초래할 수 있다.

근본적으로 LLM 은 인간 confidant 에게 일반적으로 accountability 를 부여하는 social structure 로부터 분리되어 있다.

예를 들어 relationship conflict 에 대해 advice 를 제공하는 친구는 자신의 advice 가 관련된 모든 party 에 어떤 영향을 미치는지를 고려하고, personal loyalty 와 community 내 다른 사람에게 발생할 potential consequence 사이의 balance 를 고려할 수 있다.

이러한 구조는 excessive validation 을 제한하고, apology 와 같은 restorative action 의 여지를 포함하는 더 balanced 한 counsel 을 장려한다.

이러한 risk 는 사용자가 answer 를 외부 source 를 통해 쉽게 verification 할 수 없기 때문에 특히 insidious 하다.

마지막으로 저자가 평가한 ITI 와 같이 factual sycophancy 를 위한 기존 mitigation 은 social sycophancy 를 효과적으로 해결하지 못한다.

따라서 explicit sycophancy 만을 해결하면 social sycophancy 는 그대로 남을 수 있다. 이러한 이유가 social sycophancy 를 측정하고 이해하려는 저자의 연구를 동기화한다.

## 5.2 Future Work

저자의 finding 은 social sycophancy 를 해결하기 위한 기반을 제공한다. promising research direction 은 다음과 같다.

* **Grounding for framing mitigation**
  * LLM grounding, 즉 적절한 경우 follow-up question 을 통해 추가 context 를 eliciting 하는 방식은 framing sycophancy 를 해결하는 데 도움이 될 수 있다.
  * 예를 들어 “나는 정말 이 job 을 할 수 있다고 생각한다”라는 말에 바로 동의하는 대신, grounded model 은 사용자의 qualification 또는 evidence 를 질문할 수 있다.
  * 기존 연구는 현재 LLM 이 grounding 에서 낮은 performance 를 보인다는 것을 발견했다.
  * 그러나 모든 경우 clarification 이나 evidence 를 요구하면 interaction quality 가 저하될 수 있다.
  * 따라서 언제 그리고 어떻게 이를 수행해야 하는지는 open question 이다.
* **Optimizing for long-term wellbeing**
  * social sycophancy 가 현재의 preference alignment paradigm 에서 발생할 수 있으므로, 저자의 연구는 immediate preference 가 아니라 long-term benefit 을 optimize 해야 한다는 기존 주장을 확장한다.
  * 관련 연구는 multi-turn interaction 을 optimize 하고 downstream consequence 로부터 learning 하는 방법을 보여주었다.
* **Mechanistic interpretability**
  * 저자가 평가한 truthfulness ITI 외에도 mechanistic interpretability 를 사용하여 explicit sycophancy 를 mitigate 하려는 다양한 연구가 존재한다.
  * 이를 social sycophancy 로 확장하는 것은 promising 하다.
  * e.g., latent space 에서 perspective shift 에 intervention 하는 것이 social sycophancy 를 어떻게 감소시키는지 연구할 수 있다.

어떠한 mitigation 을 효과적으로 implementation 하기 위해서도 ideal model behavior 에 대한 더 나은 이해가 필요하다.

* 언제 affirmation 이 적절한가?
* affirmation 의 long-term impact 는 무엇인가?
* LLM 은 인간과 어떻게 달라야 하는가?
* 특히 사용자가 sycophantic AI model 을 자주 선호한다는 점을 고려할 때, user experience 를 compromise 하지 않으면서 supportive 하지만 non-sycophantic 한 model 을 어떻게 구축할 수 있는가?

이러한 open question 은 향후 연구의 중요한 방향이며, careful user-experience design 과 dedicated user study 가 필요할 가능성이 높다.

intervention development 를 지원하는 것 외에도 저자의 benchmark 는 inference-time 에 social sycophancy 를 detection 할 수 있도록 함으로써 practical guardrail 을 제공한다.

저자의 evaluation 은 점점 더 많은 사람이 LLM 을 사용하게 되면서, 사람들이 human norm 과 다르거나 완전히 분리된 방식으로 face 를 보존하는 response 에 노출되고 있음을 보여준다.

이러한 tendency 를 체계적으로 특성화함으로써 **ELEPHANT** 는 사용자와 사회에 long-term benefit 을 제공하는 model 을 개발하기 위한 기반을 제공한다.
