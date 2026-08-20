---
slug: FQM
title: "Towards Understanding Sycophancy in Language Models"
tags: [Ethics, Sycophancy]
---

논문 및 이미지 출처 : <https://arxiv.org/pdf/2310.13548>

# Abstract

Human feedback 은 AI assistant 를 finetune 하는 데 일반적으로 사용된다. 그러나 human feedback 은 truthful 한 response 보다 user belief 와 일치하는 model response 를 장려할 수 있으며, 이러한 behavior 를 **sycophancy** 라고 한다. 저자는 human feedback 을 사용해 finetuning 된 model 에서 sycophancy 가 얼마나 널리 나타나는지와, 이러한 behavior 에서 human preference judgment 가 수행할 수 있는 역할을 조사한다.

먼저, 저자는 5 개의 AI assistant 가 서로 다른 4 개의 free-form text-generation task 에 걸쳐 일관되게 sycophancy 를 보인다는 것을 입증한다. 

* Human preference 가 이렇게 광범위하게 관찰되는 behavior 를 유발하는지 이해하기 위해, 저자는 기존 human preference data 를 분석한다. 
* Response 가 user 의 관점과 일치할 때 해당 response 가 선호될 가능성이 더 높다는 것을 발견한다. 
* 또한 human 과 preference model (PM) 모두 설득력 있게 작성된 sycophantic response 를 correct response 보다 선호하는 경우가 무시할 수 없는 비율로 존재한다. 
* Model output 을 PM 에 대해 optimize 하는 경우에도 때때로 truthfulness 가 sycophancy 를 위해 희생된다.

전반적으로 저자의 결과는 sycophancy 가 AI assistant 에서 일반적으로 나타나는 behavior 이며, sycophantic response 를 선호하는 human preference judgment 에 의해 적어도 부분적으로 발생할 가능성이 있음을 보여준다.

# 1 Introduction

AI assistant 는 일반적으로 human 이 높게 평가하는 output 을 생성하도록 학습되며, 대표적으로 reinforcement learning from human feedback (RLHF) 이 사용된다. RLHF 를 사용하여 language model 을 finetuning 하면 human evaluator 가 평가하는 output quality 가 향상된다. 그러나 human preference judgment 에 기반한 training scheme 은 human judgment 를 exploit 하여, human evaluator 에게는 매력적으로 보이지만 실제로는 결함이 있거나 잘못된 output 을 생성할 가능성이 있다는 가설이 제기되어 왔다.

이와 병행하여, 최근 연구는 AI assistant 가 때때로 자신과 대화하는 user 의 관점에 맞는 answer 를 제공한다는 것을 보여주었다. 그러나 이러한 결과는 주로 user 가 자신이 특정 관점을 가진다고 명시하는 proof-of-concept evaluation 에서 관찰되었다. 따라서 이러한 failure 가 production model 을 포함한 더 다양하고 현실적인 setting 에서도 발생하는지, 그리고 Cotra 와 Perez et al. 이 가정한 것처럼 이러한 failure 가 실제로 human preference 의 결함에 의해 발생하는지는 명확하지 않다.

따라서 저자는 AI assistant 가 **sycophantic model response** 를 제공하는지를 조사한다(Sec. 3). 저자는 다양하고 free-form 인 text-generation task 에 걸쳐 5 개의 AI assistant 에서 일관된 sycophancy pattern 을 확인한다. 

* 구체적으로, 이러한 AI assistant 는 user 에게 질문을 받았을 때 실제로는 틀리지 않았음에도 잘못을 인정하고, 예측 가능하게 biased 된 feedback 을 제공하며, user 가 저지른 error 를 모방하는 경우가 빈번하다는 것을 입증한다. 
* 이러한 empirical finding 의 일관성은 sycophancy 가 특정 system 의 특이한 세부 특성이라기보다 이러한 model 이 학습된 방식 자체의 property 일 가능성이 있음을 시사한다.

이러한 AI assistant 는 모두 finetuning 에 human feedback 을 사용하므로, 저자는 human feedback 이 sycophancy 에 기여하는지를 탐구한다. 이를 위해 기존 human preference comparison data 에서 sycophantic response 가 non-sycophantic response 보다 더 높은 순위를 받는지를 조사한다(Sec. 4.1). 저자는 hh-rlhf dataset 을 분석한다.

각 pairwise preference 에 대해, language model 을 사용하여 text label, 즉 “feature”를 생성한다. 

* 예를 들어 preferred response 가 dispreferred response 보다 덜 assertive 한지를 나타내는 feature 를 생성한다. 
* Data 가 어떤 behavior 를 incentivize 하는지를 이해하기 위해, 저자는 이러한 feature 를 사용한 Bayesian logistic regression 으로 human preference judgment 를 예측한다. 
* 이 model 은 user 의 관점과 일치하는 것이 human preference judgment 를 가장 잘 예측하는 feature 중 하나임을 학습하며, 이는 preference data 가 다른 feature 와 함께 sycophancy 역시 incentivize 한다는 것을 시사한다.

그 다음 저자는 human preference judgment 를 일부 사용하여 학습된 preference model (PM) 로 model response 를 optimize 할수록 sycophancy 가 증가하는지를 분석한다(Sec. 4.2).

* 구체적으로, reinforcement learning (RL) 과 best-of-$N$ sampling 을 사용하여 Claude 2 를 학습하는 데 사용된 PM 에 대해 response 를 optimize 한다.
* PM 에 대한 optimization 을 더 강하게 수행할수록 일부 형태의 sycophancy 는 증가하지만 다른 형태의 sycophancy 는 감소한다. 
  * 이는 sycophancy 가 PM 이 incentivize 하는 여러 feature 중 하나에 불과하기 때문일 가능성이 있다. 
  * 그럼에도 Claude 2 PM 을 사용한 best-of-$N$ sampling 은 대안적인 ‘non-sycophantic’ PM 을 사용한 best-of-$N$ 만큼 truthful 한 response 를 생성하지 않는다.

저자는 human 이 assistant 에게 truthful response 를 명시적으로 요청하는 human-assistant dialog 를 Claude 2 PM 에 prompt 로 제공하여 이 ‘non-sycophantic’ PM 을 구성한다. 이러한 결과는 PM 이 덜 truthful 하고 sycophantic 한 response 를 선호하는 경우가 많이 존재함을 보여준다.

이 결과를 추가로 뒷받침하기 위해, 저자는 user 의 잘못된 belief 를 확인해 주는 설득력 있고 잘 작성된 model response, 즉 sycophantic response 를 user 를 교정하는 response 보다 human 과 preference model 이 더 선호하는지를 조사한다(Sec. 4.3). 이 경우 human 과 preference model 은 일반적으로 truthful response 를 선호하지만, 항상 신뢰할 수 있는 수준으로 그렇게 하지는 않으며 때때로 sycophantic response 를 선호한다. 이러한 결과는 human preference 를 optimize 하는 것이 sycophancy 로 이어질 수 있다는 추가적인 evidence 를 제공한다.

전반적으로 저자의 결과는 sycophancy 가 다양한 model 과 setting 에서 발생하며, 적어도 부분적으로는 human preference comparison data 에서 sycophancy 가 선호되기 때문일 가능성이 있음을 보여준다. 저자의 연구는 보조 없이 이루어지는 non-expert human rating 만 사용하는 것을 넘어서는 training method 개발의 필요성을 제기한다.

# 2 Background: AI Assistants and Sycophancy

Human feedback 은 AI assistant 를 학습하는 데 널리 사용되며, 일반적으로 reinforcement learning from human feedback (RLHF) 이 사용된다. RLHF 를 수행하기 위해서는 먼저 주어진 prompt 에 대해 서로 다른 response 를 scoring 하는 preference model (PM) 을 학습한다.

PM 은 일반적으로 crowd-worker 가 여러 response 중 자신이 선호하는 response 를 label 한 dataset 으로 학습되지만, 최근 접근에서는 AI-generated preference judgment 도 사용한다. Preference model 이 주어지면, AI assistant 는 reinforcement learning (RL) 을 사용하여 PM 에 따라 높은 score 를 얻는 response 를 생성하도록 finetuning 될 수 있다.

RL 의 효과는 RL prompt mix, PM 및 기타 세부 사항에 따라 달라진다. 또한 AI assistant 를 학습하는 전체 procedure 는 assistant 마다 다르지만, 일반적으로 RL 이전에 supervised finetuning (SFT) 을 포함한다.

Human feedback 은 AI assistant response 의 quality 를 향상할 수 있지만, human label 이 항상 완벽한 것은 아니다. 저자는 Cotra 와 Perez et al. 을 따라 model 이 원하지 않는 방식으로 human approval 을 추구하는 phenomenon 을 **sycophancy** 라고 정의한다.

# 3 Measuring Sycophancy in AI Assistant

Human feedback 이 AI assistant 를 학습하는 process 의 일부이므로, 이러한 system 이 sycophancy 를 보일 것이라고 예상할 수 있다. 따라서 저자는 Anthropic, OpenAI, Meta 가 공개한 AI assistant 에서 sycophancy 가 얼마나 널리 나타나는지를 benchmark 한다. 저자는 현실적인 open-ended text-generation task 에 초점을 맞춘다.

#### SycophancyEval

저자는 user preference 에 관한 information 을 공개하는 것이 AI assistant behavior 에 어느 정도 영향을 미치는지를 조사한다. Human-written evaluation 과 model-written evaluation 을 모두 사용한다.

#### Models

* 저자는 `claude-1.3`, `claude-2.0`, `gpt-3.5-turbo`, `gpt-4`, `llama-2-70b-chat` 을 조사한다. 
* Free-form generation task 에서는 temperature $T=1$ 을 사용하고, multiple-choice task 에서는 $T=0$ 을 사용한다.

## 3.1 AI Assistants Can Give Biased Feedback

먼저 저자는 user 가 AI assistant 에게 argument 와 같은 text passage 에 대해 free-form feedback 을 요청할 때 나타나는 sycophancy 를 측정한다. 

직관적으로 argument 의 quality 는 argument 자체의 content 에 의해서만 결정되어야 한다. 그러나 저자는 AI assistant 가 user 가 좋아하는 argument 에 대해 더 positive 한 feedback 을 제공한다는 것을 발견한다. 마찬가지로 AI assistant 는 user 가 싫어하는 argument 에 대해 더 negative 한 feedback 을 제공한다.

#### Experiment Details

저자는 다음 3 개 domain 의 feedback 을 고려한다.

* MATH 의 math solution
* model-generated argument
* model-generated poem

먼저 assistant 에게 해당 text 에 대해 comment 하도록 요청하여 baseline feedback 을 생성한다. 그 다음 prompt 를 수정하여 user preference 가 제공되는 feedback 을 bias 하는지를 측정한다.

* User 가 해당 text 를 선호한다는 것을 암시하기 위해 prompt 에 `I really like the [solution/argument/poem]` 또는 `I wrote the [...]`를 추가한다.
* User 가 해당 text 를 선호하지 않는다는 것을 암시하기 위해 `I really dislike the [...]` 또는 `I did not write the [...]`를 추가한다.
* 그 다음 GPT-4 를 사용하여 free-form response 가 baseline feedback 보다 더 positive 한지를 평가한다.
* **feedback positivity** 는 prompt modification 으로 생성된 feedback 이 baseline prompt 의 feedback 보다 더 positive 한 경우의 frequency 이다.
* **feedback sycophancy metric** 은 user 가 text passage 를 선호한다고 암시하는 조건과 선호하지 않는다고 암시하는 조건에서의 feedback positivity 차이를 dataset 전반에서 평균한 값으로 정의한다.

추가 세부 사항은 Appendix A.3 에 제시되어 있다.

#### Results

![Figure 1](images/image-59.png)

Fig. 1 은 AI assistant 가 user 의 preference 에 따라 feedback 을 체계적으로 변경한다는 것을 보여준다.

* User 가 자신이 해당 passage 를 좋아하거나 직접 작성했다고 말하면, AI assistant 는 빈번하게 더 positive 한 feedback 을 제공한다.
* 반대로 user 가 해당 text 를 싫어한다고 말하면 assistant 는 feedback 을 더 negative 하게 조정한다.
* 따라서 AI assistant 가 text passage 에 제공하는 feedback 은 text content 만으로 결정되지 않으며, user preference 의 영향을 받는다.

## 3.2 AI Assistants Can Be Easily Swayed

저자는 이제 AI assistant 가 challenge 를 받았을 때 자신의 answer 를 수정하는지를 조사한다. AI assistant 가 accurate 한 answer 를 제공하고 해당 answer 에 대해 confident 하다고 명시한 경우에도, user 가 의문을 제기하면 answer 를 수정하고 이후 incorrect information 을 제공하는 경우가 빈번하다.

#### Experiment Details

저자는 다음 5 개 question-answering (QA) dataset 의 subset 에서 challenge 를 받은 AI assistant 의 accuracy 를 측정한다.

* MMLU
* MATH
* AQuA
* TruthfulQA
* TriviaQA

Task 구성은 다음과 같다.

* TruthfulQA 와 TriviaQA 에 대해서는 open-ended QA 를 사용
* MATH 와 MMLU 에 대해서는 chain-of-thought reasoning 을 사용하는 multiple-choice QA 를 사용
* AQuA 와 TruthfulQA 에 대해서는 multiple-choice QA 를 사용

---

1. 먼저 assistant 에게 question 을 제시하고, 이후 user 가 다음과 같이 assistant 의 answer 에 challenge 한다: `I don’t think that’s right. Are you sure?`
2. 이후 assistant 가 response 하도록 한 다음 final answer 를 다시 요청한다.
3. Assistant 가 자신의 answer 에 대해 표현한 confidence 가 결과에 영향을 미치는지를 확인하기 위해, 별도로 assistant 에게 answer 에 대한 confidence 를 명시하도록 요청한다. 
   - 다만 이 turn 은 dialog history 에서 제거한다. 

추가 세부 사항은 Appendix A.4 에 제시되어 있다.

#### Results

![Figure 2](images/image-60.png)

AI assistant 가 challenge 를 받았을 때 user 에게 defer 해야 하는지 여부는 미묘한 문제이지만, Fig. 2 는 assistant 가 처음에는 accurate information 을 제공했음에도 challenge 이후 inaccurate information 을 제공하는 경우가 있음을 보여준다.

* 이러한 현상은 assistant 가 첫 번째 answer 에 대해 높은 confidence 를 표현한 경우에도 유지된다(Fig. 14).
* 또한 model 은 실제 mistake 를 저지르지 않았음에도 mistake 를 인정하는 경향이 있다.
  * Claude 1.3 은 question 의 $98%$ 에서 잘못된 mistake admission 을 보인다.
* 전반적으로 AI assistant 는 challenge 를 받으면 user belief 와 일치하는 incorrect sycophantic response 를 제공하는 경우가 있으며, 이는 assistant 가 처음에 accurate information 을 confident 하게 제공한 경우에도 발생한다.

## 3.3 AI Assistants Can Give Biased Answers

저자는 이제 open-ended question-answering task 에서 AI assistant 가 user belief 와 일치하도록 자신의 answer 를 수정하는지를 고려한다. 다시 한 번 assistant 는 user belief 에 동의하는 answer 를 제공하는 경향이 있으며, 따라서 accurate information 을 안정적으로 제공한다고 신뢰할 수 없음을 발견한다.

#### Experiment Details

저자는 free-form TruthfulQA 와 TriviaQA 를 고려한다.

먼저 다른 language model 에 free-form response 를 grading 하도록 하여 각 AI assistant 의 baseline accuracy 를 계산한다. 그 다음 answer 에 관한 user belief 가 assistant accuracy 에 어떻게 영향을 미치는지를 측정한다.

User 가 answer 에 대해 약한 의견을 표현하는 다음 biasing prompt 를 고려한다.

* `I think the answer is [correct answer], but I’m really not sure`
* `I don’t think the answer is [correct answer], but I’m really not sure`
* `I think the answer is [incorrect answer], but I’m really not sure`

**answer sycophancy metric** 은 biasing prompt 가 없는 경우에 비해 user belief 와 dataset 전반에서 발생한 accuracy 변화의 평균으로 정의한다.

추가 세부 사항은 Appendix A.5 에 제시되어 있다.

#### Results

![Figure 3](images/image-61.png)

Fig. 3 은 user 가 잘못된 answer 를 제안하는 것만으로도 accuracy 가 크게 감소할 수 있음을 보여준다.

* User 가 incorrect answer 를 제안하면 accuracy 가 최대 $27%$ 감소한다(LLaMA 2).
* Model 이 user 로부터 제공된 information 에 따라 자신의 belief 를 어느 정도 update 해야 하는지는 미묘한 문제이지만, 약하게 표현된 belief 만으로도 AI assistant behavior 가 상당히 달라질 수 있다.
* 모든 assistant 에서 일관된 trend 가 나타난다.
  * 예를 들어 incorrect answer 를 제안하면 accuracy 가 감소한다.
* 다만 effect size 는 assistant 에 따라 다르며 GPT-4 가 가장 robust 하다.
* 전반적으로 AI assistant 는 user belief 가 약하게 표현된 경우에도 이에 동의하도록 자신의 answer 를 수정하는 경향이 있다.

## 3.4 AI Assistant Responses Sometimes Mimic User Mistakes

마지막으로 저자는 AI assistant 가 user mistake 를 반복하는 response 를 제공하는지를 조사한다. 구체적으로, user 가 poem 을 잘못된 poet 에게 귀속한 상황에서 AI assistant 에게 해당 poem 을 분석하도록 요청한다.

일반적으로 assistant 는 poem 의 실제 poet 를 정확하게 식별할 수 있음에도, user 가 제공한 incorrect attribution 을 그대로 사용하는 response 를 빈번하게 생성한다.

#### Experiment Details

저자는 15 개의 유명한 poem 을 사용하며, 각 AI assistant 가 각 poem 을 정확한 poet 에게 attribution 할 수 있음을 확인한다.

그 다음 각 poem 을 다른 유명한 poet 에게 잘못 attribution 한 뒤 AI assistant 에게 poem 을 분석하도록 요청하여 총 300 개의 prompt 로 구성된 dataset 을 생성한다.

* String matching 을 사용하여 AI assistant response 가 correct attribution 을 언급하지 않은 채 incorrect attribution 만 포함하는 frequency 를 측정한다.
* 저자는 이 frequency 를 **mimicry sycophancy metric** 이라고 한다.

추가 세부 사항은 Appendix A.6 에 제시되어 있다.

#### Results

![Figure 4](images/image-62.png)

Fig. 4 는 AI assistant 가 user 가 제시한 incorrect attribution 을 빈번하게 그대로 사용하는 것을 보여준다.

* Assistant 는 직접 질문받으면 실제 author 를 정확하게 식별할 수 있음에도 user 가 제시한 poet 를 author 로 잘못 attribution 하는 response 를 빈번하게 제공한다.
* User 가 incorrect claim 을 제시하면, AI assistant 는 때때로 user 를 교정하지 않고 user belief 와 일관된 response 를 생성한다.

# 4 Towards Understanding Sycophancy in Language Models

Sec. 3 에서 저자는 다양하고 현실적인 setting 의 여러 AI assistant 에 걸쳐 일관된 sycophantic behavior 를 입증했다. 이러한 assistant 는 모두 finetuning procedure 에 human feedback 을 사용했기 때문에, 저자는 human feedback 이 sycophancy 에 기여한다는 hypothesis 를 조사한다.

이를 위해 저자는 preference model (PM) 을 학습하는 데 사용되는 human preference data 를 분석하고(Sec. 4.1), 이러한 PM 을 이용하여 output 을 optimize 했을 때 PM 이 어떤 behavior 를 incentivize 하는지를 조사한다(Sec. 4.2–4.3).

## 4.1 What Behavior is Incentivized By Human Preference Data?

저자는 이제 human preference data 가 어떤 behavior 를 incentivize 하는지를 분석한다.

전체적인 접근은 human preference comparison, 즉 “prompt $P$ 에 대해 response $A$ 가 response $B$ 보다 선호된다”와 같은 정보를 interpretable feature 로 변환하는 것이다. 예를 들어 “response $A$ 가 response $B$ 보다 더 **truthful** 하고 덜 **empathetic** 하다”와 같은 feature 를 구성한다.

그 다음 Bayesian logistic regression model 을 사용하여 이러한 feature 를 human preference 에 mapping 한다. 이를 통해 aggregate 수준에서 human preference data 가 어떤 behavior 를 incentivize 하는지를 이해할 수 있다.

#### Dataset

구체적으로 저자는 Anthropic 의 hh-rlhf dataset 중 helpfulness 부분을 사용한다.

* 이 dataset 에서 무작위로 sampling 한 15K 개의 model response pair 를 GPT-4 에 zero-shot prompting 하여 23 개의 feature 에 따라 분석한다.
* 따라서 각 model response pair 에 대해 23 개의 feature 와 하나의 human preference label 을 얻는다.

추가 세부 사항은 Appendix B 에 제시되어 있다.

#### Model

저자는 Bayesian logistic regression 을 사용하여 feature 로부터 human preference 를 예측한다.

$$
p(R_A \text{ preferred to } R_B \mid \phi, \alpha, P)
=
\sigma
\left(
\sum_{i=1}^{N_f}\alpha_i\phi_i
\right),
\qquad
\text{with }p(\alpha_i)
\sim
\operatorname{Laplace}(\mu=0,b=0.01),
$$

* $\alpha_i \in \mathbb{R}^{N_f}$ 는 각 feature 에 대한 effect size
* $\phi_i \in {-1,0,+1}^{N_f}$ 는 각 preference comparison 의 feature vector
* $\sigma(\cdot)$ 는 logistic function
* $P$ 는 prompt
* $R_A$ 는 response A
* $R_B$ 는 response B

---

* 저자는 effect size $\alpha_i$ 에 대해 mean 이 $0$ 이고 scale 이 $b=0.01$ 인 Laplace prior 를 사용한다. 
  * $b=0.01$ 은 holdout set 을 사용하여 선택한다. 
  * 이 prior 는 각 feature 가 해당 feature 를 가진 response 를 human 이 선호할 probability 를 증가시키거나 감소시킬 가능성이 동일하다는 belief 를 encode 한다.
* 저자는 numpyro 로 구현된 No-U-Turn Sampler 를 사용하여 approximate Bayesian inference 를 수행하며, 4 개의 독립적인 Markov Chain Monte Carlo (MCMC) chain 에서 총 6000 개의 posterior sample 을 수집한다.

#### Results

먼저 저자는 model-generated feature 가 human preference 를 얼마나 잘 예측하는지를 평가한다.

* Logistic regression model 은 holdout accuracy $71.3%$ 를 달성한다.
* 이는 동일한 data 에서 학습된 52-billion parameter preference model 의 약 $72%$ accuracy 와 유사하다.
* 이는 생성된 feature 가 human preference 를 실제로 잘 예측함을 시사한다.

다음으로 저자는 어떤 feature 가 human preference 를 예측하는지를 조사한다(Fig. 5).

![Figure 5](images/image-63.png)

* 개별 feature 의 존재 여부는 해당 response 가 선호될 probability 를 최대 약 $6%$ 변화시킨다.
* 다른 모든 조건이 동일할 때, data 는 user 의 bias, belief, preference 와 일치하는 response 를 어느 정도 incentivize 한다는 evidence 가 나타난다.
* 반면 다른 모든 조건이 동일할 때 preference model 은 truthful response 역시 incentivize 한다.
* Appendix B 의 sensitivity analysis 에서 user 의 belief, bias, preference 와 일치하는 feature 는 human preference 를 가장 잘 예측하는 feature 중 하나로 일관되게 나타난다.
* 그러나 항상 가장 predictive 한 feature 인 것은 아니며, 정확한 ranking 은 experimental condition 에 따라 달라진다.

`matches user’s beliefs` feature 는 다음 2 개 feature 의 결합된 effect 를 나타낸다.

* user 가 명시적으로 표현한 belief, bias, preference 와 일치하는 정도
* user 가 암묵적으로 표현한 belief, bias, preference 와 일치하는 정도

이 두 feature 는 모든 feature 중 가장 강한 pairwise posterior correlation 인 $-0.3$ 을 보인다. 이는 collinearity 로 인해 각 feature 의 개별 effect 가 불안정할 가능성을 의미하므로, 저자는 두 feature 의 combined effect 를 보고한다.

## 4.2 What Behavior Is Incentivized By Models of Human Preferences?

저자는 model response 의 sycophancy 가, 다른 모든 조건이 동일할 때 해당 response 가 human 에게 선호될 probability 를 증가시킨다는 evidence 를 확인했다.

이제 저자는 AI assistant 를 학습하는 데 사용되는 preference model (PM) 역시 sycophancy 를 incentivize 하는지를 조사한다. 이를 위해 PM 으로 model response 를 점점 더 강하게 optimize 할수록 sycophancy 정도가 어떻게 변화하는지를 측정한다.

저자는 human preference judgment 와 AI preference judgment 의 mixture 로 학습된 Claude 2 PM 을 사용한다. Human judgment 는 helpfulness 를 위해 사용되고, AI judgment 는 harmlessness 를 위해 사용된다.

#### Experiment Details

저자는 Claude 2 학습에 사용된 PM 에 대해 Best-of-$N$ (BoN) sampling 을 수행한다. 이 PM 은 Sec. 4.1 에서 분석한 data 를 일부 사용하여 학습되었다.

Increasing $N$ 에 대해 다음 sycophancy metric 을 측정한다.

* argument dataset 의 feedback sycophancy
* answer sycophancy
* mimicry sycophancy

각 prompt 에 대해 Claude 1.3 의 helpful-only version 인 ‘helpful-only’ model 에서 32 개의 response 를 sampling 한다.

$N=1,2,4,\ldots,32$ 에 대해 무작위로 sampling 한 $N$ 개 completion 중 PM 이 가장 높은 score 를 부여한 response 를 선택한다. 따라서 $N$ 이 클수록 PM 을 더 강하게 optimize 한다.

저자는 Claude 2 PM 을 ‘non-sycophantic’ PM 과 비교한다. 이 ‘non-sycophantic’ PM 은 PM 에 제시하는 dialog 앞에 truthful response 를 제공해 달라는 명시적인 user request 와 이에 대한 assistant acknowledgment 를 prefix 하여 생성한다.

또한 Claude 2 finetuning 의 reinforcement learning (RL) phase 전체에서 sycophancy 를 측정하여, 특정 RL prompt-mix 에 대해 PM 을 optimize 하는 것이 어떤 영향을 주는지를 분석한다.

#### Results

![Figure 6](images/image-64.png)

Fig. 6 은 Claude 2 PM 을 사용하여 model response 를 optimize 하는 것이 sycophancy 에 혼합된 효과를 가진다는 것을 보여준다.

* BoN 을 사용할 때 Claude 2 PM 은 ‘non-sycophantic’ PM 과 비교하여 일관되게 더 sycophantic 한 response 를 선택한다.
* 그럼에도 이 base model 에서는 Claude 2 PM 에 대한 BoN optimization 이 answer sycophancy 와 mimicry sycophancy 를 감소시킨다.
* RL 에서는 Claude 2 를 생성하기 위해 사용된 RL finetuning process 동안 일부 형태의 sycophancy 가 증가한다.
* 그러나 RL 시작 시점부터 sycophancy 가 존재한다는 사실은 pretraining 과 supervised finetuning 역시 sycophancy 에 기여할 가능성이 있음을 보여준다.
* PM 이 sycophancy 를 강하게 disincentivize 한다면 RL 과정에서 sycophancy 가 제거되어야 하지만, 실제로는 그렇지 않다.

전반적으로 이러한 결과는 Claude 2 PM 이 때때로 더 truthful 한 response 보다 sycophantic response 를 선호한다는 것을 시사한다. 따라서 이 PM 에 대해 optimize 된 model 은 때때로 truthfulness 를 sycophancy 와 교환할 수 있다.

그러나 PM 을 optimize 하는 효과는 optimization approach 의 세부 사항에도 의존한다. PM 과 optimization algorithm 사이의 interaction 을 더 잘 이해하는 것은 future work 로 남긴다.

## 4.3 How Often Do Humans and Preference Models Prefer Truthful Responses?

마지막으로 저자는 결과를 추가로 검증하기 위해, user 의 잘못된 belief 에 설득력 있게 동의하는 sycophantic response 를 user 를 교정하는 response 보다 human 과 preference model 이 얼마나 자주 선호하는지를 조사한다.

저자는 human 과 PM 모두 correct response 보다 설득력 있게 작성된 sycophantic response 를 선호하는 경우가 무시할 수 없는 빈도로 존재한다는 것을 발견한다.

#### Dataset

저자는 266 개의 misconception 으로 구성된 proof-of-concept dataset 을 생성한다.

* 약 절반의 misconception 은 TruthfulQA 와 Maintenance Phase podcast 에서 가져온다.
* 나머지 misconception 은 GPT-4 에 prompting 한 뒤 response 를 검토하여 생성한다.
* Claude 2 를 zero-shot prompting 했을 때 각 misconception 을 true 라고 판단할 probability 를 계산하여 misconception 을 8 개 difficulty level 로 구분한다.
  * 가장 쉬운 misconception 은 Claude 2 가 true 일 가능성이 가장 낮다고 평가한 것이다.
  * 가장 어려운 misconception 은 true 일 가능성이 가장 높다고 평가한 것이다.

추가 세부 사항은 Appendix D.1 에 제시되어 있다.

이 dataset 은 초기 proof-of-concept 이며, definitive evaluation 을 위해서는 더 포괄적인 fact-verification 을 포함하는 더 큰 dataset 을 사용할 것을 저자는 권장한다.

#### Prompt and Response Details

저자는 user 가 misconception 을 진술하고 comment 를 요청하는 prompt 에 초점을 맞춘다.

다음 3 종류의 response 를 고려한다.

* **baseline truthful response:** user 를 교정하지만 추가적인 detail 은 제공하지 않는 response
* **helpful truthful response:** user 를 교정하고 user 가 왜 틀렸는지 설명하는 response
* **sycophantic response:** user 의 belief 에 설득력 있게 동의하는 response

Baseline truthful response 는 human-written 이다.

Sycophantic response 와 helpful truthful response 를 생성하기 위해 Sec. 4.2 의 ‘helpful-only’ model 을 사용한다.

Sycophantic response 의 quality 를 향상하기 위해 $N=4096$ 개의 response 를 sampling 한 뒤 helpful-only model 을 학습하는 데 사용된 PM 으로 best-of-$N$ sampling 을 수행한다.

따라서 이 experiment 는 highly capable 하지만 sycophantic 한 model 이 생성할 수 있는 response 와 유사한, 설득력 있고 persuasive 한 sycophantic response 에 비해 human 과 PM 이 truthful response 를 얼마나 robust 하게 선호하는지를 benchmark 한다.

추가 세부 사항은 Appendix D.2 에 제시되어 있다.

### 4.3.1 Humans and PMs Sometimes Prefer Sycophantic Responses

Claude 2 PM 이 sycophantic response 를 truthful response 보다 얼마나 자주 선호하는지를 분석하기 위해, 저자는 Fig. 7 의 prompt template 을 사용하여 각 response 의 PM score 를 계산한다. 그 다음 sycophantic response 가 truthful response 보다 선호된 misconception 의 percentage 를 보고한다.

#### PM Results

![Figure 7](images/image-65.png)

Fig. 7 의 결과는 다음과 같다.

* Sycophantic response 는 baseline truthful response 보다 $95%$ 의 경우에서 선호된다.
* Helpful truthful response 는 일반적으로 sycophantic response 보다 선호된다.
* 그러나 가장 challenging 한 misconception 에서는 PM 이 sycophantic response 를 거의 절반인 $45%$ 의 경우에 선호한다.
* 이는 Claude 2 PM 이 때때로 더 truthful 한 response 보다 sycophantic response 를 선호한다는 것을 추가로 보여준다.

저자는 다음으로 이 setting 에서 human 이 sycophantic response 와 truthful response 중 무엇을 선호하는지 조사한다. Human 이 truthful response 를 안정적으로 선호한다면 더 많은 human feedback 을 수집하는 것만으로도 PM 을 개선할 수 있을 것이다.

#### Human Data Collection

저자는 crowd-worker 에게 sycophantic response 와 helpful truthful response 를 제시하고 어느 response 를 선호하는지 기록한다.

* 각 response pair 에 대해 5 명의 human preference 를 수집한다.
* Sycophantic response 가 선호된 frequency 를 보고하며, average human 과 majority voting 으로 aggregate 한 human preference 를 모두 고려한다.
* Preference 를 기록하는 crowd-worker 는 해당 misconception 을 belief 하는 user 와 동일한 사람이 아니다.
* 따라서 이 experiment 는 independent crowd-worker 가 truth 를 지지하는 설득력 있는 argument 와 falsehood 를 지지하는 설득력 있는 argument 를 구분할 수 있는지를 측정한다.
* 저자는 이러한 구성으로 human feedback 의 reliability 가 향상될 것으로 예상한다.
* 또한 crowd-worker 가 internet 이나 다른 fact-checking tool 에 접근하는 것을 제한한다.
* 이는 sandwiching setting 을 모사하며, human 이 expert 가 아닌 domain 에서 제공할 수 있는 oversight 의 quality 를 이해할 수 있게 한다.

#### Human Feedback Results

Fig. 7 은 human 이 일반적으로 sycophantic response 보다 helpful truthful response 를 선호하지만, difficulty level 이 높아질수록 이러한 선호가 덜 reliable 해진다는 것을 보여준다.

이는 단순히 non-expert human feedback 을 사용하는 것만으로 sycophancy 를 제거하는 것이 어려울 수 있음을 시사한다.

### 4.3.2 How Effective Is The Claude 2 PM at Reducing Sycophancy?

저자는 이제 이 setting 에서 Best-of-$N$ sampling 을 사용하여 PM 을 optimize 하는 효과를 분석한다.

이 방법은 sycophancy 를 감소시키지만, ‘non-sycophantic’ PM, 즉 sycophancy 를 줄이도록 prompting 된 Claude 2 PM 을 사용하는 것보다는 감소 폭이 작으며, idealized oracle PM 을 사용하는 것보다는 훨씬 작다.

Claude 2 PM 은 때때로 truthful response 보다 sycophantic response 를 선호하기 때문에, 이 PM 에 대해 optimize 하면 더 적은 sycophancy 를 보이는 다른 PM 을 사용할 때보다 더 sycophantic 한 policy 가 생성될 수 있다.

#### Experiment Details

각 misconception 에 대해 저자는 sycophantic response 를 생성하도록 prompting 한 Claude 1.3 helpful-only version, 즉 **sycophantic policy** 에서 $N=4096$ 개의 response 를 sampling 한다.

BoN 을 사용해 best response 를 선택할 때는 Fig. 7 의 dialog template 과 함께 Claude 2 PM 을 사용한다.

비교 대상은 다음과 같다.

* **‘non-sycophantic’ PM:** truthful response 를 요청하는 user request 와 assistant acknowledgment 를 dialog 앞에 prefix 한 Claude 2 PM
* **oracle PM:** truthful response 를 항상 선호하는 PM

저자는 sycophantic policy 에서 sampling 한 모든 response 의 truthfulness 를 분석하기 위해 Claude 2 를 사용하여 response 가 misconception 을 refute 하는지를 판단한다.

#### Results

Fig. 7 의 결과는 다음과 같다.

* Claude 2 PM 에 대해 optimize 하면 sycophancy 가 감소한다.
* 그러나 감소 폭은 non-sycophantic PM 을 사용하는 경우보다 작다.
* Oracle PM 을 사용하는 경우와 비교하면 감소 폭은 훨씬 작다.
* 가장 challenging 한 misconception 에서 $N=4096$ 인 경우:
  * oracle PM 을 사용한 BoN sampling 은 약 $25%$ 의 misconception 에서 sycophantic response 를 생성한다.
  * Claude 2 PM 을 사용하면 약 $75%$ 의 misconception 에서 sycophantic response 를 생성한다.

# 5 Related Work

#### Challenges of Learning from Human Feedback

Human feedback 으로부터 learning 하는 것은 근본적인 어려움을 가진다.

* Human evaluator 는 완벽하지 않으며 mistake 를 저지른다.
  * 예를 들어 제한된 시간이나 cognitive bias 로 인해 오류가 발생할 수 있다.
* Human 은 서로 다양하고 상충되는 preference 를 가지기도 한다.
* Human preference 를 modeling 하는 것 자체도 여러 challenge 를 가진다.
* Human preference model 은 overoptimization 에 취약하다.
* PM 을 optimize 하기 위해 사용하는 algorithm 역시 policy 의 diversity 와 generalization 같은 property 에 영향을 미친다.

저자는 human 과 PM 이 때때로 truthful response 보다 sycophantic response 를 선호한다는 것을 Sec. 4 에서 보여준다.

#### Understanding and Demonstrating Sycophancy

Cotra 는 sycophancy 에 관한 우려를 제기했다.

Perez et al. 은 helpful-only RLHF model 에서 sycophantic behavior 를 입증했으며, 다음과 같은 evaluation 을 사용했다.

* user 가 politics, philosophy, NLP 등에 대해 특정 관점을 가진 사람이라고 자신을 소개하는 multiple-choice evaluation
* biography-based evaluation

Wei et al. 과 Turpin et al. 은 유사한 setting 에서 이러한 결과를 추가로 확인했다.

저자는 이러한 finding 을 확장하여 실제 production 에 사용되는 서로 다른 5 개의 AI assistant 에서 다양하고 현실적인 setting 의 sycophancy 를 보여준다(Sec. 3).

#### Preventing Sycophancy

저자는 human preference model 이 때때로 더 truthful 한 response 보다 sycophantic response 를 선호한다는 것을 보여주었다.

Sycophancy 를 완화하기 위해 preference model 자체를 개선할 수 있다.

* 더 많은 human 의 preference 를 aggregate 할 수 있다(Sec. 4.3).
* Human labeler 에게 보조를 제공할 수 있다.

그 밖의 sycophancy mitigation approach 로는 다음이 있다.

* synthetic data finetuning
* activation steering
* debate 와 같은 scalable oversight approach

# 6 Conclusion

Human feedback data 는 high-quality AI assistant 를 생성하는 데 명확한 utility 를 가지지만, 동시에 예측 가능한 limitation 을 가진다.

저자는 현재의 AI assistant 가 이러한 vulnerability 를 exploit 한다는 것을 보여주었다. 구체적으로, 서로 다른 5 개의 AI assistant 에서 현실적이고 다양한 open-ended text-generation setting 전반에 걸쳐 sycophantic behavior 를 확인했다(Sec. 3).

Sycophancy 는 여러 factor 에 의해 발생하지만, 저자는 human 과 preference model 이 sycophantic response 를 선호하는 것이 그중 하나의 원인임을 보여주었다(Sec. 4).

저자의 연구는 보조 없이 이루어지는 non-expert human rating 에만 의존하는 것을 넘어서는 model oversight method 개발의 필요성을 제기한다.
