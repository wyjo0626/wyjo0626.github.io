---
slug: PromptDA
title: "PromptDA : Label-guided Data Augmentation for Prompt-based Few Shot Learners"
tags: [PEFT, prompt tuning, soft prompt, data augmentation]
---

논문 및 이미지 출처 : <https://aclanthology.org/2023.eacl-main.41.pdf>

# Abstract

본 논문은 low-resource Natural Language Understanding (NLU) task 에 초점을 둠,

**Prompt**-based **D**ata **A**ugmentation model (**promptDA**) 를 제안

- frozen Pre-trained Language Model (PLMs) 에 small-sacle _Soft Prompt_ (예; trainable vector) 만 훈련
- 사람이 _unlabeled in-domain data_ 을 수집하는 수고를 덜고 생성된 인공 데이터의 질을 유지
- PromptDA 는 두 가지 측면으로 인공 데이터 생성하며 NLU 모델로 low-quality data 필터링
- 네 가지 벤치마크에서 실험하여, _unlabeled in-domain data_ 를 사용하는 SOTA semi-supervised model 를 포함한 baselinbe 보다 PromptDA 로 생성된 인공 데이터가 성능 능가
- PromptDA 의 인공 데이터는 _unlabeled in-domain data_ 와도 상호보완적이다.

결과적으로 NLU 모델은 인공 데이터와 결합하여 더욱 향상되었다.

# 1. Introduction

Deep neural networks 는 SOTA 성능 달성을 위해선 large-scale high-quality labeled training data 가 필요하지만, 많은 상황에서 labeled data 구성은 어렵다.

본 논문은 sentence classification 및 sequence labelling task 를 포함한 low-resource NLU task 를 연구

이전 연구들은 NLU model 학습을 위해 추가적인 "labeled data" 를 보통 생성한다.

- Wang et al (2021a) : unlabeled in-domain data 로부터 _pseudo labelled training data_ 생성을 위해 _self-training_
- Xu et al (2021) : general corpus 로부터 domain-specific unlabeled data 추출
- Wei and Zou (2019); Dai and Adel (2020) : 랜덤 동의어 교체 같은 automatic heuristic rules 을 사용하여 small training data 확장

하지만 위 과정은 문법적이나 의미적으로 잘못된 데이터가 생성되어 텍스트를 왜곡할 수 있다.

위 딜레마를 해결하기 위해, low-resource setting 에서 data augmentation 을 위해 LM 및 PLMs 에 의지하는 기존 연구도 있다.

labeled data 가 주어지면, 사람의 노력 없이 PLMs 를 fine-tuning 하여 새로운 인공 데이터를 생성할 수 있지만, 저자는 small training data (100 sample 미만)에서 직접 all parameter fine-tuning은 over-fitting 을 일으킬 수 있으며, 단순히 instance 를 기억할 수 있다고 주장. 결과적으로 인공 데이터는 original instance 와 유사할 수 있으며 NLU model 에 new training signal 을 제공할 수 없다.

---

최근 prompt-tuning 이 제안되어, 전체 모델 대신 _soft prompt_ (즉, PLMs input 에 prepended continuous vectors) 에만 back-propagate 한다.

이는 full model tuning 과 competitive 하면서도 parameter 를 상당히 줄일 수 있다.

따라서, prompt tuning 은 low-resource generative fine-tuning 의 over-fitting 이슈를 피하기에 적합하다.

이에 영감을 받아 저자는 **Prompt**-based **D**ata **A**ugmentation model (**PromptDA**) 제안

- pmls 전체를 고정한 채, small labeled training data 에 fine-tuning 할 때 추가적인 soft prompts 만 tuning
  - soft prompt initialization 이 fine-tuning 에 상당히 영향을 주는 것을 관찰. 특히 low-resource situation
- data augmentation task 를 위한 prompt parameter 를 더 잘 초기화하기 위해, PLMs 의 pre-training corpus

데이터 증강 작업을 위해 프롬프트 매개변수를 더 잘 초기화하기 위해, 우리는 PLMs의 사전 훈련 말뭉치에서 프롬프트 매개변수를 직접 사전 훈련하는 작업에 대해 과제 중립적인 동의어 키워드에서 문장 사전 훈련 작업을 제안합니다. 이 작업은 부분 조각 정보(예: 키워드)에서 전체 훈련 샘플을 생성하는 과정을 모방합니다. 이전 연구(Ding et al., 2020; Yang et al., 2020; Anaby-Tavor et al., 2020)와 유사하게, 우리는 출력 태그에 조건을 걸어 완전한 합성 데이터를 생성하도록 PLMs를 세밀 조정할 수 있습니다. 이를 출력 뷰 생성(Output View Generation)이라고 합니다. 생성된 샘플의 다양성을 증가시키기 위해, 우리는 입력 뷰 생성(Input View Generation)이라는 또 다른 세밀 조정 생성 작업을 소개합니다. 이 작업은 샘플에서 추출된 키워드를 입력으로, 샘플을 출력으로 취합니다. 작은 훈련 데이터로부터 훈련된 NLG 모델은 여전히 낮은 품질의 샘플을 생성할 수 있는 가능성이 있으므로, 생성된 샘플을 필터링하기 위해 NLU 일관성 필터링(Anaby-Tavor et al., 2020)을 활용합니다.
우리는 네 가지 벤치마크에서 실험을 실시했습니다: 시퀀스 라벨링 작업 CoNLL03 (Tjong Kim Sang and De Meulder, 2003) 및 Wikiann (Pan et al., 2017), 문장 분류 작업 SST-2 (Socher et al., 2013) 및 RT (Pang and Lee, 2005). 실험 결과, PromDA로부터 합성된 데이터로 훈련된 NLU 모델이 일련의 경쟁력 있는 기준 모델을 일관되게 능가한다는 것을 보여줍니다. 이는 시퀀스 라벨링 작업에서 최첨단 준지도형 NLU 모델 MetaST (Wang et al., 2021a)을 포함합니다. 게다가, PromDA로부터의 합성 데이터가 레이블이 없는 도메인 데이터와도 상호 보완적임을 발견했습니다. 두 가지 데이터를 결합하면 NLU 모델의 성능을 더욱 향상시킬 수 있습니다. 마지막으로, 다양성 분석 및 사례 연구를 실시하여 PromDA로부터의 합성 데이터 품질을 더 확인합니다. 저희의 소스 코드는 https://github.com/GaryYufei/PromDA 에서 확인하실 수 있습니다.