---
slug: XPrompt
title: "XPrompt: Exploring the Extreme of Prompt Tuning"
tags: [PEFT, prompt tuning, LTH, lottery ticket hypothesis]
---

논문 및 이미지 출처 : <https://aclanthology.org/2022.emnlp-main.758.pdf>

# Abstract

Prompt tuning 은 frozen Pre-trained Language Models (PLMs) 를 conditioning 하기 위해 soft prompts 를 학습한다.

모델 규모가 커짐에 따라 prompt tuning 은 점차 fine-tuning 수준에 도달하지만, moderate 및 small scales (< 11B) 에선 여전히 성능 차이가 발생한다.

본 논문에서 저자는 trained prompt tokens 는 downstream task 에 negative 영향을 줄 수 있으며 성능 저하를 일으킬 것이라는 것을 경험적으로 보여준다.

- gap 을 줄이기 위해, 저자는 lottery tickets hypothesis 하에, **Prompt** tuning model with an e**X**tremely small scale (**XPrompt**) 를 제안
- 구체적으로, XPrompt 는 hierarchical structured pruning 을 통해 다양한 granularity levels 에서 negative prompt tokens 를 제거하여 더욱 parameter-efficient prompt 를 생성하여 competitive 성능 달성
- SuperGLUE task 에서 포괄적 실험으로, smaller model scales 에서 성능 gap 을 줄여줌

# 1. Introduction

PLMs 는 _pretrain-then-finetune_ 을 통해 널리 사용되어 큰 성공을 거두지만, memory 공간에 gradient 및 optimizer 저장을 위해 trainable parameter 가 크게 차지하고 있어 fine-tuning 이 parameter-inefficient 하다.

---

최근 Prompt-Tuning (Lester et al. 2021) 으로 input 에 _soft prompt_ 를 앞에 붙이고 훈련 중 prompt parameter 만 업데이트하여 위 이슈를 해결하는 것을 제안하였다.

- fine-tuning 대체제로, soft prompt scale 은 수만배 적음
- 더 간단하고 다른 peft (Adapter) 보다 유연하여 transformer layers 에 직관적으로 수정 가능
- 적은 tunable parameter 로 fine-tuning 성능과 competitive
