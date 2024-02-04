---
slug: Adapter
title: "Parameter-Efficient Transfer Learning for NLP"
tags: [PEFT, Adapter]
---

# Abstract

pre-trained model fine-tuning 은 NLP 에서 효과적인 transfer mechanism 이다.

하지만, downstream task 가 많을 경우, 모든 task 마다 entire new model 이 요구되어 비효율적이다.

이 대안으로 저자는 **adapter module** 을 사용한 transfer 제안

- 모델이 간결하고 확장 가능
- task 당 few trainable parameters 만 추가 및 이전 작업 수정 없이 new task 에 추가 가능
- 기존 모델의 parameter 는 고정하며, 고도의 parameter sharing 이 가능

Adapter 효과성 입증을 위해 BERT 를 26 task 를 포함한 GLUE 에서 transfer

- task 당 few parameter 만으로, 거의 SOTA 에 가까운 성능을 얻음
- GLUE 에서 task 당 3.6% parameter 만 추가하여 full fine-tuning 성능 0.4% 안으로 차이남
- 반면 full fine-tuning 은 parameter 100% 훈련

