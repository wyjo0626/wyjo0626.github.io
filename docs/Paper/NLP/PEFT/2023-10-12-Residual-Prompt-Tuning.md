---
slug: Residual Prompt Tuning
title: "Residual Prompt Tuning: Improving Prompt Tuning with Residual Reparameterization"
tags: [PEFT, prompt tuning, residual prompt tuning, residual]
---

논문 및 이미지 출처: <https://arxiv.org/pdf/2305.03937v1.pdf>

# Abstract

Prompt Tuning 은 성공적인 PEFT 접근법 중 하나지만 (tuned soft prompts < 0.1% of total parameters) 로 인해 다른 PEFT 보다 성능이 안좋거나 hyper-parameter 에 민감

본 논문에서 저자는 **RESIDUAL PROMPT TUNING** 을 제안

- 간단하고 성능 및 prompt tuning 의 안정성을 크게 향상시키는 efficient method
- shallow network with a residual connection 을 사용하여 soft prompt embeddings 를 reparameterize

Residual Prompt Tuning 은 T5-Large, T5-Base 및 BERT-Base 전역에 SuperGLUE 에서 prompt tuning 능가

Notably,

- T5-Base 에서 prompt tuning +7 points 도달
- 성능 손실 없이 prompt length X10 줄임

저자의 접근법은 learning rate 및 prompt initialization 선택에 있어 robust 하며 few-shot settings 에 효과적

# 1. Introduction

