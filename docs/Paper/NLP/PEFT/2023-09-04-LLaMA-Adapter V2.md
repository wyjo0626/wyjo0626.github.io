---
slug: LLaMA-Adapter V2
title: "LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model"
tags: [PEFT, Adapter V2, LLaMA]
---

논문 및 이미지 출처 : <https://arxiv.org/pdf/2304.15010.pdf>

# Abstract

최근 LLaMA-Adapter 가 LLMs 와 visual input 을 다루는 잠재력을 보여주지만 여전히 open-ended visual instruction 은 잘 처리하지 못하며 GPT-4 에 뒤쳐지고 있다.

본 논문에서는 parameter-efficient visual instruction model 인 **LLaMA-Adapter V2** 제안

- LLaMA-Adapter 를 더 많은 learnable parameter (norm, bias 및 scale 등) 을 활용하여 보강
  - Adapter 외의 전체 LLaMA model 의 instruction-following 능력을 분산
- visual token 을 early LLM layer 에만 주입하는 early fusion 전략을 제안
  - visual knowledge incorporation 개선
- image-text pair 및 instruction-following data 의 joint training paradigm 을 도입
  - learnable parameter 의 분리된 그룹을 최적화
  - image-text alignment 및 instruction-following 두 작업 간의 간섭을 효과적으로 완화
  - 소규모 image-text 및 instruction-following dataset 만으로 강력한 multi-modal reasoning 달성

inference 단계

- LLaMA-Adapter 에 추가적인 expert models (e.g. captioning/OCR) 을 통합
- training cost 발생하지 않고 image understanding 능력 더욱 향상

기존 LLaMA-Adapter 와 비교하여 LLaMA-Adapter V2 는 LLaMA 에 14M parameter 추가만으로 open-ended multi-modal instruction 수행ㅅ 가능

그리고 이 설계는 language-only instruction-following 을 더욱 강화시키며 채팅 상호작용에도 뛰어난 성능을 보임

# 1. Introduction

최근 LLM 을 instruction-following model 로 변환하는 연구가 진행

Stanford Alpaca 는 InstructGPT model 로 생성된 instruction examples 를 사용하여 LLaMA 를 instruction-following model 로 fine-tuning 한다.

LLaMA-Adapter 는 가벼운 Adapter 와 zero-initialized attention 의 도입으로 frozen LLaMA 에게 parameter-efficient fine-tuning 으로 multi-modal knowledge 를 주입한다.

가장 최근은 MiniGPT-4 및 LLaVA 같은 연구로, language-only instruction model 을 multi-modal 로 확장하여 visual reasoning 능력을 부여하는 새로운 연구 파동을 일으켰다.

본 논문은 parameter-efficient visual instruction model 설계를 목표로 한다.

- LLaMA-Adapter 기반의 새로운 method 인 **_LLaMA-Adapter V2_** 개발
  - LLaMA-Adapter 는 instruction-following model
  - visual feature 를 adaptation prompts 로 주입하여 visual instruction model 로 변환
  - multi-modal instruction tuning data 의 부족으로 전통적인 vision-language model 로 제한됨
  - 예로, COCO Caption 에서 훈련된 LLaMA-Adapter 는 "“Generate caption for this image" 같은 prompt 에 대해 짧은 caption 만 생성 가능
  - 복잡한 visual reasoning 및 QA task 같은 open-ended multi-modal instruction 에는 adaptation 이 불가능
- frozen LLaMA-Adapter 를 사용하여 image-text pairs 에서 visual projection layer 를 최적화하여 vision-language alignment 를 보장하도록 개선
  - visual feature 가 adaptation prompts 에 두드러지며 instruction-following 능력이 빠르게 저하되는 것 관찰
- 이를 대응하기 위해 image-text alignment 와 language instruction funing 두 가지 task 간의 간섭을 해결하는 간단한 **_early fusion of visual knowledge_** 제안
  - LLaMA-Adapter 의 dynamic visual prompts 는 last $L$ layer 의 static adaptation prompts 에 통합
  - LLaMA-Adapter V2 에서는 dynamic visual prompt 를 처음 $K$ layer 에만 분배
    - $K < N - L$
    - $N$ : total number of Transformer layers
  - 이를 통해 image-text alignment 가 model 의 instruction-following 능력 방해하지 않음
- **_joint training with disjoint parameter_**
  - 고품질의 multi-modal instruction data 없이, image caption 및 instruction-following data 로 분리된 parameter 를 joint training 하여 우수한 visual instruction learning 가능
- **_bias tuning of linear layers_**
  - LLaMA-Adapter 를 normalization, layer bias 및 scale 같은 learnable parameter 를 unlocking 하여 보완
  - tunable capacity 를 증가시켜 instruction-following knowledge 를 LLM 전체에 분산
  - 이러한 parameter 는 모델 전체의 약 0.04% 만 차지
  - 이를 통해 parameter-efficient approach 유지
- **_additional expert models_** (captioning, detection 및 OCR system)
  - expert model 과 협력하여 LLaMA-Adapter V2 는 대규모 image-text pair 불필요
  - 다양한 expert 를 plugging 가능하여 유연성 얻음

![Figure 1](image-29.png)

다음은 주요 기여 요약

- Stronger Language Instruction Model
  - parameter-efficient tuning 및 high-quality language instruction data 사용하여 LLaMA-Adapter 능가
- Balanced Visual Instruction Tuning
  - image-text alignment 와 instruction-following object 간의 간섭 해결을 위해 early fusion 전략 사용
  - multi-modal instruction training data 없이 captioning data 및 instruction-following data 의 분리된 parameter 를 joint training
- Integration of Expert Systems
  - 다양한 expert model 통합

# 2. Related Work

# 3. A Revisit of LLaMA-Adapter

### Zero-initialized Attention

- instruction-following 능력 습득을 위한 parameter-efficient fine-tuning
- LLaMA-Adapter 는 LLaMA 를 freezing 하고 1.2M 추가 adapter module 만 도입
- adapter layer 는 LLaMA 의 Transformer layer 상단에 사용
- leanable soft prompt set 을 word token 의 prefix 에 연결
- new adapting knowledge 를 LLaMA 에 통합하기 위해 zero-initialized attention 사용
- 이를 통해 adaptation prompt word token 에 대한 기여를 학습 초기에 0으로 초기화된 gating factor 를 학습하여 조절
- 훈련 중 gating 크기는 점진적으로 커지며 LLaMA 에 주입

이 전략은 훈련 초기, LLaMA 의 언어 생성 능력을 보존하며 새로운 지식을 지속적으로 통합하여 강력한 instruction-following 능력을 만든다.

### Simple Multi-modal Variant

- multi-modal reasoning 을 위해 image 및 video 통합 가능
- pre-trained visual encoder 를 지닌 CLIP 으로 multi-scale visual feature 추출
- learnable projection layer 를 통과하여 visual semantics 를 language embedding space 와 alignment
- 이후 visual feature 는 Transformer layer 상단에서 element-wisely added

위 과정으로 LLaMA-Adapter 는 text 및 visual input 을 기반으로 response 생성 가능하여 ScienceQA 에서 comparable

### Open-ended Multi-modal Reasoning

LLaMA-Adapter 를 사용하여 COCO Caption dataset 에서 adapter module 및 visual projection layer 를 fine-tuning 하는 실험 수행

새로운 visual 단서가 adaptation prompt 를 간섭하는 경향이 나타나 instruction-following feature 를 덮어버림.

따라서 LLaMA-Adapter V2 를 제안하여, multi-modal 잠재력을 완전히 발휘하도록 함

# 4. LLaMA-Adapter V2

## 4.1 Bias Tuning of Linear Layers

LLaMA-Adapter 는 zero-initialized attention 으로 adaptation prompt 를 frozen LLaMA 에 사용

이는 new knowledge 를 통합하지만 LLM 내부 parameter 를 수정하지 않고는 parameter update 가 adaptation prompt 및 gating factor 로 제한됨

이 때문에 deep fine-tuning 수행 능력이 제한된다.

이를 고려하여 더욱 효과적인 통합을 위해 **_bias tuning_** 전략 제안

- instruction-following data 를 adaptively handle 하기 위해 LLaMA 의 모든 normalization layers 를 unfreezing
- Transformer 의 각 linear layer 에 대해 learnable parameter 로 bias 및 scale factor 추가
- 특정 linear layer 및 pre-trained weights 를 $\text{x}$ 및 $W$ 로 표시
- LLaMA-Adapter V2 에선, bias $b$ 및 scale $s$ 를 사용하여 linear layer 수정

$$
\begin{align}
    y = W \cdot \text{x} \rightarrow y = s \cdot (W \cdot \text{x} + b), \tag{1}  \\
    \text{where} \ b = \text{Init}(0),\  s = \text{Init}(1). \tag{2}
\end{align}
$$

- bias 와 scale factor 는 각각 0 과 1 로 초기화하여 초기 단계에 안정화
- bias tuning 및 high-quality instruction data 를 통합하여 우수한 instruction-following 능력을 얻음
- 특히, newly added parameter 는 전체 LLaMA 의 0.04% (약 5M) 만 차지
- LLaMA-Adapter V2 는 여전히 highly parameter-efficient approach

### Discussion

bias tuning 은 이전 parameter-efficient method 와 유사

BERT fine-tuning 을 위한 BitFit 및 visual prompt tuning 을 위한 SSF 가 있음

- BitFit 및 SSF 는 80M parameter scale 을 가진 comprehension task 를 위해 설계
- 저자의 approach 는 70B - 650B parameter scale 의 LLM 에서 효율성 나타냄
- bias tuning 은 input 에 독립적이며, row-rank 를 사용하여 input 에 의존적인 bias 를 추가하는 LoRA 와는 달리, fine-tuning 비용을 더 줄임

## 4.2 Joint Training with Disjoint Parameters

저자의 목표는 LLaMA-Adapter V2 에게 long language response 를 생성하는 능력과 multi-modal understanding 을 동시에 부여하는 것

![Figure 2](image-33.png)

저자는 LLaMA-Adapter V2 를 위해 image-text captioning data 및 language-only instruction examples 를 활용하기 위한 **_joint training paradigm_** 제안

- 500K image-text pairs 및 50K instruction data 사이의 데이터양 차이로 인해, instruction-following 능력에 피해가 갈 수 있음
- 따라서 **_이질적인 (disjoint) parameter groups_** 를 최적화
  - image-text captioning data : visual projection layer 및 초기 zero-initialized gating 과 관련된 부분만 학습
  - language instruction data : late adaptation prompts 와 zero gating, unfrozen norm 및 newly added bias 및 scale factor (선택적으로 low-rank adaptation) 가 사용

이를 통해 image-text understanding 과 instruction-following 간의 간섭 문제를 자연스럽게 해결

### Discussion

우리의 공동 훈련 전략 덕분에 LLaMA-Adapter V2는 MiniGPT-4 [78]와 LLaVA [38]와 같은 고품질 멀티모달 지시 데이터가 필요하지 않습니다. 대신 이미지-텍스트 쌍과 지시 따르기 데이터만 필요합니다(표 1에서 비교). 캡션 데이터는 그림 2에서 보여주는 대로 짧은 답변을 포함하여 이미지 이해에 대한 LLMs를 확장하는 역할을 합니다. 한편 언어 전용 지시 데이터는 긴 상세한 문장을 생성할 능력을 보존하기 위해 사용됩니다. 이러한 상호 보완적인 조합으로 LLaMA-Adapter V2는 고품질 멀티모달 지시 데이터 없이도 소규모 이미지-텍스트 및 지시 따르기 데이터만으로 우수한 멀티모달 추론을 달성할 수 있습니다.



## 4.3 Early Fusion of Visual Knowledge

![Figure 3](image-34.png)