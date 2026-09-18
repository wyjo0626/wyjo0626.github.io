---
slug: Swin-Transformer
title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
tags: [Swin-Transformer, Vision Transformer, Hierarchical Transformer, Computer Vision]
---

논문 및 이미지 출처 : <https://arxiv.org/pdf/2103.14030>

# Abstract

이 논문은 **Swin Transformer** 라는 새로운 vision Transformer 를 제시하며, 이 model 은 computer vision 을 위한 general-purpose backbone 으로 효과적으로 작동한다. Transformer 를 language 에서 vision 으로 적응시키는 과정에서는 두 domain 사이의 차이, 예를 들어 visual entity 의 scale 변화가 크다는 점과 text 의 word 에 비해 image 의 pixel 해상도가 매우 높다는 점에서 어려움이 발생한다. 이러한 차이를 해결하기 위해, 저자는 **Shifted windows** 로 representation 을 계산하는 hierarchical Transformer 를 제안한다. shifted windowing scheme 은 self-attention 계산을 겹치지 않는 local window 로 제한함으로써 더 높은 efficiency 를 제공하는 동시에, cross-window connection 도 가능하게 한다. 이 hierarchical architecture 는 다양한 scale 에서 model 을 구성할 수 있는 유연성을 가지며, image size 에 대해 linear computational complexity 를 가진다. 이러한 Swin Transformer 의 특성은 image classification, object detection 과 같은 dense prediction task, semantic segmentation 을 포함한 광범위한 vision task 와 호환되게 만든다. 성능은 ImageNet-1K 에서 87.3 top-1 accuracy, COCO test-dev 에서 58.7 box AP 와 51.1 mask AP, ADE20K val 에서 53.5 mIoU 를 기록한다. 이 성능은 COCO 에서 이전 state-of-the-art 를 +2.7 box AP 와 +2.6 mask AP 만큼, ADE20K 에서 +3.2 mIoU 만큼 크게 능가하며, Transformer 기반 model 이 vision backbone 으로서 지닌 잠재력을 보여준다. hierarchical design 과 shifted window approach 는 all-MLP architecture 에도 유익함이 입증된다. code 와 model 은 공개되어 있다.

# 1. Introduction

computer vision 에서의 modeling 은 오랫동안 convolutional neural networks (CNNs) 가 지배해 왔다. AlexNet 과 그것이 ImageNet image classification challenge 에서 보여준 혁신적인 성능을 시작으로, CNN architecture 는 더 큰 scale, 더 광범위한 connection, 더 정교한 형태의 convolution 을 통해 점점 더 강력해지도록 발전해 왔다. CNN 이 다양한 vision task 의 backbone network 로 사용되면서, 이러한 architecture 발전은 분야 전반의 성능 향상으로 이어졌다.

반면, natural language processing (NLP) 에서 network architecture 의 진화는 다른 경로를 밟아 왔으며, 오늘날 지배적인 architecture 는 Transformer 이다. sequence modeling 과 transduction task 를 위해 설계된 Transformer 는 data 내의 long-range dependency 를 modeling 하기 위해 attention 을 사용한다는 점에서 두드러진다. language domain 에서의 엄청난 성공은 연구자들이 이를 computer vision 에 적응시키는 방향으로 탐구하게 만들었고, 최근에는 특히 image classification 및 joint vision-language modeling 과 같은 특정 task 에서 유망한 결과를 보여주었다.

이 논문에서 저자는 Transformer 가 NLP 에서 그러하듯, 그리고 CNN 이 vision 에서 그러하듯, computer vision 을 위한 general-purpose backbone 으로 사용될 수 있도록 그 적용 범위를 확장하고자 한다. 저자는 language domain 에서의 높은 성능을 visual domain 으로 이전하는 데 존재하는 중요한 어려움이 두 modality 사이의 차이로 설명될 수 있다고 본다. 그 차이 중 하나는 scale 과 관련된다. language Transformer 에서 처리의 기본 요소가 되는 word token 과 달리, visual element 는 scale 이 크게 달라질 수 있으며, 이 문제는 object detection 과 같은 task 에서 중요하게 다뤄진다. 기존 Transformer 기반 model 에서는 token 이 모두 고정된 scale 을 가지며, 이러한 특성은 이런 vision application 에 적합하지 않다. 또 다른 차이는 image 의 pixel 해상도가 text 문장의 word 보다 훨씬 높다는 점이다. semantic segmentation 과 같이 pixel level 의 dense prediction 을 요구하는 많은 vision task 가 존재하며, self-attention 의 computational complexity 가 image size 에 대해 quadratic 이기 때문에, 고해상도 image 에 대해 Transformer 를 적용하는 것은 다루기 어렵다.

이러한 문제를 해결하기 위해, 저자는 **Swin Transformer** 라는 general-purpose Transformer backbone 을 제안하며, 이 model 은 hierarchical feature map 을 구성하고 image size 에 대해 linear computational complexity 를 가진다. Fig. 1(a) 에서 보이듯이, Swin Transformer 는 작은 크기의 patch 에서 시작하고, 더 깊은 Transformer layer 에서 이웃한 patch 를 점진적으로 병합함으로써 hierarchical representation 을 구성한다. 이러한 hierarchical feature map 을 통해, Swin Transformer model 은 feature pyramid networks (FPN) 이나 U-Net 과 같은 dense prediction 용 고급 기법을 손쉽게 활용할 수 있다. linear computational complexity 는 image 를 분할하는 겹치지 않는 window 내부에서만 self-attention 을 local 하게 계산함으로써 달성된다. 각 window 내의 patch 수는 고정되어 있으므로, complexity 는 image size 에 대해 linear 가 된다. 이러한 장점은 Swin Transformer 를 다양한 vision task 를 위한 general-purpose backbone 으로 적합하게 만들며, 단일 resolution 의 feature map 을 생성하고 quadratic complexity 를 가지는 기존 Transformer 기반 architecture 와 대비된다.

Swin Transformer 의 핵심 design 요소는 Fig. 2 에서 보이듯이, 연속된 self-attention layer 사이에서 window partition 을 이동시키는 것이다. shifted window 는 이전 layer 의 window 를 연결하며, 이들 사이의 connection 을 제공하여 modeling power 를 크게 향상시킨다. 이 전략은 실제 latency 측면에서도 efficient 하다. 하나의 window 안의 모든 query patch 가 동일한 key set 을 공유하므로, hardware 에서 memory access 가 용이해진다. 반면, 이전의 sliding window 기반 self-attention 접근은 query pixel 마다 서로 다른 key set 을 사용하기 때문에 일반 hardware 에서 latency 가 낮다. 저자의 실험은 제안한 shifted window approach 가 sliding window method 보다 훨씬 낮은 latency 를 가지면서도, modeling power 는 유사함을 보여준다. shifted window approach 는 all-MLP architecture 에도 유익함이 입증된다.

제안된 Swin Transformer 는 image classification, object detection, semantic segmentation 이라는 recognition task 에서 강력한 성능을 달성한다. 이 model 은 세 task 모두에서 유사한 latency 로 ViT / DeiT 및 ResNe(X)t model 을 유의미하게 능가한다. COCO test-dev set 에서의 58.7 box AP 와 51.1 mask AP 는 이전 state-of-the-art result 를 각각 +2.7 box AP 와 +2.6 mask AP 만큼 넘어선다. ADE20K semantic segmentation 에서는 val set 에서 53.5 mIoU 를 얻어, 이전 state-of-the-art 보다 +3.2 mIoU 향상된다. 또한 ImageNet-1K image classification 에서 top-1 accuracy 87.3% 를 달성한다.

저자는 computer vision 과 natural language processing 전반에 걸친 unified architecture 가 두 분야 모두에 이익이 될 수 있다고 믿는다. 왜냐하면 이는 visual signal 과 textual signal 의 joint modeling 을 용이하게 하고, 두 domain 의 modeling knowledge 가 더 깊이 공유될 수 있게 하기 때문이다. 저자는 Swin Transformer 가 다양한 vision 문제에서 보여주는 강력한 성능이 이러한 믿음을 community 내에서 더 확산시키고, vision 과 language signal 의 unified modeling 을 장려하기를 바란다.


# 2. Related Work

## CNN and variants

CNN 은 computer vision 전반에서 표준 network model 로 사용된다. CNN 자체는 수십 년 전부터 존재했지만, AlexNet 이 도입된 이후에야 CNN 이 본격적으로 확산되어 주류가 되었다. 그 이후로 VGG, GoogleNet, ResNet, DenseNet, HRNet, EfficientNet 과 같이 더 깊고 더 효과적인 convolutional neural architecture 가 제안되었고, 이는 computer vision 에서 deep learning 의 흐름을 더욱 가속했다.

이러한 architecture 발전에 더해, 개별 convolution layer 를 개선하려는 많은 연구도 이루어졌다. 예를 들어 depthwise convolution, deformable convolution 이 있다.

CNN 과 그 변형들은 여전히 computer vision application 을 위한 주요 backbone architecture 이지만, 저자는 vision 과 language 사이의 unified modeling 을 위한 Transformer 유사 architecture 의 강한 잠재력을 강조한다. 저자의 연구는 몇 가지 기본적인 visual recognition task 에서 강력한 성능을 달성하며, 이것이 modeling 패러다임의 전환에 기여하기를 기대한다.

## Self-attention based backbone architectures

NLP 분야에서 self-attention layer 와 Transformer architecture 가 성공한 것에 영감을 받아, 일부 연구는 popular ResNet 에서 일부 또는 전체 spatial convolution layer 를 self-attention layer 로 대체한다.

* 이들 연구에서는 optimization 을 가속하기 위해 각 pixel 의 local window 내에서 self-attention 을 계산한다.
* 그 결과, 대응되는 ResNet architecture 보다 accuracy/FLOPs trade-off 가 약간 더 좋아진다.
* 그러나 이들의 비용이 큰 memory access 로 인해, 실제 latency 는 convolutional network 보다 유의미하게 더 크다.

저자는 sliding window 를 사용하는 대신, 연속된 layer 사이에서 window 를 shift 하는 방식을 제안한다.

* 이 방식은 일반 hardware 에서 더 efficient 한 구현을 가능하게 한다.

## Self-attention/Transformers to complement CNNs

또 다른 연구 흐름은 표준 CNN architecture 를 self-attention layer 또는 Transformer 로 보강하는 것이다.

* self-attention layer 는 먼 거리 dependency 나 heterogeneous interaction 을 encoding 하는 능력을 제공함으로써 backbone 이나 head network 를 보완할 수 있다.
* 더 최근에는 Transformer 의 encoder-decoder design 이 object detection 과 instance segmentation task 에 적용되었다.

저자의 연구는 기본적인 visual feature extraction 을 위해 Transformer 를 적응시키는 방향을 탐구하며, 이러한 연구들과 상보적이다.

## Transformer based vision backbones

저자의 연구와 가장 관련이 깊은 것은 Vision Transformer (ViT) 와 그 후속 연구들이다. ViT 의 선구적인 연구는 image classification 을 위해 겹치지 않는 중간 크기의 image patch 에 Transformer architecture 를 직접 적용한다.

* ViT 는 convolutional network 와 비교해 image classification 에서 인상적인 speed-accuracy trade-off 를 달성한다.
* 그러나 ViT 가 좋은 성능을 내기 위해서는 large-scale training dataset, 즉 JFT-300M 이 필요하다.
* DeiT 는 몇 가지 training strategy 를 도입하여, 더 작은 ImageNet-1K dataset 에서도 ViT 가 효과적이도록 한다.

ViT 의 image classification 결과는 고무적이지만, 그 architecture 는 dense vision task 를 위한 general-purpose backbone network 로 사용되기에는 적합하지 않다.

* 그 이유는 feature map 의 resolution 이 낮고,
* complexity 가 image size 에 따라 quadratic 하게 증가하기 때문이다.

object detection 과 semantic segmentation 이라는 dense vision task 에 ViT model 을 직접 upsampling 이나 deconvolution 으로 적용한 몇몇 연구가 있지만, 성능은 상대적으로 더 낮다.

저자의 연구와 동시에 진행된 일부 연구는 더 나은 image classification 을 위해 ViT architecture 를 수정한다.

* 경험적으로 저자는 image classification 에서 이러한 방법들 가운데 Swin Transformer architecture 가 가장 뛰어난 speed-accuracy trade-off 를 달성함을 발견한다.
* 이는 저자의 연구가 classification 자체보다는 general-purpose performance 에 초점을 맞추고 있다는 점을 고려하면 더욱 의미가 있다.

또 다른 동시기 연구는 Transformer 에서 multi-resolution feature map 을 구축하기 위해 유사한 사고방식을 탐구한다.

* 그러나 그 방법의 complexity 는 여전히 image size 에 대해 quadratic 이다.
* 반면 저자의 방법은 linear 이며, local 하게 작동한다.

  * 이는 visual signal 의 높은 correlation 을 modeling 하는 데 유익함이 입증되었다.

저자의 접근법은 efficient 하면서도 effective 하며, COCO object detection 과 ADE20K semantic segmentation 모두에서 state-of-the-art accuracy 를 달성한다.

# 3. Method

## 3.1. Overall Architecture

Swin Transformer architecture 의 개요는 Fig. 3 에 제시되어 있으며, 여기서는 tiny version 인 Swin-T 를 보여준다. 이 model 은 먼저 ViT 와 마찬가지로 patch splitting module 을 사용해 입력 RGB image 를 겹치지 않는 patch 로 분할한다. 각 patch 는 하나의 "token" 으로 취급되며, 그 feature 는 원시 pixel RGB 값의 concatenation 으로 설정된다. 저자의 구현에서는 patch size 를 $4 \times 4$ 로 사용하므로, 각 patch 의 feature dimension 은 $4 \times 4 \times 3 = 48$ 이다. 이후 이 raw-valued feature 에 linear embedding layer 를 적용하여 이를 임의의 dimension 으로 projection 하며, 이를 $C$ 로 표기한다.

수정된 self-attention 계산을 사용하는 여러 Transformer block, 즉 Swin Transformer block 이 이 patch token 에 적용된다. 이 Transformer block 은 token 의 수 $\left(\frac{H}{4} \times \frac{W}{4}\right)$ 를 유지하며, linear embedding 과 함께 이를 "Stage 1" 이라고 부른다.

hierarchical representation 을 생성하기 위해, network 가 깊어질수록 patch merging layer 를 통해 token 수를 줄인다. 첫 번째 patch merging layer 는 인접한 $2 \times 2$ patch 그룹 각각의 feature 를 concatenation 하고, $4C$ 차원의 concatenated feature 에 linear layer 를 적용한다. 이는 token 수를 $2 \times 2 = 4$ 배 줄이며, 즉 resolution 을 $2 \times$ downsampling 하고, output dimension 은 $2C$ 로 설정된다. 그 이후 feature transformation 을 위해 Swin Transformer block 을 적용하며, resolution 은 $\frac{H}{8} \times \frac{W}{8}$ 로 유지된다. 이러한 첫 번째 patch merging 과 feature transformation block 을 "Stage 2" 라고 부른다. 이 절차는 두 번 더 반복되어 "Stage 3" 과 "Stage 4" 를 이루며, 각각의 output resolution 은 $\frac{H}{16} \times \frac{W}{16}$ 및 $\frac{H}{32} \times \frac{W}{32}$ 이다.

이들 stage 는 함께 hierarchical representation 을 생성하며, 이는 VGG 와 ResNet 같은 전형적인 convolutional network 와 동일한 feature map resolution 을 가진다. 그 결과, 제안된 architecture 는 다양한 vision task 를 위한 기존 방법의 backbone network 를 손쉽게 대체할 수 있다.

#### Swin Transformer block

Swin Transformer 는 Transformer block 의 표준 multi-head self attention (MSA) module 을 shifted window 기반 module 로 대체하여 구축되며, 다른 layer 는 동일하게 유지된다. shifted window 는 Sec. 3.2 에서 설명한다. Fig. 3(b) 에서 보이듯이, 하나의 Swin Transformer block 은 shifted window 기반 MSA module 과, 그 뒤를 잇는 중간에 GELU nonlinearity 를 둔 2-layer MLP 로 구성된다. 각 MSA module 과 각 MLP 앞에는 LayerNorm (LN) layer 가 적용되며, 각 module 뒤에는 residual connection 이 적용된다.

## 3.2. Shifted Window based Self-Attention

표준 Transformer architecture 와 image classification 을 위한 그 적응형 모두는 global self-attention 을 수행하며, 여기서는 하나의 token 과 다른 모든 token 사이의 관계를 계산한다. 이러한 global 계산은 token 수에 대해 quadratic complexity 를 유발하므로, dense prediction 을 위해 막대한 수의 token 이 필요하거나 고해상도 image 를 표현해야 하는 많은 vision 문제에는 적합하지 않다.

#### Self-attention in non-overlapped windows

efficient 한 modeling 을 위해, 저자는 local window 내부에서 self-attention 을 계산할 것을 제안한다. window 는 image 를 겹치지 않게 균등하게 분할하도록 배치된다. 각 window 가 $M \times M$ patch 를 포함한다고 가정하면, $h \times w$ patch 를 가진 image 에 대한 global MSA module 과 window 기반 module 의 computational complexity 는 다음과 같다.

$$
\Omega(\mathrm{MSA}) = 4hwC^2 + 2(hw)^2 C \tag{1}
$$

$$
\Omega(\mathrm{W{-}MSA}) = 4hwC^2 + 2M^2hwC \tag{2}
$$

* 앞의 식은 patch 수 $hw$ 에 대해 quadratic 이고,
* 뒤의 식은 $M$ 이 고정될 때 linear 이다. 기본값으로는 $M = 7$ 이다.

global self-attention 계산은 일반적으로 큰 $hw$ 에 대해 감당하기 어렵지만, window 기반 self-attention 은 scalable 하다.

#### Shifted window partitioning in successive blocks

window 기반 self-attention module 은 window 사이의 connection 이 부족하며, 이는 modeling power 를 제한한다. 겹치지 않는 window 의 efficient 한 계산을 유지하면서 cross-window connection 을 도입하기 위해, 저자는 연속된 Swin Transformer block 에서 두 partitioning configuration 을 번갈아 사용하는 shifted window partitioning approach 를 제안한다.

Fig. 2 에서 보이듯이, 첫 번째 module 은 좌상단 pixel 에서 시작하는 일반적인 window partitioning strategy 를 사용하며, $8 \times 8$ feature map 을 크기 $4 \times 4$ 의 $2 \times 2$ window 로 균등하게 분할한다. 즉, $M = 4$ 이다. 이후 다음 module 은 이전 layer 의 configuration 에서 shift 된 windowing configuration 을 채택하며, 정규 partition 된 window 로부터 window 를 $\left(\left\lfloor \frac{M}{2} \right\rfloor, \left\lfloor \frac{M}{2} \right\rfloor\right)$ pixel 만큼 이동시킨다.

shifted window partitioning approach 를 사용하면, 연속된 Swin Transformer block 은 다음과 같이 계산된다.

$$
\hat{z}^l = \mathrm{W{-}MSA}(\mathrm{LN}(z^{l-1})) + z^{l-1},
$$

$$
z^l = \mathrm{MLP}(\mathrm{LN}(\hat{z}^l)) + \hat{z}^l,
$$

$$
\hat{z}^{l+1} = \mathrm{SW{-}MSA}(\mathrm{LN}(z^l)) + z^l,
$$

$$
z^{l+1} = \mathrm{MLP}(\mathrm{LN}(\hat{z}^{l+1})) + \hat{z}^{l+1}. \tag{3}
$$

* 여기서 $\hat{z}^l$ 와 $z^l$ 는 각각 block $l$ 에 대한 (S)W-MSA module 과 MLP module 의 output feature 를 나타낸다.
* $\mathrm{W{-}MSA}$ 와 $\mathrm{SW{-}MSA}$ 는 각각 regular window partitioning configuration 과 shifted window partitioning configuration 을 사용하는 window 기반 multi-head self-attention 을 뜻한다.

shifted window partitioning approach 는 이전 layer 의 이웃한 non-overlapping window 사이에 connection 을 도입하며, 이는 Tab. 4 에서 보이듯 image classification, object detection, semantic segmentation 에서 효과적임이 확인된다.

#### Efficient batch computation for shifted configuration

shifted window partitioning 의 한 가지 문제는 shifted configuration 에서 window 수가 $\left\lceil \frac{h}{M} \right\rceil \times \left\lceil \frac{w}{M} \right\rceil$ 에서 $\left(\left\lceil \frac{h}{M} \right\rceil + 1\right) \times \left(\left\lceil \frac{w}{M} \right\rceil + 1\right)$ 로 증가하고, 일부 window 가 $M \times M$ 보다 작아진다는 점이다. 순진한 해법은 더 작은 window 를 크기 $M \times M$ 이 되도록 padding 하고, attention 계산 시 padding 된 값을 mask out 하는 것이다. regular partitioning 에서의 window 수가 작을 때, 예를 들어 $2 \times 2$ 일 때, 이 순진한 해법으로 인한 증가된 계산량은 상당하다. $2 \times 2 \rightarrow 3 \times 3$ 이 되어, 이는 2.25 배 더 크다.

여기서 저자는 Fig. 4 에서 보이듯, 좌상단 방향으로 cyclic-shifting 하는 더 efficient 한 batch computation approach 를 제안한다. 이 shift 이후 하나의 batched window 는 feature map 상에서 인접하지 않은 여러 sub-window 로 구성될 수 있으므로, self-attention 계산을 각 sub-window 내부로 제한하기 위해 masking mechanism 을 사용한다. cyclic-shift 를 사용하면 batched window 수는 regular window partitioning 의 경우와 동일하게 유지되므로, 이 역시 efficient 하다. 이 approach 의 낮은 latency 는 Tab. 5 에 제시된다.

#### Relative position bias

self-attention 을 계산할 때, 저자는 각 head 에 relative position bias $B \in \mathbb{R}^{M^2 \times M^2}$ 를 포함시키는 방식을 따른다.

$$
\mathrm{Attention}(Q, K, V) = \mathrm{SoftMax}\left(\frac{QK^T}{\sqrt{d}} + B\right)V \tag{4}
$$

* 여기서 $Q, K, V \in \mathbb{R}^{M^2 \times d}$ 는 각각 query, key, value matrix 이다.
* $d$ 는 query/key dimension 이다.
* $M^2$ 는 하나의 window 안에 있는 patch 수이다.

각 axis 를 따라 relative position 이 $[-M + 1, M - 1]$ 범위에 있으므로, 저자는 더 작은 크기의 bias matrix $\hat{B} \in \mathbb{R}^{(2M-1) \times (2M-1)}$ 를 parameterize 하며, $B$ 의 값은 $\hat{B}$ 로부터 가져온다.

저자는 이 bias term 이 없거나 absolute position embedding 을 사용하는 대응 방법에 비해 유의미한 향상을 관찰하며, 이는 Tab. 4 에 제시된다. 또한 입력에 absolute position embedding 을 추가하는 방식을 더하면 성능이 약간 감소하므로, 저자의 구현에서는 이를 채택하지 않는다.

pre-training 에서 학습된 relative position bias 는 bi-cubic interpolation 을 통해 window size 가 다른 model 을 fine-tuning 할 때 initialization 에도 사용할 수 있다.

## 3.3. Architecture Variants

저자는 base model 인 Swin-B 를, model size 와 computation complexity 가 ViT-B/DeiT-B 와 유사하도록 구성한다. 또한 model size 와 computational complexity 가 각각 약 0.25 배, 0.5 배, 2 배인 Swin-T, Swin-S, Swin-L 도 도입한다. Swin-T 와 Swin-S 의 complexity 는 각각 ResNet-50 (DeiT-S) 와 ResNet-101 의 complexity 와 유사하다는 점에 유의한다. window size 는 기본적으로 $M = 7$ 로 설정한다. 각 head 의 query dimension 은 $d = 32$ 이고, 각 MLP 의 expansion layer 는 $\alpha = 4$ 이며, 이는 모든 실험에서 동일하다. 이 model variant 들의 architecture hyper-parameter 는 다음과 같다.

* **Swin-T:** $C = 96$, layer numbers $= {2, 2, 6, 2}$

* **Swin-S:** $C = 96$, layer numbers $= {2, 2, 18, 2}$

* **Swin-B:** $C = 128$, layer numbers $= {2, 2, 18, 2}$

* **Swin-L:** $C = 192$, layer numbers $= {2, 2, 18, 2}$

* 여기서 $C$ 는 첫 번째 stage 에 있는 hidden layer 의 channel 수이다.

ImageNet image classification 에 대한 model variant 의 model size, 이론적 computational complexity (FLOPs), 그리고 throughput 은 Tab. 1 에 제시되어 있다.

# 4. Experiments

저자는 ImageNet-1K image classification, COCO object detection, ADE20K semantic segmentation 에서 실험을 수행한다. 이하에서는 먼저 제안한 Swin Transformer architecture 를 세 가지 task 에서 기존 state-of-the-art 와 비교한 뒤, Swin Transformer 의 중요한 design element 를 ablation 한다.

## 4.1. Image Classification on ImageNet-1K

#### Settings

image classification 을 위해, 저자는 ImageNet-1K 에서 제안한 Swin Transformer 를 benchmark 한다. 이 dataset 은 1,000 개 class 에서 1.28M 개의 training image 와 50K 개의 validation image 를 포함한다. single crop 에 대한 top-1 accuracy 를 보고한다. 저자는 두 가지 training setting 을 고려한다.

* **Regular ImageNet-1K training**

  * 이 setting 은 대체로 DeiT 를 따른다.
  * 저자는 AdamW optimizer 를 사용하여 300 epochs 동안 학습하며, cosine decay learning rate scheduler 와 20 epochs 의 linear warm-up 을 사용한다.
  * batch size 는 1024, 초기 learning rate 는 0.001, weight decay 는 0.05 를 사용한다.
  * repeated augmentation 과 EMA 를 제외하고, DeiT 의 대부분의 augmentation 및 regularization strategy 를 training 에 포함한다.

    * 이 두 기법은 성능을 향상시키지 않는다.
  * 이는 repeated augmentation 이 ViT training 을 안정화하는 데 중요하다고 본 DeiT 와는 상반된다.

* **Pre-training on ImageNet-22K and fine-tuning on ImageNet-1K**

  * 저자는 14.2 million image 와 22K class 를 포함하는 더 큰 ImageNet-22K dataset 에서도 pre-train 한다.
  * 저자는 AdamW optimizer 를 사용하여 90 epochs 동안 학습하며, 5-epoch linear warm-up 이 포함된 linear decay learning rate scheduler 를 사용한다.
  * batch size 는 4096, 초기 learning rate 는 0.001, weight decay 는 0.01 이다.
  * ImageNet-1K fine-tuning 에서는 model 을 30 epochs 동안 학습하며, batch size 는 1024, constant learning rate 는 $10^{-5}$, weight decay 는 $10^{-8}$ 이다.

#### Results with regular ImageNet-1K training

Tab. 1(a) 는 regular ImageNet-1K training 을 사용하는 다른 backbone 과의 비교를 제시하며, 여기에는 Transformer 기반 backbone 과 ConvNet 기반 backbone 이 모두 포함된다.

* 이전 state-of-the-art Transformer 기반 architecture 인 DeiT 와 비교하면, Swin Transformer 는 유사한 complexity 에서 대응되는 DeiT architecture 를 뚜렷하게 능가한다.

  * Swin-T 는 $224^2$ input 에서 DeiT-S 의 79.8% 대비 81.3% 를 기록하여 +1.5% 향상된다.
  * Swin-B 는 $224^2 / 384^2$ input 에서 DeiT-B 의 81.8% / 83.1% 대비 83.3% / 84.5% 를 기록하여 각각 +1.5% / +1.4% 향상된다.

* state-of-the-art ConvNet 인 RegNet 과 EfficientNet 과 비교하면, Swin Transformer 는 약간 더 나은 speed-accuracy trade-off 를 달성한다.

  * RegNet 과 EfficientNet 은 모두 철저한 architecture search 를 통해 얻어진 반면,
  * 제안한 Swin Transformer 는 표준 Transformer 로부터 적응된 것이며, 추가 개선의 잠재력이 크다.

#### Results with ImageNet-22K pre-training

저자는 더 큰 capacity 의 Swin-B 와 Swin-L 을 ImageNet-22K 에서도 pre-train 한다. ImageNet-1K image classification 에 대해 fine-tuning 한 결과는 Tab. 1(b) 에 제시된다.

* Swin-B 에서, ImageNet-22K pre-training 은 ImageNet-1K 에서 scratch 부터 학습하는 것에 비해 1.8% ∼ 1.9% 의 향상을 가져온다.
* ImageNet-22K pre-training 의 이전 최고 결과와 비교하면, 저자의 model 은 훨씬 더 나은 speed-accuracy trade-off 를 달성한다.

  * Swin-B 는 86.4% top-1 accuracy 를 달성하며,

    * 이는 유사한 inference throughput 을 가지는 ViT 보다 2.4% 높다.
    * throughput 은 84.7 vs. 85.9 images/sec 이고,
    * FLOPs 는 47.0G vs. 55.4G 로 약간 더 낮다.
* 더 큰 Swin-L model 은 87.3% top-1 accuracy 를 달성하며,

  * 이는 Swin-B 보다 +0.9% 더 좋다.

## 4.2. Object Detection on COCO

#### Settings

object detection 및 instance segmentation 실험은 COCO 2017 에서 수행되며, 이 dataset 은 118K training image, 5K validation image, 20K test-dev image 를 포함한다. ablation study 는 validation set 을 사용해 수행하며, system-level comparison 은 test-dev 에서 보고한다.

ablation study 를 위해, 저자는 mmdetection 에서 다음의 네 가지 전형적인 object detection framework 를 고려한다.

* Cascade Mask R-CNN
* ATSS
* RepPoints v2
* Sparse RCNN

이 네 framework 에 대해, 저자는 동일한 setting 을 사용한다.

* multi-scale training

  * 입력의 짧은 변이 480 에서 800 사이가 되도록 resize 하고, 긴 변은 최대 1333 으로 제한한다.
* AdamW optimizer

  * 초기 learning rate 는 0.0001
  * weight decay 는 0.05
  * batch size 는 16
* 3x schedule

  * 36 epochs

system-level comparison 을 위해, 저자는 개선된 HTC 를 채택하며, 이를 HTC++ 로 표기한다.

* instaboost
* 더 강한 multi-scale training
* 6x schedule

  * 72 epochs
* soft-NMS
* initialization 으로 ImageNet-22K pre-trained model 사용

저자는 Swin Transformer 를 표준 ConvNet 인 ResNe(X)t 및 이전 Transformer network 인 DeiT 와 비교한다. 비교는 backbone 만 바꾸고 다른 setting 은 모두 동일하게 유지하는 방식으로 수행된다.

* Swin Transformer 와 ResNe(X)t 는 hierarchical feature map 을 가지므로 위의 모든 framework 에 직접 적용 가능하다.
* 반면 DeiT 는 단일 resolution 의 feature map 만 생성하므로 직접 적용할 수 없다.

  * 공정한 비교를 위해, 저자는 DeiT 에 deconvolution layer 를 사용해 hierarchical feature map 을 구성하는 방법을 따른다.

#### Comparison to ResNe(X)t

Tab. 2(a) 는 네 가지 object detection framework 에서 Swin-T 와 ResNet-50 의 결과를 제시한다.

* Swin-T architecture 는 ResNet-50 대비 일관되게 +3.4 ∼ 4.2 box AP 향상을 가져온다.
* 이때 model size, FLOPs, latency 는 약간 더 크다.

Tab. 2(b) 는 Cascade Mask R-CNN 을 사용해 서로 다른 model capacity 에서 Swin Transformer 와 ResNe(X)t 를 비교한다.

* Swin Transformer 는 51.9 box AP 와 45.0 mask AP 라는 높은 detection accuracy 를 달성한다.
* 이는 유사한 model size, FLOPs, latency 를 가지는 ResNeXt-101-64x4d 대비

  * +3.6 box AP
  * +3.3 mask AP
    의 유의미한 향상이다.

개선된 HTC framework 를 사용하여 더 높은 baseline 인 52.3 box AP 와 46.0 mask AP 를 사용할 때에도, Swin Transformer 의 향상은 크다.

* Tab. 2(c) 에서

  * +4.1 box AP
  * +3.1 mask AP
    향상을 보인다.

inference speed 에 관해서는, ResNe(X)t 는 고도로 최적화된 Cudnn function 으로 구축된 반면, 저자의 architecture 는 내장 PyTorch function 으로 구현되며 이들 모두가 충분히 잘 최적화된 것은 아니다. 철저한 kernel optimization 은 이 논문의 범위를 벗어난다.

#### Comparison to DeiT

Cascade Mask R-CNN framework 에서 DeiT-S 의 성능은 Tab. 2(b) 에 제시된다.

* Swin-T 의 결과는 DeiT-S 대비

  * +2.5 box AP
  * +2.3 mask AP
    더 높다.
* 동시에 model size 는 유사하다.

  * 86M vs. 80M
* inference speed 는 유의미하게 더 높다.

  * 15.3 FPS vs. 10.4 FPS

DeiT 의 낮은 inference speed 는 주로 입력 image size 에 대한 quadratic complexity 때문이다.

#### Comparison to previous state-of-the-art

Tab. 2(c) 는 저자의 최고 결과를 기존 state-of-the-art model 과 비교한다.

* 저자의 최고 model 은 COCO test-dev 에서

  * 58.7 box AP
  * 51.1 mask AP
    를 달성한다.
* 이는 이전 최고 결과를 능가한다.

  * box AP 는 +2.7 향상되며, 비교 대상은 external data 를 사용하지 않은 Copy-paste 이다.
  * mask AP 는 +2.6 향상되며, 비교 대상은 DetectoRS 이다.

## 4.3. Semantic Segmentation on ADE20K

#### Settings

ADE20K 는 널리 사용되는 semantic segmentation dataset 이며, 총 150 개 semantic category 의 넓은 범위를 포괄한다. 전체 25K image 를 가지며, 20K 는 training, 2K 는 validation, 3K 는 testing 에 사용된다. 저자는 높은 efficiency 를 이유로 mmseg 의 UperNet 을 base framework 로 사용한다. 더 자세한 내용은 Appendix 에 제시된다.

#### Results

Tab. 3 은 서로 다른 method/backbone pair 에 대해 mIoU, model size, FLOPs, FPS 를 제시한다. 이 결과로부터 다음을 확인할 수 있다.

* Swin-S 는 유사한 computation cost 에서 DeiT-S 보다 +5.3 mIoU 높다.

  * 49.3 vs. 44.0
* 또한 ResNet-101 보다 +4.4 mIoU 높다.
* ResNeSt-101 보다도 +2.4 mIoU 높다.
* ImageNet-22K pre-training 을 사용한 Swin-L model 은 val set 에서 53.5 mIoU 를 달성한다.

  * 이는 이전 최고 model 보다 +3.2 mIoU 높은 값이다.
  * 비교 대상은 더 큰 model size 를 가진 SETR 의 50.3 mIoU 이다.

## 4.4. Ablation Study

이 절에서는 제안된 Swin Transformer 의 중요한 design element 를 ablation 한다. 사용한 task 는 ImageNet-1K image classification, COCO object detection 의 Cascade Mask R-CNN, ADE20K semantic segmentation 의 UperNet 이다.

#### Shifted windows

세 task 에 대한 shifted window approach 의 ablation 결과는 Tab. 4 에 보고된다.

* shifted window partitioning 을 사용하는 Swin-T 는 각 stage 에서 단일 window partitioning 으로 구축된 대응 model 보다 더 우수하다.

  * ImageNet-1K 에서 +1.1% top-1 accuracy
  * COCO 에서 +2.8 box AP / +2.2 mask AP
  * ADE20K 에서 +2.8 mIoU
* 이 결과는 이전 layer 의 window 사이에 connection 을 구축하기 위해 shifted window 를 사용하는 것의 효과를 보여준다.

shifted window 로 인한 latency overhead 도 작으며, 이는 Tab. 5 에 제시된다.

#### Relative position bias

Tab. 4 는 서로 다른 position embedding approach 를 비교한다.

* relative position bias 를 사용하는 Swin-T 는

  * position encoding 이 없는 경우에 비해

    * ImageNet-1K 에서 +1.2% top-1 accuracy
    * COCO 에서 +1.3 box AP / +1.1 mask AP
    * ADE20K 에서 +2.3 mIoU
      향상된다.
  * absolute position embedding 을 사용하는 경우에 비해

    * ImageNet-1K 에서 +0.8% top-1 accuracy
    * COCO 에서 +1.5 box AP / +1.3 mask AP
    * ADE20K 에서 +2.9 mIoU
      향상된다.

이는 relative position bias 의 효과를 보여준다. 또한 absolute position embedding 을 포함하면 image classification accuracy 는 향상되지만, object detection 과 semantic segmentation 에는 해롭다는 점도 주목할 만하다.

* image classification 에서는 +0.4% 향상된다.
* 그러나

  * COCO 에서는 -0.2 box/mask AP
  * ADE20K 에서는 -0.6 mIoU
    감소한다.

최근 ViT/DeiT model 은 image classification 에서 translation invariance 를 버리고 있지만, 그것은 visual modeling 에 중요하다고 오래전부터 알려져 있었다. 저자는 특정한 translation invariance 를 장려하는 inductive bias 가 여전히 general-purpose visual modeling 에 더 바람직하다고 본다. 특히 object detection 과 semantic segmentation 같은 dense prediction task 에서 그러하다.

#### Different self-attention methods

서로 다른 self-attention 계산 방법 및 구현의 실제 속도는 Tab. 5 에서 비교된다.

* 저자의 cyclic implementation 은 naive padding 보다 hardware 효율이 더 높다.

  * 특히 더 깊은 stage 에서 그러하다.
* 전체적으로 이는

  * Swin-T 에서 13%
  * Swin-S 에서 18%
  * Swin-B 에서 18%
    의 speed-up 을 가져온다.

제안한 shifted window approach 로 구축한 self-attention module 은 naive/kernel implementation 의 sliding window 방식보다 네 network stage 에서 각각 더 efficient 하다.

* 첫 번째 stage 에서 40.8× / 2.5×
* 두 번째 stage 에서 20.2× / 2.5×
* 세 번째 stage 에서 9.3× / 2.1×
* 네 번째 stage 에서 7.6× / 1.8×

전체적으로 shifted window 로 구축된 Swin Transformer architecture 는 sliding window 로 구축된 variant 보다 더 빠르다.

* Swin-T 에서 4.1 / 1.5 배 빠르다.
* Swin-S 에서 4.0 / 1.5 배 빠르다.
* Swin-B 에서 3.6 / 1.5 배 빠르다.

Tab. 6 은 세 task 에서 이들의 accuracy 를 비교하며, visual modeling 에서 유사한 accuracy 를 보인다는 것을 보여준다.

Performer 는 가장 빠른 Transformer architecture 중 하나인데, 제안한 shifted window 기반 self-attention 계산과 전체 Swin Transformer architecture 는 Performer 보다 약간 더 빠르다. 이는 Tab. 5 에 제시된다. 동시에 Swin-T 를 사용한 ImageNet-1K 에서, 저자의 방법은 Performer 보다 +2.3% top-1 accuracy 를 달성한다. 이는 Tab. 6 에 제시된다.

# 5. Conclusion

이 논문은 hierarchical feature representation 을 생성하고 입력 image size 에 대해 linear computational complexity 를 가지는 새로운 vision Transformer 인 Swin Transformer 를 제시한다. Swin Transformer 는 COCO object detection 과 ADE20K semantic segmentation 에서 state-of-the-art 성능을 달성하며, 이전 최고 방법을 유의미하게 능가한다. 저자는 Swin Transformer 가 다양한 vision 문제에서 보여주는 강력한 성능이 vision 과 language signal 의 unified modeling 을 장려하기를 바란다.

Swin Transformer 의 핵심 요소로서, shifted window 기반 self-attention 은 vision 문제에서 효과적이고 efficient 함이 입증되었으며, 저자는 향후 natural language processing 에서의 활용도 탐구하기를 기대한다.
