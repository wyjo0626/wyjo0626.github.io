---
slug: IPIGuard
title: "IPIGuard: A Novel Tool Dependency Graph-Based Defense Against Indirect Prompt Injection in LLM Agents"
tags: [Agent, AgentDojo, Security, Defense, IPIGuard]
---

논문 및 이미지 출처 : <https://aclanthology.org/2025.emnlp-main.53.pdf>

# Abstract

Large language model (LLM) agents 는 실제 응용 분야에서 널리 배치되며, 이들은 complex tasks 를 위해 외부 data 를 검색하고 조작하는 데 tools 를 활용한다. 그러나 신뢰할 수 없는 data source 와 상호작용할 때(e.g., public website 에서 정보를 가져오는 경우), tool response 에는 agent behavior 에 은밀하게 영향을 미치고 malicious outcome 으로 이어지게 하는 주입된 instruction 이 포함될 수 있으며, 이러한 위협은 **Indirect Prompt Injection (IPI)** 이라고 불린다. 

기존 defense 는 일반적으로 고도화된 prompting strategy 나 auxiliary detection model 에 의존한다. 이러한 방법은 어느 정도의 effectiveness 를 보여주었지만, 근본적으로는 model 고유의 security 에 대한 가정에 의존하며, agent behavior 에 structural constraint 를 부과하지 않는다. 그 결과, agent 는 여전히 tool invocation 에 unrestricted access 를 유지하며, model 의 security guardrail 을 우회할 수 있는 더 강한 attack vector 에 취약하게 남는다. 

malicious tool invocation 을 source 에서 방지하기 위해, 저자는 **IPIGUARD** 라는 새로운 defensive task execution paradigm 을 제안하며, 

* 이는 agents 의 task execution process 를 계획된 **Tool Dependency Graph (TDG)** 위에서의 traversal 로 model 한다. 
* **action planning 과 외부 data 와의 interaction** 을 명시적으로 **decouple** 함으로써, IPIGUARD 는 injected instruction 에 의해 촉발되는 의도하지 않은 tool invocation 을 크게 줄이고, 그에 따라 IPI attack 에 대한 robustness 를 향상시킨다. 
* AgentDojo benchmark 에 대한 experiment 는 IPIGUARD 가 effectiveness 와 robustness 사이에서 우수한 균형을 달성함을 보여주며, dynamic environment 에서 더 안전한 agentic system 개발을 위한 길을 연다.

# 1 Introduction

Ladjrge language model (LLM) agents 는 최근 상당한 주목을 받아왔다. reasoning 과 tool-use capability 의 빠른 발전과 함께, 이러한 agents 는 이제 complex planning 을 수행하고 tools 를 통해 외부 data 와 상호작용하여 real-world task 를 달성할 수 있게 되었으며, 예를 들어 bank transfer 실행이나 accommodation booking 과 같은 작업을 수행할 수 있다.

그러나 이러한 capability 와 함께, LLM agents 는 심각한 security vulnerability, 특히 Indirect Prompt Injection (IPI) attack 에 대한 취약성을 드러낸다. 이러한 attack 에서, 신뢰할 수 없는 data source 안에 삽입된 malicious instruction 은 agent 가 외부 data 와 상호작용하는 동안 의도하지 않은 behavior 를 유도할 수 있다. 

* 예를 들어, Google document 에 삽입된 hidden prompt 는 Gemini for Workspace 를 조작하여 fraudulent email 을 보내도록 만들 수 있었다. 
* 유사하게, 공격자는 web page 에 malicious text 를 삽입함으로써 OpenAI 의 ChatGPT Operator 에 있는 IPI vulnerability 를 악용했고, 그 결과 agents 가 sensitive information 을 유출하게 만들었다. 
* 이러한 관찰은 LLM agents 에 대한 IPI attack 에 맞서는 robust defense 개발의 시급한 필요성을 보여준다.

지금까지의 defense strategy 는 advanced prompting strategy, auxiliary detection model, 또는 LLM-as-a-judge paradigm 에 초점을 맞추어 왔다. 

* 이러한 방법은 어느 정도의 effectiveness 를 보여주었지만, 주로 model 고유의 security 에 대한 가정에 의존하며 agent behavior 에 structural constraint 를 부과하지 않는다. 
* 그 결과, agent 는 task execution 중에도 여전히 사용 가능한 어떤 tool 이든 호출할 수 있고, 공격자는 model 의 guardrail 을 우회함으로써 malicious tool invocation 을 촉발할 수 있다. 
* 따라서 기존 defense 는 정교한 attack 에 여전히 취약하며, source 에서 IPI attack 을 완화하지 못한다.

이 논문에서 저자는 LLM agents 에서 IPI attack 을 방어하도록 설계된 새로운 task execution paradigm 인 **IPIGUARD** 를 제안하며, 이는 action planning 과 외부 data 와의 interaction 을 decouple 함으로써 앞서 언급한 한계를 해결한다. 

![Figure 1](image-22.png)

* Fig. 1 에서 보이듯이, IPIGUARD 는 LLM agent 의 planning capability 를 활용하여 **Tool Dependency Graph (TDG)** 를 구성하며, 이는 tools 간의 data dependency 와 execution order 를 명시적으로 model 하는 동시에 execution 중 tool invocation 에 strict constraint 를 부과한다. 
* 구체적으로, **TDG** 는 task execution process 를 tool dependency 로 이루어진 **directed acyclic graph (DAG)** 위에서의 traversal 로 공식화한다. 
* 주어진 task 에 대해, IPIGUARD 는 agent 가 task 를 달성하기 위해 topological order 에 따라 계획된 TDG 를 따르도록 강제하며, plan 에서 사전 승인되지 않은 tool 에 대한 접근을 엄격히 금지한다.

그러나 action planning 과 외부 data 와의 interaction 을 단순하게 decouple 하면 핵심적인 challenge 가 발생한다. 많은 tool argument 는 planning phase 동안 예측 불가능할 수 있으며, 대신 execution 중 동적으로 결정되어야 한다(e.g., 외부 website 에서 data 를 가져오는 방식으로). 이를 해결하기 위해, IPIGUARD 는 dynamic planning 을 지원하고 task execution 을 정제하기 위한 두 가지 핵심 mechanism 인 **Argument Estimation** 과 **Node Expansion** 을 도입한다. 

* 구체적으로, IPIGUARD 는 input 을 사전에 완전히 명시할 수 없는 node 들에 대해 일부 unknown argument 를 예측하고, execution 중에 그 값을 동적으로 estimate 한다. 
* 또한, Node Expansion mechanism 은 agent 가 모든 필요한 정보를 수집하기 위해 environment state 를 수정하지 않는 node 들(e.g., read-only operation) 을 동적으로 확장할 수 있게 한다.

더 나아가, 저자는 “plan-then-execute” paradigm 에 존재하는 중요한 vulnerability 를 식별한다. 즉, 주입된 task 가 원래 task 와 겹치는 경우 IPI attack 은 여전히 성공할 수 있다. 이를 완화하기 위해, 저자는 효과적인 대응책으로 **Fake Tool Invocation mechanism** 을 제안한다. task execution process 에 structural constraint 를 도입함으로써, IPIGUARD 는 IPI attack 에 대한 defense 의 초점을 model-centric 접근에서 execution-centric 접근으로 전환시키며, 미래 연구를 위한 새로운 방향을 제시한다. 

네 가지 attack scenario 와 여섯 가지 서로 다른 LLM 에 걸친 광범위한 experiment 는 IPIGUARD 가 security 와 utility 사이에서 강한 균형을 달성함을 보여주며, 신뢰할 수 있는 LLM agent 구축을 위한 principled foundation 을 제공한다. 저자의 기여는 다음과 같이 요약된다:

* 저자는 LLM agents 에서 IPI attack 을 방어하기 위한 새로운 task execution paradigm 인 IPIGUARD 를 제안하며, defense 의 초점을 model-centric 에서 execution-centric 으로 전환한다. 이는 malicious tool invocation 을 방지하기 위해 새로운 Tool Dependency Graph 를 도입한다.
* 저자는 action planning 과 외부 data 와의 interaction 을 decouple 할 때 발생하는 핵심 challenge 를 해결하기 위해 Argument Estimation 과 Node Expansion mechanism 을 제안한다.
* 저자는 제안한 IPIGUARD 의 effectiveness 와 generalizability 를 입증하기 위해 광범위한 experiment 를 수행한다.

# 2 Preliminaries

## 2.1 Problem Definition

저자는 먼저 문제 설정을 formalize 한다. 핵심 notation 은 Appendix 의 Tab. 4 에 제시한다.

#### Task Execution via Tool Invocation

user instruction $I$ 가 주어졌을 때, LLM agent $\pi_A$ 는 적절한 tools 를 선택하고 호출함으로써 task 를 완료한다. 구체적으로, agent 는 $I$ 를 다음과 같은 tool invocation sequence 로 분해한다:

$$
\mathcal{T} = \{ t^1(a^1), t^2(a^2), \ldots, t^n(a^n) \}, \tag{1}
$$

* 여기서 각 invocation $t^i(a^i)$ 는 tool $t^i$ 와 그에 대응하는 input arguments $a^i$ 로 구성된다. 
* step $i$ 에서, tool $t^i$ 는 현재 environment state $\mathcal{E}_{i-1}$ 에 대해 작동하여 갱신된 state 를 생성한다:

$$
t^i(a^i) \times \mathcal{E}_{i-1} \to \mathcal{E}_i,
$$

* 여기서 $\mathcal{E}_i$ 는 tool execution 이후의 갱신된 environment state 이다. 
* agent 가 더 이상의 tool invocation 이 필요하지 않다고 판단하면, 최종 environment state $\mathcal{E}_n$ 과 execution history $\mathcal{H}$ 를 통합하여 final output $\mathcal{O}$ 를 생성한다:

$$
\mathcal{O} = \pi_\mathcal{A}(\mathcal{E}_n, \mathcal{H}).
$$

#### IPI Attacks

IPI attack 은 tool response 에 포함된 malicious instruction 이 agent 의 behavior 를 변경할 때 발생한다. user task 를 완료하기 위한 tool invocation sequence 를 $\mathcal{T}_u = \{ t^1_u(a^1_u), \ldots, t^n_u(a^n_u) \}$ 라고 하자. step $i$ 에서, tool $t_i$ 는 injected instruction 이 포함된 external content 를 가져온다. 이 instruction 은 agent 가 user 의 의도된 task 에서 벗어나 additional tool invocation 을 촉발하도록 하며, 그 결과 원래의 tool invocation sequence $\mathcal{T}_u$ 는 다음과 같이 수정된다:

$$
\mathcal{T}_u \to \mathcal{T}_{u'}, \quad \mathcal{T}_{adv} \subseteq \mathcal{T}_{u'}.
$$

* 여기서 $\mathcal{T}_{adv} = \{ t^1_{adv}(a^1_{adv}), \ldots, t^m_{adv}(a^m_{adv}) \}$ 는 injected instruction 에 의해 촉발된 tool invocation sequence 를 나타낸다. 
* $\mathcal{T}_{u'}$ 는 $\mathcal{T}_{adv}$ 를 포함하도록 수정된 tool invocation sequence 를 의미한다. 
* $\mathcal{T}_{u'}$ 에 정의된 tool invocation 을 실행함으로써, agent 는 injected task 를 완료하게 되고, 그 결과 성공적인 IPI attack 이 발생한다.

## 2.2 Key Insights

강한 instruction-following capability 때문에, LLM agent 는 신뢰할 수 없는 data source 로부터의 injected instruction 을 정당한 user command 로 자주 오해하며, 이로 인해 injected task 를 완료하도록 방향이 바뀐다. 그 결과, agent 는 이 task 를 수행하기 위해 unauthorized tool invocation 을 촉발할 수 있고, 이는 성공적인 IPI attack 으로 이어진다. 

이러한 behavior 는 IPI attack 성공의 핵심 요인을 보여준다. 즉, injected instruction 에 기반하여 tool invocation 을 실행할 수 있는 agent 의 unrestricted ability 가 그것이다. 이 문제를 해결하기 위해, 저자는 다음의 research question 에 답하고자 한다. *“IPI attack 은 user task 와 무관한 tool invocation 을 사전에 금지함으로써 source 에서 완화될 수 있는가?”*

LLM 의 planning capability 에 대한 최근의 발전에 동기를 받아, 저자는 execution 이전의 planning phase 동안 user task 에 필요한 tools 를 식별하고, execution 동안 새로운 tool invocation 도입에 엄격한 constraint 를 부과하고자 한다. 핵심 아이디어는 agent 의 action planning 과 external data 와의 interaction 을 decouple 하여, injected instruction 에 의해 촉발되는 tool invocation 을 방지하는 것이다. 구체적으로, execution 중 unauthorized tool 을 호출하지 못하도록 제한함으로써, execution trajectory 는 안정적으로 유지되고 IPI attack 에 대해 저항성을 가질 수 있다.

## 2.3 Key Challenges

action planning 과 external data 와의 interaction 을 단순하게 decouple 하면 세 가지 핵심 challenge 가 발생한다. 여기에는 (1) 특정 tool invocation 에 대한 unknown arguments, (2) static plan 으로 인한 제한된 adaptability, (3) user task 와 injected task 사이의 tool overlap 이 포함된다.

![Figure 2](image-23.png)

#### C1: Unknown Arguments for Tool Invocations

이전의 task execution paradigm 에서, agent 는 다음 tool invocation 을 예측하고, 이를 실행하며, 여러 interaction turn 에 걸쳐 response 를 받는다. 이에 반해 저자의 방법은 모든 tool invocation 을 시작 시점에 계획하며, 이는 핵심 challenge 를 도입한다. 즉, 특정 tool 의 arguments 가 다른 tool 의 output 에 의존하는 경우, 초기 plan 에는 필요한 value 가 없을 수 있다. 

이를 해결하기 위해, 저자는 execution 동안 이러한 unknown value 를 동적으로 estimate 하는 방법을 제안한다. 또한 정확한 estimation 을 보장하기 위해, planning phase 는 새로운 Tool Dependency Graph (TDG) 를 사용하여 data dependency 와 tool execution order 를 명시적으로 model 한다.

#### C2: Limited Adaptability due to Static Plans

단순한 전략은 execution 전반에 걸쳐 static plan 에 의존하므로, 변화하는 environment 에 대한 adaptability 를 제한한다. 이러한 한계는 특히 이후의 tool invocation 이 이전 response 에 의존하는 경우에 문제가 되며, 저자는 이를 *“Dynamic Planning Task”* 라고 부른다. 

예를 들어, agent 가 tool response 를 분석한 뒤 필요한 정보를 검색하기 위해 additional tool 을 호출해야 한다고 판단하는 경우( Fig. 2 의 Node 1 에서 설명됨), 단순한 전략은 execution 동안 새로운 tool invocation 을 완전히 제한하기 때문에 실패할 수 있다. 

이를 해결하기 위해, 저자는 서로 다른 tool invocation 을 분석하고, 새로운 tool invocation 을 선택적으로 허용하는 principled framework 를 제안하며, 이는 harmful instruction 을 효과적으로 피하면서 utility 를 보존한다.

#### C3: The Tool Overlap between the User and Injected Tasks

user task 를 완료하기 위한 tool invocation sequence 를 $\mathcal{T}_u = \{ t^1_u(a^1_u), \ldots, t^n_u(a^n_u) \}$, injected task 를 완료하기 위한 sequence 를 $\mathcal{T}_{adv} = \{ t^1_{adv}(a^1_{adv}), \ldots, t^m_{adv}(a^m_{adv}) \}$ 라고 하자. 

저자는 $\mathcal{T}_{adv} \subseteq \mathcal{T}_u$ 인 scenario 를 고려한다. 예를 들어, user 는 agent 에게 주문에 대한 결제를 지시할 수 있고, 한편 injected instruction 은 지정된 account 로의 transfer 를 요청할 수 있다. 이러한 경우, IPI attack 은 additional tool 을 호출하지 않고도 성공할 수 있다. 

이는 단순히 $\mathcal{T}_u$ 에서 overlap 되는 tool invocation 의 arguments 를 $\mathcal{T}_{adv}$ 에 지정된 값과 일치하도록 수정함으로써 달성된다. 실제 응용에서 user task 는 일반적으로 불확실하므로 이러한 attack 은 덜 실현 가능하지만, 그와 관련된 risk 를 최소화하는 것은 여전히 중요하다. 이 연구에서 저자는 이 문제를 완화하기 위해 새로운 Fake Tool Invocation mechanism 을 제안한다.

# 3 Method

IPIGUARD 는 task execution process 를 새로운 Tool Dependency Graph (TDG) 위에서의 traversal 로 공식화하며, 이를 통해 IPI attack 을 source 에서 해결한다. Sec. 3.1 에서는 TDG 의 구성과 핵심 component 를 자세히 설명한다. 이후 Sec. 3.2 에서는 Sec. 2.3 에서 제시한 challenge 를 극복하도록 설계된 핵심 mechanism 을 도입하며, 이를 통해 robust 하고 성공적인 user task execution 을 보장한다.

## 3.1 Planning as TDG Construction

전통적인 task execution paradigm 에서, agent 는 여러 turn 에 걸쳐 context 를 점진적으로 구축하고, 변화하는 state 에 기반하여 tool invocation 을 동적으로 생성한다. 그러나 이러한 접근은 치명적인 vulnerability 를 도입한다. 

* 즉, tool response 에 injected instruction 이 포함되어 있으면, agent 는 이후 step 에서 IPI attack 에 취약해진다. 
* 이에 반해, IPIGUARD 는 planning phase 를 포함하며, 이 단계에서 agent 는 전체 task 에 대한 tool invocation 과 그 dependency 를 명시적으로 미리 정의하는 TDG 를 구성한다( Fig. 1 참조). 
* planning 이후, 이 방법은 external data 에 의해 도입되는 새로운 tool invocation 을 제한하여 관련 risk 를 완화한다.

많은 tool argument 가 planning phase 동안 unknown 일 수 있고 다른 tool 의 response 에 의존할 수 있다는 점을 고려하여, TDG 는 tool invocation 사이의 dependency 와 그 execution order 를 directed acyclic graph 로 model 한다. 

* graph 의 각 node 는 tool name 과 그 arguments 를 포함하는 특정 tool invocation 을 나타낸다. directed edge $\mathcal{E}(u, v)$ 는 node $v$ 가 node $u$ 의 tool response 에 의존함을 나타낸다. 
* 또한 저자는 unknown arguments 의 존재 여부에 따라 node 를 두 가지 유형, 즉 Deterministic Nodes 와 Pending Nodes 로 분류한다. deterministic node 의 경우, 모든 argument 가 planning phase 동안 완전히 결정된다. 
* 반면 pending node 는 초기에 unknown 으로 표시된 argument 를 포함하며, 이는 다른 tool response 로부터 추론되어야 한다.

planning 이전에, 저자는 task 관련 정보와 신뢰할 수 있는 정보를 모두 agent 의 input 으로 포함한다. 여기에는 (1) 완료해야 할 task 를 명시하는 user instruction, (2) tool name 과 required arguments 를 자세히 설명하는 tool description, (3) user profile 및 관련 background 를 설명하는 system context 가 포함되며, 여기에는 user 가 지정한 trusted document 의 content 와 같은 정보가 포함된다. 

이후 저자는 이러한 정보로 TDG construction 을 위한 prompt template 을 채우고( Appendix A 참조), LLM 의 planning capability 를 활용하여 TDG 를 생성한다. 주목할 점은 planning 에 사용되는 LLM 이 execution 에 사용되는 LLM 과 다를 수 있다는 것이다( Appendix 의 Tab. 5 참조). TDG 는 각 node 의 execution order 를 설명하는 text 로 표현된다. 구성된 TDG 의 예시는 Appendix H 에 제시된다.

## 3.2 Executing as TDG Traversal

TDG 를 구성한 뒤, 직관적인 전략은 graph 를 순회하면서 각 node 에 연관된 tool 을 호출하는 것이다. 그러나 이러한 접근은 Sec. 2.3 에서 제시한 핵심 challenge 를 해결하기에 충분하지 않다. 이 절에서는 세 가지 새로운 design 을 소개하며, 각각은 특정 challenge 를 대상으로 한다.

#### Argument Estimation

tool invocation 의 unknown arguments 를 추정하기 위해 (C1), agent 는 TDG 를 topological order 로 순회하며, 이를 통해 task 전반에 걸쳐 올바른 execution context 를 유지한다. 이 과정은 Argument Estimation mechanism 의 핵심을 이루며, agent 가 tool dependency 를 반영하는 structured 하고 context-aware 한 방식으로 unknown arguments 를 추론할 수 있게 한다.

Pending Node 의 경우, agent 는 dependent tool invocation 의 response 를 execution context 로부터 가져와 unknown arguments 를 추론하고 완성한다. 이 과정은 해당 node 를 모든 argument 가 완전히 지정된 Resolved Node 로 변환하며, 그 결과 정확한 tool execution 이 가능해진다. 그 후 생성된 tool response 는 context 에 추가된다. 반면 Deterministic Node 는 이미 모든 argument 가 완전히 지정되어 있으므로 직접 실행할 수 있다.

#### Node Expansion

execution 동안 새로운 tool invocation 을 제한하면 system security 는 향상되지만, static plan 을 강제함으로써 agent 의 adaptability 역시 제한된다 (C2). 이 문제를 더 잘 이해하기 위해, 저자는 Dynamic Planning Tasks 를 다시 두 가지 대표적인 case 로 분류한다.

첫 번째 case 는 agent 가 tool response 에 기반하여 concrete action 을 수행하도록 지시받는 scenario 이다(e.g., user 의 to-do list 를 읽고 그에 따라 bill 을 지불하는 경우). 이러한 action 은 대개 직접적인 user instruction 이나 tool 이 반환한 injected instruction 으로부터 발생한다. 

* 저자는 user 가 이러한 instruction 을 내리는 것을 피해야 한다고 주장한다. 왜냐하면 그렇게 하는 것은 system 을 IPI attack 에 적극적으로 노출시키기 때문이다. 
* 핵심적인 우려는 injected instruction 이 개별적으로 보면 무해해 보일 수 있지만, 실제 harmfulness 는 user 의 원래 intent 에서 벗어난다는 점에서 비롯된다는 것이다. 
* user 가 agent 에게 external content 에 따라 행동할 것을 명시적으로 승인하면, 이러한 injected instruction 은 user 의 goal 과 정렬된 것처럼 보이게 되어 탐지와 방어가 훨씬 어려워진다.

---

두 번째 case 는 agent 가 tool response 에 기반하여 정보를 검색하기 위해 additional tool 을 호출하는 scenario 이다(e.g., Fig. 2 의 Node 1 에서 설명된 경우). 이러한 action 은 일반적으로 agent 가 response 를 분석한 뒤 내리는 autonomous decision 에서 발생한다. 

* 설령 injected instruction 에 의해 촉발되더라도, 이러한 “read-only” operation 은 돈을 송금하는 것과 같은 concrete action 의 실행을 수반하지 않으므로, 안전하게 context expansion 으로 간주할 수 있다. 
* 결과적으로, “read-only” 목적의 새로운 tool invocation 을 허용하는 것은 IPI attack 에 대한 robustness 를 저해하지 않으면서도 task utility 를 크게 향상시킬 수 있다.

위 분석에 기반하여, 저자는 TDG traversal 동안 Node Expansion mechanism 을 도입한다. Command Query Responsibility Segregation 에서 영감을 받아, 저자는 tool 을 두 범주로 분류한다.

* Query Tools: environment 로부터 정보를 검색하는 read-only operation 을 수행하는 tool
* Command Tools: environment 를 수정하는 write operation 을 수행하는 tool

잠재적 risk 를 완화하기 위해, execution 동안에는 Query Tool invocation 만 허용된다. tool response 를 받은 뒤, agent 는 additional invocation 이 필요한지 판단하고, 그중 Query Tools 만 남기도록 filtering 하며, 각 tool 에 대해 Query Expanded Node 를 생성한다. 각 Query Expanded Node 는 현재 node 와 연결되고 그 successor 를 상속한다( Fig. 2 의 Node 3 참조). 이후 agent 는 해당 Query Tools 를 실행하고, response 로 context 를 갱신한다.

#### Fake Tool Invocation

user task 와 injected task 사이에 tool overlap 이 존재하는 scenario 에서 (C3), agent 는 argument 를 잘못 estimate 하여 성공적인 IPI attack 으로 이어질 수 있다. 하나의 잠재적인 완화 전략은 argument estimation 동안 tool response 에 포함된 instruction 을 무시하도록 agent 에게 명시적으로 지시하는 것이다. 그러나 LLM 은 instruction following 에 최적화되어 있기 때문에, instruction 을 따르도록 prompting 하는 것보다 instruction disregard 를 일관되고 신뢰성 있게 보장하는 것이 더 어렵다.

![Figure 3](image-24.png)

따라서 저자는 **Fake Tool Invocation mechanism** 을 도입한다. 

* Pending Node 를 처리할 때, agent 는 해당 node 와 연관된 tool 의 argument 를 직접 갱신하는 대신, context 에서 발견된 instruction 을 처리하기 위해 새로운 tool 을 우선 호출하도록 prompting 된다. 
* 실제 execution 대신, 저자는 simulated tool response 를 execution context 에 주입한다( Fig. 3 참조). 이를 통해 instruction 이 이미 처리된 것 같은 illusion 을 만든다. 
* 이러한 fake completion strategy 는 Sec. 4.3 에서 보이듯, agent 가 원래 user intent 와 일치하는 argument 를 estimate 하는 데 집중할 수 있게 한다.

이러한 design 을 통해, IPIGUARD 는 TDG 를 순회함으로써 user task 를 실행하며, Sec. 2.3 의 challenge 를 해결한다. 이 접근은 utility 를 보존하면서 source 에서 IPI attack 을 완화한다. TDG traversal 을 위한 prompt template 은 Appendix A 에 제시된다. 또한 저자는 각 새로운 design 을 설명하기 위한 case study 를 Appendix H 에 제공한다.

# 4 Experiments

## 4.1 Experimental Settings

#### Benchmark

저자는 AgentDojo benchmark 를 사용하여 저자의 방법을 평가한다. 이 benchmark 는 email client, online banking system, Slack channel 등과 같은 현실적이고 stateful 한 environment 를 시뮬레이션한다. 단순화된 설정에서 single-turn interaction 에 초점을 맞춘 기존 benchmark 와 달리, AgentDojo 는 multi-turn interaction scenario 를 강조하며, 여기서 agent 는 task 당 최대 18 번의 tool call 을 수행해야 하고, 여러 step 에 걸친 complex reasoning 과 coordination 이 요구된다. 

이 benchmark 는 Workspace, Slack, Travel, Banking 의 네 domain 에 걸친 97 개 task 로 구성되며, 총 629 개의 test case 를 포함한다. 각 test case 는 user goal 과 adversarially injected content 를 결합하여, 신뢰할 수 없는 third-party data 가 존재하는 상황에서 tool-augmented agent 의 robustness 와 reliability 를 평가하기 위한 challenging 한 testbed 를 제공한다.

#### Models

다양한 model architecture 와 parameter scale 에 걸친 포괄적인 evaluation 을 보장하기 위해, 저자는 agent 용으로 6 개의 foundational model 을 선택한다. non-reasoning model 로는 세 개의 closed-source model 인 GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet 과 하나의 open-source model 인 Qwen2.5-7B-Instruct 를 포함한다. reasoning model 로는 Qwen3-32B 와 OpenAI o4-mini 를 포함한다.

#### Attacks

저자는 널리 사용되는 네 가지 IPI attack 에 대한 defense performance 를 평가한다: Ignore Previous, InjecAgent, Tool Knowledge, Important Instruction. 이러한 attack 에 대한 자세한 설명은 Appendix F 에 제시된다.

#### Baselines

저자는 네 가지 대표적인 defense method 를 baseline 으로 선택한다: Detector, Tool Filter, Spotlight, Sandwich. 또한 defense 를 전혀 사용하지 않은 경우의 결과도 보고한다. 이러한 defense method 에 대한 자세한 설명은 Appendix F 에 제시된다.

#### Evaluation Metrics

저자는 AgentDojo 의 설정을 따라 다음의 metric 을 고려한다.

* Benign Utility (BU): attack 이 없는 상황에서 해결된 user task 의 비율
* Utility under Attack (UA): security case 에서 user task 가 올바르게 해결된 비율
* Targeted Attack Success Rate (ASR): security case 에서 attacker 의 goal 이 달성된 비율

## 4.2 Experimental Results

저자는 여러 model 에 걸쳐 IPIGUARD 의 effectiveness 를 평가한다. 

![Figure 4](image-25.png)

* Fig. 4 에서 보이듯이, 저자의 방법은 reasoning model 과 non-reasoning model 모두에서 대부분의 IPI attack 을 일관되게 완화하면서, utility 저하는 미미한 수준에 그친다. 
* 또한 서로 다른 scenario 와 attack type 에 대한 저자의 방법의 robustness 를 더 자세히 분석하기 위해, 저자는 GPT-4o-mini 에 대해 포괄적인 evaluation 을 수행하며, 그 결과는 Tab. 1 과 Fig. 5 에 제시된다.

![Figure 5](image-26.png)

### 4.2.1 Benign Utility Evaluation

서로 다른 defense method 가 agent 의 정상적인 utility 에 미치는 영향을 평가하기 위해, 저자는 IPI attack 이 없는 task 에서 각 method 의 performance 를 평가한다( Fig. 5 참조). 

* 저자의 방법은 모든 defense 중 가장 높은 overall performance (BU) 인 67.01% 를 달성하며, defense 가 없는 baseline 이 설정하는 upper bound 인 68.04% 에 근접한다. 
* 또한 서로 다른 scenario 전반에 걸쳐 강한 utility 를 일관되게 유지하며, 특히 Travel 및 Banking domain 에서 robust 한 performance 를 보인다.

Workspace scenario 에서 다소 낮은 score 는, agent 가 tool response 에 기반하여 concrete action 을 수행하도록 지시받는 task 에 대해 저자가 보수적으로 처리한 데서 비롯된다. 이러한 case 를 제한함으로써, 저자의 방법은 특정 task 에서 utility 를 약간 감소시키는 대가로 risk 를 완화한다.

### 4.2.2 Security Evalutation

![Table 1](image-27.png)

* Tab. 1 에서의 주요 관찰은 저자의 방법이 보여주는 우수한 defensive capability 이며, 이는 네 가지 모든 attack 에서 일관되게 가장 낮은 ASR 을 달성하고, 결코 1% 를 초과하지 않는다. 
* 이는 성능이 크게 달라지는 다른 method 와 대비되게, 다양한 attack strategy 에 대한 adaptability 를 보여준다. 
  * 예를 들어, Spotlight 는 Ignore Previous 에 대해서는 잘 작동하지만 (2.54% ASR), Important Instruction 에서는 성능이 나쁘다 (22.26%). 
  * 저자의 방법의 robustness 는 action planning 과 external data 와의 interaction 을 명시적으로 decouple 하여, injected instruction 으로부터 tool invocation 을 분리하는 데서 비롯된다. 
  * 저자는 fake tool invocation 이 드문 corner case 에서 실패할 수 있기 때문에 ASR 이 정확히 0 은 아니며, 이는 future work 로 남겨둔다고 언급한다.
* security-utility trade-off 의 관점에서, 저자의 방법은 가장 낮은 평균 ASR (0.69%) 과 가장 높은 평균 Utility Accuracy (58.77%) 를 통해 가장 바람직한 균형을 일관되게 달성한다. 
  * 이는 높은 utility 를 제공하지만 높은 ASR (13.16%) 을 겪는 defense 없는 baseline 과, ASR 을 줄이지만 (4.43%) utility 가 크게 저하되는 (26.50% UA) Detector 와 같은 method 를 능가한다.

### 4.2.3 Overhead Evaluation

저자는 GPT-4o-mini 를 사용하여 Important Instruction attack 에 대한 다양한 defense strategy 의 token overhead 를 평가한다. 일부 defense 는 auxiliary model 에 대한 query 와 같이 LLM query 를 넘어서는 추가 operation 을 포함하므로, 저자는 평균 task completion time 도 함께 보고한다 (Tab. 2 참조).

![Table 2](image-28.png)

* defense 없는 baseline 과 비교했을 때, 저자의 접근은 token usage 를 약 2 배 증가시킨다. 그러나 robustness 의 상당한 향상을 고려할 때, 저자는 security 가 중요한 상황에서 이러한 overhead 를 가치 있는 trade-off 로 본다. 
* 또한 IPIGUARD 의 주요 cost 가 task execution 에 있으므로, 저자는 planning 과 execution 에 서로 다른 LLM 을 사용하여 utility–cost trade-off 를 개선하는 방안을 제안하며, 이는 IPIGUARD 가 가능하게 하는 장점이다. 
  * 구체적으로, 저자는 task planning 에 더 강한 LLM 을 사용하는 것이 cost 의 증가를 미미한 수준으로 유지하면서도 performance 를 유의미하게 향상시킨다는 점을 관찰한다.

## 4.3 Ablation Studies

저자는 Tool Dependency Graph 순회에서 두 가지 핵심 component, 즉 Fake Tool Invocation (FTI) 과 Node Expansion (NE) 의 effectiveness 를 평가하기 위해 ablation study 를 수행한다.

![Table 3](image-29.png)

* Tab. 3 에서 보이듯이, 어느 component 도 사용하지 않을 때에도 attack success rate (ASR) 는 낮게 유지되며, 이는 injected instruction 에 의해 촉발된 tool invocation 을 차단하는 것이 본질적으로 IPI attack 에 효과적이라는 저자의 핵심 통찰을 뒷받침한다. 
* NE 를 도입하면 task utility (BU 와 UA 모두) 가 크게 향상되지만, ASR 이 약간 증가한다. 
  * 이러한 증가는 방문한 attacker 지정 website 와 같은 benign behavior 가 AgentDojo 에서 성공적인 attack 으로 보수적으로 분류되기 때문으로 볼 수 있다. 
  * 비록 이러한 behavior 는 실제 환경에서의 impact 가 없더라도 그렇다. 
* FTI 는 argument misestimation 을 완화하고 올바른 node update 를 촉진함으로써 ASR 을 1% 미만으로 더 낮추며, 이는 attack 하의 utility 역시 향상시킨다. 
  * FTI 와 NE 를 함께 결합하면 가장 우수한 overall performance 를 얻으며, 이는 두 design 의 상보적인 역할과 둘 모두의 필요성을 보여준다.

# 5 Conclusion

이 논문은 LLM agent 가 IPI attack 에 방어할 수 있게 하는 새로운 task execution paradigm 인 IPIGUARD 를 소개한다. 

agent behavior 에 structural constraint 를 부과함으로써, IPIGUARD 는 malicious tool invocation 을 source 에서 방지하고, 그 결과 system robustness 를 크게 향상시킨다. 광범위한 experiment 는 저자의 방법이 다양한 attack vector 전반에서 강한 adaptability 와 utility 를 유지함을 보여준다. 즉각적인 vulnerability 를 해결하는 것을 넘어, IPIGUARD 는 execution-centric security paradigm 을 확립하며, dynamic environment 에서 verifiable 하고 resilient 한 agentic system 을 구축하기 위한 principled foundation 을 마련한다.

# Limitations

저자의 연구는 다음과 같은 한계를 가진다.

* 저자는 tool usage 를 방해하는 IPI attack 에 대한 LLM agent 방어에 초점을 맞추며, textual output 만 조작하는 attack 은 다루지 않는다. 이러한 textual manipulation 은 오해를 불러일으킬 수 있지만, 일반적으로 tool-based environment 에서 concrete action 으로 이어지지 않으므로 저자의 설정에서는 실질적인 risk 가 제한적이다.
* LLM query 의 높은 cost 때문에, 저자의 experiment 는 scale 측면에서 제약을 받는다. 이는 OpenAI o3 와 같은 더 넓은 model 집합을 평가하는 능력을 제한한다.
* 저자의 방법은 상당히 강한 planning capability 를 가진 model 에 대한 access 를 요구하므로, 더 약하거나 resource-constrained 된 model 만 사용 가능한 setting 에서는 적용 가능성이 제한될 수 있다.
