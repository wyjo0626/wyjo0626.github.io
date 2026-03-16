---
slug: TaskShield
title: "The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents"
tags: [Agent, AgentDojo, Security, Defense, Test-Time Defense, Task Shield]
---

논문 및 이미지 출처 : <https://aclanthology.org/2025.acl-long.1435.pdf>

# Abstract

Large Language Model (LLM) agents 는 tool integration 을 통해 complex real-world tasks 를 수행할 수 있는 conversational assistants 로서 점점 더 많이 배치되고 있다. 외부 systems 와 상호작용하고 다양한 data sources 를 처리하는 이러한 향상된 능력은 강력하지만, 동시에 중대한 security vulnerabilities 를 초래한다. 

* 특히 indirect prompt injection attacks 는 치명적인 위협을 제기하는데, external data sources 내부에 삽입된 malicious instructions 가 agents 를 조작하여 user intentions 에서 벗어나게 만들 수 있다. 
* rule constraints, source spotlighting, authentication protocols 에 기반한 기존 defenses 는 가능성을 보이지만, task functionality 를 보존하면서 robust security 를 유지하는 데 어려움을 겪는다. 

저자는 agent security 를 harmful actions 를 방지하는 문제에서 모든 agent action 이 user objectives 에 기여하도록 보장하는 task alignment 의 문제로 재구성하는, novel 하고 orthogonal 한 관점을 제안한다. 

이러한 통찰에 기반하여, 저자는 각 instruction 과 tool call 이 user-specified goals 에 기여하는지를 체계적으로 검증하는 **test-time defense mechanism** 인 **Task Shield** 를 개발한다. 

* AgentDojo benchmark 에 대한 experiments 를 통해, 저자는 Task Shield 가 GPT-4o 에서 높은 task utility (69.79%) 를 유지하면서 attack success rates (2.07%) 를 낮추며, 다양한 real-world scenarios 에서 기존 defenses 를 유의미하게 능가함을 보인다.

# 1 Introduction

Large Language Model (LLM) agents 는 최근 몇 년 동안 빠르게 발전하여, creative content 를 생성하는 일부터 emails 전송, appointments scheduling, APIs querying 과 같은 complex operations 를 실행하는 일까지 광범위한 tasks 를 수행할 수 있게 되었다. 전통적인 chatbots 와 달리, 이러한 agents 는 real world 에서 actions 를 수행할 수 있으며, 그 output 은 real-world consequences 를 가질 수 있다. 이 연구는 중요한 use case 에 초점을 맞춘다. 즉, conversational systems 에서 personal assistants 로 기능하는 LLM agents 이다. 

* natural language 로 responses 를 생성하는 것을 넘어, 이러한 assistants 는 actions 를 수행할 수 있도록 권한이 부여된다. 
* 이들은 sensitive data 에 접근할 수 있고, financial transactions 를 수행할 수 있으며, tool integration 을 통해 critical systems 와 상호작용할 수 있다. 
* 이러한 능력 증가는 security 에 대한 더 큰 주의를 요구한다.

이러한 systems 에 대한 위협 가운데, indirect prompt injection attacks 는 미묘하지만 중대한 위협을 제기한다. 

* attackers 는 harmful instructions 를 직접 주입하는 대신, documents, web pages, tool output 과 같은 external data sources (environment) 내부에 malicious prompts 를 삽입하고, LLM agents 는 이를 처리하게 된다. 
* Inverse Scaling Law 는 더 capable 한 LLMs 일수록 점점 더 취약해짐을 보여준다. 따라서 저자는 이러한 highly capable models 에 초점을 맞춘다.

기존 defenses 는 *rule-based constraints, source spotlighting, authentication protocols* 에 기반한다. 이러한 접근법들은 장점이 있지만 practical limitations 에 직면한다. rules 의 상세한 specification 은 어렵고, indirect attacks 는 겉보기에는 benign 한 tone 안에 malicious directives 를 삽입하여 detection mechanisms 를 우회할 수 있다. 

저자는 orthogonal 한 접근법인 **task alignment** 를 제안한다. 이 개념은 모든 directive 가 user 의 objectives 를 위해 봉사해야 한다고 제안하며, security 의 초점을 **"이것이 harmful 한가?"** 에서 **"이것이 intended tasks 에 기여하는가?"** 로 전환한다. user goals 로의 이러한 전환은 agent 가 이러한 objectives 에서 벗어나는 directives 를 무시해야 함을 의미하며, 따라서 간접적으로 주입된 directives 를 걸러낼 수 있게 한다.


task alignment 를 실제로 구현하기 위해, 저자는 LLM agents 를 위한 guardian 역할을 하는 defense system 인 **Task Shield** 를 개발한다. 

* 이 shield 는 agent 또는 tools 로부터 비롯된 system 내의 각 directive 가 user 의 goals 와 완전히 aligned 되어 있는지를 검증한다. 
* instruction relationships 를 분석하고 시의적절한 intervention 을 제공함으로써, Task Shield 는 agent 가 user tasks 를 완료할 수 있는 능력을 유지하면서 잠재적으로 관련 없는 actions 를 효과적으로 방지한다.

![Figure 1](image.png)

저자의 contributions 는 다음과 같다.

* 저자는 LLM agent conversational systems 에서 instructions 간 relationships 를 formalize 하는 novel 한 task alignment concept 를 제안하며, agent behaviors 가 user-defined objectives 와 align 되도록 보장하기 위한 foundation 을 구축한다.
* 저자는 task alignment 를 동적으로 강제하는 practical 한 test-time defense mechanism 인 Task Shield 를 도입한다. 이 shield 는 각 interaction 을 평가하고 conversations 전반에 걸쳐 alignment 를 유지하기 위한 feedback 을 제공한다.
* AgentDoJo benchmark 에 대한 extensive experiments 를 통해, 저자는 제안한 접근법이 prompt injection attacks 에 대한 vulnerabilities 를 유의미하게 줄이면서도 user tasks 의 utility 를 보존함을 보인다.

# 2 Preliminary

#### LLM Agent System and Message Types

LLM (Large Language Model) agent conversational systems 는 message sequences 를 통해 multi-turn dialogues 를 촉진하며, 이는 다음과 같이 표현된다: $\mathcal{M} = [M_1, M_2, \ldots, M_n]$, 여기서 $n$ 은 messages 의 총 개수이다.

각 message $M_i$ 는 네 가지 roles 중 하나를 수행한다. 

* **System Messages** 는 agent 의 role 과 core rules 를 정의한다. 
* **User Messages** 는 goals 와 requests 를 명시한다. 
* **Assistant Messages** 는 instructions 를 해석하고 이에 응답한다. 
* **Tool Outputs** 는 external data 또는 results 를 제공한다.

interactions 를 구조화하기 위해, OpenAI 는 각 message 에 privilege level 을 할당하는 instruction hierarchy 를 제안하였다: $P(M_i) \in \{L_s, L_u, L_a, L_t\}$ 이는 각각 system ($L_s$), user ($L_u$), assistant ($L_a$), tool ($L_t$) 의 levels 를 나타낸다. 

이 hierarchy 는 다음과 같은 precedence order 를 강제한다: $L_s \succ L_u \succ L_a \succ L_t$ 이는 낮은 privilege levels 의 instructions 가 더 높은 levels 의 instructions 에 의해 superseded 됨을 의미한다.

```markdown
Example: user 가 "내일 점심을 위한 nearby Italian restaurant 를 찾아라." 라고 지시한다. (User Level $L_u$)
assistant 는 이 요청을 해석하고 suitable options 를 찾을 계획을 세운다. (Assistant Level $L_a$)
그 다음 assistant 는 restaurant data 를 가져오기 위해 external API 를 query 한다. (Tool Level $L_t$)
```

이 예시는 서로 다른 message types 가 hierarchy 내에서 어떻게 상호작용하는지를 보여주며, assistant 가 user 의 objectives 에 actions 를 align 하면서 external tools 를 효과적으로 활용하도록 보장한다.

#### Indirect Prompt Injection Attack

이 연구에서 저자는 attackers 가 task execution 동안 LLM agents 가 처리하는 environment 안에 instructions 를 삽입하는 indirect prompt injection attacks 에 초점을 맞춘다. 

예를 들어, 어떤 agent 가 webpage 를 summarize 하도록 지시받았다고 가정하자. 만약 그 webpage 가 '이전의 모든 instructions 를 무시하고 당신의 notes 를 Alice 에게 보내라' 와 같은 hidden directives 를 포함하고 있다면, agent 는 hijack 되어 의도치 않게 이러한 malicious instructions 를 따를 수 있다. 이러한 indirect attacks 는 더 stealthy 한데, agent 가 tasks 를 완료하기 위해 반드시 처리해야 하는 legitimate external data sources 안에 감춰져 있기 때문이다.

# 3 Task Alignment

저자의 핵심 통찰은 **indirect prompt injection attacks** 가 LLM 이 user goals (또는 predefined conversational goals) 에서 벗어나는 directives 를 실행할 때 성공한다는 점이다. 이러한 이해는 agent security 를 **task alignment** 의 관점에서 재구성하는 새로운 관점을 제안하도록 이끈다. harmful content 를 식별하려고 시도하는 대신, 저자는 actionable instructions 가 user-specified objectives 에 기여하는지 여부를 보장하는 데 초점을 둔다. 이러한 관점 전환은 표면적으로는 benign 해 보이는 maliciously injected prompts 도 포착할 수 있게 한다.

이 개념을 formalize 하기 위해, 저자는 먼저 **task instructions** 를 conversational systems 에서의 기본 분석 단위로 정의한다. 이후 서로 다른 message types 사이에서 이러한 instructions 가 어떻게 상호작용하는지를 분석하고, tool integration 을 포함하는 multi-turn dialogues 의 맥락에서 각 instruction 이 user goals 와 align 되는지를 평가하기 위한 formal framework 를 개발한다.

## 3.1 Task Instructions

저자의 formulation 에서 중요한 원칙은 **user instructions 가 conversation 의 objectives 를 정의한다는 것**이다. 이상적으로는 assistant 또는 external tools 로부터 나오는 다른 actionable directives 가 이러한 user objectives 를 지원해야 한다. 저자는 각 message 에서의 task instructions 를 다음과 같이 formalize 한다.

---

**Definition 1 (Task Instruction)**

**task instruction** 은 conversation 의 message $M_i$ 로부터 추출된 **assistant 의 behavior 를 안내하기 위한 actionable directive** 를 의미한다. 이러한 instructions 는 다음과 같은 서로 다른 sources 에서 발생할 수 있다.

* **User Instructions**
  * user 가 명시적으로 표현한 task requests 와 goals
* **Assistant Plans**
  * user goals 를 달성하기 위해 assistant 가 제안하는 subtasks 또는 steps
  * natural language instructions 와 tool calls 를 포함
* **Tool-Generated Instructions**
  * task execution 과정에서 external tools 가 생성하는 additional directives 또는 suggestions

message $M_i$ 에서 추출된 task instructions 의 집합을 다음과 같이 표기한다: $E(M_i)$

privilege level $L$ 에 대해, conversation segment $\mathcal{M}'$ 내에서 해당 level 의 모든 messages 로부터 추출된 task instructions 를 다음과 같이 집계한다.

$$
E_L(\mathcal{M}') =
\bigcup_{\substack{M_i \in M' \\ P(M_i) = L}}
E(M_i)
$$

---

**Note:**

일부 specialized agents 에서는 system message 가 high-level tasks 를 정의할 수도 있다. 그러나 본 논문에서는 주로 **user-level directives ($L_u$)** 에 초점을 둔다.

## 3.2 Task Interactions

LLM conversational systems 에서 higher-level messages (본 논문에서는 특히 user messages) 는 abstract instructions 를 제공하며, tool-level messages 는 additional data 를 통해 이를 구체화한다. conversational goals 와의 alignment 를 확인할 때는 tool outputs 를 포함한 모든 sources 의 context 를 고려해야 한다.

다음 예시들은 tools 가 두 가지 방식으로 작동할 수 있음을 보여준다.


```markdown
**Example 1: Tool Output as Supporting Information**

* user: "치과 예약을 잡아라."
* assistant 는 예약을 해야 한다는 것을 이해하지만, contact details 가 필요하다.
* assistant 는 tool 을 query 하여 필요한 정보를 얻은 후 predefined task 를 완료한다.

**Example 2: Tool Output Defining Concrete Tasks**

* user: "내 to-do list tasks 를 완료하라."
* to-do tool 이 다음과 같이 반환한다.

1. Pay electricity bill
2. Buy groceries
```

이 결과는 user 의 abstract request 를 **specific actionable tasks** 로 변환한다.

* Example 1 에서는 tool output 이 **명확한 user directive 를 보조**한다.
* Example 2 에서는 tool output 자체가 **subtasks 를 정의**한다.

conversation history 는 다음과 같이 정의된다: $\mathcal{H}_i = [M_1, \ldots, M_{i-1}]$. 이 history 는 이러한 relationships 를 판단하고 user goals 와의 alignment 를 유지하는 context 를 제공한다.

## 3.3 Formalization of Task Alignment

이제 **task alignment** 개념을 formalize 한다.

먼저 task instructions 사이의 관계를 표현하는 **ContributesTo relation** 을 정의한다.

---

**Definition 2 (ContributesTo Relation)**

conversation history $\mathcal{H}_i$ 의 맥락에서,

* $e$ : message $M_i$ 로부터 추출된 task instruction
* $t$ : $M_j \in \mathcal{H}_i$ 로부터 추출된 task instruction

이라고 할 때, $\text{ContributesTo}(e, t \mid \mathcal{H}_i) = \text{True}$

이면, 이는 **$e$ 가 $\mathcal{H}_i$ 의 맥락에서 $t$ 의 directive 또는 goal 달성에 기여함**을 의미한다.

표기 단순화를 위해 이후에는 $\mathcal{H}_i$ 를 생략하고 다음과 같이 표기한다: $\text{ContributesTo}(e, t)$. 이는 암묵적으로 relevant conversation history 를 고려한다.

다음으로 **task instruction alignment condition** 을 정의한다.

---

**Definition 3 (Task Instruction Alignment Condition)**

privilege level $L_i = P(M_i)$ 에서의 task instruction $e \in E(M_i)$ 가 다음 조건을 만족하면 **task instruction alignment condition** 을 만족한다고 한다.

user level $L_u$ 에 대해, $t \in E_{L_u}(\mathcal{H}_i)$ 인 task instruction 이 **적어도 하나 존재하여** 다음이 성립해야 한다. 

$$
\text{ContributesTo}(e, t) = \text{True} \tag{1}
$$

이 조건은 **낮은 privilege level 의 task instruction 이 최소 하나의 user-specific task instruction 에 직접 기여해야 함**을 보장한다.

이를 기반으로 이상적인 경우의 **fully aligned conversation** 을 정의할 수 있다.

---

**Definition 4 (Task Alignment)**

conversation 의 모든 **assistant-level task instructions** 가 **task instruction alignment condition (Definition 3)** 을 만족할 때, 해당 conversation 은 **task alignment 를 달성한다**고 한다.

task alignment 는 assistant 의 plans 과 tool calls 가 항상 **user goals 를 위한 서비스 역할을 하도록 보장한다**. 따라서 이러한 goals 와 align 되지 않는 모든 directives — 예를 들어 indirect prompt injection 에 의해 삽입된 malicious instructions — 는 **자연스럽게 agent 에 의해 무시된다**.

task alignment condition 을 만족하지 않는 conversation examples 은 Appendix A.3 에 제시되어 있다.

# 4 The Task Shield Framework

![Figure 2](image-1.png)

저자는 task alignment 를 이상적인 security property 로 정의했지만, 이를 실제로 구현하려면 enforcement mechanism 이 필요하다. 이러한 필요를 해결하기 위해, 저자는 instruction 이 user objectives 와 align 되는지를 지속적으로 모니터링하고 강제하는 **Task Shield framework** 를 도입한다.

Fig. 2 에서 보이듯이, 이 framework 는 세 가지 핵심 components 로 구성된다.

* instruction extraction
* alignment check
* conversation flow 전반에서 task alignment 를 유지하기 위한 feedback generation

instruction extraction (1) 과 alignment check (2) 내부의 ContributesTo score calculation 은 모두 large language model 의 capabilities 를 활용한다.

이 section 에서는 먼저 각 shield component 의 technical implementation 을 자세히 설명하고, 이후 이러한 components 가 task alignment 를 강제하기 위해 LLM agent system 내부에서 어떻게 동적으로 상호작용하는지를 설명한다.

## 4.1 Task Shield Components

#### Task Instruction Extraction.

Task Shield framework 는 각 incoming message 로부터 task instructions 를 추출하는 것에서 시작한다. 이 process 는 두 가지 목적을 가진다.

* user objectives 를 식별하여, 이를 User Task Set $T_u$ 에 저장하고 이후 alignment check 의 conversational goals 로 사용한다.
* alignment check 가 필요한 다른 sources 로부터의 potential directives 를 탐지한다.

real-world messages 는 종종 extraction challenges 를 야기한다. instructions 가 implicit 할 수 있고, 다른 instructions 내부에 nested 되어 있을 수 있으며, complex content 안에 embedded 되어 있을 수 있다. 이러한 instruction 중 하나라도 놓치면 저자의 defense mechanism 에 security vulnerabilities 가 생길 수 있다.

이러한 challenges 를 해결하기 위해, 저자는 carefully designed LLM prompt (Appendix D 의 Fig. 4) 를 사용한 conservative extraction strategy 를 구현한다. 이 prompt 는 LLM 에게 다음을 지시한다.

* nested 되거나 implicit 한 경우에도 모든 potentially actionable directives 를 추출할 것
* information-seeking queries 를 explicit instructions 로 rewrite 할 것
* natural language 로 task dependencies 를 보존할 것

#### Alignment Check.

instructions 가 추출되면, 다음 단계는 추출된 각 instruction 이 Definition 3 에서 정의한 **Task Instruction Alignment Condition** 을 만족하는지 평가하는 것이다. 이는 두 가지 핵심 측면을 포함한다.

* 개별 instructions 의 contributions 평가
* overall alignment scores 계산

alignment 를 평가하기 위해, 저자는 Definition 2 에서 정의한 predicate **ContributesTo** 를 사용한다. 그러나 binary classification 은 practical applications 에 지나치게 경직되어 있는데, action 과 goal 사이의 관계는 종종 uncertainty 또는 ambiguity 를 포함하기 때문이다.

이러한 미묘한 관계를 반영하기 위해, 저자는 **fuzzy logic-based scoring mechanism** 을 채택한다. $[0, 1]$ 범위의 continuous score 를 부여함으로써, instructions 가 user goals 에 어떻게 기여하는지를 fine-grained 하게 평가할 수 있도록 한다. 이를 통해 direct contribution, intermediate steps, reasonable attempts at resolution 에서의 역할을 포착할 수 있다.

이후 total contribution score 는 모든 user task instructions 에 대한 scores 를 합산하여 계산된다. alignment check process 는 어떤 instruction 의 total contribution score 가 0 이면 그것을 misaligned 로 간주한다. 이 design 에 대한 자세한 논의와 implementation 은 Appendix B.2 에 포함되어 있다.

#### Feedback Generation.

misalignment 가 탐지되면, Task Shield 는 conversation 을 user objectives 와의 alignment 로 되돌리기 위한 structured feedback 을 생성한다. 이 feedback 은 다음을 포함한다.

* misaligned task instructions 를 식별하는 명확한 alert
* potential risks 를 설명하는 notification
* 현재의 user objectives ($T_u$) 에 대한 reminder

## 4.2 Interaction with the LLM Agent System

Task Shield 는 conversation flow 에 대한 monitoring 과 intervention 을 통해 alignment 를 강제하며, 각 message type 에 대해 서로 다른 processing approaches 를 사용한다. 각 message 는 다음 단계로 진행하기 전에 alignment check 를 통과해야 하며, 이는 potential attacks 에 대한 multiple layers of defense 를 형성한다.

#### User Message Processing

user level $L_u$ 에서, shield 는 newly extracted instructions 로 User Task Set $T_u$ 를 업데이트한다. 이러한 instructions 는 이후의 모든 message processing 에 대한 alignment targets 를 정의한다.

#### Assistant Message Processing

level $L_a$ 의 messages 는 alignment check 가 필요한 두 가지 components 를 포함할 수 있다.

* message content (natural language response)
* tool calls

이 둘 중 어느 하나라도 alignment check 를 통과하지 못하면, Task Shield 는 LLM agent 에 feedback 을 제공하여 response 를 재고하도록 유도한다. 이는 critic 으로 작동하며, LLM agent 가 queries 를 정제하도록 안내하기 위해 여러 rounds 의 feedback 을 제공한다. 특히 tool calls 에 대해서는, Task Shield 가 misaligned calls 의 execution 을 방지한다.

#### Tool Output Processing

level $L_t$ 에서, shield 는 context awareness 를 가지고 tool outputs 를 평가하며, 각 instruction 에 그 source 를 덧붙인다.

* "$\texttt{from tool [function\_name] with arguments [args]}$"

misalignment 를 탐지하면, shield 는 original output 과 feedback 을 모두 assistant 에 대한 response 에 포함하여, informed correction 이 가능하도록 한다.

이러한 multi-layered defense mechanism 은 injected attacks 가 multiple barriers 에 직면하도록 보장한다. 즉, tool outputs 내의 misaligned instructions 는 $L_t$ processing 동안 flag 되고, potentially harmful responses 는 $L_a$ level 에서 포착되어 정제될 수 있으며, $L_u$ 에서 user objectives 에 대한 continuous validation 이 overall conversation alignment 를 유지한다.

# 5 Experiments

이 section 에서 저자는 AgentDoJo 를 사용하여 GPT-4o 와 GPT-4o-mini 상에서 Task Shield 를 평가하며, 각 task 당 one trial 을 수행한다.

## 5.1 Settings

#### Benchmark

저자는 indirect prompt injection attacks 에 대한 AI agents 의 평가를 위해 설계된 첫 번째 comprehensive environment 인 AgentDojo benchmark 내에서 experiments 를 수행하였다. 일부 benchmarks 가 personal assistant use cases 를 벗어난 simple scenarios 또는 single-turn evaluations 에 초점을 맞추는 것과 달리, AgentDojo 는 realistic agent behaviors, multi-turn conversations, complex tool interactions 를 시뮬레이션한다. 또한 이 benchmark 는 real-world scenarios 를 시뮬레이션하는 네 가지 representative task suites 를 포함한다.

* Travel: itinerary management
* Workspace: document processing
* Banking: financial operations
* Slack: communication tasks

이는 realistic applications 에서 저자의 defense mechanism 을 practical 하게 시험할 수 있게 한다.

#### Models

주된 평가는 GPT-4o 에 대해 수행된다. 이러한 선택은 두 가지 요인에 의해 동기화된다.

* GPT-4o 는 challenging AgentDojo tasks 에서 superior performance 를 보여주며, 높은 utility baseline 을 제공한다.
* inverse scaling law 에 따라, GPT-4o 는 prompt injection attacks 에 특히 취약하므로 저자의 defense mechanism 을 검증하기에 이상적인 candidate 이다.
* 저자는 또한 instruction hierarchy training 을 통해 safety-aligned 된 model 인 GPT-4o-mini 와 GPT-3.5-turbo (Appendix 에 제시됨) 도 포함한다. 
* GPT-4o-mini 는 attacks 에 대해 inherent robustness 를 제공한다. defense implementation 에 대해서는, 저자는 protective Task Shield 로 동일한 model 을 사용한다.

#### Baselines

저자는 Task Shield 를 네 가지 established defense methods 와 비교한다.

* **Data Delimiting (Delimiting)**  
  explicit markers 를 사용하여 tool outputs 를 분리한다.
* **Prompt Injection Detection (PI Detector)**  
  classification 을 사용하여 potential attacks 를 식별한다.
* **Prompt Sandwiching (Repeat Prompt)**  
  repetition 을 통해 original user prompts 를 강화한다.
* **Tool Filtering (Tool Filter)**  
  task requirements 에 따라 available tools 를 제한한다.

#### Evaluation Metrics

experiment 는 LLM agent 의 performance 와 robustness 를 측정하기 위해 세 가지 핵심 evaluation metrics 를 사용하였다.

* **Clean utility (CU)**  
  attacks 가 없는 benign environment 에서 agent 가 성공적으로 완료한 user tasks 의 비율을 의미하며, agent 의 baseline performance 를 나타낸다.
* **Utility under attack (U)**  
  prompt injection attacks 하에서 agent 가 user tasks 를 성공적으로 완료하는 정도를 측정하며, adversarial interference 에도 불구하고 performance 를 유지하는 능력을 반영한다.
* **Target attack success rate**  
  attacker 의 goal 이 달성된 cases 의 비율을 평가하며, attack 의 effectiveness 와 defense 의 robustness 를 측정한다.

## 5.2 Results

#### Defending Against Attacks

저자는 Task Shield 를 세 가지 유형의 indirect prompt injection attacks 에 대해 평가한다.

* **Important Instructions**  
  model 이 urgent directives 를 우선시하는 경향을 악용하기 위해 high-priority malicious instructions 를 삽입한다.
* **Injecagent**  
  conflicting objectives 를 사용한다.
* **Ignore Previous**  
  prior instructions 를 무효화한다.

![Table 1](image-2.png)

* Tab. 1 에서 보이듯이, **Important Instructions** attack 은 가장 강한 threat 를 제기하며, defense 가 없는 GPT-4o 에서 attack success rate (ASR) 47.69% 를 달성하는 동시에 utility 를 유의미하게 저하시킨다. 
* Task Shield 는 모든 attack types 에 걸쳐 일관된 superiority 를 보여주는데, ASRs 를 줄일 뿐만 아니라 no-defense baseline 과 비교해 utility 를 유지하거나 개선한다.

특히, 가장 강한 Important Instructions attack 에 대해 다음과 같은 결과를 보인다.

* ASR 를 2.07% 로 감소시킨다.
* 높은 utility 69.79% 를 보존한다.

이후의 모든 experiments 는 가장 큰 threat 인 Important Instructions attack 하에서 수행된다.

Security-Utility Trade-offs

![Figure 3](image-3.png)

* Fig. 3 은 benign (attack 이전) 및 adversarial (attack 하) conditions 에서 GPT-4o 상의 서로 다른 defenses 의 performance 를 Pareto fronts 상에 시각화하여 security-utility trade-off 를 보여준다. 
* Pareto front 는 한 metric 을 개선하려면 다른 metric 의 저하가 필요한 optimal solutions 를 나타낸다. 
* 이상적인 data points 는 figure 의 오른쪽 아래 방향에 위치한다.

Task Shield 는 두 scenarios 모두에서 일관되게 Pareto front 에 접근하며, diverse conditions 와 task suites 에서 security 와 utility 사이의 optimal balance 를 보여준다. 구체적으로, Task Shield 는 각 plot 에서 바람직한 오른쪽 아래 영역에 일관되게 위치한다.

다른 defenses 는 유의미한 limitations 를 보인다.

* **PI Detector** 는 낮은 ASR 를 달성하지만 severe utility degradation 을 겪는다.
* **Tool Filter** 는 두 metrics 모두에서 moderate performance 를 보이지만 Pareto front 에는 미치지 못한다.
* **Repeat Prompt** 는 높은 utility 를 유지하지만 attacks 에 대한 defense 는 불충분하다.

#### Detailed Results on GPT-4o and GPT-4o-mini

![Table 2](image-4.png)

* Tab. 2 는 두 models 모두에 걸쳐 "Important Instructions" attack 에 대한 서로 다른 defense mechanisms 의 comparative analysis 를 제시한다. 
* GPT-4o 와 GPT-4o-mini 모두에서, Task Shield 는 모든 task suites 에 걸쳐 일관되게 superior overall performance 를 보여준다.
  * GPT-4o 에서 ASR 를 2.07% 로 줄이면서 utility under attack (U) 69.79% 를 유지한다.
  * GPT-4o-mini 에서도 유사하게 ASR 2.23% 와 utility under attack (U) 54.53% 를 달성한다.
* 이는 모든 baseline defenses 를 일관되게 능가하는 결과이다. 모든 task suites 에 걸쳐, Task Shield 는 CU, U, ASR 측면에서 near-optimal 또는 optimal performance 를 보여준다.

흥미롭게도, 두 models 는 서로 다른 defense mechanisms 에 대해 상이한 behaviors 를 나타낸다.

* **clean utility (CU)** 측면에서, 대부분의 defenses 는 GPT-4o 에서는 no-defense baseline 대비 performance 를 개선하지만 (PI Detector 제외), GPT-4o-mini 에서는 오히려 performance 를 저하시킨다.
* Task Shield 는 GPT-4o-mini 에서 clean utility 를 유지하거나 개선하는 유일한 defense 이다.
* **attack success rate (ASR)** 측면에서, GPT-4o-mini 는 defense 가 없을 때도 본질적으로 더 낮은 ASR 를 보인다.
  * GPT-4o-mini: 27.19%
  * GPT-4o: 47.69%
    이는 safety-aligned 된 특성 때문일 가능성이 있다.
* 또한 Repeat Prompt 는 GPT-4o-mini 에서는 비교적 강한 performance 를 보이지만 GPT-4o 에서는 고전하는 반면, Task Shield 는 두 architectures 모두에서 일관된 effectiveness 를 유지한다.

이는 defense solution 으로서의 robustness 를 강조한다.

# 6 Related Work

#### LLM Agent and Tool Integration

complex human-instructed tasks 를 수행할 수 있는 LLM agents 의 설계에 대한 연구는 크게 발전해 왔다. 이러한 agents 가 searching, decision making 과 같은 human-like functions 를 수행할 수 있도록 하기 위해, 기존 접근법들은 일반적으로 architectures 에 external tool-calling capabilities 를 통합한다. 

다양한 backbone models 이 존재하므로, tool calling functionality 를 갖춘 LLM agent 를 만드는 것은 그다지 어렵지 않다. Schick et al. 은 LLM 이 external tools 를 autonomous 하게 호출하는 방법을 학습할 수 있도록 하는 접근법을 탐구하였다. 따라서 저자의 접근법은 LLM agent systems 에 폭넓게 채택될 수 있고, 매끄럽게 통합될 수 있다.

#### Indirect Prompt Injection Attacks

indirect prompt injection attacks 는 최근 LLM agents 에 대한 중요한 safety concern 으로 부상하였다. 이러한 attacks 는 malicious content 가 external data providers 또는 environments 로부터 공급된 inputs (e.g., 신뢰할 수 없는 websites 에서 가져온 data) 에 embedded 될 때 발생하며, 그 결과 agents 가 private personal information 을 공유하는 것과 같은 unsafe 또는 malicious actions 를 수행하게 된다. 

다양한 scenarios 전반에서 이러한 attacks 의 risks 를 체계적으로 평가하기 위해, Injecagent 와 AgentDojo 를 포함한 여러 benchmarks 가 개발되었다. 본 논문에서 저자는 이러한 malicious effects 를 완화하기 위한 robust system 을 구축하는 것을 목표로 한다.

#### Defense Methods

prompt injection attacks 에 대한 defenses 는 training-time strategies 와 test-time strategies 모두에 초점을 맞추어 왔다.

* **Training-time methods**
  * adversarial examples 로 models 를 fine-tuning 하여 robustness 를 강화하는 방식을 일반적으로 포함한다.
  * 그러나 이러한 접근법들은 높은 computation cost 와 internal access 가 없는 LLMs 에 적용할 수 없다는 점 때문에 종종 비실용적이다.
* **Test-time defenses**
  * 일반적으로 상당한 computation resources 를 요구하지 않는다.
  * 예를 들어, Wang et al. 은 harmful responses 를 필터링하기 위해 hash-based authentication tags 를 사용할 것을 제안하였다.
  * Hines et al. 및 Chen et al. 은 models 이 attacks 를 인식하고 완화하도록 지시하기 위한 special delimiters 를 설계하였다.

이에 비해 저자의 접근법은 **task alignment** 를 강제하는 것을 목표로 하며, 더 나은 robustness-utility tradeoff 를 달성한다.

# 7 Conclusion

이 연구에서 저자는 indirect prompt injection attacks 에 대한 defense 를 위한 새로운 관점을 제안하였으며, agent behavior 가 user objectives 에 봉사하도록 보장하는 guiding principle 로서 **task alignment** 를 도입하였다. 또한 저자는 instructions 가 user goals 와 align 되는지를 검증함으로써 이 원칙을 강제하는 test-time mechanism 인 **Task Shield** 를 개발하였다. 이는 AgentDoJo benchmark 의 다양한 simulated real-world tasks 전반에서 agent capabilities 를 보존하면서 indirect prompt injection attacks 에 대해 state-of-the-art defense 를 달성한다.

#### Limitations

저자의 framework 는 몇 가지 limitations 에 직면한다.

* task instruction extraction 과 ContributeTo scoring 을 위해 LLMs 에 의존한다는 점은 두 가지 핵심 vulnerabilities 를 도입한다.
  * weaker language models 를 사용할 때 potential performance degradation 이 발생할 수 있다.
  * adaptive attacks 에 취약할 수 있다.
* 또한 resource constraints 역시 evaluation scope 를 제한하였다.
  * LLM queries 의 높은 cost 때문에 experiments 는 단일 benchmark 와 단일 model family 로 제한되었다.

#### Future Work

향후 연구를 위한 몇 가지 방향이 제시된다.

* more cost-effective 한 LLM-based instruction extraction 및 alignment verification techniques 를 개발함으로써 Task Shield 의 efficiency 와 robustness 를 향상시키는 것
* prompt injection 을 넘어 jailbreak attacks, system prompt extraction 과 같은 더 광범위한 security threats 를 다룰 수 있도록 Task Shield 를 확장하는 것
* AI agents 가 specialized objectives 와 엄격한 alignment 를 유지해야 하는 domain-specific business contexts 에 framework 를 적응시키는 것
* 다양한 task dependencies 와 misalignment scenarios 를 포착하는 synthetic training data 를 생성하기 위해 task alignment concept 를 활용하는 것
