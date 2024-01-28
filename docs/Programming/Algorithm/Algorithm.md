import Tabs from '@theme/Tabs';

import TabItem from '@theme/TabItem';

# What is the Algorithm?

문제 해결을 위한 절차나 방법

수학 및 컴퓨터과학 측면의 알고리즘은, 보통 반복되는 문제를 풀기 위한 작은 프로시저를 의미

크게 3 가지의 표현 방법을 사용

- 의사 코드 pseudocode
- 순서도
- 프로그래밍 언어

# Time Complexity

알고리즘의 소요 시간을 정확히 평가할 수 없으므로, 자료 수 $n$ 에 따라 증가하는 대략적인 패턴을 시간 복잡도로 표기

- Big-$\Omega(n)$ (빅 오메가) : 최선일 때 (best case)의 연산 횟수 표기
- Big-$\Theta(n)$ (빅 세타)   : 보통일 때 (average case)의 연산 횟수 표기
- Big-$O(n)$ (빅 오)          : 최악일 때 (worst case)의 연산 횟수 표기

즉, 주어진 문제를 해결하기 위한 연산 횟수를 의미

※ 파이썬에서는 대략 2,000만 ~ 1억 번의 연산을 1초의 수행 시간으로 예측 가능

일반적으로 $O(n)$ 의 시간 복잡도를 기준으로 평가하여 좋은 알고리즘을 나눈다.

# Space Complexity

알고리즘을 통한 문제 해결 중 필요한 메모리 양을 공간 복잡도로 표기

일반적으로 시간 복잡도처럼 Big-$O(n)$ 를 주로 사용

메모리가 상당히 많이 필요한 _동적 계획법_ 을 제외하고는, 공간 복잡도는 시간 복잡도보다 중요도는 떨어진다.

임베디드, 펌웨어 등의 하드웨어 환경과 같이 극도로 한정되어 있을 경우는 공간 복잡도를 중요하게 보게된다.

# Algorithm

## Array & List

:::tip

<Tabs>
  <TabItem value="input" label="input()">
input() 내장 함수는 parameter 로 prompt message 를 받을 수 있다. 

- 따라서 입력받기 전 prompt message 를 출력해야 한다. 
  - 물론 prompt message가 없는 경우도 있지만, 이 경우도 약간의 부하로 작용할 수 있다. 하지만, sys.stdin.readline()은 prompt message를 인수로 받지 않는다.
- 또한, input() 내장 함수는 입력받은 값의 개행 문자를 삭제시켜서 리턴한다. 
  - 즉, 입력받은 문자열에 rstrip() 함수를 적용시켜서 리턴한다. 
  - 반면에 sys.stdin.readline()은 개행 문자를 포함한 값을 리턴한다.

**결론**

input() 내장 함수는 sys.stdin.readline() 과 비교해서 prompt message 를 출력하고, 개행 문자를 삭제한 값을 리턴하기 때문에 느리다.

input 값이 많다면 sys.stdin.readline() 을 사용하자.
  </TabItem>
  <TabItem value="orange" label="Orange">This is an orange 🍊</TabItem>
  <TabItem value="banana" label="Banana">This is a banana 🍌</TabItem>
</Tabs>



:::

### 구간 합

구간 합은 합 배열을 통해 복잡도를 더 줄이는 알고리즘

합 배열 S 정의

$$
S[i] = A[0] + A[1] + \cdots A[i - 1] + A[i]
$$

- 리스트 A $[15, 13, 10, 07, 03, 12]$
- 합배열 S $[15, 28, 38, 45, 48, 60]$

합 배열 S 를 만드는 정의

$$
S[i] = S[i - 1] + A[i]
$$

위와 같이 구현한 합배열로 구간 합을 쉽게 구할 수 있다.

구간 합 구하는 공식

$$
S[j] - S[i - 1]
$$

A[2] ~ A[5] 구간 합을 구하는 과정

$$
\begin{align*}
    &S[5] &&= A[0] + A[1] + A[2] + A[3] + A[4] + A[5] \\
    &S[1] &&= A[0] + A[1] \\
    &S[5] - S[1] &&=  A[2] + A[3] + A[4] + A[5]
\end{align*}
$$