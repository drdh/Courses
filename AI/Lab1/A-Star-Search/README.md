## A*搜索

### A*搜索要求

评估函数
$$
f(n)=g(n)+h(n)
$$
其中

$g(n)$=cost so far to reach n --到达节点n的耗散

$h(n)$ = estimated cost to goal from n--启发函数：从节点n到目标节点的最低耗散路径的耗散估计值

$f(n)$ = estimated total cost of path through n to goal--经过节点n的最低耗散的估计函数

还要求$h(n) ≤ h^*(n) $where $h^*(n)$ is the true cost from n.
以及 $h(n) ≥ 0$, so $h(G) = 0$ for any goal G.

