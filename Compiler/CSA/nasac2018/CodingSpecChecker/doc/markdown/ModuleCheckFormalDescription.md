程序状态$\mathbb{S}=\{S_1, S_2\}$

$S_1$表示Memory Region绑定右值是否为污点
$$
S_1:Region \rightarrow (2^{\{Tainted, Untained\}}, \subseteq)
$$

$S_2$表示Memory Region绑定右值定值所依赖的Memory Region集合
$$
S_2:Region \rightarrow (2^{RegionSet}, \subseteq)
$$

---

程序状态$S_1$转换规则
$$
[[Entry]] = \bot
$$

$$
[[ParmDecl\ X]]=\mathbb{S}[Region(X) \mapsto Func.IsStatic\ ?\ Untainted : Tainted]
$$

$$
[[VarDecl\ X]]=\mathbb{S}[Region(X) \mapsto Tainted]
$$

$$
[[X=E]]=\mathbb{S}[Region(X) \mapsto eval(\mathbb{S},E)]
$$

$$
[[E(E_1,E_2,\dots,E_n)]]=\mathbb{S}[\forall E_i \in \{E_1,E_2,\dots,E_n\}, Func.IsOutParm(i) \rightarrow Region(\mathbb{S},E_i) \mapsto Tainted]
$$

$$
[[condition\ E]]=\mathbb{S}[\forall R_1 \in FirstRegions(E), R_1 \mapsto Untainted;\forall R_2 \in S_2(R_1),R_2\mapsto Untainted]
$$

---

$eval$函数定义如下：
$$
eval(\mathbb{S},Op(E_1,E_2,\dots E_n))=\bigsqcup\{eval(\mathbb{S},E_1),eval(\mathbb{S},E_2),\dots eval(\mathbb{S},E_n)\}
$$

$$
eval(\mathbb{S},X)=\mathbb{S}(X)
$$

$$
eval(\mathbb{S},E(E_1,E_2,\dots E_n))=\bigsqcup\{eval(\mathbb{S},E_1),eval(\mathbb{S},E_2),\dots eval(\mathbb{S},E_n)\}
$$

$$
eval(\mathbb{S},Member(E,M))=eval(\mathbb{S},E)
$$

$$
eval(\mathbb{S},E_1[E_2])=\bigsqcup\{eval(\mathbb{S},E_1), eval(\mathbb{S},E_2)\}
$$

$$
\dots 
$$

$Regions$函数定义如下：
$$
Regions(\mathbb{S}, Op(E_1,E_2,\dots ,E_n))=Regions(\mathbb{S}, E_1)\cup Regions(\mathbb{S}, E_2)\cup\dots \cup Regions(\mathbb{S}, E_n)
$$

$$
Regions(\mathbb{S}, X) = \{Region(\mathbb{S}, X)\}
$$

$$
Regions(\mathbb{S}, E(E_1,E_2,\dots ,E_n)) = Regions(\mathbb{S}, E_1)\cup Regions(\mathbb{S}, E_2)\cup \dots \cup Regions(\mathbb{S}, E_n)
$$

$$
Regions(\mathbb{S}, Member(E,M))=\{BaseRegion(\mathbb{S}, Member(E, M))\}
$$

$$
Regions(\mathbb{S}, E_1[E_2])=\{Region(\mathbb{S}, E_1[E_2])\}
$$

---

程序状态$S_2$转换规则：
$$
[[Entry]] = \bot
$$

$$
[[X=E]] = \mathbb{S}[Region(\mathbb{S}, X) \mapsto Regions(E)]
$$

---

检查动作$A_{pre}$:
$$
[[read\ X]]\rightarrow \begin{cases}
if\ eval(S_1,X)=Tainted, then\ warning & X \in E(E_1,E_2,\dots E_n)\or X\notin condition\ E\\
None&otherwise
\end{cases}
$$

