定义程序点状态，意为内存区域上一次访问操作类型
$$
Region \rightarrow (2^{\{Read,Write\}},\subseteq)
$$
定义状态转换规则与检查动作
$$
[[entry]]=\bot
$$

$$
[[read\ Region]]_{pre}\rightarrow if\ \mathbb{S}(Region)=\bot, then\ report\ read\ uninit;\mathbb{S}[Region \mapsto Read]
$$

$$
[[write\ Region]]_{pre}\rightarrow if\ \mathbb{S}(Region)=Write,then\ report\ useless\ write;\mathbb{S}[Region\mapsto Write]
$$

$$
[[E(E_1,E_2,...,E_n)]]_{eval}\rightarrow \forall E_i \in \{E_1,E_2,...,E_n\},if\ IsOutput(E,E_i),then\ write\ Region(E_i)
$$

