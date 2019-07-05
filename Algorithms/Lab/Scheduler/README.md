$$
O(k^n)
$$


$$
O(k!k^{n-k})
$$

$$
\begin{aligned}
\text{upper bound}&=\sum_{\sum_{i=1}^kn_i=n}\frac{n!}{k!\prod_{i=1}^k n_i!}\\
&\le \frac{n!}{k! \Gamma^k(\frac{n}{k}+1)}\binom{n-1}{k-1}\\
&=\frac{n!(n-1)!}{k!(k-1)!(n-k)! \Gamma^k(\frac{n}{k}+1)}
\end{aligned}
$$
