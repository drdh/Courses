# HW 1 kNN with Inverted Index

先根据inverted vector计算各个文档的tf-idf vector. 然后使用$\cos$的值作为文档的相似性。

查找k个最佳检索，就是kNN.

具体来说就是各个文档的tf-idf结合起来得到一个矩阵$A\in \mathbb{R}^{|D|\times |W|}$. 然后test的tf-idf vector设为$\mathbf{x}\in \mathbb{R}^{|W|}$. 然后计算为$A\mathbf{x}$ 得到一个vector为  $\mathbf{y} \in \mathbb{R}^{|W|}$. 从其中找到最大的$k$个分量，对应的文档就是kNN得到的文档。


# HW 2 Derivative of the Logistic

$$
\begin{aligned}
\min_{\mathbf{w}} J(\mathbf{w}) &= \min_{\mathbf{w}}- \mathcal{L}(\mathbf{w})\\
& = \min_{\mathbf{w}} -\Bigg[ \sum_{i=1}^{N}y_i\log \sigma (\mathbf{w}^\top \mathbf{x}_i)  +(1-y_i) \log (1-\sigma (\mathbf{w}^{\top} \mathbf{x}_i)) \Bigg] \\
\end{aligned}
$$

Update
$$
\mathbf{w}=\mathbf{w}-\alpha \frac{\partial}{\partial \mathbf{w}} J(\mathbf{w})
$$

Now calculate the derivatives. 

First, simplify the formula:
$$
\begin{aligned}
J(\mathbf{w}) &= \sum_{i=1}^{N} -\Bigg[y_i \log \sigma (\mathbf{w}^\top \mathbf{x}_i)  +(1-y_i) \log (1-\sigma (\mathbf{w}^{\top} \mathbf{x}_i)) \Bigg] \\

&=\sum_{i=1}^{N} -\Bigg [-y_i\log \Big(1+e^{- \mathbf{w}^{\top}\mathbf{x}_i}\Big ) -(1-y_i)\Big[\log \Big(1+e^{- \mathbf{w}^{\top}\mathbf{x}_i}\Big)\Big]\Bigg] \\

&= \sum_{i=1}^{N} \Bigg[\log \Big(1+e^{- \mathbf{w}^{\top} \mathbf{x}_i} \Big)+ \mathbf{w}^{\top} \mathbf{x}_i  -y_i \mathbf{w}^{\top} \mathbf{x}_i  \Bigg] \\

&= \sum_{i=1}^{N} \Bigg [\log \Big( 1+e^{\mathbf{w}^{\top} \mathbf{x}_i}\Big)-y_i \mathbf{w}^{\top} \mathbf{x}_i \Bigg] \\
\end{aligned}
$$
Then the derivatives
$$
\begin{aligned}
\frac{\partial }{\partial  \mathbf{w}} J( \mathbf{w}) 
&= \sum_{i=1}^N \Bigg[\frac{ \mathbf{x}_ie^{ \mathbf{w}^\top  \mathbf{x}_i}}{1+e^{ \mathbf{w}^\top  \mathbf{x}_i}}-y_i  \mathbf{x}_i \Bigg] \\
&= \sum_{i=1}^N \Bigg[\Big(\sigma( \mathbf{w}^\top  \mathbf{x}_i)-y_i\Big)  \mathbf{x}_i  \Bigg]
\end{aligned}
$$
And the update formula
$$
\mathbf{w}=\mathbf{w}+\alpha \sum_{i=1}^N \Bigg[\Big( y_i-\sigma( \mathbf{w}^\top  \mathbf{x}_i)\Big)  \mathbf{x}_i  \Bigg]
$$
