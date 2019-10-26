# Optimization of Dynamic Systems
这个小项目是本人上课的随堂联系，主要是为了锻炼python能力。代码写的不好看，还请大家多多提意见。
# Line Search Method
所有Line Search Method的主要思路都是在第k步迭代时，以$\theta^{(k)}$为起始点，选择合适的方向$p^{(k)}$并沿着该方向搜索的新的迭代点$\theta^{(k+1)}$, 其步长(step length)记为$\alpha$。这样做的目的是，将原始的多维最小值问题转化为每一步迭代时的一维最小值问题。其数学描述如下：
$$
\begin{equation}
\theta^{(k+1)} = \theta^{(k)} + \alpha^{(k)} p^{(k)} \longmapsto J(\theta^{(k+1)}) = \min_{\alpha^{(k)}\in R^{+}}{J(\theta^{(k)} + \alpha^{(k)}p^{(k)})\}
\tag{1.1}
\end{equation}
$$
其中J为目标方程(object function)因此对于Line Search Method 来说，在每一部迭代中我们有两个任务。第一，确定方向$p^{(k)}$;第二，确定步长$\alpha^{(k)}$。下面我们来看几个line search method。
## Steepest Descent Method(最速下降法)
对于最速下降法，方向向量我们选择:
$$
\begin{equation}
p_{SD}^{((k)} = -[\frac{dJ(\theta)}{d\theta}\Bigg|_{\theta_k}]
\end{equation}
$$
下面我们来确定步长 $\alpha^{(k)}$。我们观察一下公式$(1.1)$公式，就会发现：在最速下降法中目标函数J为：
$$
\begin{equation}
\min_{\alpha^{k}\in R^+}\lbrace J(\theta^{(k)}+\alpha^{(k)}p_{SD}^{k})\rbrace = \min_{\alpha^{k}\in R^+}\lbrace\Phi(\alpha^{k})\rbrace 
\end{equation}
$$
