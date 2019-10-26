# Optimization of Dynamic Systems
这个小项目是本人上课的随堂联系，主要是为了锻炼python能力。代码写的不好看，还请大家多多提意见。
# Steepest Descent Method
所有Line Search Method的主要思路都是在第k步迭代时，以$\theta^{(k)}$为起始点，选择合适的方向$p^{(k)}$并沿着该方向搜索的新的迭代点$\theta^{(k+1)}$, 其步长(step length)记为$\alpha$。这样做的目的是，将原始的多维最小值问题转化为每一步迭代时的一维最小值问题。其数学描述如下：
$$\theta^{(k+1)} = \theta^{(k)} + \alpha^{(k)} p^{(k)} \longmapsto J(\theta^{(k+1)}) = \min_{\alpha^{(k)}\in R^{+}}{J(\theta^{(k)} + \alpha^{(k)}p^{(k)})\}$$
