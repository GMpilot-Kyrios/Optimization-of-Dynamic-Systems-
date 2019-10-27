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
p_{SD}^{(k)} = -\frac{dJ(\theta)}{d\theta}\Bigg |_ {\theta_k}
\end{equation}
$$
下面我们来确定步长 $\alpha^{(k)}$。我们观察一下公式$(1.1)$公式，就会发现：在最速下降法中目标函数J为：
$$
\begin{equation}
\min_{\alpha^{k}\in R^+}\lbrace J(\theta^{(k)}+\alpha^{(k)}p_{SD}^{k})\rbrace = \min_{\alpha^{k}\in R^+}\lbrace\Phi(\alpha^{k})\rbrace 
\end{equation}
$$
现在$\theta^{(k)}$和$p_{SD}^{(k)}$已知，我们记新的目标方程J的最小值为$\Phi(\alpha^k)$，也就是说新的目标方程是关于步长$\alpha$的一次线性方程。一个直观的方法是用导数等于零的方法直接计算$\Phi(\alpha^k)$的最小值。[SteepestDescentMethod.py](https://github.com/FahrerFeng/Optimization-of-Dynamic-Systems-/blob/master/SteepestDescentMethod.py)文件描述了这一想法。我们选用banana function $100(\theta_2-\theta_1^2)^2+(1-\theta_1)^2$作为例子，该方程在(1,1)点处去得最小值。
![](https://github.com/FahrerFeng/Optimization-of-Dynamic-Systems-/blob/master/SDForiginal.png)
<br>上图显示了所有迭代的结果。我们可以看到结果并不理想，每一步跨度过长，造成了迭代在一个区域内不断往复。这种现象的原因可以形象地理解为:假设截面$\theta^{(k)}+\alpha^{(k)}p_{SD}^{k}$与方程相交地曲线有两个最小值相同的波谷。第33行利用funcMinPoint(f)计算出最小值，并以此为最新的迭代点。下一次运行又是同样的效果。也就是说，迭代点在两个波谷上来回跳跃，因此收敛速度极慢，而且跨度很长。另外，最小值计算需要花很长时间，因为要利用solve()函数求出导数零点。这就有很多隐患，比如方程的确存在最小值，但是在某一步迭代中沿着梯度反方向，不一定存在极小值(或者导数为零的点)。因此我们需要一个新的方法来寻找步长$\alpha$。
ps: 小编在使用sympy库的solve方法时发现了一个一些问题。程序的第34、35行计算了目标方程的导数，这里需要用到simplify方法来简化表达式，否则solve方法返回出来的的结果为一个表达式，不容易进行类型转换。除此之外，面对系数复杂的等式，solve一般会返回虚部为零的虚数解，这就会让我们很头疼。同学们不妨亲自验证一下，范例的banana function对于任意带入点的导数都有实数零点。这时候下编自己相想出了一个方法，就是利用.as_real_imag()方法提取虚部，一旦虚部为零就选取实数的部分存在新的列表里，由此得到实数零点(见41至46行)。程序的第56行，我们人为地加入一个判断条件就是，在新选取的迭代点处，方程的值一定要小于原先的迭代点$J(\theta^{(k)})$这样过程才是下降的，另外我们选择步长小于1使得步长不至于过大，同学们可以试试去掉这个条件，这样对于不同的起始点，该算法有可能不收敛。即便有了以上两个条件，我们依然可以从图中观察到，该算法收敛效果并不好。
## Backtracking algorithm
我们使用回溯法来获得步长$\alpha$，这个方法有一个好处就是不用计算零点。
```{r, eval=FALSE,tidy=FALSE}
Choose a_hat = 1, c in [0,1], set a_k = a_hat
repeat 
  a_k = c * a_k
until \Phi(a_k) small enough
return a_k
```
