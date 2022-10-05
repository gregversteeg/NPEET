NPEET
=====

Non-parametric Entropy Estimation Toolbox

This package contains Python code implementing several entropy estimation functions for both discrete and continuous variables. Information theory provides a model-free way find structure in complex systems, but difficulties in estimating these quantities has traditionally made these techniques infeasible. This package attempts to allay these difficulties by making modern state-of-the-art entropy estimation methods accessible in a single easy-to-use python library.

The implementation is very simple. It only requires that numpy/scipy be installed. It includes estimators for entropy, mutual information, and conditional mutual information for both continuous and discrete variables. Additionally it includes a KL Divergence estimator for continuous distributions and mutual information estimator between continuous and discrete variables along with some non-parametric tests for evaluating estimator performance.

**The main documentation is in <a href="https://github.com/gregversteeg/NPEET/blob/master/npeet_doc.pdf">npeet_doc.pdf</a>.**
It includes description of functions, references, implementation details, and technical discussion about the difficulties in estimating entropies. The <a href="http://www.isi.edu/~gregv/npeet.tgz">code is available here</a>. It requires <a href="http://www.scipy.org">scipy</a> 0.12 or greater. This package is mainly geared to estimating information-theoretic quantities for continuous variables in a non-parametric way. If your primary interest is in discrete entropy estimation, particularly with undersampled data, please see <a href="http://thoth-python.org">this package</a>.</p>

Example installation and usage:

```bash
git clone https://github.com/gregversteeg/NPEET.git
cd NPEET
pip install .
```

```python
>>> from npeet import entropy_estimators as ee
>>> x = [[1.3],[3.7],[5.1],[2.4],[3.4]]
>>> y = [[1.5],[3.32],[5.3],[2.3],[3.3]]
>>> ee.mi(x,y)
Out: 0.168
```

Another example:

```python
import numpy as np
from npeet import entropy_estimators as ee

my_data = np.genfromtxt('my_file.csv', delimiter=',')  # If you look in the documentation, there is a way to skip header rows and other things

x = my_data[:,[5]].tolist()
y = my_data[:,[9]].tolist()
z = my_data[:,[15,17]].tolist()
print(ee.cmi(x, y, z))
print(ee.shuffle_test(ee.cmi, x, y, z, ci=0.95, ns=1000))
```

This prints the mutual information between column 5 and 9, conditioned on columns 15 and 17. You can also use the function shuffle_test to return confidence intervals for any estimator. Shuffle_test returns the mean CMI under the null hypothesis (CMI=0), and 95% confidence intervals, estimated using 1000 random permutations of the data.
*Note that we converted the numpy arrays to lists! The current version really works only on python lists (lists of lists actually, as in the first example.*

See documentation for references on all implemented estimators.

```latex
@article{kraskov_estimating_2004,
    title = {Estimating mutual information},
    url = {https://link.aps.org/doi/10.1103/PhysRevE.69.066138},
    doi = {10.1103/PhysRevE.69.066138},
    journaltitle = {Physical Review E},
    author = {Kraskov, Alexander and Stögbauer, Harald and Grassberger, Peter},
    date = {2004-06-23},
}

@misc{steeg_information-theoretic_2013,
    title = {Information-Theoretic Measures of Influence Based on Content Dynamics},
    url = {http://arxiv.org/abs/1208.4475},
    doi = {10.48550/arXiv.1208.4475},
    author = {Steeg, Greg Ver and Galstyan, Aram},
    date = {2013-02-15},
}

@misc{steeg_information_2011,
    title = {Information Transfer in Social Media},
    url = {http://arxiv.org/abs/1110.2724},
    doi = {10.48550/arXiv.1110.2724},
    author = {Steeg, Greg Ver and Galstyan, Aram},
    date = {2011-10-12},
}%
```

The non-parametric estimators actually fare poorly for variables with strong relationships. See the following paper and the improved code available at <https://github.com/BiuBiuBiLL/NPEET_LNC>

```latex
@misc{gao_efficient_2015,
    title = {Efficient Estimation of Mutual Information for Strongly Dependent Variables},
    url = {http://arxiv.org/abs/1411.2003},
    doi = {10.48550/arXiv.1411.2003},
    author = {Gao, Shuyang and Steeg, Greg Ver and Galstyan, Aram},
    date = {2015-03-05},
}%
```
