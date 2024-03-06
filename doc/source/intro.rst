.. math:: \def\*#1{\mathbf{#1}}
          \newcommand{\norm}[1]{\left\lVert#1\right\rVert}
          \newcommand{\mT}{\mathcal{T}}
          \newcommand{\minimize}[2]{\underset{\substack{{#1}}}{\mathrm{minimize}}\;\;#2}

          

Introduction.
=============

Problem statement.
------------------

Optimization problem for the robust shifted proper orthogonal projection

.. math::
    :label: criterion

    \minimize{\{\*Q^k\}_k,\*E} \sum_{k=1}^K \lambda_{k}\norm{\*Q^k}_*
    + \lambda_{K+1} \norm{\*E}_1
    \quad \text{s.t. } \*Q =\sum_{k=1}^K \mT^k \*Q^k + \*E \, ,

TO BE DONE

Examples.
---------

Wildland fire.
~~~~~~~~~~~~~~~


