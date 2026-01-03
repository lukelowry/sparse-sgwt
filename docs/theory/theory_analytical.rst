Analytical Filters
==============

Low-Pass Spectral Graph Filter
------------------------------

The low-pass filter is *refinable*, as it is a self-similar rational function. The refinability makes it useful for signal smoothing across a range of spatial scales.

.. math::

   \phi(\mathbf{\Lambda}) = \dfrac{I}{\mathbf{\Lambda}+I} 

High-Pass Spectral Graph Filter
-------------------------------

The proposed high-pass filter acts as a container for variations over the graph below a given spatial scale.

.. math::

   \mu(\mathbf{\Lambda}) = \dfrac{\mathbf{\Lambda}}{\mathbf{\Lambda}+I}

Band-Pass Spectral Graph Filter
-------------------------------

A convenient closed-form wavelet generating kernel was found to be a useful kernel as an alternative to the vector-fitting procedure if a particular filter does not need to be designed. 

.. math::

   \Psi(\mathbf{\Lambda}) = \dfrac{4\mathbf{\Lambda}}{(\mathbf{\Lambda}+I)^2} 

This filter qualifies as a wavelet generating kernel for the SGWT, since :math:`\Psi(0)=0` and the admissibility condition is satisfied. The admissibility constant of this band-pass filter is :math:`C_f=8/3`.

.. math::

   \Psi(0)=0  \qquad\text{and}\quad \int_0^{\infty}\dfrac{\Psi^2(x)}{x}\mathrm{d}x <\infty

.. seealso::
   The analytical filters are implemented as methods on the convolution contexts:

   * :meth:`~sgwt.static.Convolve.lowpass`
   * :meth:`~sgwt.static.Convolve.bandpass`
   * :meth:`~sgwt.static.Convolve.highpass`