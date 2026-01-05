Analytical Filters
==============

Low-Pass Spectral Graph Filter
------------------------------

The low-pass filter is a refinable, self-similar rational function useful for signal smoothing. The `scale` parameter, :math:`s`, controls the cutoff frequency.

.. math::

   \phi_s(\mathbf{\Lambda}) = \frac{I}{s\mathbf{\Lambda} + I}

High-Pass Spectral Graph Filter
-------------------------------

The high-pass filter isolates high-frequency variations on the graph.

.. math::

   \mu_s(\mathbf{\Lambda}) = \frac{s\mathbf{\Lambda}}{s\mathbf{\Lambda} + I}

Band-Pass Spectral Graph Filter
-------------------------------

A convenient closed-form wavelet generating kernel that serves as an alternative to custom-designed filters. The scale :math:`s` adjusts the center frequency of the band.

.. math::

   \Psi_s(\mathbf{\Lambda}) = \frac{4s\mathbf{\Lambda}}{(s\mathbf{\Lambda}+I)^2}

This filter qualifies as a wavelet generating kernel for the SGWT, since :math:`\Psi(0)=0` and the admissibility condition is satisfied. The admissibility constant of this band-pass filter is :math:`C_f=8/3`.

.. math::

   \Psi(0)=0  \qquad\text{and}\quad \int_0^{\infty}\dfrac{\Psi^2(x)}{x}\mathrm{d}x <\infty

.. seealso::
   The analytical filters are implemented as methods on the convolution contexts:

   * :meth:`~sgwt.Convolve.lowpass`
   * :meth:`~sgwt.Convolve.bandpass`
   * :meth:`~sgwt.Convolve.highpass`