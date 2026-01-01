Cholesky Implementation
=======================

For the context that this was designed for, a direct solve approach is preferred to an iterative solver like ``ARMA``. Time-varying graph signals must be as efficient as possible with memory, to ensure scalability to signals of large sparse networks. Especially if the process is online.

The ``cholmod_solve2`` and ``updown`` functions are the primary engine, in addition to other various design choices that accelerate graph convolutions.