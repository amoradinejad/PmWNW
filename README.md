# PmWNW


Developed by Azadeh Moradinezha Dizgah

This reporsitory contains python scripts for splitting the matter power spectrum into wiggly and broadband (no-wiggle) components.
I also provide a jupyyter script to guid you to how the functions can be called.

The wiggle/no-wiggle splitting is necassary for performing approximate IR resummation. Since large bulk velocities only affect the BAO, 
to account for their effect on matter power spectrum, one can perform an approximate IR resummation (see for instance [arXiv:1509.02120](https://arxiv.org/abs/1509.02120) 
and [arXiv:1605.02149](arXiv:https://arxiv.org/abs/1605.02149) for more details). In the repository, I also provide a jupyyter script to guid you to how the 
functions can be called.

The three methods for splitting that are implmented here are the following:
 - DST: discrete sine transform or spectral method
 - Gfilter: a Gaussian filter is applied to remove the BAO
 - 
