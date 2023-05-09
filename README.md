# PmWNW


Developed by Azadeh Moradinezha Dizgah

This reporsitory contains python scripts for splitting the matter power spectrum into wiggly and broadband (no-wiggle) components.
I also provide a jupyyter script to guid you to how the functions can be called. 

The wiggle/no-wiggle splitting is necassary for performing approximate IR resummation. Since large bulk velocities only affect the BAO, 
to account for their effect on matter power spectrum, one can perform an approximate IR resummation (see for instance [arXiv:1509.02120](https://arxiv.org/abs/1509.02120) and [arXiv:1605.02149](arXiv:https://arxiv.org/abs/1605.02149) for more details). Detailed comparison of the three methods and how the choice of the splitting for IR resummation affect (or not) cosmological constraints from power spectrum in real space, see Appendix B of [arXiv:2010.14523](https://arxiv.org/abs/2010.14523).  

The three methods for splitting that are implmented here are the following:
 - DST: this method is based on performing discrete sine transform to remove the BAO osciallations.
 - Gfilter: this method relies on applying a Gaussian filter to the the matter powerspectrum to smooth out the BAO. 
 - BSpline: this method relies on performing regresssion to fit a smooth curve to the part of the power spectrum containing teh BAO.

Please feel free to use these tools to undestand how the different methods work. If you used the code in a publication, plase add a comment about it in the acknowlegment and cite the paper for which the tools were developed, [arXiv:2010.14523](https://arxiv.org/abs/2010.14523). 
