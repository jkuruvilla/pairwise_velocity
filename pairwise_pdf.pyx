'''__license__   = "GNU GPLv3 <https://www.gnu.org/licenses/gpl.txt>"
__copyright__ = "2016, Joseph Kuruvilla"
__author__    = "Joseph Kuruvilla <joseph.k@uni-bonn.de>"
__version__   = "2.0"

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Changelog:
----------
v2.0: Fixed the indexing problem (substituting jj instead of j) and better
      docstrings for the functions.
'''

# --------------------
# Importing modules
# --------------------

from __future__ import division
from libc.math cimport sqrt, floor, fabs


import numpy as np
cimport numpy as np

DTYPEf = np.float32
ctypedef np.float32_t DTYPEf_t

DTYPEf = np.float64
ctypedef np.float64_t DTYPEff_t

DTYPEi = np.int64
ctypedef  np.int64_t DTYPEi_t

DTYPEu = np.uint8
ctypedef np.uint8_t DTYPEu_t

cimport cython

# --------------------
# Defining functions
# --------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def pdf_1d_los(tree, np.ndarray[DTYPEf_t, ndim=2] ppos, np.ndarray[DTYPEf_t, ndim=2] vvel, int ffirst, int ssecond, float r, int dist_bin, int vel_bin):
  '''Function to compute the pairwise velocity PDF for 1D case along the z axis.
  Z-axis is taken as the line of sight axis.
  The pairwise velocity calculated in this function is as below:
  v_{12} = (u_{2z}-u_{1z})*sign(r_{2z}-r_{1z})

  Args:
  ----------------------------------------------------------------------------------------
    tree:     The ball tree data structure which was trained on the position data set.
    ppos:     The array containing position of the particles used in the simulation.
    vvel:     The array containing velocities of the particles.
    ffirst:   Denotes the index from where the looping should start, developed in
              keeping mind of the parallelisation of the for-loop.
    ssecond:  Denotes the end index for the for-loop.
    dist_bin: No of bins for the distance. For now the bin size is fixed to 1 h^{-1} Mpc
    vel_bin:  No of bins for the velocity. Binning goes from -(vel_bin/2) to +(vel_bin/2).
              Currently the bin size is 1.

  Returns:
  ----------------------------------------------------------------------------------------
    counter:  A flattened array containing the counts of the pairwise velocity which fall
              into respective bins of distance r.
  '''
  cdef int i, j, leng, jj
  cdef float diff, dist, buff_vel, rubbish_counter
  cdef int offset=(vel_bin/2)
  cdef np.ndarray[DTYPEff_t, ndim=2] counter = np.zeros((dist_bin, vel_bin))

  for i in range(ffirst, ssecond):
      pairs = tree.query_radius(ppos[i], r, return_distance=True)
      leng = len(pairs[0][0])
      buff_vel = vvel[i,2]
      buff_pos = ppos[i,2]
      for j in range(leng):
          jj = pairs[0][0][j]
          if (jj > i):
              dist = pairs[1][0][j]
              diff = ((vvel[jj,2]-buff_vel)*(bool((ppos[jj,2]-buff_pos)>0)-bool((ppos[jj,2]-buff_pos)<0))) + offset
              #offset is added to take care of the negative velocities, should look into a better technique for
              #binning negative numbers. Might need to use copysign for the sign than using bool due to presence of 0.
              if (diff > vel_bin or diff < 0 or dist > r):
                  rubbish_counter += 1
              else:
                  counter[(<int>dist),(<int>diff)]+=1
  
  return counter.flatten()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def pdf_1d(tree, np.ndarray[DTYPEf_t, ndim=2] ppos, np.ndarray[DTYPEf_t, ndim=2] vvel, int ffirst, int ssecond, float r, int dist_bin, int vel_bin):
  '''Function to compute the pairwise velocity PDF for 1D case along the z axis, but without the sign convention.
  Z-axis is taken as the line of sight axis.
  The pairwise velocity calculated in this function is as below:
  v_{12} = (u_{2z}-u_{1z})

  Args:
  ----------------------------------------------------------------------------------------
    tree:     The ball tree data structure which was trained on the position data set.
    ppos:     The array containing position of the particles used in the simulation.
    vvel:     The array containing velocities of the particles.
    ffirst:   Denotes the index from where the looping should start, developed in
              keeping mind of the parallelisation of the for-loop.
    ssecond:  Denotes the end index for the for-loop.
    dist_bin: No of bins for the distance. For now the bin size is fixed to 1 h^{-1} Mpc
    vel_bin:  No of bins for the velocity. Binning goes from -(vel_bin/2) to +(vel_bin/2).
              Currently the bin size is 1.

  Returns:
  ----------------------------------------------------------------------------------------
    counter:  A flattened array containing the counts of the pairwise velocity which fall
              into respective bins of distance r.
  '''
  cdef int i, j, leng, jj
  cdef float diff, dist, buff_vel, rubbish_counter
  cdef int offset=(vel_bin/2)
  cdef np.ndarray[DTYPEff_t, ndim=2] counter = np.zeros((dist_bin,vel_bin))

  for i in range(ffirst, ssecond):
      pairs = tree.query_radius(ppos[i], r, return_distance=True)
      leng = len(pairs[0][0])
      buff_vel = vvel[i,2]
      for j in range(leng):
          jj = pairs[0][0][j]
          if (jj > i):
              dist = pairs[1][0][j]
              diff = (vvel[jj,2] - buff_vel) + offset
              if (diff > vel_bin or diff < 0 or dist > r):
                  rubbish_counter += 1
              else:
                  counter[(<int>dist),(<int>diff)]+=1

  return counter.flatten()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def pdf_1d_projection(tree, np.ndarray[DTYPEf_t, ndim=2] ppos, np.ndarray[DTYPEf_t, ndim=2] vvel, int ffirst, int ssecond, float r, int dist_bin, int vel_bin):
  '''Function to compute the pairwise velocity PDF along the separation vector for 1D case.
  The pairwise velocity calculated in this function is as below:
  v_{12} = ((u_{2x}-u_{1x})\cdot(r_{2x}-r_{1x})) + ((u_{2y}-u_{1y})\cdot(r_{2y}-r_{1y})) + ((u_{2z}-u_{1z})\cdot(r_{2z}-r_{1z}))
          ------------------------------------------------------------------------------------------------------------------------
                                                  |r_{12}|

  Args:
  ----------------------------------------------------------------------------------------
    tree:     The ball tree data structure which was trained on the position data set.
    ppos:     The array containing position of the particles used in the simulation.
    vvel:     The array containing velocities of the particles.
    ffirst:   Denotes the index from where the looping should start, developed in
              keeping mind of the parallelisation of the for-loop.
    ssecond:  Denotes the end index for the for-loop.
    dist_bin: No of bins for the distance. For now the bin size is fixed to 1 h^{-1} Mpc
    vel_bin:  No of bins for the velocity. Binning goes from -(vel_bin/2) to +(vel_bin/2).
              Currently the bin size is 1.

  Returns:
  ----------------------------------------------------------------------------------------
    counter:  A flattened array containing the counts of the pairwise velocity which fall
              into respective bins of distance r.
  '''
  cdef int i, j, leng
  cdef float diff, dist, buff_vel, rubbish_counter
  cdef int offset=(vel_bin/2)
  cdef np.ndarray[DTYPEff_t, ndim=2] counter = np.zeros((dist_bin, vel_bin))

  for i in range(ffirst, ssecond):
      pairs = tree.query_radius(ppos[i], r, return_distance=True)
      leng = len(pairs[0][0])
      for j in range(leng):
          jj = pairs[0][0][j]
          if (jj > i):
              dist = pairs[1][0][j]
              diff = ((((vvel[jj,0]-vvel[i,0])*(ppos[jj,0]-ppos[i,0]))+((vvel[jj,1]-vvel[i,1])*(ppos[jj,1]-ppos[i,1]))+((vvel[jj,2]-vvel[i,2])*(ppos[jj,2]-ppos[i,2])))/dist) + offset
              if (diff > vel_bin or diff < 0 or dist > r):
                  rubbish_counter += 1
              else:
                  counter[(<int>dist),(<int>diff)]+=1

  return counter.flatten()
