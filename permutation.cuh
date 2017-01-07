/* Memxor : dest = dest xor src.
 * Device code.
 */

#include "hash.h"
////////////////////////////////////////////////////////////////////////////////
//! Memxor on device dest = dest xor src
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
#define COLWORDS     (STATEWORDS/8)
#define mul2(x,t) \
{\
  t = x & 0x80808080;\
  x ^= t;\
  x <<= 1;\
  t = t >> 7;\
  t ^= (t << 1);\
  x ^= t;\
  x ^= (t << 3);\
}

__global__ void
permutation(u32 *x, int q, const u32 *columnconstant, const u8 (*shiftvalues)[8], const u8 *S)
{
  // Thread index
  int tx = threadIdx.x;
  //int ty = blockIdx.x;

  __attribute__ ((aligned (8))) u32 tmp[8];
  u32 constant;
  int i;
  for(constant=0; constant<(0x01010101*ROUNDS); constant+=0x01010101)
  {
    if (q==0)
    {
      if (tx<COLWORDS)
        x[tx] ^= columnconstant[tx]^constant;
    }
    else
    {
      if(tx<STATEWORDS)
        x[tx] = ~x[tx];
      if(tx<COLWORDS)
        x[STATEWORDS-COLWORDS+tx] ^= columnconstant[tx]^constant;
    }
    for (i=0; i<8; i++)
    {
      if(tx<COLWORDS)
        tmp[tx] = x[i*COLWORDS+tx];
      if(tx<STATECOLS)
        ((u8*)x)[i*STATECOLS+tx] = S[((u8*)tmp)[(tx+shiftvalues[q][i])%STATECOLS]];
    }

    // mixbytes(((u32(*)[COLWORDS])x), tmp, j);
    
    if (tx<COLWORDS)    	
    {
    	int i;
  		u32 t0, t1, t2;
    	for (i=0; i<8; i++)
    		tmp[i] = ((u32(*)[COLWORDS])x)[i][tx];
    	/* y_i = a_{i+6} */
  		for (i=0; i<8; i++)
  			((u32(*)[COLWORDS])x)[i][tx]=tmp[(i+2)%8];
  		/* t_i = a_i + a_{i+1} */
  		for (i=0; i<7; i++)
    		tmp[i] ^= tmp[(i+1)%8];
  		tmp[7] ^= ((u32(*)[COLWORDS])x)[6][tx];

  		/* y_i = a_{i+6} + t_i */
  		for (i=0; i<8; i++)
    		((u32(*)[COLWORDS])x)[i][tx] ^= tmp[(i+4)%8];

    	/* x_i = t_i + t_{i+3} */
  		t0 = tmp[0];
  		t1 = tmp[1];
  		t2 = tmp[2];
  		for (i=0; i<5; i++)
    		tmp[i] ^= tmp[(i+3)%8];
  		tmp[5] ^= t0;
  		tmp[6] ^= t1;
  		tmp[7] ^= t2;

  		/* z_i = 02 * x_i */
  		for (i=0; i<8; i++)
   			 mul2(tmp[i],t0);

  		/* w_i = z_i + y_{i+4} */
  		for (i=0; i<8; i++)
    		tmp[i] ^= ((u32(*)[COLWORDS])x)[i][tx];

  		/* v_i = 02 * w_i */
  		for (i=0; i<8; i++)
    		mul2(tmp[i],t0);

  		/* b_i = v_{i+3} + y_{i+4} */
  		for (i=0; i<8; i++)
    		((u32(*)[COLWORDS])x)[i][tx] ^= tmp[(i+3)%8];
    }
		    
   }

}
