#include "hash.h"
#include "tables.h"
#include <stdio.h>
#define COLWORDS     (STATEWORDS/8)
#define BYTESLICE(i) (((i)%8)*STATECOLS+(i)/8)

#if CRYPTO_BYTES<=32
__device__ static const u32 columnconstant[2] = { 0x30201000, 0x70605040 };
__device__ static const u8 shiftvalues[2][8] = { {0, 1, 2, 3, 4, 5, 6, 7}, {1, 3, 5, 7, 0, 2, 4, 6} };
#else
__device__ static const u32 columnconstant[4] = { 0x30201000, 0x70605040, 0xb0a09080, 0xf0e0d0c0 };
__device__ static const u8 shiftvalues[2][8] = { {0, 1, 2, 3, 4, 5, 6, 11}, {1, 3, 5, 11, 0, 2, 4, 6} };
#endif

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
__device__ void mixbytes(u32 a[8][COLWORDS], u32 b[8], int s)
{
  int i;
  u32 t0, t1, t2;
 #pragma unroll
  for (i=0; i<8; i++)
    b[i] = a[i][s];

  /* y_i = a_{i+6} */
  #pragma unroll
  for (i=0; i<8; i++)
    a[i][s] = b[(i+2)%8];

  /* t_i = a_i + a_{i+1} */
  #pragma unroll
  for (i=0; i<7; i++)
    b[i] ^= b[(i+1)%8];
  b[7] ^= a[6][s];

  /* y_i = a_{i+6} + t_i */
  #pragma unroll
  for (i=0; i<8; i++)
    a[i][s] ^= b[(i+4)%8];

  /* y_i = y_i + t_{i+2} */
  #pragma unroll
  for (i=0; i<8; i++)
    a[i][s] ^= b[(i+6)%8];

  /* x_i = t_i + t_{i+3} */
  t0 = b[0];
  t1 = b[1];
  t2 = b[2];
  #pragma unroll
  for (i=0; i<5; i++)
    b[i] ^= b[(i+3)%8];
  b[5] ^= t0;
  b[6] ^= t1;
  b[7] ^= t2;

  /* z_i = 02 * x_i */
  #pragma unroll
  for (i=0; i<8; i++)
    mul2(b[i],t0);

  /* w_i = z_i + y_{i+4} */
  #pragma unroll
  for (i=0; i<8; i++)
    b[i] ^= a[i][s];

  /* v_i = 02 * w_i */
  #pragma unroll
  for (i=0; i<8; i++)
    mul2(b[i],t0);

  /* b_i = v_{i+3} + y_{i+4} */
  #pragma unroll
  for (i=0; i<8; i++)
    a[i][s] ^= b[(i+3)%8];
}
__device__ void permutation(u32 *x, int q)
{
  __attribute__ ((aligned (8))) u32 tmp[8];
  u32 constant;
  int i;
  u32 prefetch_col = columnconstant[threadIdx.x];
  for(constant=0; constant<(0x01010101*ROUNDS); constant+=0x01010101)
  {

    if (q==0)
    {
     /* for (j=0; j<COLWORDS; j++)
        x[j] ^= columnconstant[j]^constant;*/
		if(threadIdx.x<COLWORDS){
			x[threadIdx.x] ^=prefetch_col^constant;			
		}
		__syncthreads();
    }
    else
    {
      /*for(i=0;i<STATEWORDS;i++)
        x[i] = ~x[i];*/
	  if(threadIdx.x<STATEWORDS){
			x[threadIdx.x] = ~x[threadIdx.x];
		}	
		__syncthreads();
     /* for (j=0; j<COLWORDS; j++)
        x[STATEWORDS-COLWORDS+j] ^= columnconstant[j]^constant;
*/
		if(threadIdx.x<COLWORDS){
			x[STATEWORDS-COLWORDS+threadIdx.x] ^= prefetch_col^constant;			
		}
		__syncthreads();
    }
	
	if(threadIdx.x<8){
//		#pragma unroll
	  for (i=0; i<COLWORDS; i++)
        tmp[i] = x[threadIdx.x*COLWORDS+i];
//	#pragma unroll
      for (i=0; i<STATECOLS; i++)
        ((u8*)x)[threadIdx.x*STATECOLS+i] = S[((u8*)tmp)[(i+shiftvalues[q][threadIdx.x])%STATECOLS]];
	}
	__syncthreads();	
			

	  mixbytes((u32(*)[4])x, tmp,threadIdx.x%4);
//	}
	__syncthreads();
  }
}

__device__ void memxor(u32* dest, const u32* src)
{
  
  	dest[threadIdx.x] ^= src[threadIdx.x];
  __syncthreads();
  
}

struct state {
  u8 bytes_in_block;
  u8 first_padding_block;
  u8 last_padding_block;
};

__device__ void setmessage(u8* buffer, const u8* in, struct state s, unsigned long long inlen, bool flag, unsigned char *nonce)
{
  int i;
  if(flag==0){
	 buffer[0]=nonce[blockIdx.x]; 
	 #pragma unroll
  for (i = 1; i < s.bytes_in_block; i++)
    buffer[BYTESLICE(i)] = in[i];
  
  }else {
	  #pragma unroll
	 for (i = 0; i < s.bytes_in_block; i++)
    buffer[BYTESLICE(i)] = in[i];
  }

  if (s.bytes_in_block != STATEBYTES)
  {
    if (s.first_padding_block)
    {
      buffer[BYTESLICE(i)] = 0x80;
      i++;
    }
	#pragma unroll
    for(;i<STATEBYTES;i++)
      buffer[BYTESLICE(i)] = 0;

    if (s.last_padding_block)
    {
      inlen /= STATEBYTES;
      inlen += (s.first_padding_block==s.last_padding_block) ? 1 : 2;
	  #pragma unroll
      for(i=STATEBYTES-8;i<STATEBYTES;i++)
        buffer[BYTESLICE(i)] = (inlen >> 8*(STATEBYTES-i-1)) & 0xff;
    }
  }
}

__device__ bool check_hash(char* hash){
    //check if first n-characters are zero
	#pragma unroll
    for(int i=0;i<3;i++)
		//Note: each 'char' of 8 bits contains 2 hex characters representing 4 bits each.
		//Hence all this bit shuffling
		if ((hash[i>>1]&(0xF0>>((i&0x1)<<2)))!=0)
            return false;
    return true;
}

__global__ void hash(unsigned char *out, const unsigned char *in, unsigned long long inlen, unsigned char *nonce)
{
  int tid = threadIdx.x;
  //__shared__ char temp;
  __shared__ char output_hash[65];
  out[0]='\0';

 __shared__ __attribute__ ((aligned (8))) u32 ctx[STATEWORDS];
  __shared__ __attribute__ ((aligned (8))) u32 buffer[STATEWORDS];
 // unsigned long long rlen = inlen;
  struct state s = { STATEBYTES, 0, 0 };
  uint16_t i;
  bool ini_flag=0;


    ctx[tid] = 0;
  ((u8*)ctx)[BYTESLICE(STATEBYTES-2)] = ((CRYPTO_BYTES*8)>>8)&0xff;
  ((u8*)ctx)[BYTESLICE(STATEBYTES-1)] = (CRYPTO_BYTES*8)&0xff;

  /* iterate compression function */
  #pragma unroll
	for(i = 0 ;i<1025; i++)
//  while(s.last_padding_block == 0)
  {
    if (i==1024)
    {
      if (s.first_padding_block == 0)
      {
        s.bytes_in_block = 1;
        s.first_padding_block = 1;
        s.last_padding_block = (s.bytes_in_block < STATEBYTES-8) ? 1 : 0;
      }
      else
      {
        s.bytes_in_block = 0;
        s.first_padding_block = 0;
        s.last_padding_block = 1;
      }
    }
	
    /* compression function */
		setmessage((u8*)buffer, in, s, inlen, ini_flag, nonce);
		//__syncblocks();
		memxor(buffer, ctx);
		permutation(buffer, 0);
		memxor(ctx, buffer);
		setmessage((u8*)buffer, in, s, inlen, ini_flag, nonce);
		//__syncblocks();
		permutation(buffer, 1);
		memxor(ctx, buffer);
		ini_flag=1;

    /* increase message pointer */
    in += STATEBYTES;
	
  }

  /* output transformation */
    buffer[tid] = ctx[tid];
    permutation(buffer, 0);
    memxor(ctx, buffer);

  /* return truncated hash value */
    output_hash[tid]=((u8*)ctx)[BYTESLICE(tid+64)];
	output_hash[tid+32]=((u8*)ctx)[BYTESLICE(tid+96)];

   if(check_hash(output_hash))
	   out[0]=nonce[blockIdx.x];
}