/* Memxor : dest = dest xor src.
 * Device code.
 */
#include "hash.h"
#define BYTESLICE(i) (((i)%8)*STATECOLS+(i)/8) 

struct state {
  u8 bytes_in_block;
  u8 first_padding_block;
  u8 last_padding_block;
};

////////////////////////////////////////////////////////////////////////////////
//! Memxor on device dest = dest xor src
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void
setmessage(u8* d_buffer, const u8* in, struct state s, unsigned long long inlen)
{
  // Thread index
  int tx = threadIdx.x;

  // Write the result to device memory;
  // each thread writes one element
  if (tx < s.bytes_in_block){
  	d_buffer[tx] = in [tx];
  }

  if (s.bytes_in_block != STATEBYTES)
  {
    if (s.first_padding_block)
    {
      d_buffer[BYTESLICE(s.bytes_in_block)] = 0x80;
    }

    if(tx<STATEBYTES && tx>s.bytes_in_block)
      d_buffer[BYTESLICE(tx)] = 0;

    if (s.last_padding_block)
    {
      inlen /= STATEBYTES;
      inlen += (s.first_padding_block==s.last_padding_block) ? 1 : 2;
      if(tx>STATEBYTES-9 && tx<STATEBYTES)
        d_buffer[BYTESLICE(tx)] = (inlen >> 8*(STATEBYTES-tx-1)) & 0xff;
    }
  }
  //__syncthreads();

}
