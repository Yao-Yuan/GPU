/* Memxor : dest = dest xor src.
 * Device code.
 */


////////////////////////////////////////////////////////////////////////////////
//! Memxor on device dest = dest xor src
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void
memxor(u32* dest, const u32* src, unsigned int n)
{
  // Thread index
  int tx = threadIdx.x;

  // Write the result to device memory;
  // each thread writes one element
  if (tx < n){
  	dest[tx] ^= src[tx];
  }
  //__syncthreads();

}
