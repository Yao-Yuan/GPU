// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#include "hash.h"
#include "tables.h"
#include "memxor.cuh"
#include "permutation.cuh"
//#include "setmessage.cuh"

#if CRYPTO_BYTES<=32
static const  u32 columnconst[2] = { 0x30201000, 0x70605040 };
static const  u8 shiftval[2][8] = { {0, 1, 2, 3, 4, 5, 6, 7}, {1, 3, 5, 7, 0, 2, 4, 6} };
__constant__  u32 d_columnconst[2];
__constant__  u8 d_shiftval[2][8];
#define     COLUMN_SIZE (4*2)
#define     SHIFT_SIZE (8*2)
#else
static const  u32 columnconst[4] = { 0x30201000, 0x70605040, 0xb0a09080, 0xf0e0d0c0 };
static const  u8 shiftval[2][8] = { {0, 1, 2, 3, 4, 5, 6, 11}, {1, 3, 5, 11, 0, 2, 4, 6} };
__constant__  u32 d_columnconst[4];
__constant__  u8 d_shiftval[2][8];
#define     COLUMN_SIZE (4*4)
#define     SHIFT_SIZE (8*2)
#endif

#define MAX_GPU_IN_SIZE  1024

//define talbe
__constant__ u8 d_S[256];

#define BYTESLICE(i) (((i)%8)*STATECOLS+(i)/8) 

struct state {
  u8 bytes_in_block;
  u8 first_padding_block;
  u8 last_padding_block;
};

void setmessage(u8* buffer, const u8* in, struct state s, unsigned long long inlen)
{
  int i;
  for (i = 0; i < s.bytes_in_block; i++)
    buffer[BYTESLICE(i)] = in[i];

  if (s.bytes_in_block != STATEBYTES)
  {
    if (s.first_padding_block)
    {
      buffer[BYTESLICE(i)] = 0x80;
      i++;
    }

    for(;i<STATEBYTES;i++)
      buffer[BYTESLICE(i)] = 0;

    if (s.last_padding_block)
    {
      inlen /= STATEBYTES;
      inlen += (s.first_padding_block==s.last_padding_block) ? 1 : 2;
      for(i=STATEBYTES-8;i<STATEBYTES;i++)
        buffer[BYTESLICE(i)] = (inlen >> 8*(STATEBYTES-i-1)) & 0xff;
    }
  }
}


int hash(unsigned char *out, const unsigned char *in, unsigned long long inlen)
{
  __attribute__ ((aligned (8))) u32 ctx[STATEWORDS];
  __attribute__ ((aligned (8))) u32 buffer[STATEWORDS];
  unsigned long long rlen = inlen;
  unsigned int d_in_len = (inlen > MAX_GPU_IN_SIZE) ? MAX_GPU_IN_SIZE: inlen;
  unsigned long long offset = 0;
  struct state s = { STATEBYTES, 0, 0 };
  u8 i;

  /* set inital value */
  for(i=0;i<STATEWORDS;i++)
    ctx[i] = 0;
  ((u8*)ctx)[BYTESLICE(STATEBYTES-2)] = ((CRYPTO_BYTES*8)>>8)&0xff;
  ((u8*)ctx)[BYTESLICE(STATEBYTES-1)] = (CRYPTO_BYTES*8)&0xff;

  /****************************************************/
  /*  Preparations                                    */
  /****************************************************/

  //printf("[Hash Using CUDA] - Starting...\n");

  // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
  int devID = 0;
  cudaError_t error;
  cudaDeviceProp deviceProp;
  error = cudaGetDevice(&devID);

  if (error != cudaSuccess)
  {
      printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
      exit(1);
  }

  error = cudaGetDeviceProperties(&deviceProp, devID);

  if (deviceProp.computeMode == cudaComputeModeProhibited)
  {
      fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
      exit(1);
  }

  if (error != cudaSuccess)
  {
      printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
      exit(1);
  }


  // utilities
  cudaEvent_t start;
  cudaEvent_t stop;
  float msecTotal;
  // create and start timer
  cudaEventCreate(&start);
  cudaEventRecord(start, NULL); 

  // allocate host memory for string
  unsigned int mem_size_ctx = sizeof(u32) * STATEWORDS;
  unsigned int mem_size_buffer = sizeof(u32) * STATEWORDS;
  float flop = (float)mem_size_buffer;

  // allocate device memory
  u32* d_ctx;
  cudaMalloc((void**) &d_ctx, mem_size_ctx);
  u32* d_buffer;
  cudaMalloc((void**) &d_buffer, mem_size_buffer);
  u8* d_in;
  cudaMalloc((void**)&d_in, MAX_GPU_IN_SIZE);
  

  cudaMemcpyToSymbol (d_columnconst, columnconst, COLUMN_SIZE);
  cudaMemcpyToSymbol (d_shiftval, shiftval, SHIFT_SIZE);
  cudaMemcpyToSymbol (d_S, S, 256);
  cudaMemcpy(d_in, in, d_in_len, cudaMemcpyHostToDevice);

  /* iterate compression function */
  while(s.last_padding_block == 0)
  {
    if (rlen<STATEBYTES)
    {
      if (s.first_padding_block == 0)
      {
        s.bytes_in_block = rlen;
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
    else
      rlen-=STATEBYTES;

    setmessage((u8*)buffer, in, s, inlen);

    cudaMemcpy(d_ctx, ctx, mem_size_ctx,cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer, buffer, mem_size_buffer,cudaMemcpyHostToDevice);

    d_in = d_in + offset;

    /* compression function */
    
    //setmessage<<<1, STATEBYTES>>>((u8*)d_buffer, d_in, s, inlen);
    //memxor(buffer, ctx, STATEWORDS);
    
    // naive implementation
    memxor<<< 1, STATEWORDS >>>(d_buffer,d_ctx, STATEWORDS);


    permutation<<<1, STATEWORDS>>>(d_buffer, 0, d_columnconst, d_shiftval, d_S);

    // naive implementation
    memxor<<< 1, STATEWORDS >>>(d_ctx,d_buffer, STATEWORDS);

    cudaMemcpy(buffer, d_buffer, mem_size_buffer,cudaMemcpyDeviceToHost);

    setmessage((u8*)buffer, in, s, inlen);

    cudaMemcpy(d_buffer, buffer, mem_size_buffer,cudaMemcpyHostToDevice);
    //setmessage<<<1, STATEBYTES>>>((u8*)buffer, d_in, s, inlen);
    permutation<<<1, STATEWORDS>>>(d_buffer, 1, d_columnconst, d_shiftval, d_S);
    // naive implementation
    memxor<<< 1, STATEWORDS >>>(d_ctx,d_buffer, STATEWORDS);

    // copy result from device to host
    //cudaMemcpy(ctx, d_ctx, mem_size_ctx,cudaMemcpyDeviceToHost);

    /* increase message pointer */
    in += STATEBYTES;
    offset += STATEBYTES;
    
    /*if ((rlen > 0) && ((offset + STATEBYTES) > MAX_GPU_IN_SIZE))
    {
      //copy input from host to device
      d_in_len = (rlen > MAX_GPU_IN_SIZE) ? MAX_GPU_IN_SIZE : rlen;
      
      cudaMemcpy(d_in, in, d_in_len, cudaMemcpyHostToDevice);
      
    }*/
  }

  /* output transformation */
  cudaMemcpy(d_buffer, d_ctx, STATEBYTES, cudaMemcpyDeviceToDevice);
  permutation<<<1, STATEWORDS>>>(d_buffer, 0, d_columnconst, d_shiftval, d_S);
  // naive implementation
  memxor<<< 1, STATEWORDS >>>(d_ctx,d_buffer, STATEWORDS);

  // copy result from device to host
  cudaMemcpy(ctx, d_ctx, mem_size_buffer,cudaMemcpyDeviceToHost);

  /* return truncated hash value */
  for (i = STATEBYTES-CRYPTO_BYTES; i < STATEBYTES; i++)
    out[i-(STATEBYTES-CRYPTO_BYTES)] = ((u8*)ctx)[BYTESLICE(i)];

  
  // stop and destroy timer
  cudaEventCreate(&stop);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
 // printf("GPU memxor\n");
 // printf("Processing time: %f (ms), GFLOPS: %f \n", msecTotal, flop / msecTotal/ 1e+6);

  /****************************************************/
  /*  Cleaning                                        */
  /****************************************************/

  // clean up memory
  //cudaFree(d_ctx);
  //cudaFree(d_buffer);

  //cudaThreadExit();

  return 0;
}
