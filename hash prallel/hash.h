#ifndef HASH_H
#define HASH_H

//Some shorthands
#include <stdint.h>
typedef uint8_t u8;
typedef uint32_t u32;
#define  NONCE_NUM 62

__constant__ char nonce_enum[NONCE_NUM];

//hash settings
#define CRYPTO_BYTES (64)
#define ROUNDS     (14)
#define STATEBYTES (128)
#define STATEWORDS (STATEBYTES/4)
#define STATECOLS  (STATEBYTES/8)

//the hash function
//__device__ int hash(unsigned char *out, const unsigned char *in, unsigned long long inlen);
__global__ void attempt( unsigned char *d_result, unsigned char *d_input, unsigned long long inlen );
#endif
