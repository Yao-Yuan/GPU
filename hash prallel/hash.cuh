#ifndef HASH_H
#define HASH_H

//Some shorthands
#include <stdint.h>
typedef uint8_t u8;
typedef uint32_t u32;
#define  NONCE_NUM 62

//extern char nonce_enum[NONCE_NUM+1];

//hash settings
#define CRYPTO_BYTES (64)
#define ROUNDS     (14)
#define STATEBYTES (128)
#define STATEWORDS (STATEBYTES/4)
#define STATECOLS  (STATEBYTES/8)

//the hash function
//__device__ int hash(unsigned char *out, const unsigned char *in, unsigned long long inlen);
__global__ void attempt(char *d_result, unsigned char *d_input, unsigned long long inlen, char *nonce_enum );
#endif
