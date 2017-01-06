#include "hash.cuh"
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "common.h"
#include "interface.h"
#include <stdlib.h>


char next_nonce(char c){
    if(c =='\0'){
        return false;
    }
    
    //jump boundaries
    if(c =='z')
        c ='A';
    else if(c =='Z')
        c ='0';
	else if(c =='9')
        c ='\0';
    else
        c++;
    return c ;
}



void benchmark(void){
    // + 1 for termination '\0'
    char input[INPUT_SIZE+NONCE_SIZE+1];
	char* d_input;
	char* result;  
	char* d_result;
    //position of repeated string in input
    char* base = &(input[NONCE_SIZE]);
	char *nonce_enum;
	
    //holder for the nonce
 //   char nonce[NONCE_SIZE+1];

	
	/*  initialize nonce_enum[] using nonce_enumtemp */

	
	char nonce_enumtemp[NONCE_NUM+1] = {0};
	nonce_enumtemp[0] = 'a';
	for(int i=1; i<NONCE_NUM ; i++)        // eligible for a new kernal to speed up!
	{
		 
		nonce_enumtemp[i] = next_nonce (nonce_enumtemp[i-1]);
	}
	  //  cudaMemcpyToSymbol(nonce_enum, nonce_enumtemp, sizeof(char) * (NONCE_NUM+1));    

		//nonce_enumtemp[0] = 'b';
	//	cudaMemcpyFromSymbol(nonce_enumtemp, nonce_enum, sizeof(char) * (NONCE_NUM+1)); 
	
    while(true){
    
        //request new input from server (should be successful, o.w. just retry)
       while(!requestInput(base));
		//printf("ab");
		
		/*
        //init nonce with 'a'*NONCE_SIZE
        for(int i=0;i<NONCE_SIZE;i++)
            nonce[i]='a';
        nonce[NONCE_SIZE]='\0';
        */
		result = (char *)malloc( sizeof(char)*2 );
		result[0] = '0';
		result[1] = '\0';
		
		
		cudaMalloc((char**)&d_input, sizeof(char)*(INPUT_SIZE+NONCE_SIZE+1));
		cudaMalloc((char**)&d_result, sizeof(char)*2);
		cudaMalloc((char**)&nonce_enum, sizeof(char)*(NONCE_NUM+1));
		
		cudaMemcpy(d_input, input, sizeof(char)*(INPUT_SIZE+NONCE_SIZE+1), cudaMemcpyHostToDevice);
		cudaMemcpy(d_result, result, sizeof(char)*2, cudaMemcpyHostToDevice);
		cudaMemcpy(nonce_enum, nonce_enumtemp, sizeof(char)*(NONCE_NUM+1), cudaMemcpyHostToDevice);
		
		attempt<<< 1 , 1 >>>((char*)d_result, (unsigned char*)d_input, sizeof(unsigned char)*32, nonce_enum ); 
		cudaMemcpy(result, d_result, sizeof(char)*2, cudaMemcpyDeviceToHost);
	//	result[0] = d_result[0];
		
		cudaFree(d_result);
		cudaFree(d_input);
	
		validateHash(base, result);
	
	
	
	}

}

int main(int argc, char *argv[]){
    
   	if ((argc==2) && (strcmp(argv[1],"-benchmark")==0) ){
        benchmark();
    }
   
  /*  else if (argc==4){
        //64 chars for 512bit output
        unsigned char output_hash[64+1];
        
        char* nonce = argv[1];
        int nonce_size=strlen(nonce);
        char* baseInput = argv[2];
        int baseInputSize=strlen(baseInput);
        int muliplier=atoi(argv[3]);
		char* d_input;
		char* d_output_hash;
        //nonce first and append input string desired number of times
        char* input = (char*)malloc(sizeof(char)*(baseInputSize*muliplier+nonce_size+1));
        for(int i=0;i<nonce_size;i++)
            input[i]=nonce[i];
        char* repeat_ptr=&(input[nonce_size]);
        for(int j=0;j<muliplier;j++)
            for(int i=0;i<baseInputSize;i++)
                repeat_ptr[j*baseInputSize+i]=baseInput[i];
        input[baseInputSize*muliplier+nonce_size]='\0';

        //do hash
		cudaMalloc((char**)&d_input, sizeof(char)*(baseInputSize*muliplier+nonce_size+1));
		cudaMalloc((char**)&d_output_hash, sizeof(char)*65);
		cudaMemcpy(d_input, input, sizeof(char)*(baseInputSize*muliplier+nonce_size+1), 
						cudaMemcpyHostToDevice);
        hash<<<64,1>>>((unsigned char*)d_output_hash, (unsigned char*)d_input, baseInputSize*muliplier+nonce_size );
		cudaMemcpy(output_hash, d_output_hash, sizeof(char)*65, cudaMemcpyDeviceToHost);
        
        for(int i=0;i<64;i++)
			printf("%02X",output_hash[i]);
		cudaFree(d_input);
		cudaFree(d_output_hash);
        free(input);

    }*/
/*	else{
        printf("usage: %s nonce (string) input(string) multiplier(int)\n", argv[0]);
        printf("------------OR-------------\n");
        printf("usage: %s -benchmark\n", argv[0]);
    }
*/
	return 0;
}

