#include "hash.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "common.h"
#include "interface.h"
#include <stdlib.h>


bool next_nonce(char* c){
    if(c[0]=='\0'){
        return false;
    }
    
    //if end of range, wrap around and increment next 'digit'
    if(c[0]=='9'){
        c[0]='a';
		//the compiler may generate an 'above array bounds' warning here, which can be safely ignored
		if (next_nonce(&c[1]))
            return true;
        c[0]='\0';
        return false;
    }   

    //jump boundaries
    if(c[0]=='z')
        c[0]='A';
    else if(c[0]=='Z')
        c[0]='0';
    else
        c[0]++;

    return true;
}



void benchmark(void){
    // + 1 for termination '\0'
    char input[INPUT_SIZE+NONCE_SIZE+1];
	char* d_input;
	char* result;  
	char* d_result;
    //position of repeated string in input
    char* base = &(input[NONCE_SIZE]);
	
	
    //holder for the nonce
 //   char nonce[NONCE_SIZE+1];

	
	/*  initialize nonce_enum[] using nonce_enumtemp */
	
	char nonce_enumtemp[NONCE_NUM] = {0};
	
	for(int i=1; i<NONCE_NUM ; i++)        // eligible for a new kernal to speed up!
	{
		 nonce_enumtemp[0] = 'a';
		nonce_enumtemp[i] = next_nonce ((char *)&nonce_enumtemp[i-1]);
	}
	    cudaMemcpyToSymbol(nonce_enum, nonce_enumtemp, sizeof(char) * NONCE_NUM);      
	
	
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
		result[0] = '\0';
		result[1] = '\0';
		
		
		
		
		cudaMalloc((char**)&d_input, sizeof(char)*(INPUT_SIZE+NONCE_SIZE+1));
		cudaMalloc((char**)&d_result, sizeof(unsigned char));
			
		cudaMemcpy(d_input, input, sizeof(char)*(INPUT_SIZE+NONCE_SIZE+1), cudaMemcpyHostToDevice);
		cudaMemcpy(d_result, result, sizeof(unsigned char), cudaMemcpyHostToDevice);
		
		attempt<<< 62 , 1 >>>((unsigned char*)d_result, (unsigned char*)d_input, strlen(input)); 
		cudaMemcpy(result, d_result, sizeof(char)*65, cudaMemcpyDeviceToHost);
		
		result[0] = 'a';
		result[1] = '\0';
		
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

