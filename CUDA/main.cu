#include "hash.h"
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
    char input[INPUT_SIZE+1+NONCE_SIZE];
	char* d_input;
	char* d_nonce;
	
    //position of repeated string in input
    char* base = &(input[NONCE_SIZE]);
    //holder for the nonce
    char nonce[NONCE_SIZE+1];
	
	char *nonce_enum;
	char *dummycall;
	/*  initialize nonce_enum[] using nonce_enumtemp */
	char nonce_enumtemp[NONCE_NUM+1] = {'a'};
	for(int i=1; i<NONCE_NUM+1 ; i++)        // eligible for a new kernal to speed up!
		nonce_enumtemp[i] = next_nonce (nonce_enumtemp[i-1]);
	
	
	
    while(true){
    
        //request new input from server (should be successful, o.w. just retry)
        while(!requestInput(base));

        //init nonce with 'a'*NONCE_SIZE
        for(int i=0;i<NONCE_SIZE;i++)
            nonce[i]='a';
        nonce[NONCE_SIZE]='\0';
        
        //test all possible nonces 
  //      do{
            //copy nonce into input
            for(int i=0;i<NONCE_SIZE;i++)
                input[i]=nonce[i];
	/*	for(num=0;num<INPUT_SIZE+1+NONCE_SIZE;num++){
			printf("%c",input[num]);
		}		
			printf("\n");*/
            //64 chars + '\0' for binary output
           
		   cudaMalloc((char**)&dummycall, 0);
		   /*    Malloc in global memeory  */
            cudaMalloc((char**)&d_input, sizeof(char)*(INPUT_SIZE+1+1));
			cudaMalloc((char**)&d_nonce, sizeof(char));
			cudaMalloc((char**)&nonce_enum, sizeof(char)*62);
			/*   Mem copy  */
			cudaMemcpy(d_input, input, sizeof(char)*(INPUT_SIZE+1+1), 
						cudaMemcpyHostToDevice);
			cudaMemcpy(nonce_enum, nonce_enumtemp, sizeof(char)*62, 
						cudaMemcpyHostToDevice);
			/*   Kernal  */			
			hash<<<62,32>>>((unsigned char*)d_nonce, (unsigned char*)d_input, strlen(input), (unsigned char*)nonce_enum);
			
			/*   Copy result to CPU  */
			cudaMemcpy(nonce, d_nonce, sizeof(char), cudaMemcpyDeviceToHost);
			
			/*   Free mem  */
			cudaFree(d_nonce);
			cudaFree(d_input);
			cudaFree(nonce_enum);
            
			
        validateHash(base, nonce);
		
    }

}

int main(int argc, char *argv[]){


   	if ((argc==2) && (strcmp(argv[1],"-benchmark")==0) ){
        benchmark();
    }
   
     else if (argc==4){
        //64 chars for 512bit output      
        char* nonce = argv[1];
        int nonce_size=strlen(nonce);
        char* baseInput = argv[2];
        int baseInputSize=strlen(baseInput);
        int muliplier=atoi(argv[3]);

		char* d_input;
	    char* d_nonce;
		char *nonce_enum;
	/*  initialize nonce_enum[] using nonce_enumtemp */
	char nonce_enumtemp[NONCE_NUM+1] = {'a'};
	for(int i=1; i<NONCE_NUM+1 ; i++)        // eligible for a new kernal to speed up!
		nonce_enumtemp[i] = next_nonce (nonce_enumtemp[i-1]);
		
        //nonce first and append input string desired number of times
        char* input = (char*)malloc(sizeof(char)*(baseInputSize*muliplier+nonce_size+1));
        for(int i=0;i<nonce_size;i++)
            input[i]=nonce[i];
        char* repeat_ptr=&(input[nonce_size]);
        for(int j=0;j<muliplier;j++)
            for(int i=0;i<baseInputSize;i++)
                repeat_ptr[j*baseInputSize+i]=baseInput[i];
        input[baseInputSize*muliplier+nonce_size]='\0';
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
		
		/*    Malloc in global memeory  */
            cudaMalloc((char**)&d_input, sizeof(char)*(INPUT_SIZE+1+1));
			cudaMalloc((char**)&d_nonce, sizeof(char));
			cudaMalloc((char**)&nonce_enum, sizeof(char)*62);
			/*   Mem copy  */
			cudaMemcpy(d_input, input, sizeof(char)*(INPUT_SIZE+1+1), 
						cudaMemcpyHostToDevice);
			cudaMemcpy(nonce_enum, nonce_enumtemp, sizeof(char)*62, 
						cudaMemcpyHostToDevice);
			/*   Kernal  */	
    cudaEventRecord(start);			
			hash<<<62,32>>>((unsigned char*)d_nonce, (unsigned char*)d_input, strlen(input), (unsigned char*)nonce_enum);
    cudaEventRecord(stop);
			/*   Copy result to CPU  */
			cudaMemcpy(nonce, d_nonce, sizeof(char), cudaMemcpyDeviceToHost);
			
	cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
			
//*			printf("%c",nonce[0]);

	printf("Kernal Time: %f\n", milliseconds);
//*			printf("Effective Bandwidth (GB/s): %fn", N*4*3/milliseconds/1e6);
			/*   Free mem  */
			cudaFree(d_nonce);
			cudaFree(d_input);
			cudaFree(nonce_enum);
        
        free(input);
    }else{
        printf("usage: %s nonce (string) input(string) multiplier(int)\n", argv[0]);
        printf("------------OR-------------\n");
        printf("usage: %s -benchmark\n", argv[0]);
    }

	return 0;
}

