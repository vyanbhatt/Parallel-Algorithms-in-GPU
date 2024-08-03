
/**
*   CS6023: GPU Programming 
*   Assignment 2
*   
*   Please don't change any existing code in this file.
*
*   Please add necessary memory APIs for your implementation. Use cudaFree() 
*   to free up memory as soon as you're done with an allocation. 
*   This will ensure that you don't run out of memory while running
*   large test cases. Use the minimum required memory for your 
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

using std::cin;
using std::cout;

typedef long long ll;


__global__ void Calculate_Convolution(long long int* g_ans,long long int* g_mat,long long int *g_filter,int m,int n,int k){

    //Creating shared memory for filter
    extern __shared__ long long int filter[];

    //Filling the filter in shared memory,to get maximum coalesced access 
    for(int i=threadIdx.x ;i<(k*k);i+=1024){
      filter[i] = g_filter[i];
    }

    int id = blockIdx.x*blockDim.x + threadIdx.x;

    //If the thread is outside the bound of m*n
    if(id>=(m*n))
      return;

    int row = id/n,col = id%n;
    int range = k/2;
    long long int sum = 0;
    int i = 0;
    
    //Waiting for the other threads of block to fill the values in filter
    __syncthreads();

    for(int x = row - range ; x < min(row+1+range,m) ; x++,i++){
        if(x<0)
            continue;
        for(int y = col - range ,j=0; y < min(col+1+range,n) ; y++,j++){
            if(y<0)
                continue;

            long long int temp = filter[i*k + j]*g_mat[x*n+y];
            sum += temp;
        }
    }

    g_ans[row*n + col] = sum;
}


int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long long int* h_mat = new long long int[m * n];
    long long int* h_filter = new long long int[k * k];
    long long int* h_ans = new long long int[m * n];

    //Gpu variables
    long long int* g_mat,*g_ans,*g_filter;
    
    int N_blocks = ceil(float(m*n)/1024);

    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    cudaMalloc(&g_mat,(m*n)*sizeof(long long int));
    cudaMalloc(&g_ans,(m*n)*sizeof(long long int));
    cudaMemset(g_ans, 0, m*n*sizeof(long long int));
    cudaMalloc(&g_filter,(k*k)*sizeof(long long int));


    cudaMemcpy(g_mat,h_mat,(m*n)*sizeof(long long int),cudaMemcpyHostToDevice);
    cudaMemcpy(g_filter,h_filter,(k*k)*sizeof(long long int),cudaMemcpyHostToDevice);

    /**
     * 
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     * 
    **/

    /****************************************************Start Here***********************************************************/
    
    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch
      
      Calculate_Convolution<<<N_blocks,1024,(k*k)*sizeof(long long int)>>>(g_ans,g_mat,g_filter,m,n,k);
      cudaDeviceSynchronize();
      
    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch
    
    cudaMemcpy(h_ans,g_ans,m*n*sizeof(long long int),cudaMemcpyDeviceToHost);


    //Freeing all the gpu variables
    cudaFree(g_mat);
    cudaFree(g_ans);
    cudaFree(g_filter);
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */


    
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}