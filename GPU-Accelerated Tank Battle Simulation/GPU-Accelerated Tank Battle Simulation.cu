#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here

__global__ void Fire(volatile int *flag,int *live,int *g_score,int *g_x,int *g_y,int *g_health,int T,int K,int *NL){

    int tid = threadIdx.x;
  
     if(live[tid]){
        int dir = (tid+K)%T;
        int target = -1;
        
        long long int Gdistance = INT_MAX;
        long long int X = g_x[dir] - g_x[tid];
        long long int Y = g_y[dir] - g_y[tid];
        for(int j=0;j<T;j++){

            if(live[j]==0 || j==tid)
                continue;
                  
            long long int x = g_x[j] - g_x[tid];
            long long int y= g_y[j] - g_y[tid];
            long long int Tdistance= ((x * x) + (y * y));
            if((target == -1 || !live[target] || Tdistance<Gdistance)){
                if(y==0 && Y==0 && (x*X)>0){
                      target = j;
                      Gdistance = Tdistance;
                } 
                if((X*y==Y*x) && (y*Y>0)){
                      target = j;
                      Gdistance = Tdistance;
                }
            }
          }
        
        if(target!=-1){
              atomicAdd(g_score+tid,1);
              atomicSub(g_health+target,1);
        }   
      }

      __syncthreads();

      if(tid<T && live[tid] && g_health[tid]<=0 && live[tid]){
          live[tid] = 0;
          atomicSub(NL,1);
      }

    __syncthreads();
    
    (*flag) = 1;
    
}

__global__ void Play(volatile int *flag,int *live,int *g_score,int *g_x,int *g_y,int *g_health,int T,int *Nlive){

    int K = 1;
    int N_L = T;
    (*Nlive) = T;

    while(N_L>1) {
        if(K % T == 0) {
            K++;
            continue;
        }

        Fire<<<1,T>>>(flag,live,g_score,g_x,g_y,g_health,T,K,Nlive);

        while((*flag) == 0);
        
        (*flag) = 0;
        N_L = *Nlive;
        K++;
    } 
}


//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    volatile int *flag;
    int *Nlive;
    
    cudaMalloc(&Nlive,sizeof(int));
    cudaMalloc(&flag,sizeof(int));

    thrust::device_vector<int>live(T,1);
    thrust::device_vector<int>g_score(T,0);
    thrust::device_vector<int>g_x(T);
    thrust::device_vector<int>g_y(T);
    thrust::device_vector<int>g_health(T,H);

    thrust::copy(xcoord,xcoord+T,g_x.begin());
    thrust::copy(ycoord,ycoord+T,g_y.begin());
    
    Play<<<1,1>>>(flag,thrust::raw_pointer_cast(live.data()),thrust::raw_pointer_cast(g_score.data()),thrust::raw_pointer_cast(g_x.data()),thrust::raw_pointer_cast(g_y.data()),thrust::raw_pointer_cast(g_health.data()), T,Nlive);
    cudaDeviceSynchronize();
    cudaFree(Nlive);
    thrust::copy(g_score.begin(),g_score.end(),score);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}