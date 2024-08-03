/*
	CS 6023 Assignment 3. 
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>


void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input. 
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
	

	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ; 
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL; 
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}
	
	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}


__global__ void Update(int l,int *gUpdatedX,int *gUpdatedY,int *q,int children,int parent){

    long int tid = blockDim.x*blockIdx.x+threadIdx.x;

    if(tid>=children)
      return;

    int ind = l+tid;
    int node = q[ind];

    gUpdatedX[node] += gUpdatedX[parent];
    gUpdatedY[node] += gUpdatedY[parent];

}

 __global__ void propogate(int *gUpdatedX,int *gUpdatedY,int *q,int *gOff,int *gCsr){

        long int l = 0,r=1;
        q[0] = 0;
        while(l<r){
            int size = r - l;

            while(size--){
                  int node = q[l];
                  l++;
                  int start = gOff[node];
                  int end = gOff[node+1];
                  int children = end - start;
                  int N_b = ceil(float(children)/ 1024);
                  for( ; start<end;start++){
                      q[r] = gCsr[start];
                      r++;
                  }
                  
                  Update<<<N_b,1024>>>(r-children,gUpdatedX,gUpdatedY,q,children,node);
            }
        }
 }

   __global__ void fill_res(int *g_mesh,int *gResult,int *gOpac,int c_opa,int x,int y,int finalX,int finalY,int frameSizeX,int frameSizeY){

        long int tid = blockDim.x*blockIdx.x + threadIdx.x;

        if(tid >= x*y)
          return;

        int row = tid/y + finalX;
        int col = tid%y + finalY;


        if(col < 0 || col >= frameSizeY || row < 0 || row >= frameSizeX)
            return;
        
        if(gOpac[row * frameSizeY + col]<c_opa){
            gOpac[row*frameSizeY+ col] = c_opa;
            gResult[row*frameSizeY+ col] = g_mesh[tid];
        }
   }

int main (int argc, char **argv) {
	
	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ; 

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
	
	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;  
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;


	// Code begins here.
	// Do not change anything above this comment.
	 int *gCsr,*gOffset,*hUpdatedX,*hUpdatedY,*gResult,*gOpac;
	 hUpdatedX = new int[V]();
	 hUpdatedY = new int[V]();
     int *gUpdatedX,*gUpdatedY;
	 cudaMalloc(&gOpac,(frameSizeX * frameSizeY)*sizeof(int));
     cudaMalloc(&gCsr,(E)*sizeof(int));
	 cudaMalloc(&gOffset,(V+1)*sizeof(int));
	 cudaMalloc(&gUpdatedX,(V)*sizeof(int));
	 cudaMalloc(&gUpdatedY,(V)*sizeof(int));
	 cudaMalloc(&gResult,(frameSizeX * frameSizeY)*sizeof(int));
   
     long int blocks =  ceil(float(frameSizeX * frameSizeY)/1024);
     cudaMemset(gOpac,-1 ,frameSizeX*frameSizeY);
     cudaMemset(gResult,0,frameSizeX*frameSizeY);

	 for(int i=0;i<numTranslations;i++){

			long int mesh = translations[i][0];
			long int trans = translations[i][1];
			long int amount = translations[i][2];

			if(trans==0){
				hUpdatedX[mesh] =  hUpdatedX[mesh] - amount;

			}
			else if(trans==1){
				hUpdatedX[mesh] =  hUpdatedX[mesh] + amount;
			}

			else if(trans==2){
				hUpdatedY[mesh] =  hUpdatedY[mesh] - amount;
			}

			else if(trans==3){
				hUpdatedY[mesh] =  hUpdatedY[mesh] + amount;

			}
	 }
	 cudaMemcpy(gUpdatedY,hUpdatedY,(V)*sizeof(int),cudaMemcpyHostToDevice);
	 cudaMemcpy(gUpdatedX,hUpdatedX,(V)*sizeof(int),cudaMemcpyHostToDevice);
	 cudaMemcpy(gCsr,hCsr,(E)*sizeof(int),cudaMemcpyHostToDevice);
	 cudaMemcpy(gOffset,hOffset,(V+1)*sizeof(int),cudaMemcpyHostToDevice);
   
	 //Propogating the change
     int *queue;
     cudaMalloc(&queue,(10000000)*sizeof(int));
	 propogate<<<1,1>>>(gUpdatedX,gUpdatedY,queue,gOffset,gCsr);
     cudaDeviceSynchronize();
     cudaFree(gCsr);
     cudaFree(gOffset);

	 cudaMemcpy(hUpdatedX,gUpdatedX,(V)*sizeof(int),cudaMemcpyDeviceToHost);
	 cudaMemcpy(hUpdatedY,gUpdatedY,(V)*sizeof(int),cudaMemcpyDeviceToHost);
     cudaFree(gUpdatedY);
     cudaFree(gUpdatedX);

     int *g_mesh;
     cudaMalloc(&g_mesh,(10000)*sizeof(int));
     for(int i = 0; i<V; ++i){
        int c_opa = hOpacity[i];
        int x = hFrameSizeX[i]; 
        int y = hFrameSizeY[i]; 
        int N_b = ceil (float(x*y)/1024); 
        int finalX = hGlobalCoordinatesX[i]+hUpdatedX[i];
        int finalY = hGlobalCoordinatesY[i]+hUpdatedY[i];
        int *c_mesh = hMesh[i];
        cudaMemcpy(g_mesh,c_mesh,(x*y)*sizeof(int),cudaMemcpyHostToDevice);
        fill_res<<<N_b,1024>>>(g_mesh,gResult,gOpac,c_opa,x,y,finalX,finalY,frameSizeX,frameSizeY);
    }
     cudaDeviceSynchronize();
     free(hUpdatedX);
     free(hUpdatedY);
	 cudaMemcpy(hFinalPng,gResult,(frameSizeX * frameSizeY)*sizeof(int),cudaMemcpyDeviceToHost);
	 cudaFree(gResult);
     cudaFree(queue);
     cudaFree(g_mesh);
     cudaFree(gOpac);
     cudaFree(gResult);

	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}
