/******************************************************************************
 *cr

 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <functional>
#include <iterator> 
#include <iterator>
#include <time.h>
#include "kernel_retrain.cu"
#include <bits/stdc++.h> 
#include <sys/time.h> 
using namespace std; 
#define NUM_COMMANDS 2
#define Miter 29
#define Diter 15
#include<string.h>


using namespace std;
vector<float> linSpace(float start_in, float end_in, int num_in);
void printLinspace(vector<float> v);

void create_marks_csv(char *filename,float a[Miter*Diter][6],int n,int m){
 
	printf("\n Creating %s.csv file",filename);
	 
	FILE *fp;
	 
	int i,j;
	 
	filename=strcat(filename,".csv");
	 
	fp=fopen(filename,"w+");
	 
	fprintf(fp,"M,D,Accuracy, Training Time(s), Testing Time(s), Total Time(s)");
	 
	for(i=0;i<Miter*Diter;i++){
		fprintf(fp,"\n");
		for(j=0;j<5;j++)
	 
			fprintf(fp,"%f, ",a[i][j]);
	 
		}
	 
	fclose(fp);
	 
	printf("\n %sfile created",filename);
 
}


int main ()
{

    time_t load_start, load_stop;
    time_t hv_start, hv_stop; 
    time_t train_start, train_stop;
    time_t test_start, test_stop; 
    time_t total_start, total_stop;
	
	time(&total_start);
	float Traintime_taken,Testtime_taken,Totaltime_taken;
	struct timeval startTest, endTest;
	struct timeval startTrain, endTrain;
	
	float MvsA[Miter*Diter][6];
	int v=0;

	//*****************************************************
    //******************** Model Parameter ****************
	//*****************************************************

	int M;

	float ephsilon = 0.01;
	int max_epoch=30;
	int min_epoch=0;
	
	
	//*****************************************************
    //******************** Dataset Parameter ****************
	//*****************************************************
	
	
	int numClasses=26;
    int numTrainSampOrig=6237;
	int numTestSamples=1558;
	int numFeatures=617;
	
	int numValidSamples=300;
	int numTrainSamples=numTrainSampOrig-numValidSamples;
	

	
	
	//**********************************************************
	//********* Initialize Host and Device Variables ***********
	//**********************************************************
	


 
	float lMax;
    float lMin;
	vector<float>L;
	float accuracy;
	
	float *trainX_h, *validX_h,*testX_h;
	int *trainY_h, *validY_h, *testY_h, *Classes_h;
	int  *ClassHV_h;
	
   
	size_t trainX_sz, trainY_sz, validX_sz, validY_sz, testX_sz, testY_sz, L_sz;
	
	float *trainX_d,*validX_d, *testX_d;
	int *validY_d, *trainY_d,*testY_d;
	
	int *ClassHV_d; 

	
	
	//*****************************************************
	//*********** Initialize data loading variables********
	//*****************************************************
	ifstream fin;
	ifstream ftestin;
    ofstream fout;
	vector<vector<float> > trainset_array;
    vector<int> trainset_labels(numTrainSampOrig+1);
    vector<vector<float> > testset_array;
    vector<int> testset_labels(numTestSamples+1);
	int row = 0;
	vector<float> rowArray(numFeatures);	
	
	
	
	
	
	
	//******************************************
	//***** Dynamic Host Memory Allocation******
	//******************************************
	
	trainX_h = (float*) malloc( sizeof(float)*numTrainSamples*numFeatures );
	trainY_h = (int*) malloc( sizeof(int)*numTrainSamples );
	
	validX_h = (float*) malloc( sizeof(float)*numValidSamples*numFeatures );
	validY_h = (int*) malloc( sizeof(int)*numValidSamples );
	
	testX_h = (float*) malloc( sizeof(float)*numTestSamples*numFeatures );
	testY_h = (int*) malloc( sizeof(int)*numTestSamples );
	
	
	
	
	
	//********************************************
	//***** Dynamic Device Memory Allocation******
	//********************************************
	
	trainX_sz= numTrainSamples*numFeatures*sizeof(float);
	trainY_sz=numTrainSamples*sizeof(int);
	validX_sz= numValidSamples*numFeatures*sizeof(float);
	validY_sz=numValidSamples*sizeof(int);
	testX_sz= numTestSamples*numFeatures*sizeof(float);
	testY_sz=numTestSamples*sizeof(int);

	
	cudaMalloc((void **)&trainX_d, trainX_sz);
	cudaMalloc((void **)&trainY_d, trainY_sz);
	cudaMalloc((void **)&validX_d, trainX_sz);
	cudaMalloc((void **)&validY_d, trainY_sz);
	cudaMalloc((void **)&testX_d, testX_sz);
	cudaMalloc((void **)&testY_d, testY_sz);
	
	
	
	
	
	
	
	
	//**************************************
	//************ Data Loading **********
	//************************************

	time(&load_start);
	printf("\nSetting up the problem...\n"); fflush(stdout);
	
	fin.open("isolet1+2+3+4.data");
    if(!fin.is_open())
    {
        printf( "Error: Can't open file containind training X dataset"  );
    }
    else
    {	printf("\nloading train data..\n");
        while(!fin.eof())
        {
            
            if(row > numTrainSampOrig)
            {
                break;
            }
            
            trainset_array.push_back(rowArray);
			
            for(int col = 0; col < numFeatures ; ++col)
            {
                
                
                fin >> trainset_array[row][col];
				//trainset_array[row][col]=trainset_array[row][col]*100;
                fin.ignore(50000000, ',');
				fin.ignore();
				//printf("\t%f",trainset_array[row][col]);

            }
            fin >> trainset_labels[row];
            fin.ignore(50000000, '.');
			//printf("\nlabel:%d\n",trainset_labels[row]);
            row++;

        }
        

		
    }
	fin.close();
	
	row=0;
	ftestin.open("isolet5.data");
    if(!ftestin.is_open())
    {
        printf("Error: Can't open file containind training X dataset");
    }
    else
    {	printf("\nloading test data..\n");
        while(!ftestin.eof())
        {
            
            if(row > numTestSamples)
            {
                break;
            }
            
            testset_array.push_back(rowArray);
			
            for(int col = 0; col < numFeatures ; ++col)
            {
                
                
                ftestin >> testset_array[row][col];
				//testset_array[row][col]=testset_array[row][col]*100;
                ftestin.ignore(50000000, ',');
                ftestin.ignore();
                //printf("\t%f", testset_array[row][col]);
            }
            ftestin >> testset_labels.at(row);
            ftestin.ignore(50000000, '.');
            
            //printf("\nlabel: %d\n", testset_labels.at(row));
            row++;
         

        }
		
    }
	ftestin.close();
	
    for (int i=0; i < numTrainSamples*numFeatures; i++) 
	{ 	
		trainX_h[i] = trainset_array[i/numFeatures][int(i%numFeatures)]; 
	}
	for (int i=0; i < numTrainSamples; i++) { trainY_h[i] = trainset_labels[i]-1; }
	
	for (int i=0; i < numValidSamples*numFeatures; i++) 
	{ 	
		validX_h[i] = trainset_array[(i/numFeatures)+numTrainSamples][int(i%numFeatures)]; 
	}
	for (int i=0; i < numValidSamples; i++) { validY_h[i] = trainset_labels[(i+numTrainSamples)]-1; }
	
	
    for (int i=0; i < numTestSamples*numFeatures; i++) 
	{ 	
		testX_h[i] = testset_array[i/numFeatures][int(i%numFeatures)]; 
	}
	for (int i=0; i < numTestSamples; i++) { testY_h[i] = testset_labels[i]-1; }
	
	
	

	
	
	

	//***********************************************
	//********* Copy dataset to Device***************
	//***********************************************
	
	cudaMemcpy(trainX_d, trainX_h, trainX_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(trainY_d, trainY_h, trainY_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(validX_d, validX_h, validX_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(validY_d, validY_h, validY_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(testX_d, testX_h, testX_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(testY_d, testY_h, testY_sz, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	printf("\ndata loading done ..\n");
	
	time(&load_stop);
	float load_seconds = load_stop - load_start;
	printf("Loading time in seconds: %fn", load_seconds);
	
	
	time(&load_stop);
	load_seconds = load_stop - load_start;
	printf("Loading time in seconds: %fn", load_seconds);
	
	lMin= *min_element(trainset_array[0].begin(),trainset_array[0].end());
	lMax= *max_element(trainset_array[0].begin(),trainset_array[0].end());
	
	
	for (M=5; M<31; M=M+1)
	{
		int D=8000;
		float *L_h, *L_d;
		L_h = (float*) malloc(sizeof(float)*M);
		L = linSpace(lMin, lMax, M);
		printLinspace(L);
		for (int i=0; i<M;i++)
		{
			L_h[i]=L[i];
		}
		size_t L_sz;
		L_sz=M*sizeof(float);
		cudaMalloc((void **)&L_d, M*sizeof(float));
		cudaMemcpy(L_d, L_h, M*sizeof(float), cudaMemcpyHostToDevice);
		
		while (D<10000)
		
		{
			int *LD_h, *LD_d, *ID_h, *ID_d;
			LD_h = (int*) malloc( sizeof(int)*M*D );
			ID_h = (int*) malloc( sizeof(int)*numFeatures*D );
			ClassHV_h = (int*) malloc( sizeof(int)*numClasses*D );
			
			cudaMalloc((void **)&LD_d, M*D*sizeof(int));
			cudaMalloc((void **)&ID_d, numFeatures*D*sizeof(int));
			cudaMalloc((void **)&ClassHV_d, numClasses*D*sizeof(int));
			
			
			//******************************************
			//******** Level Hypervector ***************
			//******************************************
			
			printf("\nSetting up Level and Identity Hypervector....\n");
			
			//*******Defining Quantization Levels *****
			
			//*********Setting up Level Hypervector*********
			
			
			printf("\ncheckpoint 1\n");
			for (int i=0; i<D; i++) {LD_h[i]=int(rand()%2);}
			int *nAlter;
			nAlter=(int*)malloc(D*sizeof(int));
			for (int i=0; i<D; i++)
			{
				nAlter[i]=rand()%D;
			}
			printf("\ncheckpoint 11\n");
			int jAlter;
			
			for (int i=1; i<M; i++)
			{
				for (int j=0; j<D; j++)
				{
					LD_h[i*D+j]=LD_h[(i-1)*D+j];
				}

				for (int j=0; j<ceil(D/M); j++)
				{
					jAlter=nAlter[int((i-1)*ceil(D/M)+j)];
					LD_h[(i*D)+jAlter]=int(LD_h[(i*D)+jAlter]==0);
				}
			}
			int LD_test1=0;
			int LD_test2=0;
			
			for (int i=0;i<D; i++)
			{
				LD_test1=LD_test1+(LD_h[0*D+i]^LD_h[1*D+i]);
				LD_test2=LD_test2+(LD_h[0*D+i]^LD_h[(M-1)*D+i]);	
			}
			printf("\n LDtest1=%d", LD_test1);
			printf("\n LDtest2=%d", LD_test2);
			
		
			
			//******* Creating Identity Hypervector ID *******
			
			

			for (int i=0; i<numFeatures; i++)
			{
				for (int j=0; j<D; j++)
				{
					ID_h[i*D+j]=int(rand()%2);
				}	
			}
			
			int ID_test1=0;
			int ID_test2=0;
			
			for (int i=0;i<D; i++)
			{
				ID_test1=ID_test1+(ID_h[0*D+i]^ID_h[1*D+i]);
				ID_test2=ID_test2+(ID_h[0*D+i]^ID_h[5*D+i]);	
			}
			printf("\n IDtest1=%d", ID_test1);
			printf("\n IDtest2=%d", ID_test2);
			
			cudaMemcpy(LD_d, LD_h, M*D*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(ID_d, ID_h, numFeatures*D*sizeof(int), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			
					
			printf("Creating level and Identity Hypervector Done\n");
			
			
			


			//*****************************************
			//****************  Training **************
			//*****************************************

			//time(&train_start);
			printf("\nTraining...\n");

			// start timer. 
			gettimeofday(&startTrain, NULL); 
		  
			// unsync the I/O of C and C++. 
			ios_base::sync_with_stdio(false);
			

			Classes_h=(int*) malloc( sizeof(int)*numClasses );
			for(int i=0; i<numClasses; i++)
			{
				Classes_h[i]=0;
			}
			Training_HV(trainX_d, trainY_d, validX_d, validY_d, L_d, numTrainSamples, numValidSamples, numFeatures, numClasses, M, D, LD_d, ID_d, ClassHV_d, Classes_h, ephsilon, max_epoch, min_epoch);
			

			cudaMemcpy(ClassHV_h, ClassHV_d, numClasses*D*sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			
			//time(&train_stop);
			//float train_seconds = train_stop - train_start;
			//printf("Training time in seconds: %fn", train_seconds);
			
			
			// stop timer. 
			gettimeofday(&endTrain, NULL); 
		  
			// Calculating total time taken by the program. 
			double Traintime_taken; 
		  
			Traintime_taken = (endTrain.tv_sec - startTrain.tv_sec) * 1e6; 
			Traintime_taken = (Traintime_taken + (endTrain.tv_usec -  
									  startTrain.tv_usec)) * 1e-6; 
		  
			cout << "Training Time : " << fixed 
				 << Traintime_taken << setprecision(6); 
			cout << " sec" << endl;
			
			
			//***************Teasting  ClassHyper Vectors ***********
			
			int Class_test1=0;
			int Class_test2=0;
			
			for (int i=0;i<D; i++)
			{
				Class_test1=Class_test1+(ClassHV_h[0*D+i]^ClassHV_h[1*D+i]);
				Class_test2=Class_test2+(ClassHV_h[0*D+i]^ClassHV_h[5*D+i]);	
			}
			printf("\n Classtest1=%d", Class_test1);
			printf("\n Classtest2=%d", Class_test2);

		 
			
			
			
			
			//********************************
			//*********   Testing  ***********
			//********************************
			
			// start timer. 
			gettimeofday(&startTest, NULL); 
		  
			// unsync the I/O of C and C++. 
			ios_base::sync_with_stdio(false);
			
			//time(&test_start);
			
			accuracy=TestingAccuracy(testX_d, testY_d,ClassHV_d,L_d,LD_d,ID_d,D,M,numTestSamples,numFeatures,numClasses);
			printf("\nM=%d;\tD=%d;\taccuracy=%f%\n",M,D,accuracy);
			printf("\n Testing done......\n");
			
			// stop timer. 
			gettimeofday(&endTest, NULL); 
		  
			// Calculating total time taken by the program. 
			double Testtime_taken; 
		  
			Testtime_taken = (endTest.tv_sec - startTest.tv_sec) * 1e6; 
			Testtime_taken = (Testtime_taken + (endTest.tv_usec -  
									  startTest.tv_usec)) * 1e-6; 
		  
			cout << "Testing Time : " << fixed 
				 << Testtime_taken << setprecision(6); 
			cout << " sec" << endl; 
			
			//time(&test_stop);
			//float test_seconds = test_stop - test_start;
			//printf("\nTesting time in seconds: %fn", test_seconds);
			
			
			
			//**********************************************************************
			//************** Free dynamically allocated memory and finish **********
			//**********************************************************************
			
			//time(&total_stop);
			//float elapsed_time = total_stop - total_start;
			//printf("\nTotal elapsed time in seconds: %fn", elapsed_time);
			Totaltime_taken=Traintime_taken+Testtime_taken;
			cout << "Total Time : " << fixed 
				 << Totaltime_taken << setprecision(6); 
			cout << " sec" << endl;
			
			MvsA[v][0]=M;
			MvsA[v][1]=D;
			MvsA[v][2]=accuracy;
			MvsA[v][3]=Traintime_taken;
			MvsA[v][4]=Testtime_taken;
			MvsA[v][5]=Totaltime_taken;
			v+=1;
			D=D+2000;
			
			
			
		 
		
			free(LD_h);
			free(ID_h);
			free(ClassHV_h);
			free(nAlter);
			

			cudaFree(LD_d);
			cudaFree(ID_d);
			cudaFree(ClassHV_d);
		}
		free(L_h);
		cudaFree(L_d);
	}
	
	cudaFree(trainX_d);
	cudaFree(trainY_d);
	cudaFree(testX_d);
	cudaFree(testY_d);
	cudaFree(validX_d);
	cudaFree(validY_d);
	
	free(trainX_h);
	free(trainY_h);
	free(testX_h);
	free(testY_h);
	free(validX_h);
	free(validY_h);
	
	
    char str[100]="ISOLET_retrain";
	create_marks_csv(str,MvsA,Miter,Diter);	
}

vector<float> linSpace(float start_in, float end_in, int num_in)
{
    vector<float> linspaced;
    
    float start = start_in;
    float end = end_in;
    int num = num_in;
    
    if(num == 0)
    {
        return linspaced;
    }
    
    if(num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }
    
    float delta = (end - start) / (num - 1);
    
    for(int i = 0; i < num - 1; i++)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);
    return linspaced;
}

void printLinspace(vector<float> v)
{
    cout << "size: " << v.size() << endl;
    for(int i=0; i< v.size(); i++)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}