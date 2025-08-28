#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

template<typename T>
void printVector(const vector<T>& a)
{
	for (int v : a)
		cout << setw(3) << v;
	cout << endl;
}

__global__ void addKernel(const int* a, const int* b, int* c, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x; 
	if (i < size)
		c[i] = a[i] + b[i];
}
int main()
{
	const int size = (1<<22)+3;
	// 생각해볼 점: 블럭이 몇 개가 필요할까?

	vector<int> a(size);
	vector<int> b(size);
	vector<int> c_single(size);     // 결과 확인용
	vector<int> c(size, -1); // CUDA에서 계산한 결과 저장

	for (int i = 0; i < size; i++)
	{
		a[i] = rand() % 10;
		b[i] = rand() % 10;
		c_single[i] = a[i] + b[i];
	}

	cout << "Add vectors using CUDA" << endl;

	{
		int* dev_a = nullptr;
		int* dev_b = nullptr;
		int* dev_c = nullptr;

		cudaMalloc((void**)&dev_a, size * sizeof(int)); // input a
		cudaMalloc((void**)&dev_b, size * sizeof(int)); // input b
		cudaMalloc((void**)&dev_c, size * sizeof(int)); // output c

		cudaEvent_t start, stop;// 시간 측정을 위한 CUDA 이벤트 생성 (시간측정도 Nsight로 할 수 있습니다.)
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0); // 시작 시간 기록

		cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice);

		const unsigned TPB = 512; // 256이나 512 권장(하드웨어, 점유율에 따라)
		const unsigned blocks = (size + TPB - 1) / TPB; // 올림
		cout << "size: "<<size<<" blocks: "<<blocks<<endl;
		addKernel<<<blocks, TPB>>>(dev_a, dev_b, dev_c, size);

		cudaMemcpy(c.data(), dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

		cudaEventRecord(stop, 0);  // 끝나는 시간 기록

		cudaDeviceSynchronize();       // kernel이 끝날때까지 대기 (동기화)
		// cudaEventSynchronize(stop); // 불필요 (동기화 중복)

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop); // 걸린 시간 계산
		cout << "Time elapsed: " << milliseconds << " ms" << endl;

		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);

		cudaDeviceReset();
	}

	if (size < 40) { // size가 작을 경우에는 출력해서 확인
		printVector(a);
		printVector(b);
		printVector(c_single);
		printVector(c);
	}

	for (int i = 0; i < size; i++)
		if (c_single[i] != c[i])
		{
			cout << "Wrong result" << endl;
			return 1;
		}

	cout << "Correct" << endl;

	return 0;
}

