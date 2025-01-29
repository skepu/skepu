
#include <stdio.h>

void v1()
{
	int M = 10, N = 20;
	int a[M][N];
	
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			a[i][j] = i * N + j;
	
			for (int i = 0; i < M; ++i)
			{
				for (int j = 0; j < N; ++j)
				{
					printf("%d\t", a[i][j]);
				}
				printf("\n");
			}
			
			
			printf("\n ---- ver1 \n");
	for (int j = 1; j < N-1; ++j)
		for (int i = 1; i < M; ++i)
			a[i][j] = a[i-1][j-1] + a[i-1][j] + a[i-1][j+1];
		
	
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			printf("%d\t", a[i][j]);
		}
		printf("\n");
	}
}


void v2()
{
	int M = 10, N = 20;
	int a[M][N];
	
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			a[i][j] = i * N + j;
			
			
	for (int i = 1; i < M; ++i)
		for (int j = 1; j < N-1; ++j)
			a[i][j] = a[i-1][j-1] + a[i-1][j] + a[i-1][j+1];
		
	
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			printf("%d\t", a[i][j]);
		}
		printf("\n");
	}
}

int main()
{
	v1();
	printf("\n ---- ver2 \n");
	v2();
}