
#include <stdio.h>
#include <stdlib.h>

// SkePU C interface header (for now containers only, no skeletons)
#include <cskepu.h>



// Application skeleton wrappers, programmer may create a header file for these
SkePU_Handle_Vector_Int
square(SkePU_Handle_Vector_Int res, SkePU_Handle_Vector_Int v);



// Main C program
int main(int argc, char **argv)
{
  if(argc <= 1)
  {
    printf("Usage: %s size\n", argv[0]);
    return 1;
  }
  
  int N = atoi(argv[1]);
  
  // Allocate
  SkePU_Handle_Vector_Int vec_handle = skepu_create_vector_int(N);
  
//  SkePU_Handle_Matrix_Int mat_handle = skepu_create_matrix_int(N, N);
//  skepu_write_matrix_int(vec_handle, 0, 0, 1);
  
  // Initialize
  for (int i = 0; i < N; ++i)
    skepu_write_vector_int(vec_handle, i, i);
  
  // Some in-line processing
  for (int i = 0; i < N; ++i)
  {
    int val = skepu_read_vector_int(vec_handle, i);
    int newval = val * 2;
    skepu_write_vector_int(vec_handle, i, newval);
  }
  
  // Call a SkePU skeleton through a C wrapper
  square(vec_handle, vec_handle);
  
  // Logging
  skepu_flush_vector_int(vec_handle);
  printf("Contents of vector: ");
  for (int i = 0; i < N; ++i)
  {
    printf("%d ", skepu_read_vector_int(vec_handle, i));
  }
  printf("\n");
  
  // Deallocate
  skepu_delete_vector_int(vec_handle);
  
  
  return 0;
}