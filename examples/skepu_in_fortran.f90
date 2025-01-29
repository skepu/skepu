
program skepu_in_fortran
  use skepu
  use f_skeleton
  implicit none

  integer :: N
  integer :: vec_handle, mat_handle
  integer :: i
  integer :: val, newval

  write(*,*) 'Enter size: '
  read(*,*) N

  ! Allocate
  vec_handle = skepu_create_vector_int(N)
  mat_handle = skepu_create_matrix_int(N, N)
  
  call skepu_write_matrix_int(vec_handle, 0, 0, 1)

  ! Initialize
  do i = 0, N-1
    call skepu_write_vector_int(vec_handle, i, i)
  end do

  ! Some in-line processing
  do i = 0, N-1
    val = skepu_read_vector_int(vec_handle, i)
    newval = val * 2
    call skepu_write_vector_int(vec_handle, i, newval)
  end do
  
  ! Call FORTRAN wrapper to C wrapper with skeleton call
  vec_handle = square(vec_handle, vec_handle)

  ! Logging
  call skepu_flush_vector_int(vec_handle)
  write(*,*) 'Contents of vector: '
  do i = 0, N-1
    val = skepu_read_vector_int(vec_handle, i)
    write(*,fmt="(i0,1x)",advance="no") val
  end do
  write(*,*) ''

  ! Deallocate
  call skepu_delete_vector_int(vec_handle)

end program skepu_in_fortran
