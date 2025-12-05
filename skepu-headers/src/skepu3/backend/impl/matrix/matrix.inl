/*! \file matrix.inl
 *  \brief Contains the definitions of non-backend specific member functions for the Matrix container.
 */

namespace skepu
{

template<typename T>
void Matrix<T>::setValidFlag(bool val)
{
   m_valid = val;
}


/*!
*  \brief Randomizes the Matrix.
*
*  Sets each element of the Matrix to a random number between \p min and \p max.
*  The numbers are generated as \p integers but are cast to the type of the matrix.
*
*  \param min The smallest number an element can become.
*  \param max The largest number an element can become.
*/
template<typename T>
void Matrix<T>::randomize(int min, int max)
{
   invalidateDeviceData();

   for(typename Matrix<T>::size_type i = 0; i < this->size(); i++)
   {
      this->m_data[i] = (T)( rand() % (int)(max-min+1) + min);
   }
}

template<typename T>
void Matrix<T>::randomizeReal(double min, double max)
{
   invalidateDeviceData();

   for(typename Matrix<T>::size_type i = 0; i < this->size(); i++)
   {
      this->m_data[i] = min + (T)( (rand() % RAND_MAX) / (double)RAND_MAX) * (max - min);
   }
}

/*!
*  \brief Saves content of Matrix to a file.
*
*  Outputs the matrix as text on one line with space between elements to the specified file.
*  Mainly for testing purposes.
*
*  \param filename Name of file to save to.
*/
template<typename T>
void Matrix<T>::save(const std::string& filename)
{
   updateHost();

   std::ofstream file(filename.c_str());

   if (file.is_open())
   {
      for(size_type i = 0; i < this->size(); ++i)
      {
         file << this->m_data[i] << " ";
      }
      file.close();
   }
   else
   {
      std::cout << "Unable to open file\n";
   }
}

/*!
*  \brief Loads the Matrix from a file.
*
*  Reads a variable number of elements from a file. In the file, all elemets should be in ASCII
*  on one line with whitespace between each element. Mainly for testing purposes.
*
*  \param filename Name of file to save to.
*  \param rowWidth The width of a row. All rows get same amount of width.
*  \param numRows The number of rows to be loaded. Default value 0 means all rows.
*/
template<typename T>
void Matrix<T>::load(const std::string& filename, size_type rowWidth, size_type numRows)
{
   invalidateDeviceData();

   std::ifstream file(filename.c_str());

   if (file.is_open())
   {
      std::string line;
      getline (file,line);
      std::istringstream ss(line);
      T num;

      //Load all elements
      if(numRows == 0)
      {
         while(ss >> num)
         {
            push_back(num);
         }
      }
      // Load only numElements elements
      else
      {
         for(size_type i = 0; i < (numRows*rowWidth); ++i)
         {
            ss >> num;
            push_back(num);
         }
      }

      this->m_cols = rowWidth;
      this->m_rows = this->size() / rowWidth;

      file.close();
   }
   else
   {
      std::cout << "Unable to open file\n";
   }
}


/*
 * Default constructor of an empty matrix.
 */
template<typename T>
Matrix<T>::Matrix()
: m_rows(0), m_cols(0),
  m_dataChanged(false),
  m_transpose_matrix(0),
  m_noValidDeviceCopy(true),
  m_valid(false),
  m_deallocEnabled(false),
	m_object_id(UniqueIdentifier::generate())
{
  DEBUG_TEXT_LEVEL1("Matrix: Constructor to an empty state");
}

/*!
 *  Constructor, used to allocate memory ($_rows * _cols$). With a value ot initialize all elements.
 * \param _rows Number of rows in the matrix.
 * \param _cols Number of columns in the matrix.
 * \param val A value to initialize all elements.
 */
template<typename T>
Matrix<T>::Matrix(typename Matrix<T>::size_type _rows, typename Matrix<T>::size_type _cols, const T& val)
: m_rows(_rows), m_cols(_cols),
  m_dataChanged(false),
  m_noValidDeviceCopy(true),
  m_valid(true),
  m_deallocEnabled(true),
	m_object_id(UniqueIdentifier::generate())
{
  DEBUG_TEXT_LEVEL1("Matrix: Constructor with " << _rows << " x " << _cols << " (total " << this->size() << ") elements and default value " << val);
  this->init(_rows, _cols, val);
}

/*!
 *  Constructor, used to allocate memory ($_rows * _cols$). With a value ot initialize all elements.
 * \param _rows Number of rows in the matrix.
 * \param _cols Number of columns in the matrix.
 * \param val A value to initialize all elements.
 */
template<typename T>
Matrix<T>::Matrix(typename Matrix<T>::size_type _rows, typename Matrix<T>::size_type _cols, std::string label, const T& val)
: m_rows(_rows), m_cols(_cols),
  m_label(label),
  m_dataChanged(false),
  m_noValidDeviceCopy(true),
  m_valid(true),
  m_deallocEnabled(true),
	m_object_id(UniqueIdentifier::generate())
{
  DEBUG_TEXT_LEVEL1("Matrix: Constructor with " << _rows << " x " << _cols << " (total " << this->size() << ") elements and default value " << val);
  this->init(_rows, _cols, val);
}


/**!
 * Used to construct matrix on a raw data pointer passed to it as its payload data.
 * Useful when creating the matrix object with existing raw data pointer.
 */
template <typename T>
inline Matrix<T>::Matrix(T * const ptr, typename Matrix<T>::size_type _rows, typename Matrix<T>::size_type _cols, bool deallocEnabled)
: m_rows(_rows), m_cols(_cols),
  m_dataChanged(false),
  m_noValidDeviceCopy(true),
  m_valid(true),
  m_deallocEnabled(deallocEnabled),
	m_object_id(UniqueIdentifier::generate())
{
  DEBUG_TEXT_LEVEL1("Matrix: Constructor from existing pointer with " << _rows << " x " << _cols << " (total " << this->size() << ") elements");
  
  if (this->m_rows < 1 || this->m_cols < 1)
    SKEPU_ERROR("The matrix size must be positive.");
  
  if (!ptr)
  {
    SKEPU_ERROR("Error: The supplied pointer for initializing matrix object is invalid");
    return;
  }
  
  this->m_data = ptr;
}


/*!
 *  Copy Constructor, used to assign copy of another matrix.
 * \param copy Matrix that is being assigned.
 *
 * Update the matrix before assigning it to assign latest copy.
 */
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& copy)
: m_rows(copy.m_rows), m_cols(copy.m_cols),
  m_noValidDeviceCopy(true),
  m_valid(true),
  m_deallocEnabled(true),
  m_dataChanged(false),
	m_object_id(UniqueIdentifier::generate())
{
  DEBUG_TEXT_LEVEL1("Matrix: Copy constructor with " << this->m_rows << " x " << this->m_cols << " (total " << this->size() << ") elements");
   copy.updateHost();
   this->init(copy.m_rows, copy.m_cols);
   std::copy(copy.m_data, copy.m_data + copy.size(), this->m_data);
   
#ifdef SKEPU_OPENCL
   this->m_transposeKernels_CL = copy.m_transposeKernels_CL;
#endif
}

template<typename T>
Matrix<T>::Matrix(Matrix<T>&& move): Matrix<T>()
{
  DEBUG_TEXT_LEVEL1("Matrix: Move constructor with " << move.m_rows << " x " << move.m_cols << " (total " << move.size() << ") elements");
   this->swap(move);
}

template <typename T>
inline Matrix<T>::Matrix(std::initializer_list<T> l)
:	m_rows(sqrt(l.size())), m_cols(sqrt(l.size())),
  m_noValidDeviceCopy(true),
  m_valid(true),
  m_deallocEnabled(true),
  m_dataChanged(false),
	m_object_id(UniqueIdentifier::generate())
{
  DEBUG_TEXT_LEVEL1("Matrix: Constructor from initializer list with " << this->m_rows << "x" << this->m_cols << " elements");
  this->init(this->m_rows, this->m_cols);

  int i = 0;
  for (const T& elem : l)
    this->m_data[i++] = elem;
}


// Initializers

template<typename T>
void Matrix<T>::init(size_type _rows, size_type _cols)
{
  DEBUG_TEXT_LEVEL1("Matrix: Allocating with " << _rows << " x " << _cols << " (total " << this->size() << ") elements");

  if (!this->m_data)
  {
    if (_rows * _cols < 1)
      SKEPU_ERROR("The container size must be positive.");
    this->m_rows = _rows;
    this->m_cols = _cols;
    backend::allocateHostMemory<T>(this->m_data, this->m_rows * this->m_cols);
    this->m_deallocEnabled = true;
    SKEPU_TRACE_ALLOCATION(this->getObjectID(), this->getLabel(), __LINE__);

#ifdef SKEPU_OPENCL
    this->m_transposeKernels_CL = &(backend::Environment<T>::getInstance()->m_transposeKernels_CL);
#endif
  }
  else SKEPU_ERROR("Container is already initialized");
}

template<typename T>
void Matrix<T>::init(size_type _rows, size_type _cols, const T& val)
{
  this->init(_rows, _cols);
	std::fill(this->m_data, this->m_data + this->m_rows * this->m_cols, val);
}

/*!
 *  Releases all allocations made on device.
 */
template <typename T>
Matrix<T>::~Matrix()
{
  this->releaseDeviceAllocations();
  
  if (this->m_transpose_matrix)
    delete this->m_transpose_matrix;
  
  if (this->m_data && this->m_deallocEnabled)
  {
    backend::deallocateHostMemory<T>(this->m_data);
    SKEPU_TRACE_DEALLOCATION(this->getObjectID(), this->getLabel(), __LINE__);
    DEBUG_TEXT_LEVEL1("Matrix: Destroyed with " << this->m_rows << " x " << this->m_cols << " (total " << this->size() << ") elements");
  }
  else
  {
    DEBUG_TEXT_LEVEL1("Matrix: Note, did not deallocate data.");
  }

}


/*!
 *  copy matrix,,, copy row and column count as well along with data
 */
template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other)
{
   if(*this == other)
      return *this;
   
   other.updateHost();
   invalidateDeviceData();

   m_data = other.m_data;
   m_rows = other.m_rows;
   m_cols = other.m_cols;
   return *this;
}


///////////////////////////////////////////////
// Public Helpers START
///////////////////////////////////////////////

/*!
    *  Updates the matrix from its device allocations.
    */
template <typename T>
inline void Matrix<T>::updateHost(bool enable) const
{
	if (!enable)
		return;
	
#ifdef SKEPU_OPENCL
   updateHost_CL();
#endif

#ifdef SKEPU_CUDA
   /*! the m_valid logic is only implemented for CUDA backend. The OpenCL still uses the old memory management mechanism */
   if(m_valid) // if already up to date then no need to check...
      return;
   
   updateHost_CU();
   
   m_valid = true;
#endif
}

/*!
 *  Invalidates (mark copies data invalid) all device data that this matrix has allocated.
 */
template <typename T>
inline void Matrix<T>::invalidateDeviceData(bool enable) const
{
	if (!enable)
		return;
	
   /// this flag is used to track whether contents in main matrix are changed so that the contents of the
   /// transpose matrix that was taken earlier need to be updated again...
   /// normally invalidation occurs when contents are changed so good place to update this flag (?)
   m_dataChanged = true;
   
#ifdef SKEPU_OPENCL
   invalidateDeviceData_CL();
#endif

#ifdef SKEPU_CUDA
//   if(m_noValidDeviceCopy)
//       assert(m_valid);
   
   if(!m_noValidDeviceCopy)
   {
      invalidateDeviceData_CU();
      m_noValidDeviceCopy = true;
      m_valid = true;
   }
#endif
}

/*!
 *  First updates the matrix from its device allocations. Then invalidates (mark copies data invalid) the data allocated on devices.
 */
template <typename T>
inline void Matrix<T>::updateHostAndInvalidateDevice()
{
   updateHost();
   invalidateDeviceData();
}

/*!
 *  Removes the data copies allocated on devices.
 */
template <typename T>
inline void Matrix<T>::releaseDeviceAllocations()
{
#ifdef SKEPU_OPENCL
   releaseDeviceAllocations_CL();
#endif

#ifdef SKEPU_CUDA
   m_valid = true;
   
   releaseDeviceAllocations_CU();
#endif
}

/*!
 *  First updates the matrix from its device allocations. Then removes the data copies allocated on devices.
 */
template <typename T>
inline void Matrix<T>::updateHostAndReleaseDeviceAllocations()
{
   updateHost();
   releaseDeviceAllocations();
}




///////////////////////////////////////////////
// Regular interface functions START
///////////////////////////////////////////////

template <typename T>
typename Matrix<T>::iterator Matrix<T>::begin()
{
   return iterator(this, this->m_data);
}

template <typename T>
typename Matrix<T>::iterator Matrix<T>::begin(size_t row)
{
   if (row >= total_rows())
   {
      SKEPU_ERROR("ERROR! Row index is out of bound!");
   }
   return iterator(this, this->m_data + row * this->m_cols);
}

template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::begin() const
{
   return iterator(this, this->m_data);
}

template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::begin(size_t row) const
{
   if (row >= total_rows())
   {
      SKEPU_ERROR("ERROR! Row index is out of bound!");
   }
   return iterator(this, this->m_data + row * this->m_cols);
}

template <typename T>
typename Matrix<T>::iterator Matrix<T>::end()
{
   return iterator(this, this->m_data + this->size());
}

template <typename T>
typename Matrix<T>::iterator Matrix<T>::end(size_t row)
{
   if (row >= total_rows())
   {
      SKEPU_ERROR("ERROR! Row index is out of bound!");
   }
   return iterator(this, this->m_data + this->size() - (this->m_rows - row + 1) * this->m_cols);
}

template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::end() const
{
   return iterator(this, this->m_data + this->size());
}

template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::end(size_t row) const
{
   if (row >= total_rows())
   {
      SKEPU_ERROR("ERROR! Row index is out of bound!");
   }
   return iterator(this, this->m_data + this->size() - (this->m_rows - row + 1) * this->m_cols);
}

#ifdef SKEPU_PRECOMPILED

/*!
 *  Returns a \p proxy_elem instead of an ordinary element. The \p proxy_elem usually
 *  behaves like an ordinary, but there might be exceptions.
 */
template <typename T>
typename Matrix<T>::proxy_elem Matrix<T>::at(size_type row, size_type col)
{
   return proxy_elem(*this, row * this->m_cols + col);
}

#else

/*!
 *  Uses \p row and \p col instead of single index.
 *  \param row Index of row to get.
 *  \param col Index of column to get.
 *  \return a const reference to T element at position identified by row,column index.
 */
template <typename T>
T& Matrix<T>::at(size_type row, size_type col)
{
   updateHost();
   if (row >= this->m_rows || col >= this->m_cols)
      SKEPU_ERROR("ERROR! Row or Column index is out of bound!");

   return this->m_data[row * this->m_cols + col];
}

#endif

/*!
 *  To initialize a matrix with some scalar value.
 *
 *  \param elem The element you want to assign to all matrix.
 */
template <typename T>
Matrix<T>& Matrix<T>::operator=(const T& elem)
{
   for (size_type i = 0; i < size(); i++)
   {
      this->m_data[i] = elem;
   }
   return *this;
}




/*!
 * Updates and invalidate both Matrices before swapping.
 */
template <typename T>
void Matrix<T>::swap(Matrix<T>& from)
{
  updateHostAndInvalidateDevice();
  from.updateHostAndInvalidateDevice();

  std::swap(this->m_rows, from.m_rows);
  std::swap(this->m_cols, from.m_cols);
  std::swap(this->m_data, from.m_data);
  std::swap(this->m_deallocEnabled, from.m_deallocEnabled);
  std::swap(this->m_valid, from.m_valid);
  std::swap(this->m_noValidDeviceCopy, from.m_noValidDeviceCopy);
	std::swap(this->m_label, from.m_label);
	std::swap(this->m_object_id, from.m_object_id);

#ifdef SKEPU_OPENCL
  std::swap(this->m_deviceMemPointers_CL, from.m_deviceMemPointers_CL);
  std::swap(this->m_deviceConstMemPointers_CL, from.m_deviceConstMemPointers_CL);
#endif

#ifdef SKEPU_CUDA
  std::swap(this->m_deviceMemPointers_CU, from.m_deviceMemPointers_CU);
  std::swap(this->m_deviceMemPointers_Modified_CU, from.m_deviceMemPointers_Modified_CU);
#endif
}

///////////////////////////////////////////////
// Regular interface functions END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Additions to interface START
///////////////////////////////////////////////


/*!
 *  Flushes the matrix, synchronizing it with the device.
 *
 *  Then release all device allocations if deallocDevice is `true`.
 */
template <typename T>
void Matrix<T>::flush(FlushMode mode)
{
#ifdef SKEPU_OPENCL
   this->flush_CL(mode);
#endif

#ifdef SKEPU_CUDA
   this->flush_CU(mode);
#endif
}


/*!
 *  Behaves like \p operator[] and unlike \p skepu::Vector, it cares about synchronizing with device.
 *  Can be used when accessing to access elements row and column wise.
 *
 *  \param row Index to a specific row of the Matrix.
 *  \param col Index to a specific column of the Matrix.
 */
template <typename T>
const T& Matrix<T>::operator()(const size_type row, const size_type col) const
{
   if(row >= this->total_rows() || col >= this->total_cols())
      SKEPU_ERROR("ERROR! Row or Column index is out of bound!");
   return m_data[row * m_cols + col];
}


/*!
 *  Behaves like \p operator[] and unlike \p skepu::Vector, it cares about synchronizing with device.
 *  Can be used when accessing to access elements row and column wise.
 *
 *  \param row Index to a specific row of the Matrix.
 *  \param col Index to a specific column of the Matrix.
 */
template <typename T>
T& Matrix<T>::operator()(const size_type row, const size_type col)
{
   if(row >= this->total_rows())
      SKEPU_ERROR("Matrix (label: " << this->getLabel() << ")"
      << "\nAttempted row access out of bounds"
      << "\nRows: " << this->total_rows()
      << "\nAttempted access at " << colorRed(row));

   if(col >= this->total_cols())
      SKEPU_ERROR("Matrix (label: " << this->getLabel() << ")"
      << "\nAttempted col access out of bounds"
      << "\nCols: " << this->total_cols()
      << "\nAttempted access at " << colorRed(col));

   return m_data[row * m_cols + col];
}

#ifdef SKEPU_ENABLE_DEPRECATED_OPERATOR
/*!
 *  Behaves like \p operator[] but does not care about synchronizing with device.
 *  Can be used when accessing many elements quickly so that no synchronization
 *  overhead effects performance. Make sure to properly synch with device by calling
 *  updateHost etc before use.
 *
 *  \param index Index of element assuming continuous Matrix row-wise storage. To facilitate access using single indexing
 */
template <typename T>
T& Matrix<T>::operator()(const size_type index)
{
   return m_data[index];
}

/*!
 *  Behaves like \p operator[] but does not care about synchronizing with device.
 *  Can be used when accessing many elements quickly so that no synchronization
 *  overhead effects performance. Make sure to properly synch with device by calling
 *  updateHost etc before use.
 *
 *  \param index Index of element assuming continuous Matrix row-wise storage. To facilitate access using single indexing
 */
template <typename T>
T& Matrix<T>::operator()(const Index2D index)
{
   return m_data[index.row * m_cols + index.col];
}


/*!
 *  A \p operator[] that care about synchronizing with device.
 *  Can be used when accessing elements considering consecutive storage
 *
 *  \param index Index of element assuming continuous Matrix row-wise storage. To facilitate access using single indexing
 */
template <typename T>
const T& Matrix<T>::operator[](const size_type index) const
{
   updateHost();
   if(index >= (this->total_rows() * this->total_cols()))
      SKEPU_ERROR("ERROR! Index is out of bound!");
   return m_data[index];
}

/*!
 *  A \p operator[] that care about synchronizing with device.
 *  Can be used when accessing elements considering consecutive storage
 *
 *  \param index Index of element assuming continuous Matrix row-wise storage. To facilitate access using single indexing
 */
template <typename T>
T& Matrix<T>::operator[](const size_type index)
{
   updateHostAndInvalidateDevice();
   if(index >= (this->total_rows() * this->total_cols()))
      SKEPU_ERROR("ERROR! Index is out of bound!");
   return m_data[index];
}

#endif // SKEPU_ENABLE_DEPRECATED_OPERATOR

///////////////////////////////////////////////
// Additions to interface END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Comparison operators START
///////////////////////////////////////////////


template <typename T>
bool Matrix<T>::operator==(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data == m_data);
}

template <typename T>
bool Matrix<T>::operator!=(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data != m_data);
}

} // end namespace skepu



#include "matrix_iterator.inl"

#ifdef SKEPU_PRECOMPILED

#include "matrix_proxy.inl"
#include "matrix_transpose.inl"
#include "matrix_cl.inl"
#include "matrix_cu.inl"

#endif // SKEPU_PRECOMPILED
