namespace skepu
{
	template <typename T>
	Index3D Tensor3Iterator<T>::getIndex() const
	{
		size_t index = *this - m_parent.begin();
		
		size_t i = index / (m_parent.m_size_j * m_parent.m_size_k);
		index = index % (m_parent.m_size_j * m_parent.m_size_k);
		
		size_t j = index / (m_parent.m_size_k);
		index = index % (m_parent.m_size_k);
		
		return Index3D{ i, j, index };
	}
	
	template <typename T>
	Tensor3Iterator<T>::Tensor3Iterator(parent_type& parent, T *std_iterator) : m_parent(parent), m_std_iterator(std_iterator) {}
	
	template <typename T>
	typename Tensor3Iterator<T>::parent_type& Tensor3Iterator<T>::getParent() const
	{
		return m_parent;
	}
	
	template <typename T>
	Tensor3Iterator<T>& Tensor3Iterator<T>::begin()
	{
		return *this;
	}
	
	template <typename T>
	size_t Tensor3Iterator<T>::size()
	{
		return this->m_parent.end() - *this;
	}
	
	template <typename T>
	T* Tensor3Iterator<T>::getAddress()
	{
		return m_std_iterator;
	}
	
	template <typename T>
	T* Tensor3Iterator<T>::data()
	{
		return m_std_iterator;
	}
	
	template <typename T>
	T& Tensor3Iterator<T>::operator()(const ssize_t index)
	{
		return m_std_iterator[index];
	}
	
	template <typename T>
	const T& Tensor3Iterator<T>::operator()(const ssize_t index) const
	{
		return m_std_iterator[index];
	}

	template <typename T>
	T& Tensor3Iterator<T>::operator[](const ssize_t index)
	{
		m_parent.updateHost();
		m_parent.invalidateDeviceData();
		return m_std_iterator[index];
	}
	
	template <typename T>
	const T& Tensor3Iterator<T>::operator[](const ssize_t index) const
	{
		m_parent.updateHost();
		return m_std_iterator[index];
	}
	
	template <typename T>
	bool Tensor3Iterator<T>::operator==(const iterator& i)
	{
		return (m_std_iterator == i.m_std_iterator);
	}
	
	template <typename T>
	bool Tensor3Iterator<T>::operator!=(const iterator& i)
	{
		return (m_std_iterator != i.m_std_iterator);
	}
	
	template <typename T>
	bool Tensor3Iterator<T>::operator<(const iterator& i)
	{
		return (m_std_iterator < i.m_std_iterator);
	}
	
	template <typename T>
	bool Tensor3Iterator<T>::operator>(const iterator& i)
	{
		return (m_std_iterator > i.m_std_iterator);
	}
	
	template <typename T>
	bool Tensor3Iterator<T>::operator<=(const iterator& i)
	{
		return (m_std_iterator <= i.m_std_iterator);
	}
	
	template <typename T>
	bool Tensor3Iterator<T>::operator>=(const iterator& i)
	{
		return (m_std_iterator >= i.m_std_iterator);
	}
	
	template <typename T>
	const Tensor3Iterator<T>& Tensor3Iterator<T>::operator++() //Prefix
	{
		++m_std_iterator;
		return *this;
	}
	
	template <typename T>
	Tensor3Iterator<T> Tensor3Iterator<T>::operator++(int) //Postfix
	{
		iterator temp(*this);
		++m_std_iterator;
		return temp;
	}
	
	template <typename T>
	const Tensor3Iterator<T>& Tensor3Iterator<T>::operator--() //Prefix
	{
		--m_std_iterator;
		return *this;
	}
	
	template <typename T>
	Tensor3Iterator<T> Tensor3Iterator<T>::operator--(int) //Postfix
	{
		iterator temp(*this);
		--m_std_iterator;
		return temp;
	}
	
	template <typename T>
	const Tensor3Iterator<T>& Tensor3Iterator<T>::operator+=(const ssize_t i)
	{
		m_std_iterator += i;
		return *this;
	}
	
	template <typename T>
	const Tensor3Iterator<T>& Tensor3Iterator<T>::operator-=(const ssize_t i)
	{
		m_std_iterator -= i;
		return *this;
	}
	
	template <typename T>
	Tensor3Iterator<T> Tensor3Iterator<T>::operator-(const ssize_t i) const
	{
		iterator temp(*this);
		temp -= i;
		return temp;
	}
	
	template <typename T>
	Tensor3Iterator<T> Tensor3Iterator<T>::operator+(const ssize_t i) const
	{
		iterator temp(*this);
		temp += i;
		return temp;
	}
	
	template <typename T>
	typename Tensor3Iterator<T>::parent_type::difference_type Tensor3Iterator<T>::operator-(const iterator& i) const
	{
		return m_std_iterator - i.m_std_iterator;
	}
	
	template <typename T>
	T& Tensor3Iterator<T>::operator*()
	{
		m_parent.updateHost();
		m_parent.invalidateDeviceData();
		return *m_std_iterator;
	}
	
	template <typename T>
	const T& Tensor3Iterator<T>::operator*() const
	{
		m_parent.updateHost();
		return *m_std_iterator;
	}
	
	template <typename T>
	const T& Tensor3Iterator<T>::operator-> () const
	{
		m_parent.updateHost();
		return *m_std_iterator;
	}
	
	template <typename T>
	T& Tensor3Iterator<T>::operator-> ()
	{
		m_parent.updateHost();
		m_parent.invalidateDeviceData();
		return *m_std_iterator;
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	template <typename T>
	Index4D Tensor4Iterator<T>::getIndex() const
	{
		size_t index = *this - m_parent.begin();
		
		size_t i = index / (m_parent.m_size_j * m_parent.m_size_k * m_parent.m_size_l);
		index = index % (m_parent.m_size_j * m_parent.m_size_k * m_parent.m_size_l);
		
		size_t j = index / (m_parent.m_size_k * m_parent.m_size_l);
		index = index % (m_parent.m_size_k * m_parent.m_size_l);
		
		size_t k = index / (m_parent.m_size_l);
		index = index % (m_parent.m_size_l);
		
		return Index4D{ i, j, k, index };
	}
	
	template <typename T>
	Tensor4Iterator<T>::Tensor4Iterator(parent_type& parent, T *std_iterator) : m_parent(parent), m_std_iterator(std_iterator) {}
	
	template <typename T>
	typename Tensor4Iterator<T>::parent_type& Tensor4Iterator<T>::getParent() const
	{
		return m_parent;
	}
	
	template <typename T>
	Tensor4Iterator<T>& Tensor4Iterator<T>::begin()
	{
		return *this;
	}
	
	template <typename T>
	size_t Tensor4Iterator<T>::size()
	{
		return this->m_parent.end() - *this;
	}
	
	template <typename T>
	T* Tensor4Iterator<T>::getAddress()
	{
		return m_std_iterator;
	}
	
	template <typename T>
	T* Tensor4Iterator<T>::data()
	{
		return m_std_iterator;
	}
	
	template <typename T>
	T& Tensor4Iterator<T>::operator()(const ssize_t index)
	{
		return m_std_iterator[index];
	}
	
	template <typename T>
	const T& Tensor4Iterator<T>::operator()(const ssize_t index) const
	{
		return m_std_iterator[index];
	}

	template <typename T>
	T& Tensor4Iterator<T>::operator[](const ssize_t index)
	{
		m_parent.updateHost();
		m_parent.invalidateDeviceData();
		return m_std_iterator[index];
	}
	
	template <typename T>
	const T& Tensor4Iterator<T>::operator[](const ssize_t index) const
	{
		m_parent.updateHost();
		return m_std_iterator[index];
	}
	
	template <typename T>
	bool Tensor4Iterator<T>::operator==(const iterator& i)
	{
		return (m_std_iterator == i.m_std_iterator);
	}
	
	template <typename T>
	bool Tensor4Iterator<T>::operator!=(const iterator& i)
	{
		return (m_std_iterator != i.m_std_iterator);
	}
	
	template <typename T>
	bool Tensor4Iterator<T>::operator<(const iterator& i)
	{
		return (m_std_iterator < i.m_std_iterator);
	}
	
	template <typename T>
	bool Tensor4Iterator<T>::operator>(const iterator& i)
	{
		return (m_std_iterator > i.m_std_iterator);
	}
	
	template <typename T>
	bool Tensor4Iterator<T>::operator<=(const iterator& i)
	{
		return (m_std_iterator <= i.m_std_iterator);
	}
	
	template <typename T>
	bool Tensor4Iterator<T>::operator>=(const iterator& i)
	{
		return (m_std_iterator >= i.m_std_iterator);
	}
	
	template <typename T>
	const Tensor4Iterator<T>& Tensor4Iterator<T>::operator++() //Prefix
	{
		++m_std_iterator;
		return *this;
	}
	
	template <typename T>
	Tensor4Iterator<T> Tensor4Iterator<T>::operator++(int) //Postfix
	{
		iterator temp(*this);
		++m_std_iterator;
		return temp;
	}
	
	template <typename T>
	const Tensor4Iterator<T>& Tensor4Iterator<T>::operator--() //Prefix
	{
		--m_std_iterator;
		return *this;
	}
	
	template <typename T>
	Tensor4Iterator<T> Tensor4Iterator<T>::operator--(int) //Postfix
	{
		iterator temp(*this);
		--m_std_iterator;
		return temp;
	}
	
	template <typename T>
	const Tensor4Iterator<T>& Tensor4Iterator<T>::operator+=(const ssize_t i)
	{
		m_std_iterator += i;
		return *this;
	}
	
	template <typename T>
	const Tensor4Iterator<T>& Tensor4Iterator<T>::operator-=(const ssize_t i)
	{
		m_std_iterator -= i;
		return *this;
	}
	
	template <typename T>
	Tensor4Iterator<T> Tensor4Iterator<T>::operator-(const ssize_t i) const
	{
		iterator temp(*this);
		temp -= i;
		return temp;
	}
	
	template <typename T>
	Tensor4Iterator<T> Tensor4Iterator<T>::operator+(const ssize_t i) const
	{
		iterator temp(*this);
		temp += i;
		return temp;
	}
	
	template <typename T>
	typename Tensor4Iterator<T>::parent_type::difference_type Tensor4Iterator<T>::operator-(const iterator& i) const
	{
		return m_std_iterator - i.m_std_iterator;
	}
	
	template <typename T>
	T& Tensor4Iterator<T>::operator*()
	{
		m_parent.updateHost();
		m_parent.invalidateDeviceData();
		return *m_std_iterator;
	}
	
	template <typename T>
	const T& Tensor4Iterator<T>::operator*() const
	{
		m_parent.updateHost();
		return *m_std_iterator;
	}
	
	template <typename T>
	const T& Tensor4Iterator<T>::operator-> () const
	{
		m_parent.updateHost();
		return *m_std_iterator;
	}
	
	template <typename T>
	T& Tensor4Iterator<T>::operator-> ()
	{
		m_parent.updateHost();
		m_parent.invalidateDeviceData();
		return *m_std_iterator;
	}
}
