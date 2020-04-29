#include <iostream>
#include <skepu>

int uf(int a, int b)
{
#ifdef SKEPU_USING_BACKEND_OMP
	return omp_get_thread_num();
#endif
	return a * b;
}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	skepu::BackendSpec spec{argv[2]};
	
	auto skel = skepu::Map<2>(uf);
	
	
	skepu::Vector<int> v1(size), v2(size), v3(size);
	
	
	// Global spec
	
	// Run with default global
	std::cout << skel.selectBackend() << "\n";
	skel(v3, v1, v2);
	
	// Set global spec
	skepu::setGlobalBackendSpec(spec);
	std::cout << skel.selectBackend() << "\n";
	skel(v3, v1, v2);
	
	// Restore default
	skepu::restoreDefaultGlobalBackendSpec();
	std::cout << skel.selectBackend() << "\n";
	skel(v3, v1, v2);
	
	
	// Instance spec
	
	skel.setBackend(spec);
	std::cout << skel.selectBackend() << "\n";
	skel(v3, v1, v2);
	std::cout << v3 << std::endl;
	
	spec.setSchedulingMode(skepu::Backend::Scheduling::Static);
	skel.setBackend(spec);
	std::cout << skel.selectBackend() << "\n";
	skel(v3, v1, v2);
	std::cout << v3 << std::endl;
	
	spec.setSchedulingMode(skepu::Backend::Scheduling::Dynamic);
	skel.setBackend(spec);
	std::cout << skel.selectBackend() << "\n";
	skel(v3, v1, v2);
	std::cout << v3 << std::endl;
	
	spec.setSchedulingMode(skepu::Backend::Scheduling::Guided);
	skel.setBackend(spec);
	std::cout << skel.selectBackend() << "\n";
	skel(v3, v1, v2);
	std::cout << v3 << std::endl;
	
	spec.setSchedulingMode(skepu::Backend::Scheduling::Auto);
	skel.setBackend(spec);
	std::cout << skel.selectBackend() << "\n";
	skel(v3, v1, v2);
	std::cout << v3 << std::endl;
	
	spec.setSchedulingMode(skepu::Backend::Scheduling::Static);
	spec.setCPUChunkSize(8);
	skel.setBackend(spec);
	std::cout << skel.selectBackend() << "\n";
	skel(v3, v1, v2);
	std::cout << v3 << std::endl;
	
	return 0;
}

