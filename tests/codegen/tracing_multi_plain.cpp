// A source file that includes SkePU but is not precompiled (like the runtime sources of a
// SkePU based language). With SKEPU_TRACING defined for the whole program it is compiled
// without tracing support, as the header demands, and must still link with the traced files.
#include <skepu>

int tracing_multi_plain()
{
	skepu::Vector<int> v(4, 1);
	return (int)v.size();
}
