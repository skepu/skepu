// The trace file of a tracer: the default name has the module's name when it is built as a module of a
// program that has several (SKEPU_TRACE_MODULE=name), and %m in SKEPU_TRACE_FILE is the module's name
// (see tracing_file_names in CMakeLists.txt, which checks the files). SKEPU_TRACE_FLUSH writes the trace
// early; the events after it are in the file at the end of the program.
#include <skepu>

int main()
{
	SKEPU_TRACE_ALLOCATION(skepu::UniqueIdentifier::generate(), "before the flush", 1);
	SKEPU_TRACE_FLUSH();
	SKEPU_TRACE_ALLOCATION(skepu::UniqueIdentifier::generate(), "after the flush", 2);
	return 0;
}
