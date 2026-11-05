// The list of compiler directories (tracing_paths.hpp) must outlive an object that reads it on destruction: the tracer does, and it can be made before the
// list is (used while statics are made, before a translation unit notes its directory). Exits 0 if the object read the list.
// With an ordinary static the list is gone (empty with libc++, dangling with libstdc++).
#include "../../skepu-headers/src/skepu3/impl/tracing_paths.hpp"

#include <cstdio>
#include <cstdlib>

namespace paths = skepu::tracing::paths;

namespace
{
	int g_directories_at_exit = -1;

	// made before the list: destroyed after it were the list an ordinary static
	struct Reader
	{
		~Reader()
		{
			g_directories_at_exit = static_cast<int>(paths::compileDirectories().size());
			std::printf("directories at exit: %d\n", g_directories_at_exit);
			if (g_directories_at_exit != 1)
				std::_Exit(2);   // main's exit code is already decided
		}
	};

	Reader &reader() { static Reader instance; return instance; }

	const bool noted = (reader(), paths::noteCompileDirectory("/a/b"));
}

int main()
{
	return noted && paths::compileDirectories().size() == 1 ? 0 : 1;
}
