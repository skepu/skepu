// Configure-time probe: whether a Metal device exists, and whether its memory is shared with the CPU (hasUnifiedMemory, true on Apple silicon).
// Prints "device=<0|1> unified=<0|1>". Plain C++: the device is a pointer and the property is read with a message send.
#include <objc/message.h>
#include <objc/runtime.h>
#include <cstdio>

extern "C" void *MTLCreateSystemDefaultDevice(void);

int main()
{
	void *device = MTLCreateSystemDefaultDevice();
	int unified = 0;
	if (device)
	{
		using Getter = signed char (*)(void *, SEL);
		unified = reinterpret_cast<Getter>(objc_msgSend)(device, sel_registerName("hasUnifiedMemory")) != 0;
	}
	std::printf("device=%d unified=%d\n", device ? 1 : 0, unified);
	return 0;
}
