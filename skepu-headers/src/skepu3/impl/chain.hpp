#pragma once

// Setters that return the skeleton itself, so that the settings of a skeleton can be chained, also on a skeleton
// that is created and called in the same expression (an anonymous skeleton instance):
//
//     skepu::MapOverlap(f).setOverlap(1).setEdgeMode(skepu::Edge::Pad).setPad(0)(out, in);
//
// The setters are written in each concrete skeleton class, hiding the (void) setters of the base classes: the return
// type is then the type of the skeleton class, whatever base class has the setter.

// A setter 'name' of the class ...: the member of the base class that is given as the second argument
#define SKEPU_CHAIN(name, ...) \
	template<typename... SkepuChainArgs> \
	auto &name(SkepuChainArgs&&... skepuChainArgs) \
	{ \
		this->__VA_ARGS__::name(std::forward<SkepuChainArgs>(skepuChainArgs)...); \
		return *this; \
	}

// The setters that all skeletons have (of SkeletonBase and of SeqSkeletonBase)
#define SKEPU_CHAIN_SKELETON_SETTERS(...) \
	SKEPU_CHAIN(setBackend, __VA_ARGS__) \
	SKEPU_CHAIN(resetBackend, __VA_ARGS__) \
	SKEPU_CHAIN(setExecPlan, __VA_ARGS__) \
	SKEPU_CHAIN(setLabel, __VA_ARGS__) \
	SKEPU_CHAIN(setSourceLoc, __VA_ARGS__) \
	SKEPU_CHAIN(setPRNG, __VA_ARGS__)
