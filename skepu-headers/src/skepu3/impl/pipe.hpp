
#pragma once

#include <thread>

namespace skepu {

template<typename Skeleton>
class PipeStage
{
public:
	PipeStage(Skeleton &stage, size_t i)
	: skel(stage), index{i}
	{
		std::cout << "Hi from stage " << index << std::endl;
		
		
	}
	
	
	template<typename Out, typename In>
	void run(Out *out, In *in)
	{
		std::cout << "Running stage " << index << std::endl;
		
		this->apply(out, in);
	}
	
	
	template<typename Out, typename In>
	void apply(Out *out, In *in)
	{
		
		skel(*out, *in);
	}
	
	
	const size_t index;
	Skeleton& skel;
};



template<typename P>
void pipeline_run(P &&pipe)
{
	pipe.run();
}


template<typename T>
struct add_pipestage
{
	using type = PipeStage<T>;
};


template<typename ... T, size_t... SI>
decltype(auto) make_pipeline(size_t start, pack_indices<SI...>, T&... args)
{
	return std::make_tuple(PipeStage(args, start + SI)...);
}



template<typename... T>
class Pipe
{
public:
	static constexpr size_t length = sizeof...(T);
	static constexpr typename make_pack_indices<length, 0>::type stage_indexes{};
	
	Pipe(T&...args)
	: stages{make_pipeline(0, stage_indexes, args...)}
	{
		
	}
	
	Pipe(size_t start, T&...args)
	: stages{make_pipeline(start, stage_indexes, args...)}
	{
		
	}
	
	template<typename Out, typename In>
	void operator()(Out &&out, In &&in)
	{
		this->apply_helper(out, in, stage_indexes);
	}
	
	template<typename Out, typename In, size_t... SI>
	void apply_helper(Out &&out, In &&in, pack_indices<SI...>)
	{
		auto threads = std::make_tuple(std::thread(
			&std::decay<decltype(std::get<SI>(stages))>::type::template run<skepu::Vector<float>, skepu::Vector<float>>,
			&std::get<SI>(stages),
			&out,
			&in)...
		);
			
			pack_expand((std::get<SI>(threads).join(), 0)...);
	}
	
	
	std::tuple<PipeStage<T>...> stages;
};


template<typename... T1, int InArity, int GivenArity, typename... T2, size_t... SI>
decltype(auto)
operator_helper(Pipe<T1...> &pipe, skepu::MapImpl<InArity, GivenArity, T2...> &skel, pack_indices<SI...>)
{
	return Pipe(/*pipe.length, */std::get<SI>(pipe.stages).skel..., skel);
}

template<typename... T1, int InArity, int GivenArity, typename... T2>
decltype(auto)
operator>>(Pipe<T1...> pipe, skepu::MapImpl<InArity, GivenArity, T2...> &skel)
{
	return operator_helper(pipe, skel, Pipe<T1...>::stage_indexes);
}

template<int InArityL, int GivenArityL, typename... TL, int InArityR, int GivenArityR, typename... TR>
decltype(auto)
operator>>(skepu::MapImpl<InArityL, GivenArityL, TL...> skellhs, skepu::MapImpl<InArityR, GivenArityR, TR...> &skelrhs)
{
	return Pipe(skellhs, skelrhs);
}





} // namespace skepu






