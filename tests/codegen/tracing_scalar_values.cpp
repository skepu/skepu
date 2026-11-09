// Scalar values in the trace: a scalar returned by a skeleton has a "scalar_value" event (object id, number) before the call that outputs it.
// LogValue::Yes derives a later value of the source id; operators make no event. Non-finite numbers are strings; non-numeric scalars have no event.
#include "../../external/catch2/catch.hpp"
#include "../../external/nlohmann/json.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <skepu>
#include <skepu-lib/util.hpp>

namespace
{
	using json = nlohmann::json;

	json load_trace()
	{
		SKEPU_TRACE_FLUSH();
		std::ifstream file(std::getenv("SKEPU_TRACE_FILE"));
		REQUIRE(file.good());
		json trace = json::parse(file);
		return trace.is_object() ? trace["data"] : trace;
	}

	// The values of the events of an id, in the order of the trace
	std::vector<json> values_of(json const& trace, unsigned long id)
	{
		std::vector<json> result;
		for (auto const& e : trace)
			if (e["type"] == "scalar_value" && e["object_id"] == id)
				result.push_back(e["value"]);
		return result;
	}

	// The scalar output of the call with the label (the first one)
	unsigned long output_of(json const& trace, std::string const& label)
	{
		for (auto const& e : trace)
			if (e["type"] == "skeleton_call" && e["label"] == label)
				return e["outputs"][0].get<unsigned long>();
		FAIL("no call " << label);
		return 0;
	}

	int square(int x) { return x * x; }
}

TEST_CASE("The values of scalars are in the trace")
{
	skepu::Vector<int> numbers(10);
	skepu::Vector<float> floats(4);
	for (int i = 0; i < 10; ++i) numbers[i] = i + 1;
	for (int i = 0; i < 4; ++i) floats[i] = 0.5f * i;

	auto sum_int = skepu::Reduce(skepu::util::add<int>);
	auto sum_float = skepu::Reduce(skepu::util::add<float>);
	auto sum_squares = skepu::MapReduce<1>(square, skepu::util::add<int>);

	const int total = sum_int(numbers);
	const float total_float = sum_float(floats);
	const int squares = sum_squares(numbers);
	REQUIRE(total == 55);
	REQUIRE(squares == 385);

	// A scalar of the id of another one (as blas::nrm2 does), and numbers that JSON cannot write
	skepu::Scalar<float> sum_scalar = sum_float(floats);
	skepu::Scalar<float> derived(std::sqrt((float)sum_scalar), sum_scalar.m_ids, skepu::LogValue::Yes);
	skepu::Scalar<float> not_a_number(std::numeric_limits<float>::quiet_NaN());
	skepu::Scalar<double> infinite(std::numeric_limits<double>::infinity());
	skepu::Scalar<double> minus_infinite(-std::numeric_limits<double>::infinity());
	skepu::Scalar<long long> large(-9007199254740993LL);
	skepu::Scalar<unsigned long long> unsigned_large(18446744073709551615ULL);
	skepu::Scalar<float> precise(0.1f);

	// operator results carry the ids of their operands and have no event
	skepu::Scalar<float> combined = sum_scalar + not_a_number;
	auto halved = sum_scalar / 2;
	auto negated = -sum_scalar;
	auto small = sum_scalar < 100.f;
	REQUIRE(halved.m_ids == sum_scalar.m_ids);
	REQUIRE(negated.m_ids == sum_scalar.m_ids);
	REQUIRE(small.m_ids == sum_scalar.m_ids);
	(void)combined;

	const json trace = load_trace();

	// the three calls: their scalar outputs have a value each (the label of an instance is its variable)
	REQUIRE(values_of(trace, output_of(trace, "sum_int")) == std::vector<json>{55});
	REQUIRE(values_of(trace, output_of(trace, "sum_squares")) == std::vector<json>{385});
	{
		auto values = values_of(trace, output_of(trace, "sum_float"));
		REQUIRE(values.size() >= 1);
		REQUIRE(values[0].get<double>() == Approx(total_float));
	}

	// sum_scalar comes from the second sum_float call; the derived scalar has its id and a later value
	{
		const auto id = sum_scalar.m_ids[0];
		auto values = values_of(trace, id);
		REQUIRE(values.size() == 2);
		REQUIRE(values[0].get<double>() == Approx(3.0));
		REQUIRE(values[1].get<double>() == Approx(std::sqrt(3.0)));
		REQUIRE(derived.m_ids[0] == id);
	}

	REQUIRE(values_of(trace, not_a_number.m_ids[0]) == std::vector<json>{"nan"});
	REQUIRE(values_of(trace, infinite.m_ids[0]) == std::vector<json>{"inf"});
	REQUIRE(values_of(trace, minus_infinite.m_ids[0]) == std::vector<json>{"-inf"});
	REQUIRE(values_of(trace, large.m_ids[0]) == std::vector<json>{-9007199254740993LL});
	REQUIRE(values_of(trace, unsigned_large.m_ids[0]) == std::vector<json>{18446744073709551615ULL});

	// a float is written as the shortest decimal number that reads back as it
	REQUIRE(values_of(trace, precise.m_ids[0]) == std::vector<json>{0.1});

	// a scalar of two ids has no event of its own
	for (auto id : combined.m_ids)
		(void)id;
	REQUIRE(combined.m_ids.size() == 2);
}
