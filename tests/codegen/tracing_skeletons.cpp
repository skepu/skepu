// What the trace says about each skeleton, on every backend: the pattern name, the elements (the same
// whatever the backend), the containers read and written, the random stream, the parameters, the
// backend, and that the time of a call is measured (see tracing_skeletons in CMakeLists.txt).
#include "../../external/catch2/catch.hpp"
#include "../../external/nlohmann/json.hpp"

#include <fstream>
#include <map>
#include <skepu>

// (a function for each skeleton: the precompiler does not take a function that two skeletons share)
float map_add(float a, float b) { return a + b; }
float reduce_add(float a, float b) { return a + b; }
float cols_add(float a, float b) { return a + b; }
float mr_map(float a, float b) { return a * b; }
float mr_add(float a, float b) { return a + b; }
float scan_add(float a, float b) { return a + b; }
float pairs_red_add(float a, float b) { return a + b; }
float inc(float a) { return a + 1.0f; }
float inc_moved(float a) { return a + 1.0f; }
float random_f(skepu::Random<1> &r) { return r.getNormalized(); }
float stencil1(skepu::Region1D<float> r) { return r(-r.oi) + r(r.oi); }
float stencil2(skepu::Region2D<float> r) { return r(-r.oi, 0) + r(r.oi, 0) + r(0, -r.oj) + r(0, r.oj); }
float pool2(skepu::Pool2D<float> p) { return p(0, 0) + p(p.si - 1, p.sj - 1); }
float pair_f(float a, float b) { return a * b; }
float pair_g(float a, float b) { return a - b; }

namespace
{
	using json = nlohmann::json;

	json load_trace()
	{
		SKEPU_TRACE_FLUSH();
		std::ifstream file(std::getenv("SKEPU_TRACE_FILE"));
		REQUIRE(file.good());
		json trace = json::parse(file);
		return trace.is_object() ? trace["data"] : trace;    // (with a header, the events are the data)
	}

	// The container labels of a list of object ids
	std::string labels(json const& trace, json const& ids)
	{
		std::string result;
		for (auto const& id : ids)
			for (auto const& e : trace)
				if (e["type"] == "allocation" && e["object_id"] == id)
					result += (result.empty() ? "" : ",") + e["label"].get<std::string>();
		return result;
	}

	json call(json const& trace, std::string const& label)
	{
		for (auto const& e : trace)
			if (e["type"] == "skeleton_call" && e["label"] == label)
				return e;
		FAIL("no call of " << label);
		return {};
	}

#ifdef SKEPU_OPENCL
	const std::string backend = "OpenCL";
#elif defined(SKEPU_OPENMP)
	const std::string backend = "OpenMP";
#else
	const std::string backend = "CPU";
#endif
}

// (the skeleton instances are global: the precompiler does not take them inside a TEST_CASE)
auto map2 = skepu::Map<2>(map_add);
auto map_strided = skepu::Map<1>(inc);
auto map_random = skepu::Map<0>(random_f);
auto reduce = skepu::Reduce(reduce_add);
auto reduce_cols = skepu::Reduce(cols_add);
auto mapreduce = skepu::MapReduce<2>(mr_map, mr_add);
auto scan = skepu::Scan(scan_add);
auto overlap1 = skepu::MapOverlap(stencil1);
auto overlap2 = skepu::MapOverlap(stencil2);
auto pool = skepu::MapPool(pool2);
auto mappairs = skepu::MapPairs<1, 1>(pair_f);
auto mappairsreduce = skepu::MapPairsReduce<1, 1>(pair_g, pairs_red_add);
auto map_moved = skepu::Map<1>(inc_moved);

TEST_CASE("the trace of the skeletons")
{
	const size_t n = 256, rows = 16, cols = 32;
	skepu::Vector<float> a(n, "a", 1.0f), b(n, "b", 2.0f), c(n, "c"), d(n / 2, "d"), pairs_h(cols, "pairs_h", 1.0f), pairs_v(rows, "pairs_v", 2.0f), pairs_red(rows, "pairs_red");
	skepu::Matrix<float> m(rows, cols, "m", 1.0f), m2(rows, cols, "m2"), pooled(rows / 2, cols / 2, "pooled"), pairs_out(rows, cols, "pairs_out");

	map2.setLabel("map2");
	map2(c, a, b);

	map_strided.setLabel("map_strided");
	map_strided.setStride(2, 2);
	map_strided(d, a);

	map_random.setLabel("map_random");
	map_random(c);

	reduce.setLabel("reduce");
	reduce(a);

	reduce_cols.setLabel("reduce_cols");
	reduce_cols.setReduceMode(skepu::ReduceMode::ColWise);
	skepu::Vector<float> col_sums(cols, "col_sums");
	reduce_cols(col_sums, m);

	mapreduce.setLabel("mapreduce");
	mapreduce(a, b);

	scan.setLabel("scan");
	scan.setScanMode(skepu::ScanMode::Exclusive);
	scan.setStartValue(3.0f);
	scan(c, a);

	overlap1.setLabel("overlap1");
	overlap1.setOverlap(2);
	overlap1.setEdgeMode(skepu::Edge::Pad);
	overlap1.setPad(7.0f);
	overlap1(c, a);

	overlap2.setLabel("overlap2");
	overlap2.setOverlap(1, 2);
	overlap2(m2, m);

	pool.setLabel("pool");
	pool.setPoolSize(2, 2);
	pool.setStride(2, 2);
	pool(pooled, m);

	mappairs.setLabel("mappairs");
	mappairs(pairs_out, pairs_v, pairs_h);

	mappairsreduce.setLabel("mappairsreduce");
	mappairsreduce(pairs_red, pairs_v, pairs_h);

	const json trace = load_trace();

	SECTION("what each call is")
	{
		struct Expected { std::string label, pattern, elements, inputs, outputs; };
		const std::vector<Expected> expected = {
			{"map2", "Map", "[256]", "a,b", "c"},
			{"map_strided", "Map", "[64]", "a", "d"},
			{"map_random", "Map", "[256]", "", "c"},
			{"reduce", "Reduce 1D-Full", "[256]", "a", ""},
			{"reduce_cols", "Reduce 1D-Partial", "[32,16]", "m", "col_sums"},
			{"mapreduce", "MapReduce", "[256]", "a,b", ""},
			{"scan", "Scan", "[256]", "a", "c"},
			{"overlap1", "MapOverlap 1D", "[256]", "a", "c"},
			{"overlap2", "MapOverlap 2D", "[16,32]", "m", "m2"},
			{"pool", "MapPool 2D", "[8,16]", "m", "pooled"},
			{"mappairs", "MapPairs", "[32,16]", "pairs_v,pairs_h", "pairs_out"},
			{"mappairsreduce", "MapPairsReduce", "[32,16]", "pairs_v,pairs_h", "pairs_red"},
		};
		for (auto const& x : expected)
		{
			INFO(x.label);
			json e = call(trace, x.label);
			CHECK(e["pattern"] == x.pattern);
			CHECK(e["elements"].dump() == x.elements);
			CHECK(labels(trace, e["elwise_inputs"]) == x.inputs);
			if (!x.outputs.empty())
				CHECK(labels(trace, e["outputs"]) == x.outputs);
			CHECK(e["backend"] == backend);
			CHECK(e["start"].get<unsigned long>() <= e["end"].get<unsigned long>());
			CHECK(e["line"].get<int>() > 0);
			CHECK(!e["file"].get<std::string>().empty());
		}
	}

	SECTION("the random stream of a call that draws numbers, and of no other")
	{
		CHECK(call(trace, "map_random").contains("prng"));
		CHECK(!call(trace, "map2").contains("prng"));
	}

	SECTION("the parameters that are not the default")
	{
		CHECK(!call(trace, "map2").contains("parameters"));
		CHECK(call(trace, "map_strided")["parameters"]["strides"].dump() == "[2,2]");
		CHECK(!call(trace, "reduce").contains("parameters"));
		CHECK(call(trace, "reduce_cols")["parameters"]["mode"] == "ColWise");
		CHECK(call(trace, "scan")["parameters"]["mode"] == "Exclusive");
		CHECK(call(trace, "scan")["parameters"]["start_value"] == 3.0);
		json overlap1 = call(trace, "overlap1")["parameters"];
		CHECK(overlap1["overlap"].dump() == "[2]");
		CHECK(overlap1["edge"] == "Pad");
		CHECK(overlap1["pad"] == 7.0);
		CHECK(call(trace, "overlap2")["parameters"]["overlap"].dump() == "[1,2]");
		json pool = call(trace, "pool")["parameters"];
		CHECK(pool["pool"] == true);
		CHECK(pool["pool_size"].dump() == "[2,2]");
		CHECK(pool["stride"].dump() == "[2,2]");
	}

	SECTION("what runs a call")
	{
		for (auto const& e : trace)
		{
			if (e["type"] != "skeleton_call")
				continue;
			if (backend == "CPU")
				CHECK(e["threads"] == 1);
			else if (backend == "OpenMP")
				CHECK(e["threads"].get<int>() >= 1);
			else
			{
				CHECK(!e.contains("threads"));
				CHECK(e["devices"].get<int>() >= 1);
			}
		}
	}

	SECTION("the names of the containers are the same in every run")
	{
		for (auto const& e : trace)
			if (e["type"] == "allocation" || e["type"] == "transfer")
				CHECK(e["label"].get<std::string>().rfind("0x", 0) != 0);
	}

	SECTION("a transfer is of a container that exists, and is not longer than it")
	{
		const std::map<std::string, size_t> sizes = {{"a", n}, {"b", n}, {"c", n}, {"d", n / 2}, {"pairs_h", cols}, {"pairs_v", rows}, {"pairs_red", rows}, {"col_sums", cols},
			{"m", rows * cols}, {"m2", rows * cols}, {"pooled", rows * cols / 4}, {"pairs_out", rows * cols}};
		for (auto const& e : trace)
			if (e["type"] == "transfer" && e["label"].get<std::string>().find("Internal") == std::string::npos)
			{
				CHECK(labels(trace, json::array({e["object_id"]})) == e["label"].get<std::string>());
				CHECK(e["elements"].get<size_t>() <= sizes.at(e["label"].get<std::string>()));
				CHECK(e["start"].get<unsigned long>() <= e["end"].get<unsigned long>());
			}
	}
}


// What the trace says about the life of containers: a copy is a dependence (and a container that gets its memory in an
// assignment, or wraps memory, is born), a move carries the id with the data.
TEST_CASE("the trace of the lifetime of containers")
{
	map_moved.setLabel("map_moved");
	float raw[4] = {1, 2, 3, 4};
	skepu::Vector<float> original(8, "original", 3.0f);
	skepu::Matrix<float> mat(2, 3, "mat", 1.0f), mat_copy(mat), mat_assigned(2, 3, "mat_assigned");
	mat_assigned = mat;
	skepu::Vector<float> assigned;
	{
		skepu::Vector<float> copy(original);
		assigned = original;
		skepu::Vector<float> wrapped(raw, 4, false);
		skepu::Vector<float> moved_from(8, "moved_from", 1.0f);
		skepu::Vector<float> moved(std::move(moved_from));
		map_moved(moved, moved);
		skepu::Matrix<float> filled(2, 2, "filled");
		filled = 2.0f;
	}

	// the copies are of the data, not of the memory
	skepu::external(skepu::read(mat, mat_copy, mat_assigned, original, assigned), [&]
	{
		mat_copy(0, 0) = 5.0f;
		mat_assigned(1, 1) = 6.0f;
		assigned(0) = 7.0f;
		CHECK(mat(0, 0) == 1.0f);
		CHECK(mat(1, 1) == 1.0f);
		CHECK(original(0) == 3.0f);
		CHECK(assigned.size() == 8);
	}, skepu::write(mat_copy, mat_assigned, assigned));

	const json trace = load_trace();
	std::map<int, std::string> label;
	std::map<int, int> allocated, freed;
	for (auto const& e : trace)
	{
		if (e["type"] == "allocation") { label[e["object_id"]] = e["label"]; allocated[e["object_id"]]++; }
		if (e["type"] == "deallocation") freed[e["object_id"]]++;
	}
	auto id_of = [&](std::string const& name)
	{
		for (auto const& [id, l] : label)
			if (l == name)
				return id;
		FAIL("no container " << name);
		return 0;
	};
	auto copies_from = [&](int source)
	{
		std::vector<json> result;
		for (auto const& e : trace)
			if (e["type"] == "copy" && e["source"] == source)
				result.push_back(e);
		return result;
	};

	SECTION("a copy constructor and an assignment are copies of the container")
	{
		auto of_original = copies_from(id_of("original"));
		REQUIRE(of_original.size() == 2);    // the copy, and the assignment
		for (auto const& e : of_original)
		{
			CHECK(e["elements"] == 8);
			CHECK(allocated[e["object_id"].get<int>()] == 1);                      // the copy was born
			CHECK(label[e["object_id"].get<int>()] == e["label"].get<std::string>());
			CHECK(e["start"].get<unsigned long>() <= e["end"].get<unsigned long>());
		}
		auto of_mat = copies_from(id_of("mat"));
		REQUIRE(of_mat.size() == 2);
		CHECK(of_mat[0]["elements"] == 6);
		CHECK(of_mat[1]["object_id"] == id_of("mat_assigned"));
		CHECK(allocated[id_of("mat_assigned")] == 1);   // (it had memory: not born by the assignment)
	}

	SECTION("no container is born twice or dies twice, and those of the inner block are dead")
	{
		for (auto const& [id, n] : allocated)
		{
			INFO(label[id]);
			CHECK(n == 1);
			CHECK(freed[id] <= 1);
		}
		for (auto const& name : {"moved_from", "filled"})
			CHECK(freed[id_of(name)] == 1);
	}

	SECTION("a vector that gets its memory in an assignment is born there")
	{
		bool born_in_assignment = false;
		for (auto const& e : trace)
			if (e["type"] == "copy" && e["source"] == id_of("original") && e["object_id"] == id_of("assigned") && allocated[id_of("assigned")] == 1)
				born_in_assignment = true;    // (the precompiler names it after the variable)
		CHECK(born_in_assignment);
	}

	SECTION("memory that a container wraps: it is born, and the end of the container is its death")
	{
		for (auto const& name : {"wrapped", "copy"})    // (named after the variables by the precompiler)
		{
			CHECK(allocated[id_of(name)] == 1);
			CHECK(freed[id_of(name)] == 1);
		}
	}

	SECTION("a move carries the id: the call reads and writes the memory of the vector that was moved from")
	{
		json e = call(trace, "map_moved");
		CHECK(e["elwise_inputs"][0] == id_of("moved_from"));
		CHECK(e["outputs"][0] == id_of("moved_from"));
	}

	SECTION("a fill writes the container")
	{
		bool fill = false;
		for (auto const& e : trace)
			if (e["type"] == "external" && e["label"] == "fill" && e["outputs"][0] == id_of("filled")) fill = true;
		CHECK(fill);
	}
}
