// The precompiler names the containers that are declared as variables by their variable names in the traces
// (see tracing_container_labels in CMakeLists.txt): vectors, matrices and tensors, several in a declaration,
// in loops and in function templates; but not the ones that it cannot name (const, references, conditions), and a name
// that the program gives (a constructor argument or setLabel) is kept.
#include "../../external/catch2/catch.hpp"
#include "../../external/nlohmann/json.hpp"

#include <fstream>
#include <map>
#include <skepu>

namespace
{
	using json = nlohmann::json;

	// The number of allocation events for each label
	std::map<std::string, int> allocations()
	{
		SKEPU_TRACE_FLUSH();
		std::ifstream file(std::getenv("SKEPU_TRACE_FILE"));
		REQUIRE(file.good());
		std::map<std::string, int> result;
		json trace = json::parse(file);
		for (auto const& e : trace.is_object() ? trace["data"] : trace)
			if (e["type"] == "allocation")
				result[e["label"].get<std::string>()]++;
		return result;
	}

	template<typename T>
	void in_a_template(int n)
	{
		skepu::Vector<T> in_template(n);
		skepu::Matrix<T> matrix_in_template(n, n);
		in_template.flush();
	}

	int counted(std::map<std::string, int> const& labels, std::string const& label)
	{
		auto i = labels.find(label);
		return i == labels.end() ? 0 : i->second;
	}
}

TEST_CASE("Container variables are named in the trace")
{
	skepu::Vector<float> first(4), second(4, 1.f); int after = 5; skepu::Matrix<float> matrix(2, 3);
	skepu::Tensor3<float> tensor3(2, 2, 2);
	skepu::Tensor4<float> tensor4(2, 2, 2, 2);
	auto from_auto = skepu::Vector<float>(3);
	for (int i = 0; i < 3; ++i)
	{
		skepu::Vector<float> in_loop(i + 1);
		in_loop.flush();
	}
	in_a_template<float>(2);
	in_a_template<double>(2);

	// a name that the program gives is kept: in the constructor, and with setLabel after the declaration
	skepu::Vector<float> constructor_label(2, "given"); skepu::Matrix<float> matrix_label(2, 2, "given_matrix");
	skepu::Vector<float> renamed(2);
	renamed.setLabel("later");

	// (these cannot be named, and must still compile and run: a const container, a reference, a declaration in a condition and in a loop header)
	const skepu::Vector<float> constant(2);
	skepu::Vector<float> &reference = first;
	if (skepu::Vector<float> in_condition(2); in_condition.size() == 2)
		reference(0) = 1.f;
	for (skepu::Vector<float> in_header(2); in_header.size() > 2; ) {}
	int line = __LINE__; skepu::Vector<float>
		over_lines(2); // (the next line is the next line)
	CHECK(__LINE__ == line + 2);

	first.flush();
	CHECK(after == 5);
	CHECK(constant.size() == 2);

	auto labels = allocations();
	for (auto const& name : {"first", "second", "matrix", "tensor3", "tensor4", "from_auto", "over_lines"})
		CHECK(counted(labels, name) == 1);
	CHECK(counted(labels, "in_loop") == 3);
	CHECK(counted(labels, "in_template") == 2);
	CHECK(counted(labels, "matrix_in_template") == 2);
	for (auto const& name : {"given", "given_matrix", "later"})
		CHECK(counted(labels, name) == 1);
	for (auto const& name : {"constructor_label", "matrix_label", "renamed"})
		CHECK(counted(labels, name) == 0);
	for (auto const& name : {"constant", "reference", "in_condition", "in_header"})
		CHECK(counted(labels, name) == 0);
}

TEST_CASE("The allocation event of a container is at its declaration")
{
	const int line = __LINE__ + 1;
	skepu::Vector<float> located(3); skepu::Matrix<float> also_located(2, 2);
	skepu::Vector<float> constructed_in_place = skepu::Vector<float>(2);   // (named, and located, as a variable)
	located.flush();

	SKEPU_TRACE_FLUSH();
	std::ifstream file(std::getenv("SKEPU_TRACE_FILE"));
	REQUIRE(file.good());
	json trace = json::parse(file);
	std::map<std::string, std::pair<std::string, int>> places;
	for (auto const& e : trace.is_object() ? trace["data"] : trace)
		if (e["type"] == "allocation" && e.contains("file"))   // (the containers that the program does not declare have the place of the header)
			places[e["label"].get<std::string>()] = {e["file"].get<std::string>(), e["line"].get<int>()};

	const std::string name = "tracing_container_labels.cpp";
	for (auto const& label : {"located", "also_located", "constructed_in_place"})
	{
		REQUIRE(places.count(label) == 1);
		REQUIRE(places[label].first.size() >= name.size());
		CHECK(places[label].first.compare(places[label].first.size() - name.size(), name.size(), name) == 0);
	}
	CHECK(places["located"].second == line);
	CHECK(places["also_located"].second == line);
	CHECK(places["constructed_in_place"].second == line + 1);
}
