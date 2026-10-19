// The argument parser of skepu-lib/args.hpp. (SKEPU_ARGS_THROW: a wrong command line throws skepu::args::Exit instead of ending the test.)

#include "../../external/catch2/catch.hpp"
#define SKEPU_ARGS_THROW
#define SKEPU_ENABLE_EXCEPTIONS // (the errors of the program, such as a declaration of the same name with another type, are thrown)
#include <skepu>
#include <skepu-lib/args.hpp>
#include <test_helper_automatic_setup.hpp>
#include <string>
#include <vector>

namespace
{
	// A command line (without the name of the program) as the argc and argv of main
	struct CommandLine
	{
		std::vector<std::string> words;
		std::vector<char *> pointers;
		explicit CommandLine(std::vector<std::string> arguments) : words{"/path/to/program"}
		{
			for (auto &a : arguments) words.push_back(a);
			for (auto &w : words) pointers.push_back(w.data());
		}
		int argc() { return (int)pointers.size(); }
		char **argv() { return pointers.data(); }
	};

	// The message of the exit that the parser makes (an empty one and status -1 if it does not)
	template<typename F>
	std::pair<int, std::string> exit_of(std::vector<std::string> arguments, F declare)
	{
		CommandLine line(arguments);
		try
		{
			skepu::args::Parser args(line.argc(), line.argv(), "A test program");
			declare(args);
			args.done();
		}
		catch (const skepu::args::Exit &e)
		{
			return {e.status, e.what()};
		}
		return {-1, ""};
	}
}


TEST_CASE("args: positional arguments, with and without defaults")
{
	CommandLine line({"64", "2.5", "text"});
	skepu::args::Parser args(line.argc(), line.argv());
	auto n = args.get<size_t>("n");
	auto x = args.get<float>("x", 1.f);
	auto s = args.get<std::string>("s", "none");
	auto missing = args.get<int>("missing", -3);
	args.done();
	CHECK(n == 64);
	CHECK(x == 2.5f);
	CHECK(s == "text");
	CHECK(missing == -3);
}

TEST_CASE("args: several positional arguments at once")
{
	CommandLine line({"4", "5", "-0.5"});
	skepu::args::Parser args(line.argc(), line.argv());
	auto [rows, cols, scale] = args.get<size_t, size_t, double>("rows", "cols", "scale");
	args.done();
	CHECK(rows == 4);
	CHECK(cols == 5);
	CHECK(scale == -0.5); // a negative number is an argument, not an option
}

TEST_CASE("args: named options and flags, in both forms, in any order after the positional arguments")
{
	CommandLine line({"7", "--frames", "12", "--adam", "--out=pic", "--rate=0.25"});
	skepu::args::Parser args(line.argc(), line.argv());
	auto n = args.get<int>("n");
	auto out = args.named<std::string>("out", "");
	auto frames = args.named<size_t>("frames", 0);
	auto rate = args.named<double>("rate", 1.0);
	auto none = args.named<int>("none", 9);
	bool adam = args.flag("adam");
	bool other = args.flag("other");
	args.done();
	CHECK(n == 7);
	CHECK(out == "pic");
	CHECK(frames == 12);
	CHECK(rate == 0.25);
	CHECK(none == 9);
	CHECK(adam);
	CHECK(!other);
}

TEST_CASE("args: the backend, as a positional argument or an option")
{
	{
		CommandLine line({"3", "OpenMP"});
		skepu::args::Parser args(line.argc(), line.argv());
		auto n = args.get<int>("n");
		auto spec = args.backend();
		CHECK(n == 3);
		CHECK(spec.getType() == skepu::Backend::Type::OpenMP);
	}
	{
		CommandLine line({"3", "--backend", "cpu"});
		skepu::args::Parser args(line.argc(), line.argv());
		args.get<int>("n");
		CHECK(args.backend().getType() == skepu::Backend::Type::CPU);
	}
	{
		CommandLine line({"3", "-b", "cpu", "--verbose"});
		skepu::args::Parser args(line.argc(), line.argv());
		args.get<int>("n");
		bool verbose = args.flag("verbose");
		CHECK(args.backend().getType() == skepu::Backend::Type::CPU);
		CHECK(verbose);
	}
	{
		CommandLine line({"3"});
		skepu::args::Parser args(line.argc(), line.argv());
		args.get<int>("n");
		CHECK(args.backend().getType() == skepu::BackendSpec{}.getType()); // the default backend
	}
	skepu::restoreDefaultGlobalBackendSpec();
}

TEST_CASE("args: a wrong command line is an error with a message")
{
	auto n_and_x = [](skepu::args::Parser &args) { args.get<int>("n"); args.get<float>("x", 1.f); };

	auto [status, message] = exit_of({}, n_and_x);
	CHECK(status == 1);
	CHECK(message.find("<n> is missing") != std::string::npos);

	std::tie(status, message) = exit_of({"abc"}, n_and_x);
	CHECK(status == 1);
	CHECK(message.find("not 'abc'") != std::string::npos);

	std::tie(status, message) = exit_of({"3.5"}, n_and_x); // an integer is expected
	CHECK(status == 1);

	std::tie(status, message) = exit_of({"-3"}, [](skepu::args::Parser &args) { args.get<size_t>("n"); }); // not negative
	CHECK(status == 1);

	std::tie(status, message) = exit_of({"1", "2", "3"}, n_and_x);
	CHECK(status == 1);
	CHECK(message.find("unexpected argument '3'") != std::string::npos);
	CHECK(message.find("Usage:") != std::string::npos);

	std::tie(status, message) = exit_of({"1", "--fames", "6"}, [](skepu::args::Parser &args) { args.get<int>("n"); args.named<int>("frames", 0); });
	CHECK(status == 1);
	CHECK(message.find("unknown option --fames") != std::string::npos);

	std::tie(status, message) = exit_of({"1", "--frames"}, [](skepu::args::Parser &args) { args.get<int>("n"); args.named<int>("frames", 0); });
	CHECK(status == 1);
	CHECK(message.find("--frames needs a value") != std::string::npos);

	std::tie(status, message) = exit_of({"1", "--frames=x"}, [](skepu::args::Parser &args) { args.get<int>("n"); args.named<int>("frames", 0); });
	CHECK(status == 1);

	std::tie(status, message) = exit_of({"1", "--rate", "2", "4"}, [](skepu::args::Parser &args) { args.get<int>("n"); args.named<int>("rate", 0); }); // the options come last
	CHECK(status == 1);

	std::tie(status, message) = exit_of({"1", "fortran"}, [](skepu::args::Parser &args) { args.get<int>("n"); args.backend(); });
	CHECK(status == 1);
	CHECK(message.find("the backend must be") != std::string::npos);
}

TEST_CASE("args: --help prints the usage, made from the declarations")
{
	auto [status, message] = exit_of({"--help"}, [](skepu::args::Parser &args)
	{
		args.get<size_t>("height");
		args.get<float>("persistence", 0.5f, "how fast the amplitude falls");
		args.named<size_t>("frames", 0, "number of frames");
		args.flag("adam", "use Adam");
		args.backend();
	});
	CHECK(status == 0);
	CHECK(message.find("A test program") != std::string::npos);
	CHECK(message.find("Usage: program <height> [persistence] [backend] [--frames N] [--adam] [--help]") != std::string::npos);
	CHECK(message.find("how fast the amplitude falls (number, default 0.5)") != std::string::npos);
	CHECK(message.find("--frames") != std::string::npos);
}


TEST_CASE("args: a name that is declared again is the same argument")
{
	CommandLine line({"7", "8", "cpu", "--frames", "12", "--adam", "--out=pic"});
	skepu::args::Parser args(line.argc(), line.argv());
	auto n = args.get<size_t>("n");
	auto n_again = args.get<size_t>("n"); // (not the next positional argument)
	auto m = args.get<int>("m", 0);
	auto m_again = args.get<int>("m", 0);
	auto frames = args.named<size_t>("frames", 0);
	auto out = args.named<std::string>("out", "");
	bool adam = args.flag("adam");
	bool other = args.flag("other");
	args.backend();
	// (after backend(): the arguments that are known are still there, so a function can take the parser and read what it needs)
	CHECK(n == 7);
	CHECK(n_again == 7);
	CHECK(m == 8);
	CHECK(m_again == 8);
	CHECK(args.get<size_t>("n") == 7);
	CHECK(args.get<int>("m", 0) == 8);
	CHECK(args.get<int>("m") == 8); // (it has a default: not an error to read it as one that is required)
	CHECK(args.named<size_t>("frames", 0) == 12);
	CHECK(args.named<std::string>("out", "") == "pic");
	CHECK(args.flag("adam") == adam);
	CHECK(args.flag("other") == other);
	CHECK(adam);
	CHECK(!other);
	CHECK(args.backend().getType() == skepu::Backend::Type::CPU);
	skepu::restoreDefaultGlobalBackendSpec();
}

TEST_CASE("args: a value that is not on the command line is its default, also when it is declared again")
{
	CommandLine line({"7"});
	skepu::args::Parser args(line.argc(), line.argv());
	auto n = args.get<int>("n");
	auto x = args.get<float>("x", 2.5f);
	auto frames = args.named<int>("frames", 3);
	args.done();
	CHECK(n == 7);
	CHECK(x == 2.5f);
	CHECK(args.get<float>("x", 2.5f) == 2.5f);
	CHECK(args.named<int>("frames", 3) == 3);
}

TEST_CASE("args: a name that is declared again with another type, kind or default is an error")
{
	CommandLine line({"7", "--frames", "2"});
	skepu::args::Parser args(line.argc(), line.argv());
	args.get<int>("n");
	args.named<int>("frames", 0);
	CHECK_THROWS(args.get<float>("n"));
	CHECK_THROWS(args.named<int>("n", 0));
	CHECK_THROWS(args.flag("frames"));
	CHECK_THROWS(args.named<unsigned>("frames", 0));
	CHECK_THROWS(args.named<int>("frames", 5)); // (another default)
	CHECK(args.named<int>("frames", 0) == 2);
	args.done();
}

TEST_CASE("args: a name that is declared again is in the usage once")
{
	auto [status, message] = exit_of({"--help"}, [](skepu::args::Parser &args)
	{
		args.get<size_t>("height");
		args.get<size_t>("height");
		args.named<int>("frames", 0);
		args.named<int>("frames", 0);
	});
	CHECK(status == 0);
	auto count = [&](const std::string &text) { size_t n = 0; for (auto at = message.find(text); at != std::string::npos; at = message.find(text, at + 1)) ++n; return n; };
	CHECK(count("<height>") == 1);
	CHECK(count("--frames N") == 1);
}
