// File names of a trace: one per file, without the shared directories.
#include "../../external/catch2/catch.hpp"
#include "../../skepu-headers/src/skepu3/impl/tracing_paths.hpp"

#include <fstream>
#include <random>

namespace fs = std::filesystem;
using skepu::tracing::paths::resolve;

namespace
{
	// temporary directory of files, removed at the end of the test
	struct Tree
	{
		fs::path root;
		Tree()
		{
			root = fs::weakly_canonical(fs::temp_directory_path()) / ("skepu_trace_paths_" + std::to_string(std::random_device{}()));
			fs::create_directories(root);
		}
		~Tree() { std::error_code ignored; fs::remove_all(root, ignored); }
		fs::path file(const std::string &relative)
		{
			fs::create_directories((root / relative).parent_path());
			std::ofstream(root / relative) << "x";
			return root / relative;
		}
	};

	std::string name(const std::map<std::string, std::string> &names, const std::string &spelling) { return names.at(spelling); }
}

TEST_CASE("one file is named by its file name")
{
	auto names = resolve({"/home/user/project/examples/horner.cpp"}, "/nowhere");
	REQUIRE(name(names, "/home/user/project/examples/horner.cpp") == "horner.cpp");
}

TEST_CASE("files are named from the directory that has them all")
{
	auto names = resolve({"/home/user/skepu/examples/horner.cpp", "/home/user/skepu/skepu-headers/src/skepu-lib/io.hpp"}, "/nowhere");
	REQUIRE(name(names, "/home/user/skepu/examples/horner.cpp") == "examples/horner.cpp");
	REQUIRE(name(names, "/home/user/skepu/skepu-headers/src/skepu-lib/io.hpp") == "skepu-headers/src/skepu-lib/io.hpp");
}

TEST_CASE("a directory that a file is in and the others are not in stays")
{
	auto names = resolve({"/a/b/x.cpp", "/a/b/lib/y.hpp"}, "/nowhere");
	REQUIRE(name(names, "/a/b/x.cpp") == "x.cpp");
	REQUIRE(name(names, "/a/b/lib/y.hpp") == "lib/y.hpp");
}

TEST_CASE("files with nothing in common but the root of the file system keep the rest of their paths")
{
	auto names = resolve({"/Users/a/x.cpp", "/home/b/y.cpp"}, "/nowhere");
	REQUIRE(name(names, "/Users/a/x.cpp") == "Users/a/x.cpp");
	REQUIRE(name(names, "/home/b/y.cpp") == "home/b/y.cpp");
}

TEST_CASE("an empty spelling has no name")
{
	auto names = resolve({"", "/a/b.cpp"}, "/nowhere");
	REQUIRE(names.count("") == 0);
	REQUIRE(name(names, "/a/b.cpp") == "b.cpp");
}

TEST_CASE("the dots of a path are taken out")
{
	auto names = resolve({"/a/./b/../b/x.cpp", "/a/b/lib/y.hpp"}, "/nowhere");
	REQUIRE(name(names, "/a/./b/../b/x.cpp") == "x.cpp");
}

TEST_CASE("a relative path is from the directory that the program runs in, and is the file of the absolute path")
{
	Tree tree;
	const fs::path source = tree.file("examples/horner.cpp"), header = tree.file("skepu-headers/io.hpp");
	auto names = resolve({"./horner.cpp", source.string(), ".//..//skepu-headers/io.hpp", header.string()}, tree.root / "examples");
	REQUIRE(name(names, "./horner.cpp") == name(names, source.string()));
	REQUIRE(name(names, ".//..//skepu-headers/io.hpp") == name(names, header.string()));
	REQUIRE(name(names, source.string()) == "examples/horner.cpp");
	REQUIRE(name(names, header.string()) == "skepu-headers/io.hpp");
}

TEST_CASE("a relative path from where the compiler ran is found from the directory of a known file")
{
	// the compiler ran in examples/; the program runs elsewhere, where the path leads nowhere
	Tree tree;
	const fs::path source = tree.file("examples/horner.cpp"), header = tree.file("skepu-headers/src/io.hpp");
	auto names = resolve({source.string(), ".//..//skepu-headers/src/io.hpp"}, "/somewhere/else");
	REQUIRE(name(names, source.string()) == "examples/horner.cpp");
	REQUIRE(name(names, ".//..//skepu-headers/src/io.hpp") == "skepu-headers/src/io.hpp");
	REQUIRE(fs::exists(header));
}

TEST_CASE("a relative path from where the compiler ran is the absolute path of the same file, when there is one")
{
	// the relative path leads nowhere here; the same file also has an absolute spelling
	auto names = resolve({".//..//skepu-headers/src/skepu-lib/io.hpp", "/home/u/skepu/skepu-headers/src/skepu-lib/io.hpp", "/home/u/skepu/examples/horner.cpp"}, "/somewhere/else");
	REQUIRE(name(names, ".//..//skepu-headers/src/skepu-lib/io.hpp") == name(names, "/home/u/skepu/skepu-headers/src/skepu-lib/io.hpp"));
	REQUIRE(name(names, "/home/u/skepu/skepu-headers/src/skepu-lib/io.hpp") == "skepu-headers/src/skepu-lib/io.hpp");
}

TEST_CASE("a relative path that two absolute paths end with is not guessed")
{
	auto names = resolve({"../x.hpp", "/a/p/x.hpp", "/a/q/x.hpp"}, "/somewhere/else/deep");
	REQUIRE(name(names, "../x.hpp") != name(names, "/a/p/x.hpp"));
	REQUIRE(name(names, "../x.hpp") != name(names, "/a/q/x.hpp"));
}

TEST_CASE("a relative path that leads nowhere is from the directory that the program runs in")
{
	auto names = resolve({"gone/x.cpp"}, "/work/run");
	REQUIRE(name(names, "gone/x.cpp") == "x.cpp");
}

TEST_CASE("two paths to a file through a link are one name")
{
	Tree tree;
	const fs::path file = tree.file("real/x.cpp");
	std::error_code error;
	fs::create_directory_symlink(tree.root / "real", tree.root / "alias", error);
	if (error)
		SUCCEED("no links here");
	else
	{
		auto names = resolve({file.string(), (tree.root / "alias" / "x.cpp").string()}, tree.root);
		REQUIRE(name(names, file.string()) == name(names, (tree.root / "alias" / "x.cpp").string()));
	}
}

TEST_CASE("a relative path is from the directory that the compiler ran in, when the precompiler said where")
{
	// one relative path leads to a file from the compiler's directory and to another from the program's: the compiler's wins
	Tree tree;
	const fs::path header = tree.file("proj/skepu-headers/io.hpp"), source = tree.file("proj/examples/horner.cpp"), other = tree.file("run/skepu-headers/io.hpp");
	auto names = resolve({source.string(), "../skepu-headers/io.hpp"}, tree.root / "run" / "deep", {(tree.root / "proj" / "examples").string()});
	REQUIRE(name(names, "../skepu-headers/io.hpp") == "skepu-headers/io.hpp");
	REQUIRE(name(names, source.string()) == "examples/horner.cpp");
	REQUIRE(fs::exists(header));
	REQUIRE(fs::exists(other));
}

TEST_CASE("the compiler's directories are noted once each")
{
	skepu::tracing::paths::compileDirectories().clear();
	skepu::tracing::paths::noteCompileDirectory("/a");
	skepu::tracing::paths::noteCompileDirectory("/a");
	skepu::tracing::paths::noteCompileDirectory("");
	skepu::tracing::paths::noteCompileDirectory("/b");
	REQUIRE(skepu::tracing::paths::compileDirectories() == std::vector<std::string>{"/a", "/b"});
}

TEST_CASE("the program may run on a system without the files: a relative path is still from the compiler's directory")
{
	auto names = resolve({"/lab/proj/examples/horner.cpp", ".//..//skepu-headers/src/skepu-lib/io.hpp"}, "/run/here", {"/lab/proj/examples"});
	REQUIRE(name(names, "/lab/proj/examples/horner.cpp") == "examples/horner.cpp");
	REQUIRE(name(names, ".//..//skepu-headers/src/skepu-lib/io.hpp") == "skepu-headers/src/skepu-lib/io.hpp");
}

TEST_CASE("a file is found where its spelling leads from the compiler's directory or the program's")
{
	Tree tree;
	const fs::path source = tree.file("proj/examples/horner.cpp"), header = tree.file("proj/skepu-headers/io.hpp");
	using skepu::tracing::paths::locate;
	const std::vector<std::string> compileDirs = {(tree.root / "proj" / "examples").string()};
	REQUIRE(locate(source.string(), "/nowhere", {}) == source);
	REQUIRE(locate(".//..//skepu-headers/io.hpp", "/nowhere", compileDirs) == header);
	REQUIRE(locate("skepu-headers/io.hpp", tree.root / "proj", {}) == header);
	REQUIRE(locate("gone.cpp", tree.root, compileDirs).empty());
	REQUIRE(locate("", tree.root, compileDirs).empty());
}
