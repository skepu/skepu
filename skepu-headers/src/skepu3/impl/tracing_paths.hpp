// tracing_paths.hpp — one name per source file in a trace, with no more path than tells the files apart.
//
// Events name files three ways: the precompiler writes the path it was given, `std::source_location` and __builtin_FILE give the path the compiler was
// given (relative to where it ran), and headers come through include paths, so one header can have spellings like ".//..//skepu-headers/src/skepu-lib/io.hpp"
// and an absolute one. `resolve` gives each spelling one name and strips the directories all files share, so the trace does not expose the host's file system.
//
// Standard library only, so it is tested without a tracer.
#pragma once

#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <vector>

namespace skepu
{
	namespace tracing
	{
		namespace paths
		{
			namespace fs = std::filesystem;

			/**
			 /** Directories the compiler ran in, as the precompiler wrote them into the program (SKEPU_COMPILE_DIR); relative `__FILE__` and `std::source_location` paths are from there. Each translation unit notes its own at startup. */
			inline std::vector<std::string> &compileDirectories()
			{
				// never destroyed: the tracer may be made before a translation unit notes its directory and read the list when it is destroyed
				static std::vector<std::string> &directories = *new std::vector<std::string>();
				return directories;
			}

			inline bool noteCompileDirectory(const char *directory)
			{
				std::vector<std::string> &directories = compileDirectories();
				if (directory && *directory && std::find(directories.begin(), directories.end(), directory) == directories.end())
					directories.push_back(directory);
				return true;
			}

			namespace detail
			{
				inline std::vector<std::string> components(const fs::path &path)
				{
					std::vector<std::string> out;
					for (const fs::path &part : path)
						out.push_back(part.generic_string());
					return out;
				}

				inline bool endsWith(const std::vector<std::string> &path, const std::vector<std::string> &tail)
				{
					return !tail.empty() && tail.size() <= path.size() && std::equal(tail.rbegin(), tail.rend(), path.rbegin());
				}

				// leading "." and ".." dropped: what is left names the file from the directory the path started in
				inline std::vector<std::string> tailOf(const fs::path &relative)
				{
					std::vector<std::string> out = components(relative.lexically_normal());
					out.erase(out.begin(), std::find_if(out.begin(), out.end(), [](const std::string &c) { return c != "." && c != ".."; }));
					return out;
				}

				// with links resolved, so two paths to one file are one
				inline fs::path real(const fs::path &path)
				{
					std::error_code error;
					if (!fs::exists(path, error))
						return path;
					fs::path canonical = fs::weakly_canonical(path, error);
					return error ? path : canonical;
				}
			}

			/**
			 /** The file of a `spelling` on this system, or an empty path: absolute as is, relative from `compileDirs`, then from `base`. */
			inline fs::path locate(const std::string &spelling, const fs::path &base, const std::vector<std::string> &compileDirs)
			{
				if (spelling.empty())
					return {};
				const fs::path path = fs::path(spelling).lexically_normal();
				std::vector<fs::path> candidates;
				if (path.is_absolute())
					candidates.push_back(path);
				else
				{
					for (const std::string &dir : compileDirs)
						candidates.push_back((fs::path(dir) / path).lexically_normal());
					candidates.push_back((base / path).lexically_normal());
				}
				for (const fs::path &candidate : candidates)
					if (std::error_code error; fs::exists(candidate, error))
						return candidate;
				return {};
			}

			/**
			 /**
			  * The name a trace gives each of the `spellings` of a path, as a map from the spelling.
			  *
			  * A path becomes absolute and normalized ("./", ".//..//", "dir/../" removed), with links of an existing file resolved. A relative path is from where the
			  * compiler ran (`compileDirs`, written into the program). In order:
			  *   - from a directory of `compileDirs` that has the file, or from `base` (the directory the program runs in) if it has it;
			  *   - from the first of `compileDirs`, by text: the program may run on another system than the compiler, which lacks the file;
			  *   - without `compileDirs`: from the innermost directory above a known file that has it, else the absolute path of the one other spelling with the same
			  *     trailing components, else from `base` as it is.
			  * The name is that path from the innermost directory that has all the files (the file's own directory for one file). An empty spelling has no name.
			 */
			inline std::map<std::string, std::string> resolve(const std::set<std::string> &spellings, const fs::path &base, const std::vector<std::string> &compileDirs = {})
			{
				std::map<std::string, fs::path> absolute;
				std::vector<std::string> unresolved;
				for (const std::string &spelling : spellings)
				{
					if (spelling.empty())
						continue;
					const fs::path path = fs::path(spelling).lexically_normal();
					if (path.is_absolute())
					{
						absolute[spelling] = detail::real(path);
						continue;
					}
					fs::path found;
					for (const std::string &dir : compileDirs)
						if (std::error_code error; fs::exists(fs::path(dir) / path, error)) { found = detail::real((fs::path(dir) / path).lexically_normal()); break; }
					if (found.empty())
						if (std::error_code error; fs::exists(base / path, error))
							found = detail::real((base / path).lexically_normal());
					if (found.empty())
						unresolved.push_back(spelling);
					else
						absolute[spelling] = found;
				}

				std::set<fs::path> known;
				for (const auto &entry : absolute)
					known.insert(entry.second);
				// directories above the known files, innermost first: the compiler may have run in one
				std::vector<fs::path> compilerDirs;
				for (const fs::path &file : known)
					for (fs::path dir = file.parent_path(); !dir.empty() && dir != dir.root_path(); dir = dir.parent_path())
						if (std::find(compilerDirs.begin(), compilerDirs.end(), dir) == compilerDirs.end())
							compilerDirs.push_back(dir);
				std::stable_sort(compilerDirs.begin(), compilerDirs.end(), [](const fs::path &a, const fs::path &b) {
					return std::distance(a.begin(), a.end()) > std::distance(b.begin(), b.end()); });

				for (const std::string &spelling : unresolved)
				{
					const fs::path path = fs::path(spelling).lexically_normal();
					fs::path found;
					if (compileDirs.empty())
						for (const fs::path &dir : compilerDirs)
							if (std::error_code error; fs::exists(dir / path, error)) { found = detail::real((dir / path).lexically_normal()); break; }
					// the precompiler recorded where the compiler ran: the path is from there even if the file is not here (the program may run elsewhere)
					if (found.empty() && !compileDirs.empty())
						found = (fs::path(compileDirs.front()) / path).lexically_normal();
					if (found.empty())
					{
						const std::vector<std::string> tail = detail::tailOf(spelling);
						std::set<fs::path> matches;
						for (const fs::path &candidate : known)
							if (detail::endsWith(detail::components(candidate), tail))
								matches.insert(candidate);
						found = matches.size() == 1 ? *matches.begin() : (base / path).lexically_normal();
					}
					absolute[spelling] = found;
				}

				// directories that contain all the files
				std::vector<std::string> root;
				bool first = true;
				for (const auto &entry : absolute)
				{
					const std::vector<std::string> directory = detail::components(entry.second.parent_path());
					if (first)
						root = directory;
					else
					{
						size_t same = 0;
						while (same < root.size() && same < directory.size() && root[same] == directory[same])
							++same;
						root.resize(same);
					}
					first = false;
				}
				fs::path rootPath;
				for (const std::string &part : root)
					rootPath /= part;

				std::map<std::string, std::string> names;
				for (const auto &entry : absolute)
				{
					const fs::path relative = entry.second.lexically_relative(rootPath);
					names[entry.first] = relative.empty() ? entry.second.generic_string() : relative.generic_string();
				}
				return names;
			}
		}
	}
}
