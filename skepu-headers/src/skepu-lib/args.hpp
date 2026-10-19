#pragma once

// Command line arguments of a program: positional arguments, named options (--name value or --name=value) and flags (--name),
// and the backend of the program, which nearly every SkePU program takes.
//
//   skepu::args::Parser args(argc, argv, "Perlin noise textures");   // the description is printed by --help
//
//   auto height      = args.get<size_t>("height");                   // a positional argument, required
//   auto persistence = args.get<float>("persistence", 0.5f);        // with a default value: optional
//   auto [rows, cols] = args.get<size_t, size_t>("rows", "cols");   // several at once (structured bindings)
//
//   auto frames = args.named<size_t>("frames", 0, "the number of frames"); // --frames 6 or --frames=6
//   bool adam   = args.flag("adam", "use the Adam optimizer");              // --adam
//
//   auto spec = args.backend();   // the backend, as a positional argument or --backend NAME (-b NAME); sets it as the global backend
//
// The positional arguments come first on the command line, then the options and the flags. The value of a declaration is
// returned at once, and the declarations are made in the order of the positional arguments on the command line.
// A name that is declared again is the same argument, not a new one: the call gets the value that was read the first time (also after
// backend()), so a function can take the parser and read the arguments it needs, instead of getting them one by one. The same name
// with another type or default value is an error.
//
// The parser ends the program itself when the command line is wrong: a missing positional argument, a value that is not of
// the type (a number is not read as 0 when it is not a number) and, when the declarations are complete, an unknown option or an
// argument too many are errors that print the message and the usage and exit with status 1 (a call to skepu_mpi_abort under MPI).
// --help (-h) prints the usage, which is made from the declarations, and exits with status 0. (With SKEPU_ARGS_THROW defined the
// parser throws skepu::args::Exit instead of ending the program, with the text in what() and the status.)
//
// The declarations are complete when backend() or done() is called, or, as a last resort, when the parser is destroyed. So
// backend() should be the last of the declarations (or done() if the program has no backend). The messages are printed with
// skepu::io::cout, that is by the first process of an MPI program only.

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <typeindex>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <skepu>
#include <skepu-lib/io.hpp>

namespace skepu {
namespace args {

namespace detail {

// What the usage says about the type of a value
template<typename T>
constexpr const char *kind_of()
{
	if constexpr (std::is_same_v<T, bool>) return "boolean";
	else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) return "non-negative integer";
	else if constexpr (std::is_integral_v<T>) return "integer";
	else if constexpr (std::is_floating_point_v<T>) return "number";
	else return "text";
}

// Reads a whole value of type T from the text; false if the text is not one (an empty text, trailing characters, out of range)
template<typename T>
bool convert(const std::string &text, T &value)
{
	if constexpr (std::is_same_v<T, std::string>)
	{
		value = text;
		return true;
	}
	else if constexpr (std::is_same_v<T, bool>)
	{
		if (text == "1" || text == "true" || text == "yes" || text == "on") value = true;
		else if (text == "0" || text == "false" || text == "no" || text == "off") value = false;
		else return false;
		return true;
	}
	else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>)
	{
		if (text.empty()) return false;
		char *end = nullptr;
		errno = 0;
		const long long v = std::strtoll(text.c_str(), &end, 10);
		if (*end != '\0' || errno == ERANGE || v < (long long)std::numeric_limits<T>::min() || v > (long long)std::numeric_limits<T>::max())
			return false;
		value = (T)v;
		return true;
	}
	else if constexpr (std::is_integral_v<T>)
	{
		if (text.empty() || text[0] == '-') return false;
		char *end = nullptr;
		errno = 0;
		const unsigned long long v = std::strtoull(text.c_str(), &end, 10);
		if (*end != '\0' || errno == ERANGE || v > (unsigned long long)std::numeric_limits<T>::max())
			return false;
		value = (T)v;
		return true;
	}
	else
	{
		static_assert(std::is_floating_point_v<T>, "an argument is a number, a boolean or a string");
		if (text.empty()) return false;
		char *end = nullptr;
		errno = 0;
		const double v = std::strtod(text.c_str(), &end);
		if (*end != '\0' || errno == ERANGE) return false;
		value = (T)v;
		return true;
	}
}

template<typename T>
std::string default_text(const T &value)
{
	std::ostringstream os;
	if constexpr (std::is_same_v<T, std::string>)
		os << '"' << value << '"';
	else if constexpr (std::is_same_v<T, bool>)
		os << (value ? "true" : "false");
	else
		os << value;
	return os.str();
}

} // namespace detail


// What the parser throws instead of ending the program, if SKEPU_ARGS_THROW is defined (for tests of programs and of this header)
struct Exit : std::runtime_error
{
	Exit(int arg_status, const std::string &message) : std::runtime_error(message), status(arg_status) {}
	int status; // 0 for --help, 1 for an error
};


class Parser
{
public:
	Parser(int argc, char *argv[], std::string description = "")
	: m_description(std::move(description))
	{
		m_program = (argc > 0 && argv[0]) ? argv[0] : "program";
		if (const auto slash = m_program.find_last_of("/\\"); slash != std::string::npos)
			m_program = m_program.substr(slash + 1);

		// The positional arguments are those before the first option
		bool options = false;
		for (int i = 1; i < argc; ++i)
		{
			Token token;
			token.text = argv[i];
			token.is_option = is_option(token.text);
			if (token.is_option)
			{
				options = true;
				if (token.text == "--help" || token.text == "-h")
				{
					m_help = true;
					token.used = true;
				}
			}
			(options ? m_options : m_positionals).push_back(token);
		}
	}

	Parser(const Parser &) = delete;
	Parser &operator=(const Parser &) = delete;

	~Parser() noexcept(false)
	{
		if (!m_finished && std::uncaught_exceptions() == 0)
			done();
	}

	// A positional argument that is required
	template<typename T>
	T get(const std::string &name)
	{
		if (Entry *again = declared(Entry::Positional, name, typeid(T), ""))
			return stored<T>(*again, T{});
		Entry &entry = declare(Entry::Positional, name, typeid(T), detail::kind_of<T>(), "", "");
		T value{};
		if (m_help) return value;
		std::string text;
		if (!next_positional(text))
			fail("the argument <" + name + "> is missing");
		if (!detail::convert(text, value))
			fail("<" + name + "> is a " + detail::kind_of<T>() + ", not '" + text + "'");
		remember(entry, text);
		return value;
	}

	// A positional argument that is optional
	template<typename T>
	T get(const std::string &name, const T &default_value, const std::string &help = "")
	{
		if (Entry *again = declared(Entry::OptionalPositional, name, typeid(T), detail::default_text(default_value)))
			return stored<T>(*again, default_value);
		Entry &entry = declare(Entry::OptionalPositional, name, typeid(T), detail::kind_of<T>(), help, detail::default_text(default_value));
		T value = default_value;
		if (m_help) return value;
		std::string text;
		if (next_positional(text))
		{
			if (!detail::convert(text, value))
				fail("<" + name + "> is a " + detail::kind_of<T>() + ", not '" + text + "'");
			remember(entry, text);
		}
		return value;
	}

	// Several required positional arguments, in the order of the names: auto [rows, cols] = args.get<size_t, size_t>("rows", "cols");
	template<typename T, typename U, typename... Rest, typename... Names>
	std::tuple<T, U, Rest...> get(const std::string &first, const std::string &second, const Names &... names)
	{
		static_assert(sizeof...(Names) == sizeof...(Rest), "get needs one name for each type");
		return std::tuple<T, U, Rest...>{get<T>(first), get<U>(second), get<Rest>(names)...}; // (evaluated in this order)
	}

	// A named option: --name value or --name=value
	template<typename T>
	T named(const std::string &name, const T &default_value, const std::string &help = "")
	{
		if (Entry *again = declared(Entry::Named, name, typeid(T), detail::default_text(default_value)))
			return stored<T>(*again, default_value);
		Entry &entry = declare(Entry::Named, name, typeid(T), detail::kind_of<T>(), help, detail::default_text(default_value));
		T value = default_value;
		if (m_help) return value;
		bool given = false;
		const std::string text = option_value({name}, given);
		if (given)
		{
			if (!detail::convert(text, value))
				fail("--" + name + " is a " + detail::kind_of<T>() + ", not '" + text + "'");
			remember(entry, text);
		}
		return value;
	}

	// A flag: --name (true when it is on the command line)
	bool flag(const std::string &name, const std::string &help = "")
	{
		if (Entry *again = declared(Entry::Flag, name, typeid(bool), ""))
			return again->given;
		Entry &entry = declare(Entry::Flag, name, typeid(bool), "", help, "");
		if (m_help) return false;
		Token *token = find_option({name});
		if (!token) return false;
		token->used = true;
		const auto equals = token->text.find('=');
		bool value = true;
		if (equals != std::string::npos && !detail::convert(token->text.substr(equals + 1), value))
			fail("--" + name + " is a flag or a boolean, not '" + token->text.substr(equals + 1) + "'");
		entry.given = value;
		return value;
	}

	// The backend: the next positional argument, or --backend NAME (-b NAME). It is set as the global backend. (Without one
	// the default backend of the build is used.) The declarations are complete after it.
	skepu::BackendSpec backend()
	{
		if (m_backend_made) return m_backend;
		declare(Entry::OptionalPositional, "backend", typeid(skepu::BackendSpec), "", "cpu | openmp | opencl | cuda | hybrid, or --backend NAME (-b NAME); the default of the build if left out", "");
		skepu::BackendSpec spec;
		if (!m_help)
		{
			bool given = false;
			std::string name = option_value({"backend", "b"}, given);
			if (!given) given = next_positional(name);
			if (given)
			{
				std::string lower = name;
				for (auto &c : lower) c = (char)std::tolower((unsigned char)c);
				if (lower != "cpu" && lower != "openmp" && lower != "opencl" && lower != "cuda" && lower != "hybrid" && lower != "auto")
					fail("the backend must be cpu, openmp, opencl, cuda or hybrid, not '" + name + "'");
				spec = skepu::BackendSpec{lower};
			}
			skepu::setGlobalBackendSpec(spec);
		}
		done();
		m_backend = spec;
		m_backend_made = true;
		return spec;
	}

	// The declarations are complete: prints the usage and exits if --help was given, and stops with an error if there
	// is an unknown option or an argument too many. (backend() does this too.)
	void done()
	{
		if (m_finished) return;
		m_finished = true;
		if (m_help)
		{
			stop(0, usage(true));
		}
		for (const auto &token : m_options)
			if (!token.used)
				fail_with_usage(token.is_option ? "unknown option " + token.text.substr(0, token.text.find('=')) : "unexpected argument '" + token.text + "' (the options come after the positional arguments)");
		if (m_next_positional < m_positionals.size())
			fail_with_usage("unexpected argument '" + m_positionals[m_next_positional].text + "'");
	}

private:
	struct Token
	{
		std::string text;
		bool is_option = false;
		bool used = false;
	};

	struct Entry
	{
		enum Kind { Positional, OptionalPositional, Named, Flag } kind;
		std::string name, type, help, default_text;
		std::type_index type_id{typeid(void)};
		bool given = false;   // the value is on the command line (for a flag: it is on); then text is it
		std::string text;
	};

	// --name or -x (a letter), but not a negative number
	static bool is_option(const std::string &text)
	{
		if (text.size() > 2 && text[0] == '-' && text[1] == '-') return true;
		return text.size() >= 2 && text[0] == '-' && std::isalpha((unsigned char)text[1]);
	}

	static std::string option_name(const std::string &text)
	{
		const std::string body = text.substr(text[1] == '-' ? 2 : 1);
		return body.substr(0, body.find('='));
	}

	Entry &declare(Entry::Kind kind, const std::string &name, std::type_index type_id, const std::string &type, const std::string &help, const std::string &default_text)
	{
		if (m_finished)
			SKEPU_ERROR("skepu::args: the argument '" << name << "' is declared after the declarations were completed (backend() or done())");
		m_entries.push_back(Entry{kind, name, type, help, default_text, type_id});
		return m_entries.back();
	}

	// The entry of a name that was declared before, or nullptr. Another declaration of it is the same argument: it is an error if
	// it is another kind of argument, of another type, or has another default value.
	Entry *declared(Entry::Kind kind, const std::string &name, std::type_index type_id, const std::string &default_text)
	{
		auto family = [](Entry::Kind k) { return (k == Entry::Positional || k == Entry::OptionalPositional) ? 0 : (k == Entry::Named) ? 1 : 2; };
		for (auto &e : m_entries)
			if (e.name == name)
			{
				if (family(e.kind) != family(kind))
					SKEPU_ERROR("skepu::args: the argument '" << name << "' is declared again as another kind of argument (positional, named or flag)");
				if (e.type_id != type_id)
					SKEPU_ERROR("skepu::args: the argument '" << name << "' is declared again with another type");
				if (!e.default_text.empty() && !default_text.empty() && e.default_text != default_text)
					SKEPU_ERROR("skepu::args: the argument '" << name << "' is declared again with another default value (" << default_text << ", not " << e.default_text << ")");
				return &e;
			}
		return nullptr;
	}

	void remember(Entry &entry, const std::string &text)
	{
		entry.given = true;
		entry.text = text;
	}

	// The value of an argument that was declared before: what was read, or the default value
	template<typename T>
	static T stored(const Entry &entry, const T &default_value)
	{
		T value = default_value;
		if (entry.given) detail::convert(entry.text, value);
		return value;
	}

	bool next_positional(std::string &text)
	{
		if (m_next_positional >= m_positionals.size()) return false;
		text = m_positionals[m_next_positional++].text;
		return true;
	}

	Token *find_option(std::initializer_list<std::string> names)
	{
		for (auto &token : m_options)
			if (token.is_option && !token.used)
				for (const std::string &name : names)
					if (option_name(token.text) == name)
						return &token;
		return nullptr;
	}

	// The value of an option, --name value or --name=value; given tells if there is one
	std::string option_value(std::initializer_list<std::string> names, bool &given)
	{
		given = false;
		Token *token = find_option(names);
		if (!token) return "";
		token->used = true;
		given = true;
		const auto equals = token->text.find('=');
		if (equals != std::string::npos)
			return token->text.substr(equals + 1);
		Token *value = token + 1;
		if (value == m_options.data() + m_options.size() || value->is_option)
			fail(token->text + " needs a value");
		value->used = true;
		return value->text;
	}

	std::string usage(bool with_description) const
	{
		std::ostringstream os;
		if (with_description && !m_description.empty())
			os << m_description << "\n\n";
		os << "Usage: " << m_program;
		for (const auto &e : m_entries)
		{
			if (e.kind == Entry::Positional) os << " <" << e.name << ">";
			else if (e.kind == Entry::OptionalPositional) os << " [" << e.name << "]";
		}
		for (const auto &e : m_entries)
		{
			if (e.kind == Entry::Named) os << " [--" << e.name << " " << (e.type == "text" ? "TEXT" : "N") << "]";
			else if (e.kind == Entry::Flag) os << " [--" << e.name << "]";
		}
		os << " [--help]\n";
		auto describe = [&](const Entry &e, const std::string &label)
		{
			os << "  " << label << std::string(label.size() < 18 ? 18 - label.size() : 1, ' ');
			bool any = false;
			if (!e.help.empty()) { os << e.help; any = true; }
			if (!e.type.empty()) { os << (any ? " (" : "(") << e.type << (e.default_text.empty() ? "" : ", default " + e.default_text) << ")"; }
			else if (!e.default_text.empty()) os << " (default " << e.default_text << ")";
			os << "\n";
		};
		bool header = false;
		for (const auto &e : m_entries)
			if (e.kind == Entry::Positional || e.kind == Entry::OptionalPositional)
			{
				if (!header) { os << "\nArguments:\n"; header = true; }
				describe(e, e.name);
			}
		header = false;
		for (const auto &e : m_entries)
			if (e.kind == Entry::Named || e.kind == Entry::Flag)
			{
				if (!header) { os << "\nOptions:\n"; header = true; }
				describe(e, "--" + e.name);
			}
		return os.str();
	}

	// Prints the text and ends the program (or throws Exit)
	[[noreturn]] void stop(int status, const std::string &text)
	{
#ifdef SKEPU_ARGS_THROW
		throw Exit(status, text);
#else
		skepu::io::cout << text;
		if (status == 0) SKEPU_EXIT();
		SKEPU_ERROR_EXIT();
#endif
	}

	[[noreturn]] void fail(const std::string &message)
	{
		stop(1, "Error: " + message + "\nRun with --help for the usage.\n");
	}

	[[noreturn]] void fail_with_usage(const std::string &message)
	{
		stop(1, "Error: " + message + "\n" + usage(false));
	}

	std::string m_program, m_description;
	std::vector<Token> m_positionals, m_options;
	std::vector<Entry> m_entries;
	size_t m_next_positional = 0;
	bool m_help = false;
	bool m_finished = false;
	bool m_backend_made = false;
	skepu::BackendSpec m_backend;
};

} // namespace args
} // namespace skepu
