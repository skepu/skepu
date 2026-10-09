#pragma once

// Tracing in the MPI backend (a pilot: container allocation, Map, communication). Every process has a tracer, as in a
// single node program, and writes a trace file of its own, but the files are not alike:
//
//  * The first process (rank 0) writes the trace of the program: the events of a single node program (allocations, skeleton
//    calls, ...) with the ids of the program's containers, a number for each logical event that the other processes agree on
//    ("seq"), and the communication that the program does. A call has the interval of this process and what it did (local
//    elements, the time it worked, the communication it made). It can be read as any other trace.
//  * The other processes write lean files: for each skeleton call and each communication the same "seq", the interval, and a few
//    numbers, with no ids. (Events of the same seq in the files are the same call or the same collective: the program is
//    replicated, so the processes make the same calls in the same order.) The events of the node-level library are not recorded.
//
// The number of events is the number of processes times a constant, for every call, and does not depend on the size of
// the problem. The times of the files are those of the clock of each process: the header of a file has what is needed to
// put them on the clock of rank 0 (offset and uncertainty, measured at the start and at the end).

#include <mpi.h>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "skepu3/impl/tracing.hpp"

#ifdef SKEPU_TRACING
#include <unistd.h>
#endif

namespace skepu
{
	namespace mpi
	{
		namespace trace
		{
		SKEPU_TRACING_VARIANT_BEGIN

#ifdef SKEPU_TRACING
			namespace tr = ::skepu_node::tracing;
			using json = nlohmann::json;

			struct ClockSample
			{
				long long local_us = 0;       // the time of the process when it was measured
				double offset_us = 0;         // what to add to a time of this process to get the time of rank 0
				double uncertainty_us = 0;    // half of the shortest round trip
			};

			struct CallScope;

			struct State
			{
				int rank = 0, size = 1;
				bool active = false;
				unsigned long long seq = 0, comm_seq = 0;
				ClockSample start_sample, end_sample;
				CallScope *current = nullptr;
			};

			inline State &state()
			{
				static State s;
				return s;
			}

			inline long long now_us() { return (long long)tr::timestamp_now(); }

			// The clock of this process against that of rank 0, from the shortest of some round trips (every process takes part)
			inline ClockSample sync_clock(MPI_Comm comm, int rank, int size)
			{
				ClockSample sample;
				MPI_Barrier(comm);
				if (size > 1)
				{
					const int rounds = 16, tag = 7771;
					if (rank == 0)
					{
						for (int peer = 1; peer < size; ++peer)
						{
							double best_rtt = 1e300, best_offset = 0;
							for (int r = 0; r < rounds; ++r)
							{
								long long t0 = now_us(), tk = 0;
								MPI_Send(&t0, 1, MPI_LONG_LONG, peer, tag, comm);
								MPI_Recv(&tk, 1, MPI_LONG_LONG, peer, tag, comm, MPI_STATUS_IGNORE);
								long long t1 = now_us();
								const double rtt = (double)(t1 - t0);
								if (rtt < best_rtt)
								{
									best_rtt = rtt;
									best_offset = ((double)t0 + (double)t1) / 2.0 - (double)tk;
								}
							}
							double result[2] = {best_offset, best_rtt / 2.0};
							MPI_Send(result, 2, MPI_DOUBLE, peer, tag + 1, comm);
						}
					}
					else
					{
						for (int r = 0; r < rounds; ++r)
						{
							long long t0 = 0;
							MPI_Recv(&t0, 1, MPI_LONG_LONG, 0, tag, comm, MPI_STATUS_IGNORE);
							long long tk = now_us();
							MPI_Send(&tk, 1, MPI_LONG_LONG, 0, tag, comm);
						}
						double result[2] = {0, 0};
						MPI_Recv(result, 2, MPI_DOUBLE, 0, tag + 1, comm, MPI_STATUS_IGNORE);
						sample.offset_us = result[0];
						sample.uncertainty_us = result[1];
					}
				}
				sample.local_us = now_us();
				return sample;
			}

			inline json clock_json(const ClockSample &s)
			{
				return json{{"local_us", s.local_us}, {"offset_us", s.offset_us}, {"uncertainty_us", s.uncertainty_us}};
			}

			// The file name of a process: %r is the rank; without it the first process keeps the name and the others have
			// ".rank<k>" before the extension
			inline std::string file_name_of(std::string name, int rank)
			{
				const size_t at = name.find("%r");
				if (at != std::string::npos)
				{
					for (size_t p = name.find("%r"); p != std::string::npos; p = name.find("%r", p))
						name.replace(p, 2, std::to_string(rank));
					return name;
				}
				if (rank == 0)
					return name;
				const size_t dot = name.rfind('.');
				const std::string suffix = ".rank" + std::to_string(rank);
				return dot == std::string::npos ? name + suffix : name.substr(0, dot) + suffix + name.substr(dot);
			}

			// At the start of the program (every process): what the tracer records, the file and the header
			inline void init(MPI_Comm comm, int rank, int size)
			{
				State &s = state();
				s.rank = rank;
				s.size = size;
				s.active = true;

				// (a test of the clock alignment: the clock of a process starts at the first time that it is read, and this starts it later on
				// the processes with a higher rank)
				if (const char *skew = std::getenv("SKEPU_TRACE_TEST_SKEW_US"))
					usleep((useconds_t)((long long)rank * std::atoll(skew)));
				now_us();

				tr::Tracer &t = tr::tracer();
				t.configureMPI(rank != 0);
				t.setFileName(file_name_of(t.fileName(), rank));

				char host[256] = {0};
				gethostname(host, sizeof host - 1);
				t.header()["mpi"] = json{{"rank", rank}, {"ranks", size}, {"detail_rank", 0}, {"role", rank == 0 ? "full" : "lean"},
					{"host", std::string(host)}, {"pid", (long long)getpid()}};
				s.start_sample = sync_clock(comm, rank, size);
				t.header()["clock"] = json{{"method", "ping-pong"}, {"start", clock_json(s.start_sample)}};
			}

			// Before MPI ends (every process): the clock again, and the file
			inline void finish(MPI_Comm comm)
			{
				State &s = state();
				if (!s.active)
					return;
				s.end_sample = sync_clock(comm, s.rank, s.size);
				tr::Tracer &t = tr::tracer();
				t.header()["clock"]["end"] = clock_json(s.end_sample);
				t.flush();
				s.active = false;
			}

			// (the events of a process that stops: the file is written, without the clock at the end)
			inline void abort_flush()
			{
				State &s = state();
				if (s.active)
					tr::tracer().flush();
			}

			// -------------------- The events --------------------

			struct CommTotals
			{
				unsigned long long bytes_sent = 0, bytes_received = 0, count = 0;
				long long time_us = 0;
			};

			inline json totals_json(const CommTotals &c)
			{
				return json{{"bytes_sent", c.bytes_sent}, {"bytes_received", c.bytes_received}, {"time_us", c.time_us}, {"collectives", c.count}};
			}

			// The container's allocation and deallocation (the first process records them, as a single node program does)
			inline void allocation(int id, const std::string &label, bool allocate)
			{
				if (!state().active || state().rank != 0)
					return;
				tr::tracer().raw(json{{"object_id", id}, {"type", allocate ? "allocation" : "deallocation"}, {"label", label},
					{"time", now_us()}, {"line", 0}});
			}

			// A skeleton call, around everything that it does on this process
			struct CallScope
			{
				const char *pattern;
				unsigned long long seq = 0;
				tr::EventHandle handle;
				long long work_start = 0, work_end = 0;
				bool work_done = false; // (the node-level skeleton has run, or would have on a process that has no elements)
				CommTotals comm;
				CallScope *previous = nullptr;

				explicit CallScope(const char *pattern_): pattern(pattern_), handle(tr::tracer().startEvent())
				{
					State &s = state();
					this->seq = ++s.seq;
					this->previous = s.current;
					s.current = this;
				}

				~CallScope() { state().current = this->previous; }

				void work_begin() { this->work_start = now_us(); }
				void work_finish() { this->work_end = now_us(); }
				// What the call does after its work (the results go back to their owners) is "after" in the events of the collectives
				void work_finished() { this->work_done = true; }

				// The event of an skepu::external call: the first process (which runs the code) records it as a single node program does, with
				// the number of the call and what the processes did; the others record that they took part and for how long.
				// 'outputs' are the containers that the code writes, 'inputs' those that it reads.
				void finish_external(const std::string &label, const char *file, int line, std::vector<int> &&outputs, std::vector<int> &&inputs)
				{
					State &s = state();
					if (!s.active)
						return;
					const unsigned long long signature = (unsigned long long)std::hash<std::string>{}(std::string(this->pattern) + "|" + label);
					const long long end = now_us();
					json local = json{{"rank", s.rank}, {"ranks", s.size}, {"local_elements", 0},
						{"work_start", this->work_start}, {"work_end", this->work_end}, {"comm", totals_json(this->comm)}, {"signature", signature}};
					if (s.rank == 0)
					{
						json entry = json{{"type", "external"}, {"label", label}, {"start", this->handle.m_time}, {"end", end},
							{"elwise_inputs", std::vector<int>{}}, {"proxy_inputs", inputs}, {"scalar_inputs", std::vector<int>{}}, {"outputs", outputs},
							{"file", file}, {"line", line}, {"seq", this->seq}, {"mpi", local}};
						tr::tracer().raw(entry);
					}
					else
					{
						json entry = json{{"type", "rank_call"}, {"seq", this->seq}, {"pattern", this->pattern}, {"start", this->handle.m_time}, {"end", end}};
						for (auto it = local.begin(); it != local.end(); ++it)
							entry[it.key()] = it.value();
						tr::tracer().raw(entry);
					}
				}

				// The event; 'skeleton' is the node-level skeleton that did the work of this process
				template<typename Skeleton>
				void finish(Skeleton *skeleton, size_t global_elements, size_t local_elements,
					std::vector<int> &&outputs, std::vector<int> &&elwise, std::vector<int> &&proxies, std::vector<int> &&scalars, int prng)
				{
					this->finish(skeleton, std::vector<size_t>{global_elements}, local_elements, std::move(outputs), std::move(elwise),
						std::move(proxies), std::move(scalars), prng);
				}

				// ('elements' are the sizes of the call as the skeleton has them: {n}, or {rows, columns})
				template<typename Skeleton>
				void finish(Skeleton *skeleton, std::vector<size_t> &&elements, size_t local_elements,
					std::vector<int> &&outputs, std::vector<int> &&elwise, std::vector<int> &&proxies, std::vector<int> &&scalars, int prng)
				{
					State &s = state();
					if (!s.active)
						return;
					auto *base = static_cast<::skepu_node::backend::SkeletonBase *>(skeleton);
					// (what the processes must agree on for a call: its pattern and the label of the skeleton. Not the "hash" of an event, which is the hash of a source file.)
					const unsigned long long signature = (unsigned long long)std::hash<std::string>{}(std::string(this->pattern) + "|" + base->getLabel());
					const long long end = now_us();
					json local = json{{"rank", s.rank}, {"ranks", s.size}, {"local_elements", local_elements},
						{"work_start", this->work_start}, {"work_end", this->work_end}, {"comm", totals_json(this->comm)}, {"signature", signature}};
					if (s.rank == 0)
					{
						tr::Parameters extra;
						extra.values["seq"] = this->seq;
						extra.values["mpi"] = local;
						tr::AllowScope allow;
						tr::tracer().call(this->handle, this->pattern, base, std::move(elements), std::move(outputs), std::move(elwise), std::move(proxies),
							std::move(scalars), prng, &extra);
					}
					else
					{
						json entry = json{{"type", "rank_call"}, {"seq", this->seq}, {"pattern", this->pattern}, {"start", this->handle.m_time},
							{"end", end}};
						for (auto it = local.begin(); it != local.end(); ++it)
							entry[it.key()] = it.value();
						tr::tracer().raw(entry);
					}
				}
			};

			// The work of this process in the call that is going on (what the node-level skeleton does), timed
			template<typename Function>
			void work(Function &&function)
			{
				CallScope *scope = state().current;
				if (scope)
					scope->work_begin();
				function();
				if (scope)
					scope->work_finish();
			}

			// A collective operation: the interval of this process in it, and what it moved. 'reason' says why the program makes it.
			// (Bytes are the payload that goes to other processes and comes from them, not the copy within a process.)
			class CommScope
			{
			public:
				CommScope(const char *operation, const char *reason, std::vector<int> object_ids = {})
				: m_operation(operation), m_reason(reason), m_ids(std::move(object_ids)), m_start(now_us())
				{
					this->m_comm_seq = ++state().comm_seq;
				}

				void bytes(unsigned long long sent, unsigned long long received)
				{
					this->m_sent = sent;
					this->m_received = received;
				}

				~CommScope()
				{
					State &s = state();
					if (!s.active)
						return;
					const long long end = now_us();
					const unsigned long long call = s.current ? s.current->seq : 0;
					if (s.current)
					{
						s.current->comm.bytes_sent += this->m_sent;
						s.current->comm.bytes_received += this->m_received;
						s.current->comm.time_us += end - this->m_start;
						s.current->comm.count += 1;
					}
					json entry = json{{"comm_seq", this->m_comm_seq}, {"seq", call}, {"operation", this->m_operation}, {"reason", this->m_reason},
						{"start", this->m_start}, {"end", end}, {"bytes_sent", this->m_sent}, {"bytes_received", this->m_received}};
					if (s.current)
						entry["phase"] = s.current->work_done ? "after" : "before"; // (of the call it is in)
					if (s.rank == 0)
					{
						entry["type"] = "communication";
						entry["object_ids"] = this->m_ids;
						entry["ranks"] = s.size;
					}
					else
					{
						entry["type"] = "rank_communication";
						entry["rank"] = s.rank;
					}
					tr::tracer().raw(entry);
				}

			private:
				const char *m_operation, *m_reason;
				std::vector<int> m_ids;
				long long m_start;
				unsigned long long m_comm_seq = 0, m_sent = 0, m_received = 0;
			};

			// The ids of the scalars among the uniform arguments of a call
			template<typename... Args>
			std::vector<int> scalar_ids(Args &&... args)
			{
				std::vector<int> ids;
				(tr::scalar_labels_helper(ids, args), ...);
				return ids;
			}

#else  // !SKEPU_TRACING

			inline void init(MPI_Comm, int, int) {}
			inline void finish(MPI_Comm) {}
			inline void abort_flush() {}
			inline void allocation(int, const std::string &, bool) {}

			template<typename Function>
			void work(Function &&function) { function(); }

			struct CallScope
			{
				explicit CallScope(const char *) {}
				void work_begin() {}
				void work_finish() {}
				void work_finished() {}
				template<typename... Args> void finish(Args &&...) {}
				void finish_external(const std::string &, const char *, int, std::vector<int> &&, std::vector<int> &&) {}
			};

			class CommScope
			{
			public:
				CommScope(const char *, const char *, std::vector<int> = {}) {}
				void bytes(unsigned long long, unsigned long long) {}
			};

			template<typename... Args>
			std::vector<int> scalar_ids(Args &&...) { return {}; }
#endif

		SKEPU_TRACING_VARIANT_END
		}
	}
}
