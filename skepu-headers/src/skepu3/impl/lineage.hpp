#pragma once

#include "skepu3/impl/common.hpp"

#include <unordered_set>
#include <unordered_map>
#include <list>

namespace skepu
{
	namespace LazyEvaluation
	{
		using ContainerTag = void*;
		
		
		template<typename First, typename... Rest>
		void registerInputsAndOutputs(
			std::vector<ContainerTag> &inputs, std::vector<ContainerTag> &outputs,
			First &&first, Rest&&... rest)
		{
			if (std::is_const<First>::value)
				inputs.push_back((ContainerTag)&first);
			else
				outputs.push_back((ContainerTag)&first);
			
			
			registerInputsAndOutputs(inputs, outputs, std::forward<Rest>(rest)...);
		}
		
		
		template<typename... Args>
		std::tuple<std::vector<ContainerTag>, std::vector<ContainerTag>>
		registerInputsAndOutputs(Args&&... args)
		{
			std::vector<ContainerTag> inputs, outputs;
			registerInputsAndOutputs(inputs, outputs, std::forward<Args>(args)...);
		}
		
		
		
		
		
		using WorkThunk = std::function<void(size_t, size_t)>;
		
		struct Invocation
		{
#ifdef SKEPU_PRECOMPILED
			using InvSkeletonBase = backend::SkeletonBase;
#else
			using InvSkeletonBase = SeqSkeletonBase;
#endif
			
			WorkThunk work;
			InvSkeletonBase *skeleton;
			std::vector<ContainerTag> inputs, outputs;
			size_t size;
			unsigned int time;
			bool evaluated = false;
			int overlap;
			
			Invocation(InvSkeletonBase *skeletonarg, size_t sizearg, int overlap, WorkThunk workarg)
			: work(workarg), size(sizearg), skeleton(skeletonarg), overlap(overlap), time(globalTime++)
			{}
			
			// end is the index after the last processed element
			void evalRange(size_t start, size_t end)
			{
			//	std::cout << "Evaluating invocation at time " << this->time  << " in the interval [" << start << ", " << end << "]\n";
				work(start, end);
			}
			
			void eval()
			{
				if (!this->evaluated)
				{
					this->evalRange(0, this->size);
					this->evaluated = true;
				}
			}
			
			void registerInputs() {}
			void registerOutputs() {}
			
			template<typename T, typename... Ts>
			void registerInputs(T&& first, Ts&&... rest)
			{
				this->inputs.push_back((ContainerTag)&first);
				this->registerInputs(std::forward<Ts>(rest)...);
			}
			
			template<typename T, typename... Ts>
			void registerOutputs(T&& first, Ts&&... rest)
			{
				this->outputs.push_back((ContainerTag)&first);
				this->registerInputs(std::forward<Ts>(rest)...);
			}
			
			// Happened-before
			bool operator<(Invocation const& other) const
			{
				return this->time < other.time;
			}
			
			std::string description() const
			{
				std::stringstream o;
				o << "Invocation(at: " << this->time << ", of: " << debug.getObjectName(this->skeleton) << ", ";
				o << "in: [ ";
				for (void *input : this->inputs)
					o << debug.getObjectName(input) << " ";
				o << "], out: [ ";
				for (void *output : this->outputs)
					o << debug.getObjectName(output) << " ";
				o << "])";
				o << " overlap=" << this->overlap;
				return o.str();
			}
			
			
			static unsigned int globalTime;
		};
		
		struct Node;
		
		enum class DependencyType
		{
			RAW, WAR, WAW
		};
		
		struct Node
		{
			Invocation invocation;
			
			// All nodes which have outputs which are inputs to this node
			std::set<std::pair<Node*, std::set<DependencyType>*>> predecessors;
			
			void addPredecessor(Node *node, ContainerTag container, DependencyType type)
			{
			//	std::cout << this->invocation.description() << " found predecessor " << node->invocation.description()
			//		<< " w.r.t. " << debug.getObjectName(myOutput) << "\n";
				std::set<std::pair<Node*, std::set<DependencyType>*>>::iterator it
					= std::find_if(this->predecessors.begin(), this->predecessors.end(),
					[=](std::pair<Node*, std::set<DependencyType>*> const& pair) { return pair.first == node; });
				
				if (it == this->predecessors.end())
					this->predecessors.emplace(node, new std::set<DependencyType>{type});
				else
					(*it).second->insert(type);
			}
			
			Node(Invocation &&inv, std::unordered_set<Node*> const &allNodes,
				                    std::unordered_map<ContainerTag, Node*> &startNodesForContainer)
			: invocation(inv)
			{
				// Find true dependencies (Read-after-Write)
				for (void *myInput : this->invocation.inputs)
				{
					if (startNodesForContainer.find(myInput) != startNodesForContainer.end())
					{
						Node *node = startNodesForContainer[myInput];
						for (void *theirOutput : node->invocation.outputs)
							if (myInput == theirOutput)
								this->addPredecessor(node, myInput, DependencyType::RAW);
					}
				}
					
				for (Node *node : allNodes)
				{
					// Find anti-dependencies
					for (void *myOutput : this->invocation.outputs)
					{
						// Write-after-Write
						for (void *theirOutput : node->invocation.outputs)
							if (myOutput == theirOutput)
								this->addPredecessor(node, myOutput, DependencyType::WAW);
						
						// Write-after-Read
						for (void *theirInput : node->invocation.inputs)
							if (myOutput == theirInput)
								this->addPredecessor(node, myOutput, DependencyType::WAR);
					}
				}
			}
			
			~Node()
			{
				for (auto pair : this->predecessors)
					delete pair.second;
			}
			
			
			bool hasTransitivePredecessor(Node *target) const
			{
				for (auto predecessor : this->predecessors)
				{
					for (auto grandPredecessor : predecessor.first->predecessors)
						if (grandPredecessor.first == target)
							return true;
					if (predecessor.first->hasTransitivePredecessor(target))
						return true;
				}
				return false;
			}
			
			bool hasTransitivePredecessor(Node *target, DependencyType type) const
			{
				for (auto predecessor : this->predecessors)
				{
					for (auto grandPredecessor : predecessor.first->predecessors)
						if (grandPredecessor.first == target && grandPredecessor.second->find(type) != grandPredecessor.second->end())
							return true;
					if (predecessor.first->hasTransitivePredecessor(target, type))
						return true;
				}
				return false;
			}
			
			std::string graphLabel() const
			{
				std::stringstream ss;
				ss << "\"";
				ss << this->invocation.time << " " << debug.getObjectName(this->invocation.skeleton) << "\\n";
				ss << "in: ";
				for (void *input : this->invocation.inputs)
					ss << debug.getObjectName(input) << " ";
				ss << "\\nout: ";
				for (void *output : this->invocation.outputs)
					ss << debug.getObjectName(output) << " ";
				ss << "\"";
				return ss.str();
			}
		};
		
		
		
		class Lineage
		{
			class Visitor
			{
			public:
				
				enum class VisitOrder
				{
					Pre, Post
				};
				
				Visitor(Lineage const& lineagearg, VisitOrder orderarg, std::function<void(Node*, int, bool, int, int)> funcarg, bool revisitarg = false)
				: lineage(lineagearg), func(funcarg), order(orderarg), revisit(revisitarg)
				{}
			
			private:
				
				void visitHelper(Node *node, std::set<Node*> &visited, size_t tile, bool last_tile, int depth) const
				{
					if (order == VisitOrder::Post)
						for (auto predecessor : node->predecessors)
							visitHelper(predecessor.first, visited, tile, last_tile, depth+1);
					
					if (revisit || visited.find(node) == visited.end())
					{
						cum_indent += node->invocation.overlap;
						func(node, tile, last_tile, depth, cum_indent);
						visited.insert(node);
					}
					
					if (order == VisitOrder::Pre)
						for (auto predecessor : node->predecessors)
							visitHelper(predecessor.first, visited, tile, last_tile, depth+1);
				}
				
			public:
				
				void visitNode(Node *node, size_t tile = 0, bool last_tile = false) const
				{
					cum_indent = 0;
					std::set<Node*> visited;
					visitHelper(node, visited, tile, last_tile, 0);
				}
				
				void visitAll() const
				{
					std::set<Node*> visited;
					for (Node* node : lineage.allNodes)
					{
						if (visited.find(node) == visited.end())
							visitHelper(node, visited, 0, false, 0);
					}
				}
				
			private:
				Lineage const& lineage;
				std::function<void(Node*, int, bool, int, int)> func;
				VisitOrder order;
				bool revisit;
			//	mutable std::list<int> overlaps, indents;
				mutable int cum_indent;
			};
			
		public:
			
			void eliminateTransitiveDependencies()
			{
				for (Node *node : allNodes)
				{
					std::set<std::pair<Node*, std::set<DependencyType>*>> toRemove;
					for (auto candidate : node->predecessors)
					{
						std::set<DependencyType> toRemove2;
						for (DependencyType type : *candidate.second)
						{
							if (node->hasTransitivePredecessor(candidate.first, type))
								toRemove2.insert(type);
						}
						
						for (auto removed : toRemove2)
							candidate.second->erase(candidate.second->find(removed));
						
						if (candidate.second->size() == 0)
							toRemove.insert(candidate);
					}
					
					for (auto removed : toRemove)
					{
						delete removed.second;
						node->predecessors.erase(node->predecessors.find(removed));
					}
				}
			}
			
			void eliminateAllTransitiveDependencies()
			{
				for (Node *node : allNodes)
				{
					std::set<std::pair<Node*, std::set<DependencyType>*>> toRemove;
					for (auto candidate : node->predecessors)
					{
						if (node->hasTransitivePredecessor(candidate.first))
							toRemove.insert(candidate);
					}
					
					for (auto removed : toRemove)
					{
						delete removed.second;
						node->predecessors.erase(node->predecessors.find(removed));
					}
				}
			}
			
			void evaluateRange(ContainerTag container, size_t tile, bool last_tile, size_t start, size_t end)
			{
				Visitor visitor(*this, Visitor::VisitOrder::Post, [=](Node *node, int tile, bool last_tile, int depth, int indent)
				{
				//	std::cout << "depth: " << depth << ", tile: " << tile << ", overlap: " << node->invocation.overlap << "my indent: " << indent << std::endl;
				//	if (last_tile) std::cout << "This is the last tile" << "\n";
					size_t n_start = last_tile ? start - indent : ((tile == 0) ? 0 : start - indent);
					size_t n_end = last_tile ? end : end - indent;
					
					node->invocation.evalRange(n_start, n_end);
					
				});
				visitor.visitNode(startNodesForContainer[container], tile, last_tile);
			}
			
			void evaluate(ContainerTag container)
			{
				Visitor visitor(*this, Visitor::VisitOrder::Post, [](Node *node, int tile, bool last_tile, int depth, int) { node->invocation.eval(); });
				visitor.visitNode(startNodesForContainer[container]);
			}
			
			template<typename T> // T is a SkePU container
			void evaluateRange(T &container, size_t tile, bool last_tile, size_t start, size_t end)
			{
				this->evaluateRange(&container, tile, last_tile, start, end);
			}
			
			template<typename T> // T is a SkePU container
			void evaluateCacheAware(T &container, size_t chunksize = 8)
			{
				size_t size = startNodesForContainer[&container]->invocation.size;
				for (size_t start = 0; start < size; start += chunksize)
				{
					size_t end = std::min(size, start + chunksize);
					this->evaluateRange(&container, start / chunksize, false, start, end);
				}
				this->evaluateRange(&container, 0, true, size, size);
			}
			
			template<typename T> // T is a SkePU container
			void evaluate(T &container)
			{
				this->evaluate(&container);
			}
			
			
			void evaluate()
			{
				for (std::pair<ContainerTag, Node*> pair : startNodesForContainer)
					this->evaluate(pair.first);
			}
			
			void clearState()
			{
				for (Node *node : allNodes)
					delete node;
					
				allNodes.clear();
				startNodesForContainer.clear();
				Invocation::globalTime = 0;
			}
			
			void printLineage() const
			{
				Visitor visitor(*this, Visitor::VisitOrder::Pre, [](Node *node, int tile, bool last_tile, int depth, int indent)
				{
					for (int i = 0; i <= depth; i++)
						std::cout << "\t";
					std::cout << node->invocation.description() << " (indent=" << indent << "\n";
				}, true);
				
				for (std::pair<ContainerTag, Node*> pair : startNodesForContainer)
				{
					std::cout << debug.getObjectName(pair.first) << ":\n";
					visitor.visitNode(pair.second);
				}
			}
			
			
			void renderGraph(std::string fileName, bool containers = false) const
			{
				std::ofstream graphFile{fileName + ".gv"};
				
				graphFile << "digraph {\n";
				
				static std::map<DependencyType, std::string> colorForDependencyType =
				{
					{DependencyType::RAW, "black"},
					{DependencyType::WAR, "red"},
					{DependencyType::WAW, "blue"},
				};
				
				graphFile << "// Invocation Nodes:\n";
				for (Node* node : allNodes)
				{
					std::string attrs = "";
					if (node->invocation.evaluated)
						attrs = attrs + ", style=filled";
					graphFile << node->invocation.time << " [label=" << node->graphLabel() << attrs << "];\n\n";
				}
				
				graphFile << "\n// Edges:\n";
				Visitor visitor(*this, Visitor::VisitOrder::Pre,
				[&](Node *node, int tile, bool last_tile, int depth, int)
				{
					for (auto pre : node->predecessors)
					{
						for (DependencyType type : *pre.second)
							graphFile << pre.first->invocation.time << " -> " << node->invocation.time
							          << "[color=" << colorForDependencyType[type] << "];\n";
					}
				});
				visitor.visitAll();
				
				if (containers)
				{
					for (std::pair<ContainerTag, Node*> pair : startNodesForContainer)
					{
						graphFile << "\"" << debug.getObjectName(pair.first) << "\"" << " [shape=box];\n\n";
						graphFile << pair.second->invocation.time << " -> " << "\"" << debug.getObjectName(pair.first) << "\" [style=dashed];\n\n";
					}
				}
				
				graphFile << "}\n";
			}
			
			void listAllStartingPoints() const
			{
				std::cout << "----------- \nAll starting points:\n";
				for (std::pair<ContainerTag, Node*> pair : startNodesForContainer)
				{
					std::cout << debug.getObjectName(pair.first) << ":\n"; 
					std::cout << "\t" << pair.second->invocation.description() << "\n";
				}
				std::cout << "----------- \n\n";
			}
			
			void listAllNodes() const
			{
				std::cout << "----------- \nAll nodes:\n";
				for (Node *node : allNodes)
					std::cout << node->invocation.time << "\n";
				std::cout << "----------- \n\n";
			}
			
			void registerInvocation(Invocation&& invocation)
			{
			//	std::cout << "Registered " << invocation.description() << "\n";
				
				Node *node = new Node(std::move(invocation), this->allNodes, this->startNodesForContainer);
				
				for (ContainerTag container : invocation.outputs)
					startNodesForContainer[container] = node;
				
				allNodes.insert(node);
			}
			
		private:
			std::unordered_set<Node*> allNodes;
			std::unordered_map<ContainerTag, Node*> startNodesForContainer;
		};
		
		Lineage singleton;
		
		unsigned int Invocation::globalTime = 0;
		
	};
	
}
