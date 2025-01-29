#ifndef CLUSTER_META_HELPERS_HPP
#define CLUSTER_META_HELPERS_HPP

#define REQUIRES(...) typename std::enable_if<(__VA_ARGS__), bool>::type = 0
#define REQUIRES_DEF(...) typename std::enable_if<(__VA_ARGS__), bool>::type


#endif /* CLUSTER_META_HELPERS_HPP */
