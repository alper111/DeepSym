#ifndef MDP_COMMON_H
#define MDP_COMMON_H

#include <cstdlib>
#include <cerrno>
#include <cstring>

// Handy macros

#ifndef WHERE
#define WHERE { std::cerr<<__FUNCTION__<<" in "<<__FILE__<<":"<<__LINE__<<std::endl; }
#endif

#ifndef EXIT_ERROR
#define EXIT_ERROR { WHERE;				\
	std::cerr << strerror(errno) << std::endl ;	\
	exit(EXIT_FAILURE);				\
    }
#endif

#endif
