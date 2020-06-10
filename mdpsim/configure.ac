# Process this file with autoconf to produce a configure script.
#
# Copyright 2003-2005 Carnegie Mellon University and Rutgers University
# Copyright 2007 Håkan Younes
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

AC_INIT(MDPSim, 2.2, hyounes@tempastic.org)
AC_CONFIG_SRCDIR(mdpsim.cc)
AM_INIT_AUTOMAKE
AM_CONFIG_HEADER(config.h)

# Add option to set location of CUDD package.
AC_ARG_VAR(CUDDDIR, [location of the CUDD package])
CUDDDIR=${CUDDDIR:-./cudd}

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_INSTALL
AC_PROG_AWK
AM_PROG_LEX
AC_PROG_YACC

AX_CXX_COMPILE_STDCXX([17], [noext], [mandatory])

# Whether to use tcmalloc.
AC_ARG_WITH([tcmalloc],
            [AS_HELP_STRING([--with-tcmalloc],
                            [a malloc replacement @<:@default=no@:>@])],
            [],
            [with_tcmalloc=no])
AS_IF([test "x$with_tcmalloc" != xno],
      [AC_CHECK_LIB(
          [tcmalloc], [tc_malloc], ,
          [AS_IF([test "x$with_tcmalloc" != xcheck],
                 [AC_MSG_FAILURE(
                     [--with-tcmalloc was given, but test for tcmalloc failed])
                 ])])])

# Set the language.
AC_LANG(C++)

# Checks for libraries.
AC_SEARCH_LIBS(gettext, intl)
AC_SEARCH_LIBS(accept, socket)
AC_SEARCH_LIBS(gethostbyname, nsl)
AC_CHECK_LIB(pthread, pthread_create, PTHREADLIB=-lpthread)
AC_SUBST(PTHREADLIB)

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([arpa/inet.h libintl.h netdb.h netinet/in.h stddef.h stdlib.h string.h strings.h sys/socket.h unistd.h sys/time.h sstream])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T
AC_CHECK_TYPES(socklen_t, , , [#include <sys/socket.h>])

# Checks for library functions.
AC_CHECK_FUNCS([gethostbyname memset socket strerror])
AC_CHECK_FUNCS(getopt_long, , [AC_LIBOBJ(getopt) AC_LIBOBJ(getopt1)])

AC_CONFIG_FILES(Makefile)
AC_OUTPUT
