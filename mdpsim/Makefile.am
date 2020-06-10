## Process this file with automake to produce Makefile.in
##
## Copyright 2003-2005 Carnegie Mellon University and Rutgers University
## Copyright 2007 Håkan Younes
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

bin_PROGRAMS = mdpsim mdpclient
EXTRA_PROGRAMS = mtbddclient
mdpsim_SOURCES = mdpsim.cc mdpserver.cc mdpserver.h strxml.cc strxml.h requirements.cc requirements.h rational.cc rational.h types.cc types.h terms.cc terms.h predicates.cc predicates.h functions.cc functions.h refcount.h expressions.cc expressions.h formulas.cc formulas.h effects.cc effects.h actions.cc actions.h domains.cc domains.h problems.cc problems.h states.cc states.h parser.yy tokenizer.ll
mdpclient_SOURCES = mdpclient.cc client.cc client.h strxml.cc strxml.h requirements.cc requirements.h rational.cc rational.h types.cc types.h terms.cc terms.h predicates.cc predicates.h functions.cc functions.h refcount.h expressions.cc expressions.h formulas.cc formulas.h effects.cc effects.h actions.cc actions.h domains.cc domains.h problems.cc problems.h states.cc states.h tokenizer.ll
mtbddclient_SOURCES = mtbddclient.cc mtbdd.cc mtbdd.h client.cc client.h strxml.cc strxml.h requirements.cc requirements.h rational.cc rational.h types.cc types.h terms.cc terms.h predicates.cc predicates.h functions.cc functions.h refcount.h expressions.cc expressions.h formulas.cc formulas.h effects.cc effects.h actions.cc actions.h domains.cc domains.h problems.cc problems.h states.cc states.h tokenizer.ll

mdpsim_LDADD = @LIBOBJS@ @PTHREADLIB@ -lstdc++fs
mdpclient_LDADD = parser.o @LIBOBJS@
mtbddclient_CPPFLAGS = @CPPFLAGS@ -I"@CUDDDIR@/include"
mtbddclient_LDFLAGS = @LDFLAGS@ -L"@CUDDDIR@/cudd" -L"@CUDDDIR@/epd" -L"@CUDDDIR@/mtr" -L"@CUDDDIR@/st" -L"@CUDDDIR@/util"
mtbddclient_LDADD = parser.o -lcudd -lepd -lmtr -lst -lutil @LIBOBJS@

CLEANFILES = logs/* last_id mtbddclient
MAINTAINERCLEANFILES = parser.cc tokenizer.cc config.h.in~
EXTRA_DIST = getopt.c getopt1.c comp.cfg examples port LICENSE NOTICE

AM_YFLAGS = -d

ACLOCAL_AMFLAGS = -I m4
