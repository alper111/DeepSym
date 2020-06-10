/* -*-C++-*- */
/*
 * Rational numbers.
 *
 * Copyright 2003-2005 Carnegie Mellon University and Rutgers University
 * Copyright 2007 Håkan Younes
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef RATIONAL_H
#define RATIONAL_H

#include <config.h>
#include <iostream>
#include <utility>


/* ====================================================================== */
/* Rational */

/*
 * A rational number.
 */
struct Rational {
  /* Returns the multipliers for the two integers. */
  static std::pair<int, int> multipliers(int n, int m);

  /* Constructs a rational number. */
  Rational(int n = 0) : numerator_(n), denominator_(1) {}

  /* Constructs a rational number. */
  Rational(int n, int m);

  /* Constructs a rational number. */
  Rational(const char* s);

  /* Returns the numerator of this rational number. */
  int numerator() const { return numerator_; }

  /* Returns the denominator of this rational number. */
  int denominator() const { return denominator_; }

  /* Returns the double value of this rational number. */
  double double_value() const { return double(numerator())/denominator(); }

 private:
  /* The numerator. */
  int numerator_;
  /* The denominator. */
  int denominator_;
};

/* Less-than comparison operator for rational numbers. */
bool operator<(const Rational& q, const Rational& p);

/* Less-than-or-equal comparison operator for rational numbers. */
bool operator<=(const Rational& q, const Rational& p);

/* Equality comparison operator for rational numbers. */
bool operator==(const Rational& q, const Rational& p);

/* Inequality comparison operator for rational numbers. */
bool operator!=(const Rational& q, const Rational& p);

/* Greater-than-or-equal comparison operator for rational numbers. */
bool operator>=(const Rational& q, const Rational& p);

/* Greater-than comparison operator for rational numbers. */
bool operator>(const Rational& q, const Rational& p);

/* Addition operator for rational numbers. */
Rational operator+(const Rational& q, const Rational& p);

/* Subtraction operator for rational numbers. */
Rational operator-(const Rational& q, const Rational& p);

/* Multiplication operator for rational numbers. */
Rational operator*(const Rational& q, const Rational& p);

/* Division operator for rational numbers. */
Rational operator/(const Rational& q, const Rational& p);

/* Output operator for rational numbers. */
std::ostream& operator<<(std::ostream& os, const Rational& q);


#endif /* RATIONAL_H */
