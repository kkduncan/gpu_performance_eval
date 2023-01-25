//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char16_t>

// static char_type* move(char_type* s1, const char_type* s2, size_t n);

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    char16_t s1[] = {1, 2, 3};
    assert(std::char_traits<char16_t>::move(s1, s1+1, 2) == s1);
    assert(s1[0] == char16_t(2));
    assert(s1[1] == char16_t(3));
    assert(s1[2] == char16_t(3));
    s1[2] = char16_t(0);
    assert(std::char_traits<char16_t>::move(s1+1, s1, 2) == s1+1);
    assert(s1[0] == char16_t(2));
    assert(s1[1] == char16_t(2));
    assert(s1[2] == char16_t(3));
    assert(std::char_traits<char16_t>::move(NULL, s1, 0) == NULL);
    assert(std::char_traits<char16_t>::move(s1, NULL, 0) == s1);
#endif  // _LIBCUDACXX_HAS_NO_UNICODE_CHARS

  return 0;
}
