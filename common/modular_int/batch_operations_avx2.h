// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_RLWE_BATCH_OPERATIONS_AVX2_H_
#define THIRD_PARTY_RLWE_BATCH_OPERATIONS_AVX2_H_

// The functions below, guarded by #if defined(__AVX2__), implement
// batch modular operations, that use the AVX2 instruction set.
// The documentation for Intel Advanced Vectors Extensions (AVX) can be found
// at http://software.intel.com/en-us/avx/.

#if defined(__AVX2__)

#include <immintrin.h>

#include "absl/numeric/int128.h"
#include "shell_encryption/bits_util.h"
#include "shell_encryption/integral_types.h"

namespace rlwe::internal {

namespace {

// Converts double in each lane to 64 bit unsigned int.
inline __m256i DoubleToUint64(__m256d x) {
  x = _mm256_add_pd(x, _mm256_set1_pd(0x0010000000000000));
  return _mm256_xor_si256(
      _mm256_castpd_si256(x),
      _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
}

// Converts 64 bit unsigned int in each lane to double.
inline __m256d Uint64ToDouble(__m256i x) {
  x = _mm256_or_si256(x,
                      _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
  return _mm256_sub_pd(_mm256_castsi256_pd(x),
                       _mm256_set1_pd(0x0010000000000000));
}

// Multiplies x * y for each 64 bit lane, outputting the low 64 bit in lo and
// the high 64 bit in hi.
inline void BatchMul64(__m256i x, __m256i y, __m256i& lo, __m256i& hi) {
  __m256i lo_mask = _mm256_set1_epi64x(0x00000000ffffffff);
  __m256i x_hi = _mm256_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);
  __m256i y_hi = _mm256_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);
  __m256i z_lo_lo = _mm256_mul_epu32(x, y);        // x_lo * y_lo
  __m256i z_lo_hi = _mm256_mul_epu32(x, y_hi);     // x_lo * y_hi
  __m256i z_hi_lo = _mm256_mul_epu32(x_hi, y);     // x_hi * y_lo
  __m256i z_hi_hi = _mm256_mul_epu32(x_hi, y_hi);  // x_hi * y_hi

  __m256i z_lo_lo_shift = _mm256_srli_epi64(z_lo_lo, 32);

  __m256i sum_tmp = _mm256_add_epi64(z_lo_hi, z_lo_lo_shift);
  __m256i sum_lo = _mm256_and_si256(sum_tmp, lo_mask);
  __m256i sum_mid = _mm256_srli_epi64(sum_tmp, 32);

  __m256i sum_mid2 = _mm256_add_epi64(z_hi_lo, sum_lo);
  __m256i sum_mid2_hi = _mm256_srli_epi64(sum_mid2, 32);
  __m256i sum_hi = _mm256_add_epi64(z_hi_hi, sum_mid);

  __m256i z_lo_hi_shift = _mm256_slli_epi64(z_lo_hi, 32);
  __m256i z_hi_lo_shift = _mm256_slli_epi64(z_hi_lo, 32);
  lo = _mm256_add_epi64(z_lo_lo, z_lo_hi_shift);
  lo = _mm256_add_epi64(lo, z_hi_lo_shift);
  hi = _mm256_add_epi64(sum_hi, sum_mid2_hi);
}

// Multiplies x * y for each 64 bit lane, only keeping the low 64 bits.
inline __m256i BatchMulLo64(__m256i x, __m256i y) {
  __m256i bswap = _mm256_shuffle_epi32(y, 0xB1);
  __m256i prodlh = _mm256_mullo_epi32(x, bswap);
  __m256i zero = _mm256_setzero_si256();
  __m256i prodlh2 = _mm256_hadd_epi32(prodlh, zero);
  __m256i prodlh3 = _mm256_shuffle_epi32(prodlh2, 0x73);
  __m256i prodll = _mm256_mul_epu32(x, y);
  __m256i prod = _mm256_add_epi64(prodll, prodlh3);
  return prod;
}

// Multiplies x * y for each 64 bit lane, only keeping the high 64 bits.
inline __m256i BatchMulHi64(__m256i x, __m256i y) {
  __m256i lo_mask = _mm256_set1_epi64x(0x00000000ffffffff);
  __m256i x_hi = _mm256_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);
  __m256i y_hi = _mm256_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);
  __m256i z_lo_lo = _mm256_mul_epu32(x, y);        // x_lo * y_lo
  __m256i z_lo_hi = _mm256_mul_epu32(x, y_hi);     // x_lo * y_hi
  __m256i z_hi_lo = _mm256_mul_epu32(x_hi, y);     // x_hi * y_lo
  __m256i z_hi_hi = _mm256_mul_epu32(x_hi, y_hi);  // x_hi * y_hi

  __m256i z_lo_lo_shift = _mm256_srli_epi64(z_lo_lo, 32);

  __m256i sum_tmp = _mm256_add_epi64(z_lo_hi, z_lo_lo_shift);
  __m256i sum_lo = _mm256_and_si256(sum_tmp, lo_mask);
  __m256i sum_mid = _mm256_srli_epi64(sum_tmp, 32);

  __m256i sum_mid2 = _mm256_add_epi64(z_hi_lo, sum_lo);
  __m256i sum_mid2_hi = _mm256_srli_epi64(sum_mid2, 32);
  __m256i sum_hi = _mm256_add_epi64(z_hi_hi, sum_mid);

  return _mm256_add_epi64(sum_hi, sum_mid2_hi);
}

// Returns x mod p, assuming p is a 63 bit modulus.
// Because p is a 63 bit modulus, we can reduce using at most 3 subtractions and
// comparisons.
inline __m256i BatchReduceWith63BitModulus64(__m256i x, __m256i p) {
  __m256i zero = _mm256_setzero_si256();
  __m256i mask1 = _mm256_cmpgt_epi64(zero, x);
  x = _mm256_sub_epi64(x, _mm256_and_si256(mask1, p));
  __m256i mask2 = _mm256_cmpgt_epi64(zero, x);
  x = _mm256_sub_epi64(x, _mm256_and_si256(mask2, p));
  __m256i mask3 = _mm256_cmpgt_epi64(zero, x);
  x = _mm256_sub_epi64(x, _mm256_and_si256(mask3, p));
  return x;
}

inline __m256i Shrdi64(__m256i x, __m256i y, int bit_shift) {
  __m256i c_lo = _mm256_srli_epi64(x, bit_shift);
  __m256i c_hi = _mm256_slli_epi64(y, 64 - bit_shift);
  return _mm256_add_epi64(c_lo, c_hi);
}

}  // namespace

// Set a 256-bit register with the value, repeated as many times as possible to
// fill the register.
template <class T>
inline __m256i BatchSetAvx2(const T& a);

// Specialize the function BatchSetAvx2 for Uint16, Uint32, and Uint64 integers.
template <>
inline __m256i BatchSetAvx2<Uint16>(const Uint16& a) {
  return _mm256_set1_epi16(a);
}

template <>
inline __m256i BatchSetAvx2<Uint32>(const Uint32& a) {
  return _mm256_set1_epi32(a);
}

template <>
inline __m256i BatchSetAvx2<Uint64>(const Uint64& a) {
  return _mm256_set1_epi64x(a);
}

// Load in a register the elements T at the pointer location. Note that this
// function assumes that there are sufficiently many elements at the position
// ptr.
template <class T>
inline __m256i BatchLoadAvx2(const T* ptr) {
  const __m256i* ptr_m256i = reinterpret_cast<const __m256i*>(ptr);
  return _mm256_loadu_si256(ptr_m256i);
}

// Store at the pointer location the elements from the register. Note that this
// function REQUIRES that there is sufficient memory allocated at the position
// ptr.
template <class T>
inline void BatchStoreAvx2(T* ptr, const __m256i& u) {
  __m256i* ptr_m256i = reinterpret_cast<__m256i*>(ptr);
  return _mm256_storeu_si256(ptr_m256i, u);
}

// Batched unsigned integer addition modulo p, assuming log2(p) <= log2(T) - 1.
template <typename T>
inline __m256i BatchAddModAvx2(const __m256i& u, const __m256i& v,
                               const __m256i& p);

// Specialize the function BatchAddModAvx2 for Uint16, Uint32, and Uint64
// integers.
template <>
inline __m256i BatchAddModAvx2<Uint16>(const __m256i& u, const __m256i& v,
                                       const __m256i& p) {
  __m256i w = _mm256_add_epi16(u, v);
  // Compute min(u + v, u + v - p) when viewed as _unsigned_ integers
  // (which explains why we use _mm256_min_epu16 instead of _mm256_min_epi16).
  return _mm256_min_epu16(w, _mm256_sub_epi16(w, p));
}

template <>
inline __m256i BatchAddModAvx2<Uint32>(const __m256i& u, const __m256i& v,
                                       const __m256i& p) {
  __m256i w = _mm256_add_epi32(u, v);
  return _mm256_min_epu32(w, _mm256_sub_epi32(w, p));  // min(u + v, u + v - p)
}

template <>
inline __m256i BatchAddModAvx2<Uint64>(const __m256i& u, const __m256i& v,
                                       const __m256i& p) {
  __m256i w = _mm256_sub_epi64(_mm256_add_epi64(u, v), p);  // u + v - p
  // Viewed as signed integers, get when are the previous values are negative.
  __m256i mask = _mm256_cmpgt_epi64(_mm256_setzero_si256(), w);
  // Add p to all the negative values.
  return _mm256_add_epi64(w, _mm256_and_si256(mask, p));  // w += mask & p
}

// Batched unsigned integer subtraction modulo p, assuming log2(p) <= log2(T)-1.
template <typename T>
inline __m256i BatchSubModAvx2(const __m256i& u, const __m256i& v,
                               const __m256i& p);

// Specialize the function BatchSubModAvx2 for Uint16, Uint32, and Uint64
// integers.
template <>
inline __m256i BatchSubModAvx2<Uint16>(const __m256i& u, const __m256i& v,
                                       const __m256i& p) {
  return BatchAddModAvx2<Uint16>(u, _mm256_sub_epi16(p, v),
                                 p);  // return u + (p - v) mod p
}

template <>
inline __m256i BatchSubModAvx2<Uint32>(const __m256i& u, const __m256i& v,
                                       const __m256i& p) {
  return BatchAddModAvx2<Uint32>(u, _mm256_sub_epi32(p, v),
                                 p);  // return u + (p - v) mod p
}

template <>
inline __m256i BatchSubModAvx2<Uint64>(const __m256i& u, const __m256i& v,
                                       const __m256i& p) {
  return BatchAddModAvx2<Uint64>(u, _mm256_sub_epi64(p, v),
                                 p);  // return u + (p - v) mod p
}

// Compute the Barrett Parameters (r, q) for a modulus p, assuming
// log2(p) <= log2(T)-2. We will use the algorithm
// proposed in Function 6 of http://www.texmacs.org/joris/simd/simd.pdf, where
// s = r - 2 and t = log2(T)+1. The values r and q are computed as follows:
//     r - 1 < log2(p) <= r <= log2(T)-2
// and
//     q = floor(2^(r + log2(T) - 1) / p).
template <typename T>
inline void ComputeBarrettParametersAvx2(int* r, __m256i* q, const T& p);

// Specialize the function BatchMulModAvx2 for Uint16 and Uint32.
template <>
inline void ComputeBarrettParametersAvx2<Uint16>(int* r, __m256i* q,
                                                 const Uint16& p) {
  *r = 64 - ::rlwe::internal::CountLeadingZeros64(p);
  Uint64 barrett_q =
      (static_cast<Uint64>(1) << (*r - 2 + 8 * sizeof(Uint16) + 1)) /
      static_cast<Uint64>(p);
  *q = _mm256_set1_epi32(barrett_q);
}

template <>
inline void ComputeBarrettParametersAvx2<Uint32>(int* r, __m256i* q,
                                                 const Uint32& p) {
  *r = 64 - ::rlwe::internal::CountLeadingZeros64(p);
  Uint64 barrett_q =
      (static_cast<Uint64>(1) << (*r - 2 + 8 * sizeof(Uint32) + 1)) /
      static_cast<Uint64>(p);
  *q = _mm256_set1_epi64x(barrett_q);
}

template <>
inline void ComputeBarrettParametersAvx2<Uint64>(int* r, __m256i* q,
                                                 const Uint64& p) {
  *r = 64 - ::rlwe::internal::CountLeadingZeros64(p);
  Uint64 barrett_q =
      (static_cast<Uint128>(1) << (*r - 2 + 8 * sizeof(Uint64) + 1)) /
      static_cast<Uint128>(p);
  *q = _mm256_set1_epi64x(barrett_q);
}


// Batched unsigned integer subtraction modulo p, where (r, q) are the Barrett
// Parameters.
template <typename T>
inline __m256i BatchMulModAvx2(const __m256i& u, const __m256i& v,
                               const __m256i& p, const int r, const __m256i& q);

// Specialize the function BatchMulModAvx2 for Uint16 and Uint32. The code
// below implements the multiplication and Barrett reduction and is inspired
// from Functions 10 and 12 of http://www.texmacs.org/joris/simd/simd.pdf.
template <>
inline __m256i BatchMulModAvx2<Uint16>(const __m256i& u, const __m256i& v,
                                       const __m256i& p, const int r,
                                       const __m256i& q) {
  const __m256i zero = _mm256_setzero_si256();  // All 0 register.
  // Place all the elements on 32-bit elements. In order to do this, we work
  // over each half of the register, and interleave with all 0's.
  // More precisely, say u = [u1, u2, ..., u16]. We work with the "lo" part of
  // the register (aka [u1, ..., u8]) and interleave with 0's to produce
  // [u1, 0, u2, 0, ..., u8, 0]. Similarly, we will work with the "hi" part to
  // produce [u9, 0, u10, 0, ..., u16, 0].
  const __m256i u_lo = _mm256_unpacklo_epi16(u, zero);
  const __m256i v_lo = _mm256_unpacklo_epi16(v, zero);
  const __m256i u_hi = _mm256_unpackhi_epi16(u, zero);
  const __m256i v_hi = _mm256_unpackhi_epi16(v, zero);
  // Multiply the elements together to obtain [u1 * v1, ..., u8 * v8] and
  // [u9 * v9, ..., u16 * v16], where the products are 32-bit values.
  const __m256i uv_lo = _mm256_mullo_epi32(u_lo, v_lo);
  const __m256i uv_hi = _mm256_mullo_epi32(u_hi, v_hi);
  // Now, we will perform a Barrett reduction.
  const __m256i b_lo = _mm256_srli_epi32(uv_lo, r - 2);
  const __m256i b_hi = _mm256_srli_epi32(uv_hi, r - 2);
  // We multiply by q and shift by log2(T) + 1.
  const __m256i c_lo = _mm256_srli_epi32(_mm256_mullo_epi32(b_lo, q),
                                         /* 16 + 1 = */ 8 * sizeof(Uint16) + 1);
  const __m256i c_hi = _mm256_srli_epi32(_mm256_mullo_epi32(b_hi, q),
                                         /* 16 + 1 = */ 8 * sizeof(Uint16) + 1);
  // We pack c_lo and c_hi (both of the form [* 0 * 0 ... * 0] where * and 0 are
  // 16-bit numbers into one register.
  const __m256i c = _mm256_packus_epi32(c_lo, c_hi);
  // We compute u * v - c * p mod 2 ** 16.
  const __m256i d =
      _mm256_sub_epi16(_mm256_mullo_epi16(u, v), _mm256_mullo_epi16(c, p));
  // The result d is between [0, 2p), so we output min(d, d - p).
  return _mm256_min_epu16(d, _mm256_sub_epi16(d, p));
}

template <>
inline __m256i BatchMulModAvx2<Uint32>(const __m256i& u, const __m256i& v,
                                       const __m256i& p, const int r,
                                       const __m256i& q) {
  // Multiply the elements together to obtain [u1 * v1, u3 * v3, u5 * v5, u7 *
  // v7] and [u2 * v2, u4 * v4, u6 * v6, u8 * v8], where the products are 64-bit
  // values.
  const __m256i uv_odd = _mm256_mul_epu32(u, v);
  const __m256i u_even = _mm256_srli_epi64(u, 32);
  const __m256i v_even = _mm256_srli_epi64(v, 32);
  const __m256i uv_even = _mm256_mul_epu32(u_even, v_even);

  // Now, we will perform a Barrett reduction.
  // The first step is a right shift by r-2.
  const __m256i b_odd = _mm256_srli_epi64(uv_odd, r - 2);
  const __m256i b_even = _mm256_srli_epi64(uv_even, r - 2);
  // We multiply by q and right shift by log2(T) + 1 = 8 * sizeof(Uint32) + 1.
  const __m256i c_odd = _mm256_srli_epi64(
      _mm256_mul_epu32(b_odd, q), /* 32 + 1 = */ 8 * sizeof(Uint32) + 1);
  const __m256i c_even = _mm256_srli_epi64(
      _mm256_mul_epu32(b_even, q), /* 32 + 1 = */ 8 * sizeof(Uint32) + 1);
  // We blend c_odd and c_even to reconstruct c. Idem for uv mod 2**32.
  const __m256i c =
      _mm256_blend_epi32(c_odd, _mm256_slli_epi64(c_even, 32), 0b10101010);
  const __m256i uv =
      _mm256_blend_epi32(uv_odd, _mm256_slli_epi64(uv_even, 32), 0b10101010);
  // We compute d = u*v - c * p mod 2**32.
  const __m256i d = _mm256_sub_epi32(uv, _mm256_mullo_epi32(c, p));
  // The result d is between [0, 2p), so we output min(d, d - p).
  return _mm256_min_epu32(d, _mm256_sub_epi32(d, p));
}

template <>
inline __m256i BatchMulModAvx2<Uint64>(const __m256i& u, const __m256i& v,
                                       const __m256i& p, const int r,
                                       const __m256i& q) {
  __m256i uv_lo, uv_hi;
  BatchMul64(u, v, uv_lo, uv_hi);
  __m256i b = Shrdi64(uv_lo, uv_hi, r - 1);

  __m256i c3 = BatchMulHi64(b, q);
  __m256i c4 = BatchMulLo64(c3, p);

  c4 = _mm256_sub_epi64(uv_lo, c4);

  __m256i mask = _mm256_cmpgt_epi64(c4, p);
  return _mm256_sub_epi64(c4, _mm256_and_si256(mask, p));
}

// Computes (u * v) mod p using floating point Avx2 instructions.
// Assumes 0 <= u, v < p < 2^50.
// p_inv stores 1 / p in each lane.
// Adapted from https://arxiv.org/pdf/1407.3383.pdf, Function 18.
inline __m256i BatchMulModAvx2Uint50(const __m256i& u, const __m256i& v,
                                     const __m256d& p, const __m256d& p_inv) {
  __m256d ui = Uint64ToDouble(u);
  __m256d vi = Uint64ToDouble(v);
  __m256d h = _mm256_mul_pd(ui, vi);
  __m256d l = _mm256_fmsub_pd(ui, vi, h);
  __m256d a = _mm256_mul_pd(h, p_inv);
  __m256d b = _mm256_floor_pd(a);
  __m256d c = _mm256_fnmadd_pd(b, p, h);
  __m256d d = _mm256_add_pd(c, l);
  __m256d t = _mm256_sub_pd(d, p);
  d = _mm256_blendv_pd(t, d, t);
  t = _mm256_add_pd(d, p);
  return DoubleToUint64(_mm256_blendv_pd(d, t, d));
}

// Computes z_low + z_mid * 2^32 + z_high * 2^64 = (x * y) mod p, where p
// is a 63 bit modulus, such that z_low, z_mid, z_high < p.
// x and y are assumed to be at most 62 bit long.
inline void BatchMulWithoutReductionUint64Avx2(__m256i x, __m256i y, __m256i p,
                                               __m256i& z_low, __m256i& z_mid,
                                               __m256i& z_high) {
  __m256i x_high = _mm256_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);
  __m256i y_high = _mm256_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);

  z_low = _mm256_mul_epu32(x, y);
  // z_low can be 64 bit long, so we need to reduce it.
  z_low = BatchReduceWith63BitModulus64(z_low, p);
  __m256i z_low_high = _mm256_mul_epu32(x, y_high);
  __m256i z_high_low = _mm256_mul_epu32(x_high, y);

  // z_mid = z_low_high + z_high_low mod p can be reduced with single modular
  // addition since z_low_high and z_high_low are both at most 62 bit.
  z_mid = BatchAddModAvx2<Uint64>(z_low_high, z_high_low, p);

  // We don't have to reduce z_high since it's at most 60 bit (and p is assumed
  // to be a 63 bit modulus).
  z_high = _mm256_mul_epu32(x_high, y_high);
}

// Multiply and return the high 8 * sizeof(T) bits of the intermediate integers.
template <typename T>
inline __m256i BatchMulHighAvx2(const __m256i& u, const __m256i& v);

template <>
inline __m256i BatchMulHighAvx2<Uint16>(const __m256i& u, const __m256i& v) {
  return _mm256_mulhi_epu16(u, v);
}

template <>
inline __m256i BatchMulHighAvx2<Uint32>(const __m256i& u, const __m256i& v) {
  // Compute mul_lo = [u1 * v1 >> 32, u3 * v3 >> 32]
  const __m256i mul_lo = _mm256_srli_epi64(_mm256_mul_epu32(u, v), 32);
  // Compute mul_hi = [u2 * v2 >> 32, u4 * v4 >> 32]
  const __m256i u_hi = _mm256_srli_epi64(u, 32);  // [u2, 0, u4, 0]
  const __m256i v_hi = _mm256_srli_epi64(v, 32);  // [v2, 0, v4, 0]
  const __m256i mul_hi = _mm256_mul_epu32(u_hi, v_hi);
  // Blend the result by taking simultaneously the 32 LSB of mul_lo and 32 MSB
  // of mul_hi.
  return _mm256_blend_epi16(mul_lo, mul_hi, 0b11001100);
}

// Batched unsigned multiplication by constant modulo p, where
// v_constant_barrett = (v_constant << (8 * sizeof(T)) / p).
template <typename T>
inline __m256i BatchMulConstantModAvx2(const __m256i& u,
                                       const __m256i& v_constant,
                                       const __m256i& v_constant_barrett,
                                       const __m256i& p);

template <>
inline __m256i BatchMulConstantModAvx2<Uint16>(
    const __m256i& u, const __m256i& v_constant,
    const __m256i& v_constant_barrett, const __m256i& p) {
  // Compute the high word of u * v_constant_barrett.
  const __m256i uv_barrett_hi = BatchMulHighAvx2<Uint16>(u, v_constant_barrett);
  // Compute u * v_consant - uv_barrett_hi * p, as Uint16.
  const __m256i uv = _mm256_mullo_epi16(u, v_constant);
  const __m256i out =
      _mm256_sub_epi16(uv, _mm256_mullo_epi16(uv_barrett_hi, p));
  // The result d is between [0, 2p), so we output min(out, out - p).
  return _mm256_min_epu16(out, _mm256_sub_epi16(out, p));
}

template <>
inline __m256i BatchMulConstantModAvx2<Uint32>(
    const __m256i& u, const __m256i& v_constant,
    const __m256i& v_constant_barrett, const __m256i& p) {
  const __m256i uv_barrett_hi = BatchMulHighAvx2<Uint32>(u, v_constant_barrett);
  const __m256i uv = _mm256_mullo_epi32(u, v_constant);
  const __m256i out =
      _mm256_sub_epi32(uv, _mm256_mullo_epi32(uv_barrett_hi, p));
  return _mm256_min_epu32(out, _mm256_sub_epi32(out, p));
}

}  // namespace rlwe::internal

#endif  // defined(__AVX2__)

#endif  // THIRD_PARTY_RLWE_BATCH_OPERATIONS_AVX2_H_
