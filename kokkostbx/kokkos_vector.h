#ifndef KOKKOSTBX_VECTOR_H
#define KOKKOSTBX_VECTOR_H

#include <iostream>
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace {
    template <typename T> KOKKOS_FUNCTION
    typename std::enable_if<std::is_integral<T>::value, void>::type print_num(const T &x) {
        printf("%d ", x);
    }

    template <typename T> KOKKOS_FUNCTION
    typename std::enable_if<std::is_floating_point<T>::value, void>::type print_num(const T &x) {
        printf("%f ", x);
    }
}

namespace kokkostbx {

    template <typename Derived, typename NumType, size_t size>
    struct vector_base {

        NumType data[size] = {};

        // CONSTRUCTOR
        KOKKOS_FUNCTION vector_base() = default;

        KOKKOS_FUNCTION vector_base(NumType val) {
            for (NumType& d : data) { d = val; }
        }

        KOKKOS_FUNCTION vector_base(NumType arr[]) {
            for (size_t i=0; i<size; ++i) {
                data[i] = arr[i];
            }
        }

        // OPERATORS
        // streaming
        friend std::ostream& operator<< (std::ostream &os, const vector_base<Derived, NumType, size>& v) const {
            for (size_t i=0; i<size; ++i ) {
                if (i>0) { os << " "; }
                os << v.data[i];
            }
            return os;
        }

        // access
        KOKKOS_FUNCTION NumType& operator[](const int index) {
            return data[index];
        }

        // addition
        KOKKOS_FUNCTION friend Derived operator+(const Derived& lhs, const Derived& rhs) const {
            Derived sum = lhs;
            sum += rhs;
            return sum;
        }

        // KOKKOS_FUNCTION friend Derived operator+(const Derived& lhs, NumType& rhs) {
        //     Derived sum = lhs;
        //     sum += rhs;
        //     return sum;
        // }

        // KOKKOS_FUNCTION friend Derived operator+(NumType lhs, const Derived& rhs) {
        //     return rhs + lhs;
        // }

        KOKKOS_FUNCTION void operator+=(const Derived& v) {
            for (size_t i=0; i<size; ++i) {
                data[i] += v.data[i];
            }
        }

        KOKKOS_FUNCTION void operator+=(const NumType& v) {
            for (size_t i=0; i<size; ++i) {
                data[i] += v;
            }
        }

        // subtraction
        KOKKOS_FUNCTION friend Derived operator-(const Derived& vec) const {
            Derived negative = vec;
            for (size_t i=0; i<size; ++i) {
                negative[i] *= -1;
            }
            return negative;
        }

        KOKKOS_FUNCTION friend Derived operator-(const Derived& lhs, const Derived& rhs) const {
            Derived sum = lhs;
            sum -= rhs;
            return sum;
        }

        KOKKOS_FUNCTION void operator-=(const Derived& v) {
            for (size_t i=0; i<size; ++i) {
                data[i] -= v.data[i];
            }
        }

        KOKKOS_FUNCTION void operator-=(const NumType& v) {
            for (size_t i=0; i<size; ++i) {
                data[i] -= v;
            }            
        }

        // multiplication
        KOKKOS_FUNCTION friend Derived operator*(const Derived& lhs, const Derived& rhs) const {
            Derived prod = lhs;
            prod *= rhs;
            return prod;
        }

        // KOKKOS_FUNCTION friend Derived operator*(const Derived& lhs, NumType& rhs) {
        //     Derived prod = lhs;
        //     prod *= rhs;
        //     return prod;
        // }

        // KOKKOS_FUNCTION friend Derived operator*(NumType lhs, const Derived& rhs) {
        //     return rhs * lhs;
        // }

        KOKKOS_FUNCTION void operator*=(const Derived& v) {
            for (size_t i=0; i<size; ++i) {
                data[i] *= v.data[i];
            }
        }

        KOKKOS_FUNCTION void operator*=(const NumType& v) {
            for (size_t i=0; i<size; ++i) {
                data[i] *= v;
            }
        }

        // division
        KOKKOS_FUNCTION friend Derived operator/(const Derived& lhs, const Derived& rhs) const {
            Derived quot = lhs;
            quot /= rhs;
            return quot;
        }

        KOKKOS_FUNCTION void operator/=(const Derived& v) {
            for (size_t i=0; i<size; ++i) {
                data[i] /= v.data[i];
            }
        }

        KOKKOS_FUNCTION void operator/=(const NumType& v) {
            for (size_t i=0; i<size; ++i) {
                data[i] /= v;
            }            
        }

        // METHODS
        KOKKOS_FUNCTION void print(const char name[]) const {
            printf("%s: ", name);
            for (size_t i=0; i<size; ++i) {
                print_num(data[i]);
            }
            printf("\n");
        }  

        KOKKOS_FUNCTION void zero() {
            for (size_t i=0; i<size; ++i) {
                data[i] = 0;
            }
        }

        KOKKOS_FUNCTION void ones() {
            for (size_t i=0; i<size; ++i) {
                data[i] = 1;
            }
        }

        KOKKOS_FUNCTION bool is_zero() const {
            for (size_t i=0; i<size; ++i) {
                if (data[i] != 0) return false;
            }
            return true;
        }

        KOKKOS_FUNCTION NumType dot(const Derived& v) const {
            NumType sum = 0;
            for (size_t i=0; i<size; ++i) {
                sum += data[i] * v.data[i];
            }
            return sum;
        }

        KOKKOS_FUNCTION NumType length_sqr() const {
            NumType sum = 0;
            for (size_t i=0; i<size; ++i) {
                sum += data[i] * data[i];
            }
            return sum;
        }

        KOKKOS_FUNCTION NumType length() const {
            return ::Kokkos::Experimental::sqrt(length_sqr());
        }

        KOKKOS_FUNCTION void normalize() {
            NumType l = length();
            if (l>0) {
                for (size_t i=0; i<size; ++i) {
                    data[i] /= l;
                }
            }
        }

        KOKKOS_FUNCTION Derived get_unit_vector() const {
            NumType l = length();
            Derived unit_vector { };
            if (l>0) {
                for (size_t i=0; i<size; ++i) {
                    unit_vector[i] = data[i] / l;
                }
            }
            return unit_vector;
        }
    };

    template <typename NumType, size_t size>
    struct vector : public vector_base<vector<NumType, size>, NumType, size> { 

        using vector_base = kokkostbx::vector_base<vector<NumType, size>, NumType, size>;

        vector() = default;
        KOKKOS_FUNCTION vector(NumType val) : vector_base(val) { };
        KOKKOS_FUNCTION vector(NumType arr[]) : vector_base(arr) { };
    };

}

#endif