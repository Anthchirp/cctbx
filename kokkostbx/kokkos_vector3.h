#ifndef KOKKOSTBX_VECTOR3_H
#define KOKKOSTBX_VECTOR3_H

#include <Kokkos_Core.hpp>

#include "kokkos_vector.h"

namespace kokkostbx {

    template <typename NumType>
    struct vector3 : public vector_base<vector3<NumType>, NumType, 3> { 

        using vector_base = kokkostbx::vector_base<vector3<NumType>, NumType, 3>;

        vector3() = default;
        KOKKOS_FUNCTION vector3(NumType val) : vector_base(val) { };
        KOKKOS_FUNCTION vector3(NumType arr[]) : vector_base(arr) { };

        KOKKOS_FUNCTION vector3(NumType x, NumType y, NumType z) : vector_base() {
            vector_base::data[0] = x;
            vector_base::data[1] = y;
            vector_base::data[2] = z;
        }

        // decided against using properties, as this would increase the size of the class
        KOKKOS_FUNCTION NumType& x_val() { return vector_base::data[0]; }
        KOKKOS_FUNCTION NumType& y_val() { return vector_base::data[1]; }
        KOKKOS_FUNCTION NumType& z_val() { return vector_base::data[2]; }

        KOKKOS_FUNCTION NumType x_val() const { return vector_base::data[0]; }
        KOKKOS_FUNCTION NumType y_val() const { return vector_base::data[1]; }
        KOKKOS_FUNCTION NumType z_val() const { return vector_base::data[2]; }

        KOKKOS_FUNCTION vector3<NumType> cross(const vector3<NumType>& v) const {
            vector3<NumType> cross_vector { };
            cross_vector.x_val() = y_val()*v.z_val() - z_val()*v.y_val();
            cross_vector.y_val() = z_val()*v.x_val() - x_val()*v.z_val();
            cross_vector.z_val() = x_val()*v.y_val() - y_val()*v.x_val();

            return cross_vector;
        }
    };

}

#endif