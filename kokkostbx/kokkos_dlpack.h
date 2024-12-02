#ifndef KOKKOS_DLPACK_H
#define KOKKOS_DLPACK_H
#include <Kokkos_Core.hpp>
#include <dlpack/dlpack.h>

namespace kokkostbx {

template<typename DataType, typename SpaceType>
DLDataTypeCode getDLPackTypeCode() {
  using ValueType = typename Kokkos::View<DataType, SpaceType>::value_type;
  if (std::is_same<ValueType, float>::value) {
    return kDLFloat;
  } else if (std::is_same<ValueType, double>::value) {
    return kDLFloat;
  } else if (std::is_same<ValueType, int>::value) {
    return kDLInt;
  } else if (std::is_same<ValueType, unsigned int>::value) {
    return kDLUInt;
  // } else if (std::is_same<ValueType, bool>::value) {
    // return kDLBool;
  } else {
    // Unsupported data type
    throw std::runtime_error("Unsupported data type for DLPack conversion");
  }
}

template<typename DataType, typename SpaceType>
DLDataType getDLPackDataType() {
  DLDataType dtype;
  dtype.code = getDLPackTypeCode<DataType, SpaceType>();
  dtype.bits = sizeof(typename Kokkos::View<DataType, SpaceType>::value_type) * 8;
  dtype.lanes = 1;
  return dtype;
}

template<typename SpaceType>
DLDevice getDLPackDevice() {
  const int device_id = std::max(0, Kokkos::device_id()); // convert host id from -1 to 0

  if (std::is_same<SpaceType, Kokkos::HostSpace>::value) {
      return {kDLCPU, device_id};
  }
#ifdef KOKKOS_ENABLE_CUDA
  else if (std::is_same<SpaceType, Kokkos::CudaSpace>::value) {
      return {kDLCUDA, device_id};
  } else if (std::is_same<SpaceType, Kokkos::CudaUVMSpace>::value) {
      return {kDLCUDAManaged, device_id};
  } else if (std::is_same<SpaceType, Kokkos::CudaHostPinnedSpace>::value) {
      return {kDLCUDAHost, device_id};
  }
#endif
#ifdef KOKKOS_ENABLE_HIP
  else if (std::is_same<SpaceType, Kokkos::HIPSpace>::value) {
      return {kDLROCM, device_id};
  } else if (std::is_same<SpaceType, Kokkos::HIPHostPinnedSpace>::value) {
      return {kDLROCMHost, device_id};
  }
#endif
  else {
      // Extend to other device types as needed
      throw std::runtime_error("Unsupported Kokkos device type for DLPack conversion.");
  }
}

template<typename DataType, typename SpaceType>
DLManagedTensor* view_to_dlpack(Kokkos::View<DataType, SpaceType>& view) {
  // Get the Kokkos view size and dimensions
  constexpr size_t numDims = Kokkos::View<DataType, SpaceType>::rank;
  int64_t* shape = new int64_t[numDims];
  for (size_t i = 0; i < numDims; i++) {
    shape[i] = view.extent(i);
  }

  // Create a DLPack tensor
  DLManagedTensor* dlpackTensor = new DLManagedTensor;
  dlpackTensor->dl_tensor.data = view.data();
  dlpackTensor->dl_tensor.device = getDLPackDevice<SpaceType>();
  dlpackTensor->dl_tensor.ndim = numDims;
  dlpackTensor->dl_tensor.dtype = getDLPackDataType<DataType, SpaceType>();
  dlpackTensor->dl_tensor.shape = shape;
  dlpackTensor->dl_tensor.strides = nullptr;
  dlpackTensor->dl_tensor.byte_offset = 0;
  dlpackTensor->manager_ctx = nullptr;
  dlpackTensor->deleter = [](DLManagedTensor* tensor) {
      delete[] tensor->dl_tensor.shape;
  };
  return dlpackTensor;
}

}

#endif  // KOKKOS_DLPACK_H
