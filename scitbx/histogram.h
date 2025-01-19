#ifndef SCITBX_HISTOGRAM_H
#define SCITBX_HISTOGRAM_H
#include <scitbx/array_family/shared.h>
#include <scitbx/array_family/ref_reductions.h>
#include <iostream>
namespace scitbx {

  //! Histogram of an array of scalar values.
  template <typename ValueType = double,
            typename CountType = long>
  class histogram
  {
    public:
      template <typename DataType>
      histogram() {}

      //! Computation of the histogram.
      template <typename DataType>
      histogram(
        af::const_ref<DataType> const& data,
        std::size_t n_slots=1000)
      :
        data_min_(0),
        data_max_(0),
        slot_width_(0),
        slots_(n_slots),
        n_out_of_slot_range_(0)
      {
        SCITBX_ASSERT(n_slots > 0);
        if (data.size() == 0) return;
        data_min_ = af::min(data);
        data_max_ = af::max(data);
        slot_width_ = (data_max_ - data_min_) / slots_.size();
        for(std::size_t i=0;i<data.size();i++) {
          assign_to_slot(static_cast<ValueType>(data[i]));
        }
      }

      //! Histogram using user-defined data_min(), data_max().
      template <typename DataType>
      histogram(
        af::const_ref<DataType> const& data,
        DataType const& data_min,
        DataType const& data_max,
        std::size_t n_slots=1000,
        ValueType const& relative_tolerance=1.e-4)
      :
        data_min_(data_min),
        data_max_(data_max),
        slot_width_(0),
        slots_(n_slots),
        n_out_of_slot_range_(0)
      {
        SCITBX_ASSERT(data_max > data_min);
        SCITBX_ASSERT(n_slots > 0);
        slot_width_ = (data_max_ - data_min_) / slots_.size();
        assign_to_slots(data, relative_tolerance);
      }

      //! Histogram using slots of other.
      template <typename DataType>
      histogram(
        histogram const& other,
        af::const_ref<DataType> const& data,
        ValueType const& relative_tolerance=1.e-4)
      :
        data_min_(other.data_min_),
        data_max_(other.data_max_),
        slot_width_(other.slot_width_),
        slots_(other.slots_.size()),
        n_out_of_slot_range_(0)
      {
        assign_to_slots(data, relative_tolerance);
      }

      //! Minimum slot cutoff.
      ValueType
      data_min() const { return data_min_; }

      //! Maximum slot cutoff.
      ValueType
      data_max() const { return data_max_; }

      //! Slot width used in the determination of the histogram.
      ValueType
      slot_width() const { return slot_width_; }

      //! Direct access to the array of counts.
      af::shared<CountType>
      slots() const { return slots_; }

      //! Number of unaccounted data values.
      std::size_t
      n_out_of_slot_range() const { return n_out_of_slot_range_; }

      af::shared<ValueType>
      slot_centers() const
      {
        af::shared<ValueType> centers;
        ValueType low_cutoff = data_min_;
        ValueType high_cutoff;
        for (std::size_t i=0; i< slots_.size(); i++) {
          high_cutoff = low_cutoff + slot_width_;
          centers.push_back((high_cutoff + low_cutoff)/2);
          low_cutoff = high_cutoff;
        }
        return centers;
      }

      //! Determination of the cutoff value given a maximum number of points.
      ValueType
      get_cutoff(CountType const& max_points,
                 ValueType const& relative_tolerance=1.e-4) const
      {
        CountType cum = 0;
        std::size_t i = slots_.size();
        for (; i; i--) {
          cum += slots_[i-1];
          if (cum > max_points) break;
        }
        return data_min_ + i * slot_width_ + slot_width_ * relative_tolerance;
      }

      template <typename DataType>
      void update(DataType const& d,
                  ValueType const& relative_tolerance=1.e-4)
      {
        ValueType width_tolerance = slot_width_ * relative_tolerance;
        if (   d < data_min_ - width_tolerance
            || d > data_max_ + width_tolerance) {
          n_out_of_slot_range_++;
        }
        else {
          assign_to_slot(static_cast<ValueType>(d));
        }
      }

      void update(histogram const& other)
      {
        SCITBX_ASSERT(data_min_ == other.data_min_);
        SCITBX_ASSERT(data_max_ == other.data_max_);
        SCITBX_ASSERT(slot_width_ == other.slot_width_);
        SCITBX_ASSERT(slots_.size() == other.slots_.size());
        for (std::size_t i=0; i<slots_.size(); i++)
          slots_[i] += other.slots_[i];
        n_out_of_slot_range_ += other.n_out_of_slot_range_;
      }
///////////////Muhamed Amin: Code to create histogram object from numpy array///////////////////////////
      void update(af::shared<int> const& from_ar)
      {
        SCITBX_ASSERT(slots_.size() == from_ar.size());
        for (std::size_t i=0; i<from_ar.size(); i++)
          slots_[i] = from_ar[i];
      }
/////////////////////////////////////////////////////////////////////////////////
      std::size_t
      get_i_slot(ValueType const& d_)
      {
        std::size_t i_slot = 0;
        ValueType d = d_ - data_min_;
        if (d != 0 && d >= slot_width_) {
              i_slot = static_cast<std::size_t>(d / slot_width_);
          if (i_slot >= slots_.size()) i_slot = slots_.size() - 1;
        }
        return i_slot;
      }

    protected:
      void
      assign_to_slot(ValueType const& d)
      {
        if (!std::isnan(d)) {
          slots_[get_i_slot(d)]++;
        }
      }

      template <typename DataType>
      void
      assign_to_slots(
        af::const_ref<DataType> const& data,
        ValueType const& relative_tolerance)
      {
        ValueType width_tolerance = slot_width_ * relative_tolerance;
        for(std::size_t i=0;i<data.size();i++) {
          if (   data[i] < data_min_ - width_tolerance
              || data[i] > data_max_ + width_tolerance) {
            n_out_of_slot_range_++;
          }
          else {
            assign_to_slot(static_cast<ValueType>(data[i]));
          }
        }
      }

      ValueType data_min_;
      ValueType data_max_;
      ValueType slot_width_;
      af::shared<CountType> slots_;
      std::size_t n_out_of_slot_range_;

    public:
      //! support for Python pickling
      histogram(
        ValueType const& data_min,
        ValueType const& data_max,
        ValueType const& slot_width,
        af::shared<CountType> const& slots,
        std::size_t n_out_of_slot_range)
      :
        data_min_(data_min),
        data_max_(data_max),
        slot_width_(slot_width),
        slots_(slots),
        n_out_of_slot_range_(n_out_of_slot_range)
      {}
  };

} // namespace scitbx

#endif // SCITBX_HISTOGRAM_H
