#ifndef CPP_PROBE_TYPES
#define CPP_PROBE_TYPES

namespace raysim {

/// Enumeration of supported probe types
enum ProbeType {
  PROBE_TYPE_CURVILINEAR = 0,
  PROBE_TYPE_LINEAR_ARRAY = 1,
  PROBE_TYPE_PHASED_ARRAY = 2
};

}  // namespace raysim

#endif /* CPP_PROBE_TYPES */
