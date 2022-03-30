from __future__ import absolute_import, division, print_function

buttonsdeflist = [
  ("H_I", "Show Zone H for intensities", """
                                                clip_plane {
                                                  hkldist = 0
                                                  normal_vector = 0
                                                  is_assoc_real_space_vector = True
                                                  clip_width = 0.5
                                                }
                                                viewer {
                                                  data_array.label = "I,SIGI"
                                                  data_array.datatype = "Intensity"
                                                  fixorientation = *vector None
                                                  nth_power_scale_radii = nan
                                                  expand_to_p1 = True
                                                  expand_anomalous = True
                                                  color_scheme = *CMRmap
                                                  color_powscale = 0.2633312543
                                                }
"""),
  ("K_I", "Show Zone K for intensities", """
                                                clip_plane {
                                                  hkldist = 0
                                                  normal_vector = 1
                                                  is_assoc_real_space_vector = True
                                                  clip_width = 0.650000
                                                }
                                                viewer {
                                                  data_array.label = "I,SIGI"
                                                  data_array.datatype = "Intensity"
                                                  fixorientation = *vector None
                                                  nth_power_scale_radii = nan
                                                  expand_to_p1 = True
                                                  expand_anomalous = True
                                                  color_scheme = *CMRmap
                                                  color_powscale = 0.2633312543
                                                }
  """),
  ("L_I", "Show Zone L for intensities", """
                                                clip_plane {
                                                  hkldist = 0
                                                  normal_vector = 2
                                                  is_assoc_real_space_vector = True
                                                  clip_width = 0.5
                                                }
                                                viewer {
                                                  data_array.label = "I,SIGI"
                                                  data_array.datatype = "Intensity"
                                                  fixorientation = *vector None
                                                  nth_power_scale_radii = nan
                                                  expand_to_p1 = True
                                                  expand_anomalous = True
                                                  color_scheme = *CMRmap
                                                  color_powscale = 0.2633312543
                                                }
"""),
  ("aniso", "Show Anisotropy", """
                                                #clip_plane {
                                                  #angle_around_vector = "[3, 0]"
                                                  #animate_rotation_around_vector = "[0, -1.000000]"
                                                  #clip_width = 0
                                                #}
                                                binlabel = "ANISO"
                                                nbins = 8
                                                viewer {
                                                  data_array.label = "ANISO"
                                                  nth_power_scale_radii = nan
                                                  expand_to_p1 = True
                                                  expand_anomalous = True
                                                  color_scheme = *jet
                                                  show_vector = "['ANISO1', True]"
                                                  show_vector = "['ANISO2', True]"
                                                  show_vector = "['ANISO3', True]"
                                                }
                                                NGL {
                                                  bin_opacities = "[(1.0, 0), (1.0, 1), (0.0, 2), (0.0, 3), (0.0, 4), (0.0, 5), (1.0, 6), (1.0, 7), (0.0, 8), (0.0, 9), (0.0, 10)]"
                                                }
  """),
  ("Test", "I,SIGI test", """
                                                viewer {
                                                  data_array.label = "I,SIGI"
                                                  data_array.datatype = "Intensity"
                                                }
  """),
  ("INAT", "INAT,SIGINAT test", """
                                                viewer {
                                                  data_array.phasertng_tag = "INAT,SIGINAT"
                                                }
  """),
  ("TNCS", "Show TNCS normal", """
                                                clip_plane {
                                                  angle_around_vector = "['TNCS', 0]"
                                                  animate_rotation_around_vector = "[0, -1.000000]"
                                                  normal_vector = 3
                                                  normal_vector_length_scale = 0.87128
                                                  is_assoc_real_space_vector = False
                                                  clip_width = 0.35
                                                }
                                                viewer {
                                                  data_array.label = "TEPS"
                                                  show_vector = "['TNCS', True]"
                                                  fixorientation = *vector None
                                                  nth_power_scale_radii = 0.1
                                                  expand_to_p1 = True
                                                  expand_anomalous = True
                                                }
  """),
  ("TNCSpar", "Show TNCS paralllel", """
                                                clip_plane {
                                                  angle_around_vector = "['TNCS', 0.0]"
                                                  animate_rotation_around_vector = "['TNCS', 5.000000]"
                                                  clip_width = 5.000000
                                                  auto_clip_width = False
                                                }
                                                viewer {
                                                  data_array.label = "TEPS"
                                                  show_vector = "['TNCS', True]"
                                                  is_parallel = True
                                                  fixorientation = *vector None
                                                  nth_power_scale_radii = 0.1
                                                  expand_to_p1 = True
                                                  expand_anomalous = True
                                                  color_scheme = *rainbow
                                                }


  """)


]