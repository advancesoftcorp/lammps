set(WIN_PACKAGES
  ASPHERE
  BOCS
  BODY
  BPM
  BROWNIAN
  CG-DNA
  CG-SDK
  CLASS2
  COLLOID
  COLVARS
  CORESHELL
  DIELECTRIC
  DIFFRACTION
  DIPOLE
  DPD-BASIC
  DPD-MESO
  DPD-REACT
  DPD-SMOOTH
  DRUDE
  EFF
  EXTRA-COMPUTE
  EXTRA-DUMP
  EXTRA-FIX
  EXTRA-MOLECULE
  EXTRA-PAIR
  FEP
  GRANULAR
  INTERLAYER
  KSPACE
  MANIFOLD
  MANYBODY
  MC
  MEAM
  MISC
  ML-IAP
  ML-SNAP
  ML-SANNP
  ML-OC20DRIVER
  ML-M3GNET
  ML-CHGNET
  MOFFF
  MOLECULE
  MOLFILE
  OPENMP
  ORIENT
  PERI
  PHONON
  POEMS
  PLUGIN
  PTM
  QEQ
  QTB
  REACTION
  REAXFF
  REPLICA
  RIGID
  SHOCK
  SMTBQ
  SPH
  SPIN
  SRD
  TALLY
  UEF
  YAFF)

foreach(PKG ${WIN_PACKAGES})
  set(PKG_${PKG} ON CACHE BOOL "" FORCE)
endforeach()

