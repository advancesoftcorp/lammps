set(WIN_PACKAGES
  ASPHERE
  ATC
  AWPMD
  BOCS
  BODY
  BPM
  BROWNIAN
  CG-DNA
  CG-SDK
  CLASS2
  COLLOID
  COLVARS
  COMPRESS
  CORESHELL
  DIELECTRIC
  DIFFRACTION
  DIPOLE
  DPD-BASIC
  DPD-MESO
  DPD-REACT
  DPD-SMOOTH
  DRUDE
  ELECTRODE
  EFF
  EXTRA-COMPUTE
  EXTRA-DUMP
  EXTRA-FIX
  EXTRA-MOLECULE
  EXTRA-PAIR
  FEP
  GPU
  GRANULAR
  INTEL
  INTERLAYER
  KSPACE
  LATTE
  MACHDYN
  MANIFOLD
  MANYBODY
  MC
  MDI
  MEAM
  MESONT
  MGPT
  MISC
  ML-HDNNP
  ML-IAP
  ML-SNAP
  ML-RANN
  ML-SANNP
  ML-OC20DRIVER
  ML-M3GNET
  ML-CHGNET
  MOFFF
  MOLECULE
  MOLFILE
  OPENMP
  OPT
  ORIENT
  PERI
  PHONON
  POEMS
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
  VORONOI
  YAFF)

foreach(PKG ${WIN_PACKAGES})
  set(PKG_${PKG} ON CACHE BOOL "" FORCE)
endforeach()

# these two packages require a full MPI implementation
if(BUILD_MPI)
  set(PKG_MPIIO ON CACHE BOOL "" FORCE)
  set(PKG_LATBOLTZ ON CACHE BOOL "" FORCE)
endif()

set(DOWNLOAD_VORO ON CACHE BOOL "" FORCE)
set(DOWNLOAD_EIGEN3 ON CACHE BOOL "" FORCE)
set(LAMMPS_MEMALIGN "0" CACHE STRING "" FORCE)
set(CMAKE_TUNE_FLAGS "-Wno-missing-include-dirs" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--enable-stdcall-fixup,--as-needed,-lssp" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--enable-stdcall-fixup,--as-needed,-lssp" CACHE STRING "" FORCE)
set(BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/lammps-installer")
