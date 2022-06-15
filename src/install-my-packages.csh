#!/bin/csh -f

set packages = ( \
CLASS2 \
KSPACE \
MANYBODY \
MISC \
MOLECULE \
QEQ \
RIGID \
ML-SANNP \
ML-OC20DRIVER \
FEP \
MEAM \
PHONON \
REAXFF \
)

set i = 0
while ( ${i} < $#packages )
  @ i = ${i} + 1
  set package = $packages[${i}]
  make yes-${package}
end
