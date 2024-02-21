#!/bin/csh -f

set packages = ( \
CLASS2 \
EXTRA-COMPUTE \
KSPACE \
MANYBODY \
MISC \
MOLECULE \
QEQ \
RIGID \
ML-SANNP \
ML-OC20DRIVER \
ML-M3GNET \
ML-CHGNET \
FEP \
MEAM \
PHONON \
REAXFF \
REPLICA \
)

set i = 0
while ( ${i} < $#packages )
  @ i = ${i} + 1
  set package = $packages[${i}]
  make yes-${package}
end
