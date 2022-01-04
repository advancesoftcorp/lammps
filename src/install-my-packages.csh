#!/bin/csh -f

set packages = ( \
CLASS2 \
KSPACE \
MANYBODY \
MISC \
MOLECULE \
QEQ \
RIGID \
SANNP \
OC20DRIVER \
USER-FEP \
USER-MEAMC \
USER-MOLFILE \
USER-PHONON \
USER-REAXC \
)

set i = 0
while ( ${i} < $#packages )
  @ i = ${i} + 1
  set package = $packages[${i}]
  make yes-${package}
end
