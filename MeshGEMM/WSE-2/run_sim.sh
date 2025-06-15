set -e
fabric_w=$(($1 + 7))
fabric_h=$(($1 + 2))

Mt=$(($2 / $1))
Kt=$(($3 / $1))
Nt=$(($4 / $1))

cslc ./src/layout.csl --fabric-dims="$fabric_w","$fabric_h" --fabric-offsets=4,1 \
    --params=P:"$1",Mt:"$Mt",Kt:"$Kt",Nt:"$Nt" \
    -o out --memcpy --channels 1

cs_python ./launch_sim.py --P "$1" --M "$2" --K "$3" --N "$4"