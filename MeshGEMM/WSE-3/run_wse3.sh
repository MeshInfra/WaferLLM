set -e

P=$1
Mt=$(($2 / $1))
Kt=$(($3 / $1))
Nt=$(($4 / $1))

simulator=false

if [ -n "$5" ]; then
    simulator=$5
fi

python compile.py "$P" "$Mt" "$Kt" "$Nt" "$simulator"

if [ "$simulator" == "true" ]; then
    python launch_wse3.py --P "$1" --M "$2" --K "$3" --N "$4" --simulator
else
    python launch_wse3.py --P "$1" --M "$2" --K "$3" --N "$4"
fi