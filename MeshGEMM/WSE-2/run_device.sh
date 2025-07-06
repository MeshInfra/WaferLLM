set -e

P=$1
Mt=$(($2 / $1))
Kt=$(($3 / $1))
Nt=$(($4 / $1))

python compile.py "$P" "$Mt" "$Kt" "$Nt"
python launch_device.py --P "$1" --M "$2" --K "$3" --N "$4"