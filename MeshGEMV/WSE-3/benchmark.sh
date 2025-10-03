./run_wse3.sh 64 4096 4096 8 false | tee log/wse3_4k_64_8.log
./run_wse3.sh 128 4096 4096 8 false | tee log/wse3_4k_128_8.log
./run_wse3.sh 256 4096 4096 16 false | tee log/wse3_4k_256_16.log
./run_wse3.sh 512 4096 4096 16 false | tee log/wse3_4k_512_16.log

./run_wse3.sh 64 8192 8192 8 false | tee log/wse3_8k_64_8.log
./run_wse3.sh 128 8192 8192 8 false | tee log/wse3_8k_128_8.log
./run_wse3.sh 256 8192 8192 16 false | tee log/wse3_8k_256_16.log
./run_wse3.sh 512 8192 8192 16 false | tee log/wse3_8k_512_16.log

./run_wse3.sh 64 16384 16384 8 false | tee log/wse3_16k_64_8.log
./run_wse3.sh 128 16384 16384 8 false | tee log/wse3_16k_128_8.log
./run_wse3.sh 256 16384 16384 16 false | tee log/wse3_16k_256_16.log
./run_wse3.sh 512 16384 16384 16 false | tee log/wse3_16k_512_16.log