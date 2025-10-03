./run_wse2.sh 64 4096 4096 4096 | tee log/wse2_4k_64.log
./run_wse2.sh 128 4096 4096 4096 | tee log/wse2_4k_128.log
./run_wse2.sh 256 4096 4096 4096 | tee log/wse2_4k_256.log
./run_wse2.sh 512 4096 4096 4096 | tee log/wse2_4k_512.log

./run_wse2.sh 64 8192 8192 8192 | tee log/wse2_8k_64.log
./run_wse2.sh 128 8192 8192 8192 | tee log/wse2_8k_128.log
./run_wse2.sh 256 8192 8192 8192 | tee log/wse2_8k_256.log
./run_wse2.sh 512 8192 8192 8192 | tee log/wse2_8k_512.log

# ./run_wse2.sh 64 16384 16384 16384 | tee log/wse2_16k_64.log
./run_wse2.sh 128 16384 16384 16384 | tee log/wse2_16k_128.log
./run_wse2.sh 256 16384 16384 16384 | tee log/wse2_16k_256.log
./run_wse2.sh 512 16384 16384 16384 | tee log/wse2_16k_512.log