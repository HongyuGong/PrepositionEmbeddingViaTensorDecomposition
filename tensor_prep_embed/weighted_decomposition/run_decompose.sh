COOCCURRENCE_SHUF_FILE=data/tensor_cooccur_shuff_49.bin
#COOCCURRENCE_SHUF_FILE=data/tensor_cooccur_shuff_76.bin

gcc decompose.c -o decompose -lm -pthread -ffast-math -march=native -funroll-loops -Wno-unused-result

# 50/77
./decompose -save-file ../../model/weighted_tensor_vectors/weighted_tensor_vectors_49 -threads 25 -input-file ../../model/tensor_cooccur_shuff_49.bin -x-max 10 -iter 1 -vector-size 200 -binary 0 -vocab-file ../../dump/vocab.txt -verbose 2 -model 2 -cond 50 -write-header 1
#./decompose -save-file ../../model/weighted_tensor_vectors/weighted_tensor_vectors_76 -threads 25 -input-file ../../model/tensor_cooccur_shuff_76.bin -x-max 10 -iter 20 -vector-size 200 -binary 0 -vocab-file ../../dump/vocab.txt -verbose 2 -model 2 -cond 77 -write-header 1
