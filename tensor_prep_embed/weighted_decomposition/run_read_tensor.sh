MEMORY=4.0
VERBOSE=2

COOCCURRENCE_TXT=../../model/tensor_49.txt
COOCCURRENCE_FILE=../../model/tensor_cooccur_49.bin
COOCCURRENCE_SHUF_FILE=../../model/tensor_cooccur_shuff_49.bin

# compile
gcc read_tensor.c -o read_tensor -lm -pthread -ffast-math -march=native -funroll-loops -Wno-unused-result


# transform cooccurrence txt to cooccurrence bin
./read_tensor -input_txt $COOCCURRENCE_TXT -output_bin $COOCCURRENCE_FILE


# shuffle cooccur_bin

gcc shuffle.c -o shuffle -lm -pthread -ffast-math -march=native -funroll-loops -Wno-unused-result

./shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
