// read each slab into cooccur_rec format

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef double real;

typedef struct cooccur_rec {
    int word1;
    int word2;
    int word3;
    float val;
} CREC;

int scmp( char *s1, char *s2 ) {
    while (*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

void testBinFormat(char *bin_fn) {
	FILE *fin;
	CREC cr;
	fin = fopen(bin_fn, "r");
	while (fread(&cr, sizeof(CREC), 1, fin) == 1) {
		fprintf(stderr, "read binary file\n");
		fprintf(stderr, "word 1: %d\n", cr.word1);
		fprintf(stderr, "word 2: %d\n", cr.word2);
		fprintf(stderr, "prep: %d\n", cr.word3);
		fprintf(stderr, "val: %f\n", cr.val);
	}
	fclose(fin);
}

void formatCooccur(char *txt_fn, char *bin_fn) {
	FILE *fin;
	FILE *fout;
	int word1;
	int word2;
	int word3;
	float val;
	CREC crec;
	char line[10000];
	//fprintf(stderr, "%s\n", txt_fn);
	fin = fopen(txt_fn, "r");
	fout = fopen(bin_fn, "w");
	fprintf(stderr, "open file is ok\n");
	while (fscanf(fin, "%d %d %d %f\n", &word1, &word2, &word3, &val) != EOF) {
		//fprintf(stderr, "loop begins\n");
		//fprintf(stderr, "word 1: %d\n", word1);
		//fprintf(stderr, "word 2: %d\n", word2);
		//fprintf(stderr, "val: %f\n", val);
		crec.word1 = word1;
		crec.word2 = word2;
		crec.word3 = word3;
		crec.val = val;
		fwrite(&crec, sizeof(CREC), 1, fout);
	}
	fclose(fin);
	fclose(fout);
}


int main(int argc, char **argv) {
	int i;
	char txt_fn[2000];
	char bin_fn[2000];
	if ((i = find_arg((char *)"-input_txt", argc, argv)) > 0) strcpy(txt_fn, argv[i + 1]);
	if ((i = find_arg((char *)"-output_bin", argc, argv)) > 0) strcpy(bin_fn, argv[i + 1]);
	fprintf(stderr, "%s\n", txt_fn);
	fprintf(stderr, "%s\n", bin_fn);
	formatCooccur(txt_fn, bin_fn);
	// testing
	//testBinFormat(bin_fn);
}
