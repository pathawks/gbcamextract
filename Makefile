all: 	gbcamextract


clean:
	rm -f gbcamextract

gbcamextract:	Makefile gbcamextract.c
	gcc -Wall -std=c99 gbcamextract.c -lpng -o gbcamextract