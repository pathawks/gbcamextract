target ?= gbcamextract
VERSION_STRING= 1.1
objects := $(patsubst %.c,%.o,$(wildcard *.c))

LDLIBS += -lpng

CFLAGS  += -std=gnu99 -Os -ggdb -D__progversion=\"${VERSION_STRING}\" -D__progname=\"${target}\"

#EXTRAS += -fsanitize=undefined -fsanitize=null -fcf-protection=full -fstack-protector-all -fstack-check -Wimplicit-fallthrough -fanalyzer -Wall
EXTRAS += -Wall -flto

CFLAGS += ${EXTRAS} -I/opt/homebrew/Cellar/libpng/1.6.43/include/libpng16
LDFLAGS += ${EXTRAS} -L/opt/homebrew/Cellar/libpng/1.6.43/lib

.PHONY: all
all:	$(target)

.PHONY: clean
clean:
	rm -f $(target) $(target).exe $(objects)

.PHONY: install
install:
	cp $(target) /usr/local/bin/$(target)

$(target): $(objects)
