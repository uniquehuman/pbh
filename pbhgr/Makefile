CC ?= gcc
CFLAGS ?= -O2 -fPIC -shared -std=c11

all: pbhgr/_kernels.so

pbhgr/_kernels.so: csrc/kernels.c
	$(CC) $(CFLAGS) -o $@ $<

test: all
	python3 -m pytest -q

clean:
	rm -f pbhgr/_kernels.so
