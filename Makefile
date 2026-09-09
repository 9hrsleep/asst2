FILES= saxpy/Makefile \
	saxpy/*.h \
	saxpy/*.cpp \
	saxpy/*.cu \
	scan/Makefile \
	scan/*.h \
	scan/*.cpp \
	scan/*.cu \
	render/Makefile \
	render/*.h \
	render/*.cpp \
	render/*.cu \
	render/*.cu_inl \
	Makefile

all:
	(cd saxpy ; make all)
	(cd scan ; make all)
	(cd render ; make all)

handin.tgz: $(FILES)
	tar cvzf handin.tgz $(FILES)

clean:
	(cd saxpy ; make clean)
	(cd scan ; make clean)
	(cd render ; make clean)
	rm -f *~ handin.tgz
