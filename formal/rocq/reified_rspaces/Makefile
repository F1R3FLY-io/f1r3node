# Makefile for Reified RSpaces Rocq Proofs
#
# Usage:
#   make          - Build all proofs
#   make clean    - Remove generated files
#   make check    - Verify all proofs compile
#   make docs     - Generate documentation
#
# For resource-limited builds (recommended for CI):
#   systemd-run --user --scope -p MemoryMax=126G -p CPUQuota=1800% \
#       -p IOWeight=30 -p TasksMax=200 make -j1

COQPROJECT := _CoqProject
COQMAKEFILE := Makefile.coq

.PHONY: all clean check docs

all: $(COQMAKEFILE)
	$(MAKE) -f $(COQMAKEFILE)

$(COQMAKEFILE): $(COQPROJECT)
	coq_makefile -f $(COQPROJECT) -o $(COQMAKEFILE)

clean:
	if [ -f $(COQMAKEFILE) ]; then $(MAKE) -f $(COQMAKEFILE) clean; fi
	rm -f $(COQMAKEFILE) $(COQMAKEFILE).conf
	find . -name "*.vo" -delete
	find . -name "*.vok" -delete
	find . -name "*.vos" -delete
	find . -name "*.glob" -delete
	find . -name ".*.aux" -delete

check: all
	@echo "All proofs verified successfully!"

docs: all
	mkdir -p html
	coqdoc -Q theories ReifiedRSpaces -d html --utf8 theories/**/*.v

# Individual theory targets for debugging
theories/Prelude.vo: theories/Prelude.v
	coqc -Q theories ReifiedRSpaces $<
