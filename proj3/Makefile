# Makefile

UID1=704534585
UID2=604721196
UID3=104778067
PROJ=Project3

.PHONY: dist clean run

run:
	
dist:
	zip -X $(PROJ)_$(UID1)_$(UID2)_$(UID3).zip generate_out_csv.py Project*Report.pdf matlab/* matlab/nmfv1_4/* ml-latest-small/ratings.csv README

clean:
	rm -f *.pyc *~ *.zip

clobber: clean
	rm -f *.pdf *.png
