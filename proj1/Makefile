# Makefile

UID1=704534585
UID2=604721196
UID3=104778067
PROJ=Project1

.PHONY: dist clean run

run:
	python main-network.py
	python main-housing.py
	
dist:
	zip -X $(PROJ)_$(UID1)_$(UID2)_$(UID3).zip *.py Project*Report.pdf *.csv README Makefile

# removes all files generated by make and vim's tmp files
clean:
	rm -f *.pyc *~
