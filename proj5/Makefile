# Makefile

UID1=704534585
UID2=604721196
UID3=104778067
PROJ=Project5

.PHONY: dist clean run

run:
	python part1.py
	# part 2
	python part2n3.py --question=2
	# part 3
	python part2n3.py --savefeat --question=3
	python part4.py
	python part5.py

retrain:
	# notice: this will change your saved feature files
	python part2n3.py --savefeat --question=3 window=6
	python part4.py
	python part5.py
	
dist:
	zip -X $(PROJ)_$(UID1)_$(UID2)_$(UID3).zip Project*Report.pdf README Makefile *.py

clean:
	rm -f *.pyc *~ model_stats_*.txt scatter_*.png features_*.csv targets_*.csv

clobber: clean
	rm -f *.png *.zip
