#!/bin/bash

go() (
	cd $1
	jupyter nbconvert --to pdf Report.ipynb
	pdfunite assets/report-head.pdf Report.pdf report.pdf	
	rm Report.pdf
)


for l in lab*; do
	go $l
done
