#!/bin/bash
#
cd ../src/
filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint --rcfile .pylintrc $eachfile
done

cd ../src/models/
filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint --rcfile .pylintrc $eachfile
done
