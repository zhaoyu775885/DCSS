#! /bin/bash

clean_recursive() {
    echo ${1}
    rm -rf ${1}/*~ ${1}/.*~ ${1}/__pycache__
    for file in ${1}/*
    do
        if [ -d ${file} ]
        then
            clean_recursive ${file}
        fi
    done
}

clean_recursive .
