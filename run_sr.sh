FALSE=0
TRUE=1

WORKROOT='workdir'
NET_DATASET='EDSR'
LOG_DIR=${WORKROOT}/${NET_DATASET}/log
mkdir -p ${FULL_DIR} ${LOG_DIR}

TIME_TAG=`date +"%Y%m%d_%H%M"`
LOG_FILE=${LOG_DIR}/${TIME_TAG}.txt
python -u main_sr.py 2>&1 | tee ${LOG_FILE}
