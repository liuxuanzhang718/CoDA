WORK_DIR="."
LOCAL_DIR=$WORK_DIR/data

mkdir -p $LOCAL_DIR

DATA=nq,hotpotqa
python $WORK_DIR/preprocess/data_process/qa_search_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA 
echo "Train Data: $DATA" >> $LOCAL_DIR/datasource.txt

## process multiple dataset search format test file
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/preprocess/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA --filename "valid_500" --n_subset 500
echo "Valid Data: $DATA" >> $LOCAL_DIR/datasource.txt

DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/preprocess/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA --filename "test"
echo "Test Data: $DATA" >> $LOCAL_DIR/datasource.txt