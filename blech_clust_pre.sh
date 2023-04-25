DIR=$1
echo Running Blech Clust 
python blech_clust.py $DIR &&
echo Running Common Average Reference 
python blech_common_avg_reference.py $DIR &&
echo Running Jetstream Bash 
for x in $(seq 10);do bash blech_clust_jetstream_parallel.sh;done &&
echo Running UMAP 
bash bash_umap_spike_scatter.sh;
#rsync -avP $DIR/$(basename $DIR).h5 $DIR/$(basename $DIR).copy
