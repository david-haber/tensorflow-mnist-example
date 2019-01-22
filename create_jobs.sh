# create job.yaml files
for i in $(eval echo {1..$1})
do
  cat mnist-job-template.yml | sed "s/\$ITEM/$i/" > ./hyperparam-jobs-specs/mnist-train-job-$i.yml
done
