
Rootdir=$1

dirs='biggan crn cyclegan deepfake gaugan imle progan san seeingdark stargan stylegan stylegan2 whichfaceisreal'

echo $Rootdir
cd $Rootdir
ls -lR|grep "^-"| wc -l
for dir in ${dirs}
do
 cd ${Rootdir}/${dir}
 echo ${dir} 
 ls -lR|grep "^-"| wc -l
done


Rootdir=/opt/data/private/tcc/data/data/CNNDetection/test

dirs='biggan crn cyclegan deepfake gaugan imle progan san seeingdark stargan stylegan stylegan2 whichfaceisreal'
cd $Rootdir
ls -lR|grep "^-"| wc -l
echo $Rootdir
for dir in ${dirs}
do
 cd ${Rootdir}/${dir}
 echo ${dir} 
 ls -lR|grep "^-"| wc -l
done



# Rootdir=/opt/data/private/tcc/data/data/CNNDetection/progan_train

# dirs='airplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'
# cd $Rootdir
# ls -lR|grep "^-"| wc -l
# echo $Rootdir
# for dir in ${dirs}
# do
 # cd ${Rootdir}/${dir}
 # echo ${dir} 
 # ls -lR|grep "^-"| wc -l
# done

# Rootdir=$2

# dirs='airplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'
# cd $Rootdir
# ls -lR|grep "^-"| wc -l
# echo $Rootdir
# for dir in ${dirs}
# do
 # cd ${Rootdir}/${dir}
 # echo ${dir} 
 # ls -lR|grep "^-"| wc -l
# done