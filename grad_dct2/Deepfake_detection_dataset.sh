



# from https://github.com/peterwang512/CNNDetection
wget http://10.126.56.37:8080/directlink/1/CNNDetection/CNN_synth_testset.zip
mkdir CNN
tar -zxvf CNN_synth_testset.zip -C CNN


# from https://github.com/Ekko-zn/AIGCDetectBenchmark
wget http://10.126.56.37:8080/directlink/1/CNNDetection/AIGCDetect_testset.zip
mkdir AIGC
tar -zxvf AIGCDetect_testset.zip -C AIGC

# https://github.com/chuangchuangtan/NPR-DeepfakeDetection
wget http://10.126.56.37:8080/directlink/1/CNNDetection/Diffusion1kStep.tar.gz
mkdir Diff
tar -zxvf Diffusion1kStep.tar.gz -C Diff

# from https://github.com/ZhendongWang6/DIRE
wget http://10.126.56.37:8080/directlink/1/CNNDetection/dire_googledrive.tar.gz
mkdir dire
tar -zxvf dire_googledrive.tar.gz -C dire

# from https://github.com/chuangchuangtan/GANGen-Detection
wget http://10.126.56.37:8080/directlink/1/CNNDetection/GANGen-Detection-tar.tar.gz
mkdir gan
tar -zxvf GANGen-Detection-tar.tar.gz -C gan

# https://github.com/Yuheng-Li/UniversalFakeDetect
wget http://10.126.56.37:8080/directlink/1/CNNDetection/UniversalFakeDetect_test.tar.gz
mkdir fake
tar -zxvf UniversalFakeDetect_test.tar.gz -C fake