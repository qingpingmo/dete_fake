#include<stdio.h>
#include<time.h>
#include<math.h>
#include<string.h>
#include<malloc.h>
#define VEC_SIZE 64
#define DATA_SIZE   32*1024*1024
	
typedef struct {
float data[VEC_SIZE];
} float_vec;
	
void memAccess(float_vec * A_in, float_vec * B_out, unsigned data_num){
	    float_vec temp;
	    for(unsigned i=0; i<data_num; i++){
	      temp = A_in[i];
	      B_out[i] = temp;
	    }
}
	
int main(){
	struct timespec ts1, ts2;
	float_vec * A_in;
	float_vec * B_out;
	float time_ms = 0.0;
	int time_s = 0;
	
    unsigned data_num = DATA_SIZE/VEC_SIZE;

	A_in = (float_vec*)malloc(data_num*sizeof(float_vec));
	B_out = (float_vec*)malloc(data_num*sizeof(float_vec));
	memset(A_in, 0.0, sizeof(A_in));
	
	clock_gettime(CLOCK_REALTIME, &ts1);

    memAccess(A_in,B_out,data_num);
	//memcpy(B_out, A_in, sizeof(A_in));

	clock_gettime(CLOCK_REALTIME, &ts2);
	time_ms = (ts2.tv_nsec-ts1.tv_nsec) / 1000000000.0;
	time_s = ts2.tv_sec - ts1.tv_sec;

	float transfer_rate = 0.0;
	transfer_rate = DATA_SIZE /( (time_ms + time_s) * pow(10, 6));
	printf("------this test is examined by LiManYi 21281133--------\n");
	printf("Transfer Rate = %.6f MByte/s when VEC_SIZE=%d \n", transfer_rate, VEC_SIZE);
}
