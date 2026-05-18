#include "ds4_on_pico_w.hpp"
#include "pico/stdlib.h"

#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "pico/binary_info.h"
#include "hardware/spi.h"
#include "hardware/timer.h"

#define LOG_HEADER "[main]"

/* ---------- define for MPU9250 ---------- */
#define SPI_PORT_MPU spi1
#define READ_BIT 0x80

#define PIN_MISO_MPU 8
#define PIN_CS_MPU   9
#define PIN_SCK_MPU  10
#define PIN_MOSI_MPU 11
/* ---------------------------------------- */

/* ----------- define for L6470 ----------- */
#define SPI_PORT_L6470 spi0
#define Front_BIT 0x51
#define Back_BIT 0x50
#define Stop_BIT 0xB0

#define PIN_MISO_L6470 3
#define PIN_CS_L6470 5
#define PIN_SCK_L6470 2
#define PIN_MOSI_L6470 4

// defining parameter addresses
#define L6470_ABS_POS       0x01 // 現在位置
#define L6470_EL_POS        0x02 // 電気的位置
#define L6470_MARK          0x03 // マーク（絶対）位置
#define L6470_SPEED         0x04 // 現在の速度
#define L6470_ACC           0x05 // 加速度
#define L6470_DEC           0x06 // 減速度
#define L6470_MAX_SPEED     0x07 // 最大速度
#define L6470_MIN_SPEED     0x08 // 最小速度
#define L6470_FS_SPD        0x15 // フルステップ速度のしきい値
#define L6470_KVAL_HOLD     0x09 // 待機中のKVAL値!
#define L6470_KVAL_RUN      0x0A // 定速のKVAL値!
#define L6470_KVAL_ACC      0x0B // 加速開始のKVAL値!
#define L6470_KVAL_DEC      0x0C // 減速開始のKVAL値!
#define L6470_INT_SPD       0x0D // 傾斜の交差速度 
#define L6470_ST_SLP        0x0E // 始動スロープ!
#define L6470_FN_SLP_ACC    0x0F // 加速終端スロープ!
#define L6470_FN_SLP_DEC    0x10 // 減速終端スロープ!
#define L6470_K_THERM       0x11 // 温度補償定数
#define L6470_ADC_OUT       0x12 // ADC出力
#define L6470_OCD_TH        0x13 // 過電流しきい値!
#define L6470_STALL_TH      0x14 // 失速（脱調）のしきい値!
#define L6470_STEP_MODE     0x16 // ステップモード!
#define L6470_ALARM_EN      0x17 // アラーム有効
#define L6470_CONFIG        0x18 // コンフィグ!
#define L6470_STATUS        0x19 // ステータス
/* ---------------------------------------- */

/* --------- define used for calc ---------- */
#define accel_se     0.061
#define gyro_se      0.00763    // for gyro value correction
#define micro        0.000001   // pow(10.0, -6.0)
#define Rad          0.0174532  // for RAD conversion
#define Step_convert 0.0149011  // For converting the input value to the motor to step/s
#define feedback     0.1        // for feedback
#define feed_step    71205      // 内部速度から時間あたりのステップ数??への変換用

#define MAX_SPEED    5000      // Max motor output
#define START_SPEED  2000       // min motor output
#define VST          0.3        // target velocity of the sphere(m/s)
#define Wheel_R      30         // Enter wheel radius in centimeters(ホイールの半径、センチで入力)
#define Sphere_R     150        // Enter sphere radius in centimeters(球体の半径、センチで入力)
#define milli        0.001      // convert centimeters to meters
/* ----------------------------------------- */

enum AutoMoveState {
    MOVE_FORWARD,
    WAIT_1,
    MOVE_LEFT,
    WAIT_2,
    MOVE_DIAGONAL,
    WAIT_3,
    STOP
};

AutoMoveState auto_state = STOP;
uint32_t auto_start_time = 0;


static inline void cs_select_MPU() 
{
    asm volatile("nop \n nop \n nop");
    gpio_put(PIN_CS_MPU, 0);  // Active low
    asm volatile("nop \n nop \n nop");
}

static inline void cs_deselect_MPU() 
{
    asm volatile("nop \n nop \n nop");
    gpio_put(PIN_CS_MPU, 1);
    asm volatile("nop \n nop \n nop");
}

static inline void cs_select_L6470()
{
    asm volatile("nop \n nop \n nop");
    gpio_put(PIN_CS_L6470, 0);  // Active low
    asm volatile("nop \n nop \n nop");
}

static inline void cs_deselect_L6470()
{
    asm volatile("nop \n nop \n nop");
    gpio_put(PIN_CS_L6470, 1);
    asm volatile("nop \n nop \n nop");
}

//センサのリセット信号を送る関数
static void mpu9250_reset() 
{
    // Two byte reset. First byte register, second byte data
    // There are a load more options to set up the device in different ways that could be added here
    uint8_t buf[] = {0x6A, 0x03};
    cs_select_MPU();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_MPU, buf, 2);
    cs_deselect_MPU();
    sleep_us(1);

    buf[0]=0x6B;
    buf[1]=0x00;
    cs_select_MPU();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_MPU, buf, 2);
    cs_deselect_MPU();
    sleep_us(1);

    buf[0]=0x6C;
    buf[1]=0x00;
    cs_select_MPU();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_MPU, buf, 2);
    cs_deselect_MPU();
    sleep_us(1);

    buf[0]=0x1B;
    buf[1]=0x00;
    cs_select_MPU();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_MPU, buf, 2);
    cs_deselect_MPU();
    sleep_us(1);

    buf[0]=0x1C;
    buf[1]=0x00;
    cs_select_MPU();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_MPU, buf, 2);
    cs_deselect_MPU();
    sleep_us(1);
}

//センサのレジスタのIDを読み出す関数
static void read_registers(uint8_t reg, uint8_t *buf, uint16_t len) 
{
    reg |= READ_BIT;
    cs_select_MPU();
    spi_write_blocking(SPI_PORT_MPU, &reg, 1);
    sleep_us(1);
    spi_read_blocking(SPI_PORT_MPU, 0, buf, len);
    cs_deselect_MPU();
    sleep_us(1);
}

//センサの加速度、ジャイロ、温度を読み出す関数
static void mpu9250_read_raw1(int16_t acceleration2[], int16_t gyro2[], int16_t *temp2) 
{
    uint8_t buffer[6];

    // Start reading acceleration registers from register 0x3B for 6 bytes
    read_registers(0x3B, buffer, 6);

    for (int i = 0; i < 3; i++)
    {
        acceleration2[i] = (buffer[i * 2] << 8 | buffer[(i * 2) + 1]);
    }

    // Now gyro data from reg 0x43 for 6 bytes
    read_registers(0x43, buffer, 6);

    for (int i = 0; i < 3; i++)
    {
        gyro2[i] = (buffer[i * 2] << 8 | buffer[(i * 2) + 1]);
    }

    // Now temperature from reg 0x41 for 2 bytes
    read_registers(0x41, buffer, 2);

    *temp2 = buffer[0] << 8 | buffer[1];
}


//センサの加速度、ジャイロ、温度を読み出す関数(offset)
static void mpu9250_read_raw2(int32_t gy_offset[], int16_t acceleration[], int16_t gyro[], int16_t& temp) 
{
    uint8_t buffer[6];

    // Start reading acceleration registers from register 0x3B for 6 bytes
    read_registers(0x3B, buffer, 6);

    for (int i = 0; i < 3; i++) {
        acceleration[i] = (buffer[i * 2] << 8 | buffer[(i * 2) + 1]);//[0,1,2]=[x,y,z]　他のセンサ値も同様
    }

    // Now gyro data from reg 0x43 for 6 bytes
    read_registers(0x43, buffer, 6);

    for (int i = 0; i < 3; i++) {
        gyro[i] = (buffer[i * 2] << 8 | buffer[(i * 2) + 1]);;
    }

    // Now temperature from reg 0x41 for 2 bytes
    read_registers(0x41, buffer, 2);

    temp = buffer[0] << 8 | buffer[1];
    
    for(int i = 0; i<3 ; i++){
        //acceleration[i] -= ac_offset[i];
        gyro[i] -= gy_offset[i];
    }
}


void mpu9250_calibration(int32_t gy_offset[], int16_t acceleration2[], int16_t gyro2[], int16_t *temp2)
{
    /*printf("---calibration() Start-----\n");
    printf("ac_offset0. X = %6d, Y = %6d, Z = %6d\n",ac_offset[0],ac_offset[1],ac_offset[2]);
    printf("gy_offset0. X = %6d, Y = %6d, Z = %6d\n",gy_offset[0],gy_offset[1],gy_offset[2]);   */
    for(int t=0; t<10000; t++){
        mpu9250_read_raw1(acceleration2, gyro2, temp2);
        for(int i=0;i<3;i++){
            //ac_offset[i] += acceleration2[i]; 
            gy_offset[i] += gyro2[i];
        }
        /*if (t%1000 == 0){
            printf("acce1eration. X = %6d, Y = %6d, Z = %6d\n",acceleration[0],acceleration[1],acceleration[2]);
            printf("gyro. X = %6d, Y = %6d, Z = %6d\n",gyro[0],gyro[1],gyro[2]);
            printf("ac_offset1. X = %6d, Y = %6d, Z = %6d\n",ac_offset[0],ac_offset[1],ac_offset[2]);
            printf("gy_offset1. X = %6d, Y = %6d, Z = %6d\n",gy_offset[0],gy_offset[1],gy_offset[2]);   
        }   */
    }
    for(int i=0;i<3;i++){
        //ac_offset[i] = ac_offset[i]/1000;
        gy_offset[i] = gy_offset[i]/10000;
    }
    /*printf("ac_offset2. X = %3f, Y = %3f, Z = %3f\n",ac_offset[0],ac_offset[1],ac_offset[2]);
    printf("gy_offset2. X = %3f, Y = %3f, Z = %3f\n",gy_offset[0],gy_offset[1],gy_offset[2]);
    printf("---calibration() Finish-----\n");   */
}

//L6470リセット時に使用する関数。ただ信号送信するだけ。
void L6470_setting(uint8_t buf[3],uint8_t value[3])
{
    cs_select_L6470();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_L6470,buf,3);
    cs_deselect_L6470();
    sleep_us(1);

    cs_select_L6470();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_L6470,value,3);
    cs_deselect_L6470();
    sleep_us(1);

    return;
}

//モータドライバのリセット用関数
void L6470_reset()  
{
    uint8_t buf[3]={0xC0,0xC0,0xC0};//レジスタアドレス用
    uint8_t value[3]={0x00,0x00,0x00};//レジスタ値入力用

    L6470_setting(buf,value);

    //Kvalの設定　詳しくはL6470データシートp38
    //0x29=1.92V
    //0x36=2.5V
    //0x40=3V
    //0x0A 稼働時のKval設定
    
    for(int i = 0; i < 3; i++){//最大速度設定
        buf[i] = L6470_MAX_SPEED;
        value[i]=0x03;
    }
    cs_select_L6470();//送るバイト数が18bitなので、関数を使わず手動でデータ送信。
    sleep_us(1);
    spi_write_blocking(SPI_PORT_L6470,buf,3);
    cs_deselect_L6470();
    sleep_us(1);

    cs_select_L6470();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_L6470,value,3);
    cs_deselect_L6470();
    sleep_us(1);
    for(int i = 0; i < 3; i++){
        value[i]=0xFF;
    }
    cs_select_L6470();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_L6470,value,3);
    cs_deselect_L6470();
    sleep_us(1);

    for(int i = 0; i < 3; i++){//停止時のKval値設定
        buf[i] = L6470_KVAL_HOLD;
        value[i]=0x5A;
    }
    L6470_setting(buf,value);

    for(int i = 0; i < 3; i++){//走行時のKval値設定
        buf[i] = L6470_KVAL_RUN;
    }
    L6470_setting(buf,value);

    for(int i = 0; i < 3; i++){//加速時のKval値設定
        buf[i] = L6470_KVAL_ACC;
    }
    L6470_setting(buf,value);

    for(int i = 0; i < 3; i++){//減速時のKval値設定
        buf[i] = L6470_KVAL_DEC;
    }
    L6470_setting(buf,value);

    for(int i = 0; i < 3; i++){//始動の傾斜の設定
        buf[i]=0x0E;
        value[i]=0x19;
    }
    L6470_setting(buf,value);

    for(int i = 0; i < 3; i++){//加速の最終傾斜の設定
        buf[i]=0x0F;
        value[i]=0x29;
    }
    L6470_setting(buf,value);

    for(int i = 0; i < 3; i++){//減速の最終傾斜の設定
        buf[i]=0x10;
        value[i]=0x29;
    }
    L6470_setting(buf,value);

    for(int i = 0; i < 3; i++){//ステップモードの設定
        buf[i]=0x16;
        value[i]=0x07;
    }
    L6470_setting(buf,value);

    for(int i = 0; i < 3; i++){//過電流のしきい値の設定
        buf[i]=0x13;
        value[i]=0x0F;
    }

    L6470_setting(buf,value);

    for(int i = 0; i < 3; i++){//失速（脱調）のしきい値の設定
        buf[i]=L6470_STALL_TH;
        value[i]=0x40;
    }

    L6470_setting(buf,value);

    for(int i = 0; i < 3; i++){//デバイスの設定
        buf[i]=0x18;
        value[i]=0x2E;
    }

    cs_select_L6470();//送るバイト数が16bitなので、関数を使わず手動でデータ送信。
    sleep_us(1);
    spi_write_blocking(SPI_PORT_L6470,buf,3);
    cs_deselect_L6470();
    sleep_us(1);

    cs_select_L6470();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_L6470,value,3);
    cs_deselect_L6470();
    sleep_us(1);

    for(int i = 0; i < 3; i++){
        value[i]=0x88;
    }

    cs_select_L6470();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_L6470,value,3);
    cs_deselect_L6470();
    sleep_us(1);

    for(int i = 0; i < 3;i++)   //モータの加速度の設定
    {
        buf[i]=0x05;
        value[i]=0x20;
    }
    L6470_setting(buf,value); 
}

//w1,2,3は回転スピード
void L6470_move(int w1,int w2,int w3) 
{
    uint8_t rotate[]={Stop_BIT,Stop_BIT,Stop_BIT};//モーターの正転、逆転を管理する行列
    uint8_t nullsp[]={0,0,0};//スピード送信時にはじめに送る空の行列
    uint8_t firstsp[]={0,0,0};//スピード送信用の行列 始めの8bit
    uint8_t lastsp[]={0,0,0};//後ろの8bit 合計16bit
    uint16_t speed[]={0,0,0};//回転のスピード,最大値65535まで
    //正ならば正回転、負ならば逆回転、0ならば停止
    if(w1>0){rotate[0]=Front_BIT;}
    else if(w1<0){rotate[0]=Back_BIT;}
    else{rotate[0]=Stop_BIT;}

    speed[0]=abs(w1);

    if(w2>0){rotate[1]=Front_BIT;}
    else if(w2<0){rotate[1]=Back_BIT;}
    else{rotate[1]=Stop_BIT;}

    speed[1]=abs(w2);

    if(w3>0){rotate[2]=Front_BIT;}
    else if(w3<0){rotate[2]=Back_BIT;}
    else{rotate[2]=Stop_BIT;}

    speed[2]=abs(w3);
    
    for(int i=0;i<3;i++)
    {
        firstsp[i]= speed[i] >> 8;//16bitのデータを8つ分だけ右にシフトさせて、頭の8bitだけを取り出した。
        lastsp[i]= (255)&(speed[i]);//16bitのデータとb'00001111'の論理積をして後ろの8bitだけ取り出した。
    }

    /*
    printf("F%x\n",firstsp[0]);
    printf("L%x\n",lastsp[0]);
    printf("R%x\n",rotate[0]);

    printf("F%x\n",firstsp[1]);
    printf("L%x\n",lastsp[1]);
    printf("R%x\n",rotate[1]);

    printf("F%x\n",firstsp[2]);
    printf("L%x\n",lastsp[2]);
    printf("R%x\n",rotate[2]);
    */

    cs_select_L6470();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_L6470,rotate,3);
    cs_deselect_L6470();
    sleep_us(1);

    cs_select_L6470();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_L6470,nullsp,3);
    cs_deselect_L6470();
    sleep_us(1);

    cs_select_L6470();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_L6470,firstsp,3);
    cs_deselect_L6470();
    sleep_us(1);

    cs_select_L6470();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_L6470,lastsp,3);
    cs_deselect_L6470();
}

void L6470_manualstop(int& Ypulse_num, int& Xpulse_num)
{
    uint8_t buf[] = {Stop_BIT, Stop_BIT, Stop_BIT};
    cs_select_L6470();
    sleep_us(1);
    spi_write_blocking(SPI_PORT_L6470, buf, 3);
    cs_deselect_L6470();

    Xpulse_num = 0;
    Ypulse_num = 0;
}


void calculate_XY_Velocity(float direction, float Vst, float& Vst_y, float& Vst_x)
{
    // convert angle to radians(角度をラジアンに変換)
    float angle_in_radians = direction * M_PI / 180.0;

    // calculate the components of X and Y(XとYの成分を計算)
    Vst_x = Vst * cos(angle_in_radians);
    Vst_y = Vst * sin(angle_in_radians);
}

void P_ctrl_Y(float Kp, float Vst_Y, float Vsy, float& cor_P)
{

    float Vdiff_Y  = 0;    // deviation (偏差)
    Vdiff_Y = Vsy - Vst_Y;  // Deviation between the target speed and current speed of the sphere (球体の目標速度と現在速度の偏差)
    cor_P   = Kp * Vdiff_Y; // Determine the amount of operation correction (操作補正量の決定)

}
void P_ctrl_X(float Kp, float Vst_X, float Vsx, float& cor_P)
{
    float Vdiff_X  = 0;    // deviation (偏差)
    Vdiff_X = Vsx - Vst_X;  // Deviation between the target speed and current speed of the sphere (球体の目標速度と現在速度の偏差)
    cor_P   = Kp * Vdiff_X; // Determine the amount of operation correction (操作補正量の決定)
}

void D_ctrl_Y(float Kd, float Vst_Y, float Vsy, float& cor_D)
{ 
    static uint32_t prev_time = time_us_32(); // ループ開始前の時間（staticで一度だけ宣言）
    uint32_t now = time_us_32();
    float deltaT = (now - prev_time) / 1e6f; // 秒に変換（μs → s）
    prev_time = now; // 次回用に保存
    //const float deltaT = 450 * pow(10.0, -9.0); //1 cycle of program[μs](プログラムの1周期)
    static float Vdiff_X_before = 0;
    float Vdiff_X = 0; //偏差
    float Vdd_X = 0;
    Vdiff_X = Vsy - Vst_Y; //偏差
    Vdd_X = Vdiff_X - Vdiff_X_before;
    cor_D = Kd * ((Vdd_X) / deltaT);
    Vdiff_X_before = Vdiff_X;
}
void D_ctrl_X(float Kd, float Vst_X, float Vsx, float& cor_D)
{ 
    static uint32_t prev_time = time_us_32(); // ループ開始前の時間（staticで一度だけ宣言）
    uint32_t now = time_us_32();
    float deltaT = (now - prev_time) / 1e6f; // 秒に変換（μs → s）
    prev_time = now; // 次回用に保存
    //const float deltaT = 450 * pow(10.0, -9.0); //1 cycle of program[μs](プログラムの1周期)
    static float Vdiff_X_before = 0;
    float Vdiff_X = 0; //偏差
    float Vdd_X = 0;
    Vdiff_X = Vsx - Vst_X; //偏差
    Vdd_X = Vdiff_X - Vdiff_X_before;
    cor_D = Kd * ((Vdd_X) / deltaT);
    Vdiff_X_before = Vdiff_X;
}

void cor_tot(float cor_P_Y, float cor_D_Y, float cor_P_X, float cor_D_X, float& cor_PD_Y, float& cor_PD_X)
{
    // Initial value setting(初期値設定)
    static float previous_cor_PD_Y = 0;//(前回のcor_tot)
    static float previous_cor_PD_X = 0;//(前回のcor_tot)
    cor_PD_Y = cor_P_Y + cor_D_Y + previous_cor_PD_Y;
    cor_PD_X = cor_P_X + cor_D_X + previous_cor_PD_X;
    previous_cor_PD_Y = cor_PD_Y;
    previous_cor_PD_X = cor_PD_X;
}

void motor_calc(float cor_PD_Y, float cor_PD_X, int& PD_w1, int& PD_w2, int& PD_w3, int& synt_vec_Y, int& synt_vec_X)
{
    int SAC_fin = -cor_PD_Y + (-cor_PD_X);

    //おいらが調べたやつ、マイナス方向の制御に難あり
    //PD_w1 = (SAC_fin * (cos(direction) * (1.0 / 2.0) + sin(direction) * (sqrt(3.0) / 2.0)));
    //PD_w2 = (SAC_fin * (cos(direction) * (1.0 / 2.0) - sin(direction) * (sqrt(3.0) / 2.0)));
    //PD_w3 = (-SAC_fin * cos(direction));

    //富田氏の球体移動機構の制御式（修論本体:p25.(3.4.3式参照)）
    PD_w1 = -cor_PD_X * (1.0 / 2.0) + (-cor_PD_Y) * (sqrt(3.0) / 2.0);
    PD_w2 = -cor_PD_X * (1.0 / 2.0) - (-cor_PD_Y) * (sqrt(3.0) / 2.0); 
    PD_w3 = cor_PD_X;
    return;
}

void vib_ctrl_Y(float degree, int& w1, int& w2)
{   
    if (-2 < degree && degree < 2)
    {
        w1 = 0;
        w2 = 0;
    }
    else
    {
        w1 = -2000 * 9.8 * sin(degree * Rad) * (sqrt(3.0/2.0));
        w2 = -w1;
    }
}
void vib_ctrl_X(float degree, int& w1, int& w2, int& w3)
{   
    if (-2 < degree && degree < 2)
    {
        w1 = 0;
        w2 = 0;
        w3 = 0;
    }
    else
    {
        w1 = 2000 * 9.8 * sin(degree * Rad) * (1.0 / 2.0);
        w2 = w1;
        w3 = -2000 * 9.8 * sin(degree * Rad);
    }
}

void yaw_angle_control(float Target_angle, float current_angle, float yaw_Kp, float yaw_kd, int& yaw_output)
{
    float angle_diff = Target_angle - current_angle;
    if (angle_diff > -1 && angle_diff < 1) // 目標角度と現在角度の差が±1°以内ならば、0にする
    {
        angle_diff = 0;
    }
    // 差分を ±180° にラップ（これが超重要！）
    if (angle_diff > 180.0f)  angle_diff -= 360.0f;
    if (angle_diff < -180.0f) angle_diff += 360.0f;

    // 微分制御用の時間差
    static uint32_t prev_time = time_us_32();
    uint32_t now = time_us_32();
    float deltaT = (now - prev_time) / 1e6f;
    prev_time = now;

    // 微分成分
    static float angle_diff_before = 0;
    float angle_dd = angle_diff - angle_diff_before;
    angle_diff_before = angle_diff;

    // P + D制御
    float result = yaw_Kp * angle_diff + yaw_kd * (angle_dd / deltaT);
    if (result > MAX_SPEED) result = MAX_SPEED;
    if (result < -MAX_SPEED) result = -MAX_SPEED;
    
    // intに変換（範囲制限は必要ならここで）
    yaw_output = -(int)result;
}



void motor_ctrl(int PD_w1, int PD_w2, int PD_w3, int w1, int w2, int w3, int yaw_output, int& synt_vec_Y, int& synt_vec_X)
{
    int fin_w1 = PD_w1 + w1 + yaw_output;
    int fin_w2 = PD_w2 + w2 + yaw_output;
    int fin_w3 = PD_w3 + w3 + yaw_output;
    L6470_move(fin_w1, fin_w2, fin_w3);
    synt_vec_Y = fin_w1 + (-fin_w2);
    synt_vec_X = fin_w1 + fin_w2 + (-fin_w3);
}

void degree_calc(int16_t acceleration[], int16_t gyro[], float& Vgx, float& Vgy, float& comp_x, float& comp_y, float& angle_gz)
{
    static float Roll = 0, Pitch = 0, Yaw = 0;
    static float previous_comp_x = 0, previous_comp_y = 0;
    static float angle_gx;
    static float angle_gy;
    static float Vgz;

    static uint32_t timestamp = time_us_32();
    uint32_t now = time_us_32();
    float dt = (now - timestamp) / 1e6f;
    timestamp = now;

    const float k = 0.985;
    const float DEG_TO_RAD = M_PI / 180.0f;

    // 角加速度の補正
    float gx = gyro[0] * gyro_se;
    float gy = gyro[1] * gyro_se;
    float gz = gyro[2] * gyro_se;

    // Roll, Pitch 角度変化
    float gx_num = (gx + sin(Roll) * tan(Pitch) * gy + cos(Roll) * tan(Pitch) * gz);
    float gy_num = cos(Roll) * gy - sin(Roll) * gz;

    Vgx = gx_num * DEG_TO_RAD;
    Vgy = gy_num * DEG_TO_RAD;

    gx_num *= dt;
    gy_num *= dt;

    angle_gx += gx_num;
    angle_gy += gy_num;

    // Yaw の更新
    float delta_yaw = gz * dt;
    angle_gz += delta_yaw;

    // 相補フィルタ
    float ax = -atan2(-acceleration[1], acceleration[2]) * 180.0f / M_PI;
    float ay = -atan2(acceleration[0], sqrt(acceleration[1]*acceleration[1] + acceleration[2]*acceleration[2])) * 180.0f / M_PI;

    comp_x = k * (previous_comp_x + gx_num) + (1 - k) * ax;
    comp_y = k * (previous_comp_y + gy_num) + (1 - k) * ay;

    previous_comp_x = comp_x;
    previous_comp_y = comp_y;
    angle_gz = fmodf(angle_gz, 360.0f); // 余り（-360〜360）
    if (angle_gz < 0.0f)
        angle_gz += 360.0f; // 負の角度を正の値に変換
}

void sphere_speed(int synt_vec_Y, int synt_vec_X, float Vgy, float Vgx, float& Vsy, float& Vsx)
{
    /*ただの入力からstep数へ変換したあと、1stepあたりの角度である1.8を掛けて
    度へと変える*/
    float Vry = synt_vec_Y * Step_convert * 1.8;
    float Vrx = synt_vec_X * Step_convert * 1.8; 
    //タイヤの半径を使って速度[m/s]へと変換
    Vry = Vry * Rad * Wheel_R * milli;
    Vrx = Vrx * Rad * Wheel_R * milli;
    //ジャイロセンサの出力値は[°/s]なので[rad/s]に変換
    Vgy = Vgy * Rad;
    Vgx = Vgx * Rad;
    Vsy = Vry + (Sphere_R * milli * Vgx);
    Vsx = Vrx + (Sphere_R * milli * Vgy);
}

/* ============================ Controller functions ============================ */

/**
 * @brief DualShock 4のスティック状態を保持する構造体  
 * Structure to hold the state of the DualShock 4 analog sticks.
 * * @details  
 * L3スティックとR3スティックのX/Y軸の値（正規化済み）を保持します。  
 * Holds normalized values (-1.0 to 1.0) for the X and Y axes of the L3 and R3 sticks.
 */
struct StickState 
{
    float l3_x;
    float l3_y;
    float r3_x;
    float r3_y;
};

/**
 * @brief Normalize the value to a range of -1.0 to 1.0.
 * @param value The value to normalize.
 * @return float Normalized value.
 */
float normalize_stick_axis(int value) 
{
    return (value - 127.5f) / 127.5f;
}

/**
 * @brief 
 * DualShock 4のスティック入力を正規化して返します。
 * 与えられた `DualShock4_state` 構造体の各スティック軸（L3およびR3）を-1.0 ～ 1.0 の範囲に正規化し、`StickState` 構造体として返します。
 * * Normalize DualShock 4 stick input values and return them.
 * Normalize each stick axis (L3 and R3) from the given `DualShock4_state`  
 * to the range -1.0 to 1.0 and return as a `StickState` structure.
 * * @param state 
 * DualShock 4のスティック生データ（整数）Raw stick input values from DualShock 4 (integers)
 * * @return 
 * tickState 正規化されたスティックの状態（float） Normalized stick state as floating point values
 */
StickState normalize_sticks(const DualShock4_state& state) 
{
    return 
    {
        .l3_x = normalize_stick_axis(state.l3_x),
        .l3_y = normalize_stick_axis(state.l3_y),
        .r3_x = normalize_stick_axis(state.r3_x),
        .r3_y = normalize_stick_axis(state.r3_y),
    };
}

float Calc_target_yaw_from_rstick(float r3_x, float r3_y)
{
    // 右スティックのx/yから角度を算出（前が0°）
    float angle = atan2f(-r3_x, -r3_y) * (180.0f / M_PI); // Y軸前方向を0°に
    if (angle < 0) angle += 360.0f; // 0〜360° に正規化
    return angle;
}

/* ============================================================================== */
    

int main()
{
    float comp_x = 0, comp_y = 0;//機体の角度
    float Vgx = 0, Vgy = 0;
    float Vsy = 0, Vsx = 0;
    int Ypulse_num = 0, Xpulse_num = 0;
    float Vst_y = 0,Vst_x = 0;

    int32_t gy_offset[3];
    int16_t acceleration[3], gyro[3], temp;
    int16_t acceleration2[3], gyro2[3], temp2;

    float cor_P_Y = 0, cor_D_Y = 0, cor_P_X = 0, cor_D_X = 0;
    float cor_PD_Y = 0, cor_PD_X = 0;

    int synt_vec_Y = 0, synt_vec_X = 0; //synthetic vector(合成ベクトル)

    float target_yaw = 0, save_yaw = 0, Yaw = 0;
    int yaw_output = 0;

    //Motor rotation speed in the direction of travel based on PD control correction amount
    int PD_w1 = 0, PD_w2 = 0, PD_w3 = 0; 

    uint32_t timestamp = 0;

    // --- フラグ・計測・自動制御用変数の追加 ---
    bool is_command_mode = false;
    float Kp_value = 25.0f;
    float Kd_value = 0.00165f;
    float adjust_step = 1.0f;
    enum ParamTarget { KP, KD };
    ParamTarget current_target = KP;

    uint32_t measure_start_time = 0;
    bool is_measuring = false;
    int control_pattern = 0; // 1:PDのみ, 2:PD+制振Y, 3:PD+制振2D

    bool is_auto_mode = false;
    uint32_t auto_start_time = 0;
    enum AutoState { STOP, MOVE_FORWARD, WAIT_1, MOVE_LEFT, WAIT_2, MOVE_DIAGONAL, WAIT_3 };
    AutoState auto_state = STOP;

    DS4forPicoW controller;
    bool is_stick_moving = false;
    bool loop_contents = true;

    ////////////////////////////////////////////
    // SETUP
    ////////////////////////////////////////////

    stdio_init_all();
    sleep_ms(5000);
    printf("======================\n[EXPERIMENT & AUTO MODE] DS4 on PicoW\n======================\n");
    controller.setup();

    // UART/SPI初期化
    uart_init(uart0, 115200);
    gpio_set_function(0, GPIO_FUNC_UART);
    gpio_set_function(1, GPIO_FUNC_UART);
    
    spi_init(SPI_PORT_MPU, 1000 * 1000);
    gpio_set_function(PIN_MISO_MPU, GPIO_FUNC_SPI);
    gpio_set_function(PIN_SCK_MPU, GPIO_FUNC_SPI);
    gpio_set_function(PIN_MOSI_MPU, GPIO_FUNC_SPI);
    gpio_init(PIN_CS_MPU);
    gpio_set_dir(PIN_CS_MPU, GPIO_OUT);
    gpio_put(PIN_CS_MPU, 1);
    
    spi_init(SPI_PORT_L6470, 4000 * 1000);
    gpio_set_function(PIN_MISO_L6470, GPIO_FUNC_SPI);
    gpio_set_function(PIN_SCK_L6470, GPIO_FUNC_SPI);
    gpio_set_function(PIN_MOSI_L6470, GPIO_FUNC_SPI);
    gpio_init(PIN_CS_L6470);
    gpio_set_dir(PIN_CS_L6470, GPIO_OUT);
    gpio_put(PIN_CS_L6470, 1);

    mpu9250_reset();
    L6470_reset();
    mpu9250_calibration(gy_offset, acceleration2, gyro2, &temp2);

    gpio_init(16);
    gpio_set_dir(16, GPIO_IN);
    gpio_pull_up(16);
    gpio_init(17);
    gpio_set_dir(17, GPIO_OUT);
    gpio_put(17, 1);
    
    printf("Ready. \nCircle: Pat1, Cross: Pat2, Square: Pat3 | Triangle: AutoMove | Options: Tuning\n");

    while (1) 
    {
        if(gpio_get(16) == true) // 安全スイッチ（待機状態）
        {
            gpio_put(17, 1); sleep_ms(200); gpio_put(17, 0); sleep_ms(200);
            L6470_manualstop(Ypulse_num, Xpulse_num);
            is_measuring = false;
            is_auto_mode = false;
            control_pattern = 0;

            DualShock4_state state = controller.get_state();
            if (state.linked) 
            {
                loop_contents = true;
                printf("\n[Linked] %s\n", controller.get_mac_address());
            } else 
            {
                printf(".");
            }

            timestamp = time_us_32();
        }
        else // 通常動作状態
        {
            gpio_put(17, 1);
            tight_loop_contents(); // ループ周期安定化
            DualShock4_state state = controller.get_state();
            StickState sticks = normalize_sticks(state);

            // ==========================================
            // 1. コマンドモード（ゲイン調整）
            // ==========================================
            if (!is_command_mode && state.options) { 
                is_command_mode = true; 
                printf("[Command Mode] Start\n"); 
            }
            if (is_command_mode) 
            {
                if (state.square) { current_target = (current_target == KP) ? KD : KP; sleep_ms(300); printf("Target: %s\n", current_target == KP ? "Kp" : "Kd"); }
                if (state.hat == 0) { 
                    if(current_target == KP) { Kp_value += adjust_step; printf("Kp: %.3f (+)\n", Kp_value); } 
                    else { Kd_value += adjust_step; printf("Kd: %.6f (+)\n", Kd_value); } 
                    sleep_ms(300); 
                }
                if (state.hat == 4) { 
                    if(current_target == KP) { Kp_value -= adjust_step; printf("Kp: %.3f (-)\n", Kp_value); } 
                    else { Kd_value -= adjust_step; printf("Kd: %.6f (-)\n", Kd_value); } 
                    sleep_ms(300); 
                }
                if (state.hat == 2) { adjust_step *= 10.0f; if (adjust_step > 10.0f) adjust_step = 10.0f; printf("Step: %.6f\n", adjust_step); sleep_ms(300); }
                if (state.hat == 6) { adjust_step /= 10.0f; if (adjust_step < 0.000001f) adjust_step = 0.000001f; printf("Step: %.6f\n", adjust_step); sleep_ms(300); }
                if (state.cross) { is_command_mode = false; printf("[Command Mode] End\n"); sleep_ms(300); }
                
                Vst_y = 0; Vst_x = 0;
                continue;
            }

            // ==========================================
            // 2. 計測・オートモード トリガー判定
            // ==========================================
            if (!is_measuring && !is_auto_mode) 
            {
                // 実験・計測モード開始
                // 丸ボタンでPD制御のみ、バツボタンでPD+制振Y、四角ボタンでPD+制振2Dを開始
                if (state.circle) { control_pattern = 1; is_measuring = true; measure_start_time = time_us_32(); } // 1次元球体速度PD
                else if (state.cross) { control_pattern = 2; is_measuring = true; measure_start_time = time_us_32(); } // 1次元PD + 制振Y
                else if (state.square) { control_pattern = 3; is_measuring = true; measure_start_time = time_us_32(); } // 2次元PD + 制振XY
                
                if(is_measuring) {
                    printf(">>> Measurement Start! Pattern: %d\n", control_pattern);
                    printf("ResultLog: Time(s), Pattern, CompX(deg), CompY(deg), Vst_y(m/s), Vsy(m/s), Vsx(m/s)\n");
                }

                // オートモード開始
                if (state.triangle) {
                    is_auto_mode = true;
                    auto_state = MOVE_FORWARD;
                    auto_start_time = time_us_32();
                    printf(">>> Auto Mode Start!\n");
                }
            }

            // ==========================================
            // 3. モードごとの速度指令 (Vst_y, Vst_x)
            // ==========================================
            if (is_measuring) 
            {
                // 実験モード (10秒間一定速度)
                uint32_t elapsed_us = time_us_32() - measure_start_time;
                if (elapsed_us > 10000000) { 
                    is_measuring = false;
                    control_pattern = 0;
                    Vst_y = 0; Vst_x = 0;
                    L6470_move(0, 0, 0);
                    printf(">>> Measurement Finished\n");
                } else {
                    Vst_y = 0.3f; // 比較用に一定速度指令
                    Vst_x = 0.0f;
                }
            } 
            else if (is_auto_mode) 
            {
                // オートモード (状態遷移)
                float target_speed_x = 0.0f;
                float target_speed_y = 0.0f;
                
                switch (auto_state) 
                {
                    case MOVE_FORWARD:
                        target_speed_x = 0.0f; target_speed_y = 0.5f;
                        if ((time_us_32() - auto_start_time) / 1e6f >= 6.0f) { auto_state = WAIT_1; auto_start_time = time_us_32(); printf("Wait 1s\n"); }
                        break;
                    case WAIT_1:
                        target_speed_x = 0.0f; target_speed_y = 0.0f;
                        if ((time_us_32() - auto_start_time) / 1e6f >= 1.0f) { auto_state = MOVE_LEFT; auto_start_time = time_us_32(); printf("Move Left\n"); }
                        break;
                    case MOVE_LEFT:
                        target_speed_x = -0.5f; target_speed_y = 0.0f;
                        if ((time_us_32() - auto_start_time) / 1e6f >= 4.0f) { auto_state = WAIT_2; auto_start_time = time_us_32(); printf("Wait 1s\n"); }
                        break;
                    case WAIT_2:
                        target_speed_x = 0.0f; target_speed_y = 0.0f;
                        if ((time_us_32() - auto_start_time) / 1e6f >= 1.0f) { auto_state = MOVE_DIAGONAL; auto_start_time = time_us_32(); printf("Move Diagonal\n"); }
                        break;
                    case MOVE_DIAGONAL:
                        target_speed_x = 0.27f; target_speed_y = -0.41f; // 0.5 * cos(45°), 0.5 * sin(45°)
                        if ((time_us_32() - auto_start_time) / 1e6f >= 5.656f) { auto_state = WAIT_3; auto_start_time = time_us_32(); printf("Stop\n"); }
                        break;
                    case WAIT_3:
                        target_speed_x = 0.0f; target_speed_y = 0.0f;
                        auto_state = STOP;
                        is_auto_mode = false;
                        printf(">>> Auto Mode Finished\n");
                        break;
                    default:
                        break;
                }
                Vst_x = target_speed_x;
                Vst_y = target_speed_y;
            }
            else 
            {
                // 通常手動操作モード
                if (sticks.l3_y > 0.3 || sticks.l3_y < -0.3 || sticks.l3_x > 0.3 || sticks.l3_x < -0.3) {
                    Vst_y = -sticks.l3_y * 0.5;
                    Vst_x = sticks.l3_x * 0.5;
                    if (!is_stick_moving) { is_stick_moving = true; save_yaw = Yaw; }
                    target_yaw = save_yaw;
                } else {
                    Vst_y = 0; Vst_x = 0; is_stick_moving = false;
                    target_yaw = Yaw;
                }
                
                // 右スティックで目標Yaw角の更新
                if (sticks.r3_x * sticks.r3_x + sticks.r3_y * sticks.r3_y > 0.1f) 
                    target_yaw = Calc_target_yaw_from_rstick(sticks.r3_x, sticks.r3_y);
            }

            // ==========================================
            // 4. 共通制御計算
            // ==========================================
            P_ctrl_Y(Kp_value, Vst_y, Vsy, cor_P_Y);
            D_ctrl_Y(Kd_value, Vst_y, Vsy, cor_D_Y);
            cor_tot(cor_P_Y, cor_D_Y, cor_P_X, cor_D_X, cor_PD_Y, cor_PD_X);
            motor_calc(cor_PD_Y, cor_PD_X, PD_w1, PD_w2, PD_w3, synt_vec_Y, synt_vec_X);

            // 制振制御の適用分け
            int w1_vib_Y = 0, w2_vib_Y = 0;
            int w1_vib_X = 0, w2_vib_X = 0, w3_vib_X = 0;

            if (is_measuring)
            {
                if (control_pattern >= 2) vib_ctrl_Y(comp_x, w1_vib_Y, w2_vib_Y);
                if (control_pattern == 3)
                {
                    P_ctrl_X(Kp_value, Vst_x, Vsx, cor_P_X);
                    D_ctrl_X(Kd_value, Vst_x, Vsx, cor_D_X);
                    vib_ctrl_X(comp_y, w1_vib_X, w2_vib_X, w3_vib_X);
                }
            }
            else 
            {
                // 手動操作時やオートモード時は全軸制振を適用
                vib_ctrl_Y(comp_x, w1_vib_Y, w2_vib_Y);
                vib_ctrl_X(comp_y, w1_vib_X, w2_vib_X, w3_vib_X);
                P_ctrl_X(Kp_value, Vst_x, Vsx, cor_P_X);
                D_ctrl_X(Kd_value, Vst_x, Vsx, cor_D_X);
            }

            yaw_angle_control(target_yaw, Yaw, 1000, 0.1, yaw_output);
            motor_ctrl(PD_w1, PD_w2, PD_w3, (w1_vib_Y + w1_vib_X), (w2_vib_Y + w2_vib_X), w3_vib_X, yaw_output, synt_vec_Y, synt_vec_X);
            
            // ==========================================
            // 5. センサー更新
            // ==========================================
            mpu9250_read_raw2(gy_offset, acceleration, gyro, temp);
            degree_calc(acceleration, gyro, Vgx, Vgy, comp_x, comp_y, Yaw);
            sphere_speed(synt_vec_Y, synt_vec_X, Vgy, Vgx, Vsy, Vsx);

            // ==========================================
            // 6. データ出力
            // ==========================================
            if (is_measuring) {
                // 計測中はCSV形式で高速出力 (10Hz = 100ms周期、実際は制御ループに合わせて約450μs周期で出力)
                if (time_us_32() - timestamp > 100000) { 
                    printf("%.3f, %d, %.2f, %.2f, %.3f, %.3f, %.3f\n", 
                           (time_us_32() - measure_start_time)/1e6f, control_pattern, comp_x, comp_y, Vst_y, Vsy, Vsx);
                    timestamp = time_us_32();
                }
            } else if (time_us_32() - timestamp > 200000) {
                // 通常時はデバッグ表示 (5Hz = 200ms周期)
                printf("Pat:%d | X:%.1f | Y:%.1f | Yaw:%.1f | Auto:%d\n", control_pattern, comp_x, comp_y, Yaw, is_auto_mode);
                timestamp = time_us_32();
            }
        }
    }
    return 0;
}