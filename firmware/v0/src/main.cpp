#include <Adafruit_MLX90393.h>
#include <Arduino.h>
#include <ArduinoJson.h>
#include <NimBLEDevice.h>
#include <SPI.h>
#include <esp_timer.h>
#include "pins.h"

namespace {
constexpr uint32_t SPI_HZ = 1000000, PERIOD_US = 100000;
constexpr char BLE_SERVICE_UUID[] = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
constexpr char BLE_TX_UUID[] = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";
constexpr size_t BLE_MAX_PACKET = 180, BLE_HEADER = 8;
constexpr uint8_t SM = 0x30, RM = 0x40, TXYZ = 0x0F;
constexpr char const *NAMES[] = {"index_tip","thumb_tip","middle_tip","thenar_apb"};
constexpr char const *IDS[] = {"P-0001","P-0002","P-0003","P-0004"};
Adafruit_MLX90393 mlx[SENSOR_COUNT];
bool mlx_ok[SENSOR_COUNT]{}, imu_ok=false;
NimBLECharacteristic *ble_tx=nullptr;
char uid[32], json[2300];
uint32_t frame=0, sent=0, dropped=0, errors=0, clients=0, last_log=0;
uint16_t ble_packet_size=20;
int64_t next_frame=0;

struct Mag { float x,y,z,t; int64_t us; uint8_t status=0xFF; bool valid=false,sat=false,sed=false; };
struct Imu { float ax,ay,az,gx,gy,gz,t; int64_t us; uint32_t st; bool valid=false; };

void csHigh(){
  for(size_t i=0;i<SENSOR_COUNT;i++){pinMode(SENSOR_CS_PINS[i],OUTPUT);digitalWrite(SENSOR_CS_PINS[i],HIGH);}
  pinMode(IMU_CS_PINS[0],OUTPUT);digitalWrite(IMU_CS_PINS[0],HIGH);
}
uint8_t mx(size_t n,const uint8_t* tx,size_t tn,uint8_t* rx=nullptr,size_t rn=0){
  SPI.beginTransaction(SPISettings(SPI_HZ,MSBFIRST,SPI_MODE3));digitalWrite(SENSOR_CS_PINS[n],LOW);
  for(size_t i=0;i<tn;i++) SPI.transfer(tx[i]);
  uint8_t s=SPI.transfer(0);
  for(size_t i=0;i<rn;i++) rx[i]=SPI.transfer(0);
  digitalWrite(SENSOR_CS_PINS[n],HIGH);SPI.endTransaction();return s;
}
bool statusOk(uint8_t s){return s!=0xFF && !(s&0x10);}
bool configMlx(size_t i){
  return mlx[i].begin_SPI(SENSOR_CS_PINS[i],&SPI)&&mlx[i].setGain(MLX90393_GAIN_4X)&&
    mlx[i].setResolution(MLX90393_X,MLX90393_RES_18)&&mlx[i].setResolution(MLX90393_Y,MLX90393_RES_18)&&
    mlx[i].setResolution(MLX90393_Z,MLX90393_RES_18)&&mlx[i].setOversampling(MLX90393_OSR_0)&&
    mlx[i].setFilter(MLX90393_FILTER_2)&&mlx[i].setTrigInt(false);
}
int16_t dec(uint16_t r){return static_cast<int16_t>(static_cast<uint16_t>(r-0x8000U));}
void readMlx(Mag m[]){
  uint8_t c=SM|TXYZ;
  for(size_t i=0;i<SENSOR_COUNT;i++)if(mlx_ok[i]&&!statusOk(mx(i,&c,1)))mlx_ok[i]=configMlx(i);
  delayMicroseconds(2150);c=RM|TXYZ;
  for(size_t i=0;i<SENSOR_COUNT;i++){
    m[i].us=esp_timer_get_time();if(!mlx_ok[i])continue;uint8_t d[8]{};m[i].status=mx(i,&c,1,d,8);m[i].sed=m[i].status&8;
    if(!statusOk(m[i].status))continue;
    uint16_t rt=(d[0]<<8)|d[1],rx=(d[2]<<8)|d[3],ry=(d[4]<<8)|d[5],rz=(d[6]<<8)|d[7];
    m[i].t=25.0f+(rt-46244.0f)/45.2f;m[i].x=dec(rx)*2.403f;m[i].y=dec(ry)*2.403f;m[i].z=dec(rz)*3.872f;
    m[i].sat=fabsf(m[i].x)>=48000||fabsf(m[i].y)>=48000||fabsf(m[i].z)>=48000;m[i].valid=!m[i].sat;
  }
}
uint8_t br(uint8_t r){SPI.beginTransaction(SPISettings(SPI_HZ,MSBFIRST,SPI_MODE0));digitalWrite(IMU_CS_PINS[0],LOW);SPI.transfer(r|0x80);uint8_t v=SPI.transfer(0);digitalWrite(IMU_CS_PINS[0],HIGH);SPI.endTransaction();return v;}
void brs(uint8_t r,uint8_t*d,size_t n){SPI.beginTransaction(SPISettings(SPI_HZ,MSBFIRST,SPI_MODE0));digitalWrite(IMU_CS_PINS[0],LOW);SPI.transfer(r|0x80);for(size_t i=0;i<n;i++)d[i]=SPI.transfer(0);digitalWrite(IMU_CS_PINS[0],HIGH);SPI.endTransaction();}
void bw(uint8_t r,uint8_t v){SPI.beginTransaction(SPISettings(SPI_HZ,MSBFIRST,SPI_MODE0));digitalWrite(IMU_CS_PINS[0],LOW);SPI.transfer(r&0x7F);SPI.transfer(v);digitalWrite(IMU_CS_PINS[0],HIGH);SPI.endTransaction();delayMicroseconds(5);}
bool waitPmu(uint8_t mask,uint8_t expected,uint32_t timeout){uint32_t start=millis();do{if((br(3)&mask)==expected)return true;delay(2);}while(millis()-start<timeout);return false;}
bool configImu(){
  (void)br(0);delay(2);if(br(0)!=0xD1)return false;bw(0x7E,0xB6);delay(120);(void)br(0);delay(2);bw(0x70,1);delay(2);
  bw(0x7E,0x11);bool accel=waitPmu(0x30,0x10,100);delay(100);(void)br(2);
  bw(0x7E,0x15);bool gyro=waitPmu(0x0C,0x04,150);
  if(!gyro){bw(0x7E,0x17);delay(85);bw(0x7E,0x15);gyro=waitPmu(0x0C,0x04,100);}
  bw(0x40,0x28);bw(0x41,3);bw(0x42,0x28);bw(0x43,1);delay(5);
  return accel&&gyro&&br(0)==0xD1&&br(3)==0x14;
}
void readImu(Imu&m){
  m.us=esp_timer_get_time();if(!imu_ok)return;uint8_t d[15]{};brs(0x0C,d,15);
  auto v=[&](int o){return(int16_t)((uint16_t)d[o]|((uint16_t)d[o+1]<<8));};
  constexpr float G=9.80665f,R=.01745329252f;m.gx=v(0)/32.8f*R;m.gy=v(2)/32.8f*R;m.gz=v(4)/32.8f*R;
  m.ax=v(6)/16384.0f*G;m.ay=v(8)/16384.0f*G;m.az=v(10)/16384.0f*G;m.st=d[12]|(d[13]<<8)|(d[14]<<16);
  uint8_t td[2]{};brs(0x20,td,2);m.t=23.0f+(int16_t)(td[0]|(td[1]<<8))/512.0f;m.valid=br(2)==0;
}
void flags(JsonArray a,const Mag&m,bool found){if(!found)a.add("NOT_FOUND");else if(!statusOk(m.status))a.add("SPI_ERROR");else{if(m.sat)a.add("SATURATED");if(m.sed)a.add("SED");}}
size_t makeJson(const Mag m[],const Imu&i,int64_t now){
  JsonDocument d;d["schema_version"]=1;d["device_id"]=uid;d["frame"]=frame;d["t_device_us"]=now;d["t_host_ns"]=nullptr;d["config_id"]="dermis-4mlx-1imu-v1";
  bool bad=!i.valid;for(size_t n=0;n<SENSOR_COUNT;n++)bad|=!m[n].valid;if(bad)d["flags"].to<JsonArray>().add("SENSOR_ERROR");else d["flags"].to<JsonArray>();
  for(size_t n=0;n<SENSOR_COUNT;n++){
    JsonObject p=d["pods"][NAMES[n]].to<JsonObject>();p["pod_id"]=IDS[n];JsonObject x=p["mag"].to<JsonObject>();x["ch"]=n;
    if(m[n].valid){JsonArray b=x["b_ut"].to<JsonArray>();b.add(m[n].x);b.add(m[n].y);b.add(m[n].z);x["temp_c"]=m[n].t;}else{x["b_ut"]=nullptr;x["temp_c"]=nullptr;}
    x["t_us"]=m[n].us;x["status"]=m[n].status;flags(x["flags"].to<JsonArray>(),m[n],mlx_ok[n]);x["valid"]=m[n].valid;
  }
  JsonObject x=d["imu"].to<JsonObject>();x["imu_id"]="IMU-0001";x["ch"]=0;
  if(i.valid){JsonArray a=x["acc_mps2"].to<JsonArray>();a.add(i.ax);a.add(i.ay);a.add(i.az);JsonArray g=x["gyr_rads"].to<JsonArray>();g.add(i.gx);g.add(i.gy);g.add(i.gz);x["temp_c"]=i.t;}
  else{x["acc_mps2"]=nullptr;x["gyr_rads"]=nullptr;x["temp_c"]=nullptr;}
  x["t_us"]=i.us;x["sensortime"]=i.st;JsonArray f=x["flags"].to<JsonArray>();if(!imu_ok)f.add("NOT_FOUND");else if(!i.valid)f.add("SPI_ERROR");x["valid"]=i.valid;
  return serializeJson(d,json,sizeof(json));
}
class BleCallbacks:public NimBLEServerCallbacks{
  void onConnect(NimBLEServer*,ble_gap_conn_desc*)override{clients++;Serial.println("BLE connected");}
  void onDisconnect(NimBLEServer*)override{if(clients)clients--;Serial.println("BLE disconnected");NimBLEDevice::startAdvertising();}
  void onMTUChange(uint16_t mtu,ble_gap_conn_desc*)override{ble_packet_size=min((size_t)(mtu-3),(size_t)BLE_MAX_PACKET);Serial.printf("BLE MTU=%u payload=%u\n",mtu,ble_packet_size);}
};
void startBle(){
  NimBLEDevice::init("Dermis-S3");NimBLEDevice::setMTU(185);
  NimBLEServer *server=NimBLEDevice::createServer();server->setCallbacks(new BleCallbacks());
  NimBLEService *service=server->createService(BLE_SERVICE_UUID);
  ble_tx=service->createCharacteristic(BLE_TX_UUID,NIMBLE_PROPERTY::READ|NIMBLE_PROPERTY::NOTIFY);
  service->start();NimBLEAdvertising *advertising=NimBLEDevice::getAdvertising();advertising->addServiceUUID(BLE_SERVICE_UUID);advertising->setScanResponse(true);advertising->start();
  Serial.println("BLE advertising as Dermis-S3");
}
bool sendBle(size_t length){
  if(!clients||!ble_tx)return false;
  const size_t data_size=ble_packet_size-BLE_HEADER;
  const uint8_t count=(length+data_size-1)/data_size;
  uint8_t packet[BLE_MAX_PACKET];
  for(uint8_t index=0;index<count;index++){
    const size_t offset=index*data_size,count_bytes=min(data_size,length-offset);
    packet[0]=0xD3;packet[1]=1;packet[2]=frame;packet[3]=frame>>8;packet[4]=frame>>16;packet[5]=frame>>24;packet[6]=index;packet[7]=count;
    memcpy(packet+BLE_HEADER,json+offset,count_bytes);ble_tx->notify(packet,count_bytes+BLE_HEADER);delay(5);
  }
  return true;
}
}
void setup(){
  Serial.begin(115200);delay(250);csHigh();SPI.begin(PIN_SPI_SCLK,PIN_SPI_MISO,PIN_SPI_MOSI,-1);size_t count=0;
  for(size_t i=0;i<SENSOR_COUNT;i++){mlx_ok[i]=configMlx(i);count+=mlx_ok[i];}imu_ok=configImu();
  snprintf(uid,sizeof(uid),"DERMIS-%012llX",(unsigned long long)ESP.getEfuseMac());Serial.printf("Sensors: MLX=%u/4 IMU=%u/1\n",(unsigned)count,imu_ok?1:0);
  startBle();last_log=millis();next_frame=esp_timer_get_time();
}
void loop(){
  int64_t now=esp_timer_get_time();if(now>=next_frame){next_frame+=PERIOD_US;if(now-next_frame>(int64_t)PERIOD_US)next_frame=now+PERIOD_US;
    Mag m[SENSOR_COUNT]{};Imu i{};readMlx(m);readImu(i);bool bad=!i.valid;for(const auto&v:m)bad|=!v.valid;if(bad)errors++;
    size_t n=makeJson(m,i,now);if(n&&n+1<sizeof(json)){if(clients){if(sendBle(n))sent++;else dropped++;}}else dropped++;frame++;}
  if(millis()-last_log>=5000){Serial.printf("BLE frames=%lu connected=%lu dropped=%lu sensor_errors=%lu\n",(unsigned long)sent,(unsigned long)clients,(unsigned long)dropped,(unsigned long)errors);last_log=millis();}
}
