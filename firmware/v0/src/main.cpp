#include <Adafruit_MLX90393.h>
#include <Arduino.h>
#include <ArduinoJson.h>
#include <Preferences.h>
#include <SPI.h>
#include <WebSocketsServer.h>
#include <WiFi.h>
#include <esp_timer.h>
#include "pins.h"

namespace {
constexpr uint32_t SPI_HZ = 1000000, PERIOD_US = 100000;
constexpr uint8_t FORCE_ZERO_SAMPLES = 30;
constexpr float FORCE_ZERO_MAX_FIELD_STD_UT = 80.0f, FORCE_ZERO_MAX_TEMP_STD_C = 1.0f;
constexpr float FORCE_SPAN_MIN_UT = 250.0f, FORCE_DEADBAND_N = 0.08f;
constexpr char AP_PASSWORD[] = "Dermis2026!";
constexpr uint16_t WS_PORT = 81;
constexpr uint8_t SM = 0x30, RM = 0x40, TXYZ = 0x0F;
constexpr float MAX_CALIBRATED_FORCE_N = 9.80665f;
// Physical glove map (SPI sensor order): S1 palm, S2 middle, S3 thumb, S4 index.
constexpr char const *NAMES[] = {"palm_center","middle_tip","thumb_tip","index_tip"};
constexpr char const *IDS[] = {"P-0001","P-0002","P-0003","P-0004"};
Adafruit_MLX90393 mlx[SENSOR_COUNT];
bool mlx_ok[SENSOR_COUNT]{}, imu_ok=false;
uint8_t imu_chip_id=0xFF,imu_pmu=0xFF,imu_error=0xFF;
bool imu_accel_ready=false,imu_gyro_ready=false;
WebSocketsServer web_socket(WS_PORT);
Preferences preferences;
char uid[32], json[2300];
uint32_t frame=0, sent=0, dropped=0, errors=0, clients=0, last_log=0;
char ap_ssid[32];
int64_t next_frame=0;

struct Mag { float x,y,z,t; int64_t us; uint8_t status=0xFF; bool valid=false,sat=false,sed=false; };
struct Imu { float ax,ay,az,gx,gy,gz,t; int64_t us; uint32_t st; bool valid=false; };
struct ForceEstimate { float value=NAN, raw=NAN, signal_ut=NAN; bool valid=false, clamped_low=false, clamped_high=false; };
struct ForceZero {
  float mean[4]{}, m2[4]{};
  uint8_t samples=0;
  bool ready=false, unstable=false;
};
struct ForceSpan {
  float mean[3]{}, m2[3]{}, unit[3]{}, gain_n_per_ut=NAN, reference_n=NAN;
  uint8_t samples=0;
  bool ready=false, capturing=false, unstable=false;
};
struct StoredSpans {
  uint32_t version;
  float unit[SENSOR_COUNT][3];
  float gain[SENSOR_COUNT];
  float reference[SENSOR_COUNT];
};
ForceZero force_zero[SENSOR_COUNT];
ForceSpan force_span[SENSOR_COUNT];
constexpr uint32_t FORCE_SPAN_VERSION = 0x44524D32; // "DRM2"

void loadForceSpans(){
  StoredSpans stored{};
  preferences.begin("dermis-force",true);
  const size_t read=preferences.getBytes("spans",&stored,sizeof(stored));preferences.end();
  if(read!=sizeof(stored)||stored.version!=FORCE_SPAN_VERSION)return;
  for(size_t sensor=0;sensor<SENSOR_COUNT;sensor++){
    const float norm=sqrtf(stored.unit[sensor][0]*stored.unit[sensor][0]+stored.unit[sensor][1]*stored.unit[sensor][1]+stored.unit[sensor][2]*stored.unit[sensor][2]);
    if(!isfinite(norm)||norm<0.98f||norm>1.02f||!isfinite(stored.gain[sensor])||stored.gain[sensor]<=0||stored.gain[sensor]>.1f)continue;
    auto&s=force_span[sensor];for(size_t axis=0;axis<3;axis++)s.unit[axis]=stored.unit[sensor][axis];
    s.gain_n_per_ut=stored.gain[sensor];s.reference_n=stored.reference[sensor];s.ready=true;
  }
}
void saveForceSpans(){
  StoredSpans stored{};stored.version=FORCE_SPAN_VERSION;
  for(size_t sensor=0;sensor<SENSOR_COUNT;sensor++){
    for(size_t axis=0;axis<3;axis++)stored.unit[sensor][axis]=force_span[sensor].unit[axis];
    stored.gain[sensor]=force_span[sensor].gain_n_per_ut;stored.reference[sensor]=force_span[sensor].reference_n;
  }
  preferences.begin("dermis-force",false);preferences.putBytes("spans",&stored,sizeof(stored));preferences.end();
}
void resetForceZero(){
  for(auto&z:force_zero)z=ForceZero{};
  for(auto&s:force_span){s.capturing=false;s.samples=0;s.unstable=false;}
}
void updateForceZero(size_t sensor,const Mag&m){
  auto&z=force_zero[sensor];if(!m.valid||z.ready)return;
  const float value[4]={m.x,m.y,m.z,m.t};z.samples++;
  for(size_t n=0;n<4;n++){const float d=value[n]-z.mean[n];z.mean[n]+=d/z.samples;z.m2[n]+=d*(value[n]-z.mean[n]);}
  if(z.samples<FORCE_ZERO_SAMPLES)return;
  bool steady=true;
  for(size_t n=0;n<4;n++){
    const float sigma=sqrtf(z.m2[n]/(FORCE_ZERO_SAMPLES-1));
    steady&=sigma<((n==3)?FORCE_ZERO_MAX_TEMP_STD_C:FORCE_ZERO_MAX_FIELD_STD_UT);
  }
  if(steady){z.ready=true;z.unstable=false;}
  else{z=ForceZero{};z.unstable=true;}
}
void requestForceSpan(size_t sensor,float reference_n){
  if(sensor>=SENSOR_COUNT||!force_zero[sensor].ready||!isfinite(reference_n)||reference_n<=0||reference_n>MAX_CALIBRATED_FORCE_N)return;
  auto&s=force_span[sensor];s.mean[0]=s.mean[1]=s.mean[2]=0;s.m2[0]=s.m2[1]=s.m2[2]=0;
  s.samples=0;s.reference_n=reference_n;s.capturing=true;s.unstable=false;
}
void updateForceSpan(size_t sensor,const Mag&m){
  auto&s=force_span[sensor];const auto&z=force_zero[sensor];if(!m.valid||!z.ready||!s.capturing)return;
  const float value[3]={m.x,m.y,m.z};s.samples++;
  for(size_t axis=0;axis<3;axis++){const float d=value[axis]-s.mean[axis];s.mean[axis]+=d/s.samples;s.m2[axis]+=d*(value[axis]-s.mean[axis]);}
  if(s.samples<FORCE_ZERO_SAMPLES)return;
  float delta[3],span2=0,sigma2=0;
  for(size_t axis=0;axis<3;axis++){delta[axis]=s.mean[axis]-z.mean[axis];span2+=delta[axis]*delta[axis];sigma2+=s.m2[axis]/(FORCE_ZERO_SAMPLES-1);}
  const float span=sqrtf(span2),sigma=sqrtf(sigma2),limit=fmaxf(250.0f,span*.10f);
  s.capturing=false;
  if(!isfinite(span)||span<FORCE_SPAN_MIN_UT||!isfinite(sigma)||sigma>limit){s.unstable=true;return;}
  for(size_t axis=0;axis<3;axis++)s.unit[axis]=delta[axis]/span;
  s.gain_n_per_ut=s.reference_n/span;s.ready=true;s.unstable=false;saveForceSpans();
}
ForceEstimate estimateForce(size_t sensor,const Mag&m){
  ForceEstimate result;const auto&z=force_zero[sensor];const auto&s=force_span[sensor];if(!m.valid||!z.ready)return result;
  const float delta[3]={m.x-z.mean[0],m.y-z.mean[1],m.z-z.mean[2]};
  result.signal_ut=s.ready?(delta[0]*s.unit[0]+delta[1]*s.unit[1]+delta[2]*s.unit[2]):sqrtf(delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]);
  if(!s.ready||s.capturing||!isfinite(result.signal_ut))return result;
  result.raw=result.signal_ut*s.gain_n_per_ut;if(!isfinite(result.raw))return result;
  result.value=result.raw;result.valid=true;
  if(result.value<=FORCE_DEADBAND_N){result.clamped_low=result.value<-FORCE_DEADBAND_N;result.value=0;}
  if(result.value>MAX_CALIBRATED_FORCE_N){result.value=MAX_CALIBRATED_FORCE_N;result.clamped_high=true;}
  return result;
}

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
void bw(uint8_t r,uint8_t v){SPI.beginTransaction(SPISettings(SPI_HZ,MSBFIRST,SPI_MODE0));digitalWrite(IMU_CS_PINS[0],LOW);SPI.transfer(r&0x7F);SPI.transfer(v);digitalWrite(IMU_CS_PINS[0],HIGH);SPI.endTransaction();delayMicroseconds(500);}
bool waitPmu(uint8_t mask,uint8_t expected,uint32_t timeout){uint32_t start=millis();do{if((br(3)&mask)==expected)return true;delay(2);}while(millis()-start<timeout);return false;}
bool configImu(){
  (void)br(0);delay(2);imu_chip_id=br(0);if(imu_chip_id!=0xD1){imu_pmu=br(3);imu_error=br(2);return false;}bw(0x7E,0xB6);delay(120);(void)br(0);delay(2);bw(0x70,1);delay(2);
  bw(0x40,0x28);bw(0x41,3);bw(0x42,0x28);bw(0x43,1);
  imu_accel_ready=false;imu_gyro_ready=false;
  for(uint8_t attempt=0;attempt<3;attempt++){
    bw(0x7E,0x15);imu_gyro_ready=waitPmu(0x0C,0x04,200);
    bw(0x7E,0x11);imu_accel_ready=waitPmu(0x30,0x10,200);
    delay(10);
    if(br(3)==0x14)break;
  }
  imu_chip_id=br(0);imu_pmu=br(3);imu_error=br(2);
  return imu_accel_ready&&imu_gyro_ready&&imu_chip_id==0xD1&&imu_pmu==0x14;
}
void readImu(Imu&m){
  m.us=esp_timer_get_time();if(!imu_ok)return;uint8_t d[15]{};brs(0x0C,d,15);
  auto v=[&](int o){return(int16_t)((uint16_t)d[o]|((uint16_t)d[o+1]<<8));};
  constexpr float G=9.80665f,R=.01745329252f;m.gx=v(0)/32.8f*R;m.gy=v(2)/32.8f*R;m.gz=v(4)/32.8f*R;
  m.ax=v(6)/16384.0f*G;m.ay=v(8)/16384.0f*G;m.az=v(10)/16384.0f*G;m.st=d[12]|(d[13]<<8)|(d[14]<<16);
  uint8_t td[2]{};brs(0x20,td,2);m.t=23.0f+(int16_t)(td[0]|(td[1]<<8))/512.0f;m.valid=br(2)==0;
}
void flags(JsonArray a,const Mag&m,bool found,const ForceEstimate&force,size_t sensor){
  if(!found){a.add("NOT_FOUND");return;}if(!statusOk(m.status)){a.add("SPI_ERROR");return;}
  if(m.sat)a.add("SATURATED");
  if(m.sed)a.add("SED");
  const auto&z=force_zero[sensor];const auto&s=force_span[sensor];
  if(!z.ready)a.add(z.unstable?"FORCE_ZERO_UNSTABLE":"FORCE_ZEROING");
  else if(s.capturing)a.add("FORCE_SPAN_CALIBRATING");
  else if(s.unstable)a.add("FORCE_SPAN_UNSTABLE");
  else if(!s.ready)a.add("FORCE_SPAN_REQUIRED");
  else a.add("FORCE_CAL_READY");
  if(force.valid){if(force.clamped_low)a.add("FORCE_CLAMPED_LOW");if(force.clamped_high)a.add("FORCE_CLAMPED_HIGH");}
}
size_t makeJson(const Mag m[],const Imu&i,int64_t now){
  JsonDocument d;d["schema_version"]=1;d["device_id"]=uid;d["frame"]=frame;d["t_device_us"]=now;d["t_host_ns"]=nullptr;d["config_id"]="dermis-4mlx-1imu-force-zero-span-v3";
  bool bad=!i.valid;for(size_t n=0;n<SENSOR_COUNT;n++)bad|=!m[n].valid;if(bad)d["flags"].to<JsonArray>().add("SENSOR_ERROR");else d["flags"].to<JsonArray>();
  for(size_t n=0;n<SENSOR_COUNT;n++){
    JsonObject p=d["pods"][NAMES[n]].to<JsonObject>();p["pod_id"]=IDS[n];JsonObject x=p["mag"].to<JsonObject>();x["ch"]=n;
    updateForceZero(n,m[n]);updateForceSpan(n,m[n]);const ForceEstimate force=estimateForce(n,m[n]);
    if(m[n].valid){JsonArray b=x["b_ut"].to<JsonArray>();b.add(m[n].x);b.add(m[n].y);b.add(m[n].z);x["temp_c"]=m[n].t;if(isfinite(force.signal_ut))x["load_signal_ut"]=force.signal_ut;else x["load_signal_ut"]=nullptr;if(force.valid){x["force_n"]=force.value;x["force_raw_n"]=force.raw;}else{x["force_n"]=nullptr;x["force_raw_n"]=nullptr;}}else{x["b_ut"]=nullptr;x["temp_c"]=nullptr;x["load_signal_ut"]=nullptr;x["force_n"]=nullptr;x["force_raw_n"]=nullptr;}
    if(force_span[n].ready)x["force_reference_n"]=force_span[n].reference_n;else x["force_reference_n"]=nullptr;
    x["force_valid"]=force.valid;x["t_us"]=m[n].us;x["status"]=m[n].status;flags(x["flags"].to<JsonArray>(),m[n],mlx_ok[n],force,n);x["valid"]=m[n].valid;
  }
  JsonObject x=d["imu"].to<JsonObject>();x["imu_id"]="IMU-0001";x["ch"]=0;
  if(i.valid){JsonArray a=x["acc_mps2"].to<JsonArray>();a.add(i.ax);a.add(i.ay);a.add(i.az);JsonArray g=x["gyr_rads"].to<JsonArray>();g.add(i.gx);g.add(i.gy);g.add(i.gz);x["temp_c"]=i.t;}
  else{x["acc_mps2"]=nullptr;x["gyr_rads"]=nullptr;x["temp_c"]=nullptr;}
  x["t_us"]=i.us;x["sensortime"]=i.st;JsonArray f=x["flags"].to<JsonArray>();if(!imu_ok)f.add("NOT_FOUND");else if(!i.valid)f.add("SPI_ERROR");x["valid"]=i.valid;
  if(!imu_ok){JsonObject diag=x["diag"].to<JsonObject>();diag["chip_id"]=imu_chip_id;diag["pmu_status"]=imu_pmu;diag["error_reg"]=imu_error;diag["accel_ready"]=imu_accel_ready;diag["gyro_ready"]=imu_gyro_ready;}
  return serializeJson(d,json,sizeof(json));
}
void wsEvent(uint8_t client,WStype_t type,uint8_t *payload,size_t length){
  if(type==WStype_CONNECTED){String path((char*)payload,length);clients++;Serial.printf("WS connected path=%s clients=%lu\n",path.c_str(),(unsigned long)clients);}
  else if(type==WStype_DISCONNECTED){if(clients)clients--;Serial.printf("WS disconnected clients=%lu\n",(unsigned long)clients);}
  else if(type==WStype_TEXT){
    JsonDocument command;
    if(deserializeJson(command,payload,length)!=DeserializationError::Ok)return;
    const char*cmd=command["cmd"]|"";
    if(strcmp(cmd,"tare")==0){
      resetForceZero();Serial.printf("Force tare requested by WS client %u: keep every patch unloaded for %u samples\n",client,FORCE_ZERO_SAMPLES);
    }else if(strcmp(cmd,"span")==0){
      const int sensor=command["ch"]|-1;const float reference=command["force_n"]|NAN;
      if(sensor>=0&&sensor<SENSOR_COUNT&&force_zero[sensor].ready&&isfinite(reference)&&reference>0&&reference<=MAX_CALIBRATED_FORCE_N){
        requestForceSpan(sensor,reference);Serial.printf("Force span S%d requested: hold %.3f N steady for %u samples\n",sensor+1,reference,FORCE_ZERO_SAMPLES);
      }else Serial.printf("Rejected force span request ch=%d force=%.3f: tare must be ready and force within (0, %.3f] N\n",sensor,reference,MAX_CALIBRATED_FORCE_N);
    }
  }
}
void startNetwork(){
  const IPAddress ip(192,168,4,1),gateway(192,168,4,1),subnet(255,255,255,0);
  WiFi.mode(WIFI_AP);WiFi.setSleep(false);WiFi.softAPConfig(ip,gateway,subnet);
  const bool started=WiFi.softAP(ap_ssid,AP_PASSWORD,6,false,1);
  Serial.printf("Hotspot: %s (%s)\n",started?"READY":"FAILED",ap_ssid);
  Serial.printf("Password: %s\nIP: %s\nWebSocket: ws://%s:%u/ws\n",AP_PASSWORD,WiFi.softAPIP().toString().c_str(),WiFi.softAPIP().toString().c_str(),WS_PORT);
  web_socket.begin();web_socket.onEvent(wsEvent);
}
}
void setup(){
  Serial.begin(115200);delay(250);csHigh();SPI.begin(PIN_SPI_SCLK,PIN_SPI_MISO,PIN_SPI_MOSI,-1);size_t count=0;
  for(size_t i=0;i<SENSOR_COUNT;i++){mlx_ok[i]=configMlx(i);count+=mlx_ok[i];}imu_ok=configImu();
  const uint64_t mac=ESP.getEfuseMac();snprintf(uid,sizeof(uid),"DERMIS-%012llX",(unsigned long long)mac);snprintf(ap_ssid,sizeof(ap_ssid),"Dermis-S3-%04X",(unsigned)(mac&0xFFFF));Serial.printf("Device ID: %s\nSensors: MLX=%u/4 IMU=%u/1 id=0x%02X pmu=0x%02X err=0x%02X accel=%u gyro=%u\n",uid,(unsigned)count,imu_ok?1:0,imu_chip_id,imu_pmu,imu_error,imu_accel_ready,imu_gyro_ready);
  loadForceSpans();startNetwork();last_log=millis();next_frame=esp_timer_get_time();
}
void loop(){
  web_socket.loop();
  int64_t now=esp_timer_get_time();if(now>=next_frame){next_frame+=PERIOD_US;if(now-next_frame>(int64_t)PERIOD_US)next_frame=now+PERIOD_US;
    Mag m[SENSOR_COUNT]{};Imu i{};readMlx(m);readImu(i);bool bad=!i.valid;for(const auto&v:m)bad|=!v.valid;if(bad)errors++;
    size_t n=makeJson(m,i,now);if(n&&n<sizeof(json)){if(clients){web_socket.broadcastTXT((uint8_t*)json,n);sent++;}}else dropped++;frame++;}
  if(millis()-last_log>=5000){Serial.printf("WS frames=%lu clients=%lu stations=%u dropped=%lu sensor_errors=%lu\n",(unsigned long)sent,(unsigned long)clients,WiFi.softAPgetStationNum(),(unsigned long)dropped,(unsigned long)errors);last_log=millis();}
}
