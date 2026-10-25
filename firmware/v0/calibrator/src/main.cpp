#include <Adafruit_MLX90393.h>
#include <Arduino.h>
#include <HX711.h>
#include <Preferences.h>
#include <SPI.h>
#include <esp_timer.h>
#include <math.h>

#include "pins.h"

namespace {
constexpr uint32_t SERIAL_BAUD = 115200;
constexpr uint32_t SPI_HZ = 1000000;
constexpr uint32_t MLX_CONVERSION_US = 2150;
constexpr float XY_UT_PER_LSB = 2.403F;
constexpr float Z_UT_PER_LSB = 3.872F;
constexpr float G_MPS2 = 9.80665F;
constexpr uint8_t CMD_SM = 0x30, CMD_RM = 0x40, AXIS_TXYZ = 0x0F;

struct MagSample {
  float x = NAN, y = NAN, z = NAN, temp = NAN;
  uint8_t status = 0xFF;
  bool valid = false;
};

Adafruit_MLX90393 mlx[SENSOR_COUNT];
bool mlx_found[SENSOR_COUNT]{};
HX711 load_cell;
Preferences prefs;
int64_t zero_raw = 0;
double counts_per_newton = 0.0;
enum class OutputMode { STOPPED, TABLE, CSV };
OutputMode output_mode = OutputMode::STOPPED;
uint32_t frame = 0;
String command_line;
enum class LoadError { NONE, TIMEOUT, STUCK_ZERO };
LoadError last_load_error = LoadError::NONE;
uint32_t last_load_elapsed_us = 0;

void driveCsHigh() {
  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    digitalWrite(SENSOR_CS_PINS[i], HIGH);
    pinMode(SENSOR_CS_PINS[i], OUTPUT);
    digitalWrite(SENSOR_CS_PINS[i], HIGH);
  }
}

uint8_t transaction(size_t sensor, const uint8_t *tx, size_t tx_len,
                    uint8_t *rx = nullptr, size_t rx_len = 0) {
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE3));
  digitalWrite(SENSOR_CS_PINS[sensor], LOW);
  for (size_t i = 0; i < tx_len; ++i) SPI.transfer(tx[i]);
  const uint8_t status = SPI.transfer(0);
  for (size_t i = 0; i < rx_len; ++i) rx[i] = SPI.transfer(0);
  digitalWrite(SENSOR_CS_PINS[sensor], HIGH);
  SPI.endTransaction();
  return status;
}

bool statusOkay(uint8_t status) {
  return status != 0xFF && (status & 0x10) == 0;
}

int16_t decode18(uint16_t raw) {
  return static_cast<int16_t>(static_cast<uint16_t>(raw - 0x8000U));
}

bool configureMlx(size_t sensor) {
  return mlx[sensor].begin_SPI(SENSOR_CS_PINS[sensor], &SPI) &&
         mlx[sensor].setGain(MLX90393_GAIN_4X) &&
         mlx[sensor].setResolution(MLX90393_X, MLX90393_RES_18) &&
         mlx[sensor].setResolution(MLX90393_Y, MLX90393_RES_18) &&
         mlx[sensor].setResolution(MLX90393_Z, MLX90393_RES_18) &&
         mlx[sensor].setOversampling(MLX90393_OSR_0) &&
         mlx[sensor].setFilter(MLX90393_FILTER_2) &&
         mlx[sensor].setTrigInt(false);
}

void readMagnets(MagSample samples[SENSOR_COUNT]) {
  uint8_t command = CMD_SM | AXIS_TXYZ;
  for (size_t i = 0; i < SENSOR_COUNT; ++i)
    if (mlx_found[i]) transaction(i, &command, 1);
  delayMicroseconds(MLX_CONVERSION_US);
  command = CMD_RM | AXIS_TXYZ;
  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    if (!mlx_found[i]) continue;
    uint8_t data[8]{};
    samples[i].status = transaction(i, &command, 1, data, sizeof(data));
    if (!statusOkay(samples[i].status)) continue;
    const uint16_t rt = (static_cast<uint16_t>(data[0]) << 8) | data[1];
    const uint16_t rx = (static_cast<uint16_t>(data[2]) << 8) | data[3];
    const uint16_t ry = (static_cast<uint16_t>(data[4]) << 8) | data[5];
    const uint16_t rz = (static_cast<uint16_t>(data[6]) << 8) | data[7];
    samples[i].temp = 25.0F + (static_cast<float>(rt) - 46244.0F) / 45.2F;
    samples[i].x = decode18(rx) * XY_UT_PER_LSB;
    samples[i].y = decode18(ry) * XY_UT_PER_LSB;
    samples[i].z = decode18(rz) * Z_UT_PER_LSB;
    samples[i].valid = true;
  }
}

int64_t averageLoad(uint16_t samples) {
  last_load_error = LoadError::NONE;
  const int64_t started = esp_timer_get_time();
  int64_t sum = 0;
  uint16_t collected = 0;
  while (collected < samples) {
    if (!load_cell.wait_ready_timeout(1000)) {
      last_load_error = LoadError::TIMEOUT;
      return INT64_MIN;
    }
    sum += load_cell.read();
    ++collected;
  }
  last_load_elapsed_us = static_cast<uint32_t>(esp_timer_get_time() - started);
  // Even an 80 Hz HX711 needs about 12.5 ms per conversion. Repeated raw zero
  // values arriving far faster mean DOUT is held low, not that the scale is at
  // a legitimate zero load.
  if (sum == 0 && last_load_elapsed_us < static_cast<uint32_t>(samples) * 5000U) {
    last_load_error = LoadError::STUCK_ZERO;
    return INT64_MIN;
  }
  return sum / samples;
}

void printLoadError() {
  if (last_load_error == LoadError::STUCK_ZERO) {
    Serial.printf("# ERROR HX711 DOUT stuck LOW/all-zero data (%lu us). Check 3.3V, GND, DT->GPIO%d, SCK->GPIO%d.\n",
                  static_cast<unsigned long>(last_load_elapsed_us),
                  PIN_HX711_DOUT, PIN_HX711_SCK);
  } else {
    Serial.println("# ERROR HX711 timeout; DOUT never became ready");
  }
}

void saveCalibration() {
  prefs.putLong64("zero", zero_raw);
  prefs.putDouble("cpn", counts_per_newton);
}

void printHelp() {
  Serial.println("Commands:");
  Serial.println("  tare [samples]       remove all force and set zero (default 30)");
  Serial.println("  cal <grams> [samples] apply known mass and calibrate scale");
  Serial.println("  stream               start human-readable table output");
  Serial.println("  csv                  start machine-readable CSV output");
  Serial.println("  stop                 stop streaming");
  Serial.println("  status               show sensors and calibration");
  Serial.println("  clear                erase saved load-cell calibration");
  Serial.println("  help                 show commands");
}

void statusCommand() {
  size_t found = 0;
  for (bool value : mlx_found) found += value;
  Serial.printf("# STATUS MLX=%u/4 HX711=%s zero=%lld counts_per_N=%.6f calibrated=%u\n",
                static_cast<unsigned>(found),
                load_cell.wait_ready_timeout(500) ? "READY" : "NOT_READY",
                static_cast<long long>(zero_raw), counts_per_newton,
                counts_per_newton != 0.0);
}

void runCommand(String line) {
  line.trim();
  if (line.isEmpty()) return;
  if (line == "help" || line == "?") { printHelp(); return; }
  if (line == "status") { statusCommand(); return; }
  if (line == "stream") {
    output_mode = OutputMode::TABLE;
    Serial.println("# Human-readable synchronized stream started");
    return;
  }
  if (line == "csv") {
    output_mode = OutputMode::CSV;
    Serial.println("#BEGIN,t_us,frame,load_raw,load_delta,force_N,s1x_uT,s1y_uT,s1z_uT,s1temp_C,s1valid,s2x_uT,s2y_uT,s2z_uT,s2temp_C,s2valid,s3x_uT,s3y_uT,s3z_uT,s3temp_C,s3valid,s4x_uT,s4y_uT,s4z_uT,s4temp_C,s4valid");
    return;
  }
  if (line == "stop") { output_mode = OutputMode::STOPPED; Serial.println("#END"); return; }
  if (line == "clear") {
    zero_raw = 0; counts_per_newton = 0.0; prefs.clear();
    Serial.println("# CALIBRATION CLEARED"); return;
  }

  char command[12]{};
  double value = 0;
  unsigned samples = 30;
  const int fields = sscanf(line.c_str(), "%11s %lf %u", command, &value, &samples);
  samples = constrain(samples, 5U, 500U);
  if (strcmp(command, "tare") == 0) {
    if (fields >= 2) samples = constrain(static_cast<unsigned>(value), 5U, 500U);
    Serial.printf("# TARE remove all force; collecting %u samples\n", samples);
    const int64_t reading = averageLoad(samples);
    if (reading == INT64_MIN) { printLoadError(); return; }
    zero_raw = reading; saveCalibration();
    Serial.printf("# TARE_OK zero_raw=%lld\n", static_cast<long long>(zero_raw));
    return;
  }
  if (strcmp(command, "cal") == 0 && fields >= 2 && value > 0) {
    Serial.printf("# CAL keep %.3f g still; collecting %u samples\n", value, samples);
    const int64_t loaded = averageLoad(samples);
    if (loaded == INT64_MIN) { printLoadError(); return; }
    const int64_t calibration_delta = loaded - zero_raw;
    if (llabs(calibration_delta) < 100) {
      Serial.printf("# ERROR reference load changed only %lld counts; calibration rejected. Check E+/E-/A+/A- wiring and load-cell mounting.\n",
                    static_cast<long long>(calibration_delta));
      return;
    }
    const double force_n = (value / 1000.0) * G_MPS2;
    counts_per_newton = static_cast<double>(calibration_delta) / force_n;
    saveCalibration();
    Serial.printf("# CAL_OK loaded_raw=%lld force_N=%.6f counts_per_N=%.6f\n",
                  static_cast<long long>(loaded), force_n, counts_per_newton);
    return;
  }
  Serial.println("# ERROR unknown command; type help");
}

void pollCommands() {
  while (Serial.available()) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r') {
      if (!command_line.isEmpty()) { runCommand(command_line); command_line = ""; }
    } else if (command_line.length() < 80) command_line += c;
  }
}

void streamOne() {
  if (!load_cell.is_ready()) return;
  const int64_t raw = load_cell.read();
  const int64_t delta = raw - zero_raw;
  const double force = counts_per_newton != 0.0 ? delta / counts_per_newton : NAN;
  MagSample samples[SENSOR_COUNT]{};
  readMagnets(samples);
  const int64_t timestamp = esp_timer_get_time();
  if (output_mode == OutputMode::TABLE) {
    Serial.println("+--------------------------------------------------------------------------+");
    Serial.printf("| FRAME %-8lu | TIME %-12lld us | LOAD RAW %-10lld | DELTA %-10lld |\n",
                  static_cast<unsigned long>(frame), static_cast<long long>(timestamp),
                  static_cast<long long>(raw), static_cast<long long>(delta));
    if (isfinite(force)) Serial.printf("| FORCE %12.6f N                                                     |\n", force);
    else Serial.println("| FORCE       UNCALIBRATED (run tare, then cal <grams>)                 |");
    Serial.println("+--------+-------------+-------------+-------------+----------+-------+");
    Serial.println("| SENSOR | X (uT)      | Y (uT)      | Z (uT)      | TEMP (C) | VALID |");
    Serial.println("+--------+-------------+-------------+-------------+----------+-------+");
    for (size_t i = 0; i < SENSOR_COUNT; ++i) {
      Serial.printf("| S%-5u | %11.3f | %11.3f | %11.3f | %8.3f | %-5s |\n",
                    static_cast<unsigned>(i + 1), samples[i].x, samples[i].y,
                    samples[i].z, samples[i].temp, samples[i].valid ? "YES" : "NO");
    }
    Serial.println("+--------+-------------+-------------+-------------+----------+-------+");
    ++frame;
    return;
  }
  Serial.printf("%lld,%lu,%lld,%lld,",
                static_cast<long long>(timestamp),
                static_cast<unsigned long>(frame++), static_cast<long long>(raw),
                static_cast<long long>(delta));
  if (isfinite(force)) Serial.printf("%.6f", force); else Serial.print("nan");
  for (const MagSample &sample : samples) {
    Serial.printf(",%.3f,%.3f,%.3f,%.3f,%u", sample.x, sample.y, sample.z,
                  sample.temp, sample.valid);
  }
  Serial.println();
}
}  // namespace

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(500);
  driveCsHigh();
  SPI.begin(PIN_SPI_SCLK, PIN_SPI_MISO, PIN_SPI_MOSI, -1);
  size_t found = 0;
  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    mlx_found[i] = configureMlx(i);
    found += mlx_found[i];
  }
  load_cell.begin(PIN_HX711_DOUT, PIN_HX711_SCK);
  prefs.begin("loadcal", false);
  zero_raw = prefs.isKey("zero") ? prefs.getLong64("zero", 0) : 0;
  counts_per_newton = prefs.isKey("cpn") ? prefs.getDouble("cpn", 0.0) : 0.0;
  Serial.printf("Dermis calibrator: MLX=%u/4 HX711 DOUT=GPIO%d SCK=GPIO%d\n",
                static_cast<unsigned>(found), PIN_HX711_DOUT, PIN_HX711_SCK);
  statusCommand();
  printHelp();
}

void loop() {
  pollCommands();
  if (output_mode != OutputMode::STOPPED) streamOne(); else delay(2);
}
