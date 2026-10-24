#include <Adafruit_MLX90393.h>
#include <Arduino.h>
#include <HX711.h>
#include <SPI.h>
#include <esp_timer.h>
#include <math.h>
#include <stdio.h>

#include "pins.h"

namespace {

constexpr uint32_t SERIAL_BAUD = 921600;
constexpr uint32_t DEFAULT_SPI_HZ = 1000000;
constexpr uint32_t MAX_SPI_HZ = 10000000;

// Datasheet rev. 012 tables 17/21: GAIN_SEL=1 (library name GAIN_4X)
// with RES=2 has a digital range of approximately +/-78.7 mT on X/Y and
// +/-126.9 mT on Z. This exceeds the requested +/-60 mT without digital
// clipping. Standard ABA-0xx silicon still has an *analog* magnetic saturation
// onset around 50 mT; firmware cannot remove that physical limit, so samples
// at/above 48 mT are explicitly marked as saturation-risk in the flags.
constexpr mlx90393_gain_t SENSOR_GAIN = MLX90393_GAIN_4X;
constexpr mlx90393_resolution_t SENSOR_RESOLUTION = MLX90393_RES_18;
constexpr mlx90393_oversampling_t SENSOR_OSR = MLX90393_OSR_0;
constexpr mlx90393_filter_t SENSOR_FILTER = MLX90393_FILTER_2;
constexpr float XY_UT_PER_LSB = 2.403F;
constexpr float Z_UT_PER_LSB = 3.872F;
constexpr float ANALOG_SATURATION_WARNING_UT = 48000.0F;
constexpr uint16_t DIGITAL_RAIL_MARGIN = 32;

// OSR=0, DIG_FILT=2, OSR2=0, HALLCONF=0xC: datasheet table 19 gives
// 1.84 ms typical for TXYZ. 2.15 ms allows oscillator/timing margin while
// retaining a theoretical sensor-side rate above 400 Hz.
constexpr uint32_t TXYZ_CONVERSION_US = 2150;
constexpr uint32_t T_ONLY_CONVERSION_US = 1200;

constexpr uint8_t CMD_SM = 0x30;
constexpr uint8_t CMD_RM = 0x40;
constexpr uint8_t CMD_RR = 0x50;
constexpr uint8_t CMD_WR = 0x60;
constexpr uint8_t CMD_EX = 0x80;
constexpr uint8_t CMD_RT = 0xF0;
constexpr uint8_t AXIS_T = 0x01;
constexpr uint8_t AXIS_XYZ = 0x0E;
constexpr uint8_t AXIS_TXYZ = 0x0F;

constexpr uint8_t STATUS_BURST = 0x80;
constexpr uint8_t STATUS_WOC = 0x40;
constexpr uint8_t STATUS_SM = 0x20;
constexpr uint8_t STATUS_ERROR = 0x10;
constexpr uint8_t STATUS_SED = 0x08;
constexpr uint8_t STATUS_RESET = 0x04;

constexpr uint32_t FLAG_STATUS_ERROR_BASE = 0;
constexpr uint32_t FLAG_SATURATION_BASE = 4;
constexpr uint32_t FLAG_NOT_FOUND_BASE = 8;
constexpr uint32_t FLAG_SED_BASE = 12;

Adafruit_MLX90393 mlx[SENSOR_COUNT];
HX711 load_cell;
bool sensor_found[SENSOR_COUNT] = {false, false, false, false};
uint32_t spi_hz = DEFAULT_SPI_HZ;

struct Sample {
  uint16_t raw_t;
  uint16_t raw_x;
  uint16_t raw_y;
  uint16_t raw_z;
  float t_c;
  float x_ut;
  float y_ut;
  float z_ut;
  uint8_t start_status;
  uint8_t read_status;
};

struct Frame {
  int64_t t_us;
  Sample sensor[SENSOR_COUNT];
  uint32_t flags;
};

struct OnlineStats {
  uint32_t count;
  double mean;
  double m2;
  float minimum;
  float maximum;

  void reset() {
    count = 0;
    mean = 0.0;
    m2 = 0.0;
    minimum = INFINITY;
    maximum = -INFINITY;
  }

  void add(float value) {
    ++count;
    const double delta = static_cast<double>(value) - mean;
    mean += delta / count;
    const double delta2 = static_cast<double>(value) - mean;
    m2 += delta * delta2;
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }

  double stddev() const {
    return count > 1 ? sqrt(m2 / static_cast<double>(count - 1)) : 0.0;
  }
};

void driveAllCsHigh() {
  // This must happen before SPI.begin(): CS at reset selects SPI vs I2C.
  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    digitalWrite(SENSOR_CS_PINS[i], HIGH);
    pinMode(SENSOR_CS_PINS[i], OUTPUT);
    digitalWrite(SENSOR_CS_PINS[i], HIGH);
  }
}

bool statusHasError(uint8_t status) {
  return status == 0xFF || (status & STATUS_ERROR) != 0;
}

void printStatus(uint8_t status) {
  Serial.printf("0x%02X [burst=%u woc=%u sm=%u error=%u sed=%u reset=%u d=%u]",
                status, (status & STATUS_BURST) != 0,
                (status & STATUS_WOC) != 0, (status & STATUS_SM) != 0,
                (status & STATUS_ERROR) != 0, (status & STATUS_SED) != 0,
                (status & STATUS_RESET) != 0, status & 0x03);
}

uint8_t transaction(size_t sensor, const uint8_t *tx, size_t tx_len,
                    uint8_t *rx, size_t rx_len) {
  SPI.beginTransaction(SPISettings(spi_hz, MSBFIRST, SPI_MODE3));
  digitalWrite(SENSOR_CS_PINS[sensor], LOW);
  for (size_t i = 0; i < tx_len; ++i) SPI.transfer(tx[i]);
  const uint8_t status = SPI.transfer(0x00);
  for (size_t i = 0; i < rx_len; ++i) rx[i] = SPI.transfer(0x00);
  digitalWrite(SENSOR_CS_PINS[sensor], HIGH);
  SPI.endTransaction();
  return status;
}

uint8_t simpleCommand(size_t sensor, uint8_t command) {
  return transaction(sensor, &command, 1, nullptr, 0);
}

uint8_t readRegister(size_t sensor, uint8_t reg, uint16_t &value) {
  const uint8_t tx[2] = {CMD_RR, static_cast<uint8_t>(reg << 2)};
  uint8_t rx[2] = {0, 0};
  const uint8_t status = transaction(sensor, tx, sizeof(tx), rx, sizeof(rx));
  value = static_cast<uint16_t>(rx[0] << 8) | rx[1];
  return status;
}

uint8_t writeRegister(size_t sensor, uint8_t reg, uint16_t value) {
  const uint8_t tx[4] = {CMD_WR, static_cast<uint8_t>(value >> 8),
                         static_cast<uint8_t>(value),
                         static_cast<uint8_t>(reg << 2)};
  return transaction(sensor, tx, sizeof(tx), nullptr, 0);
}

int16_t decodeRes18(uint16_t raw) {
  // RES=2 is unsigned with zero field at 2^15. Mapping via uint16 arithmetic
  // is well-defined and produces the desired signed -32768..32767 value.
  return static_cast<int16_t>(static_cast<uint16_t>(raw - 0x8000U));
}

float decodeTemperature(uint16_t raw) {
  // Datasheet table 7 typical transfer: 46244 LSB at 25 C, 45.2 LSB/C.
  return 25.0F + (static_cast<float>(raw) - 46244.0F) / 45.2F;
}

bool isSaturated(const Sample &sample) {
  const bool digital_rail =
      sample.raw_x <= DIGITAL_RAIL_MARGIN ||
      sample.raw_x >= (0xFFFFU - DIGITAL_RAIL_MARGIN) ||
      sample.raw_y <= DIGITAL_RAIL_MARGIN ||
      sample.raw_y >= (0xFFFFU - DIGITAL_RAIL_MARGIN) ||
      sample.raw_z <= DIGITAL_RAIL_MARGIN ||
      sample.raw_z >= (0xFFFFU - DIGITAL_RAIL_MARGIN);
  const bool analog_risk = fabsf(sample.x_ut) >= ANALOG_SATURATION_WARNING_UT ||
                           fabsf(sample.y_ut) >= ANALOG_SATURATION_WARNING_UT ||
                           fabsf(sample.z_ut) >= ANALOG_SATURATION_WARNING_UT;
  return digital_rail || analog_risk;
}

uint8_t startMeasurement(size_t sensor, uint8_t axes = AXIS_TXYZ) {
  return simpleCommand(sensor, CMD_SM | axes);
}

uint8_t readMeasurement(size_t sensor, Sample &sample,
                        uint8_t axes = AXIS_TXYZ) {
  uint8_t rx[8] = {0};
  const size_t rx_len = axes == AXIS_TXYZ ? 8 : (axes == AXIS_XYZ ? 6 : 2);
  const uint8_t command = CMD_RM | axes;
  const uint8_t status = transaction(sensor, &command, 1, rx, rx_len);

  size_t offset = 0;
  if ((axes & AXIS_T) != 0) {
    sample.raw_t = static_cast<uint16_t>(rx[offset] << 8) | rx[offset + 1];
    sample.t_c = decodeTemperature(sample.raw_t);
    offset += 2;
  }
  if ((axes & AXIS_XYZ) == AXIS_XYZ) {
    sample.raw_x = static_cast<uint16_t>(rx[offset] << 8) | rx[offset + 1];
    sample.raw_y = static_cast<uint16_t>(rx[offset + 2] << 8) | rx[offset + 3];
    sample.raw_z = static_cast<uint16_t>(rx[offset + 4] << 8) | rx[offset + 5];
    sample.x_ut = static_cast<float>(decodeRes18(sample.raw_x)) * XY_UT_PER_LSB;
    sample.y_ut = static_cast<float>(decodeRes18(sample.raw_y)) * XY_UT_PER_LSB;
    sample.z_ut = static_cast<float>(decodeRes18(sample.raw_z)) * Z_UT_PER_LSB;
  }
  return status;
}

bool acquireFrame(Frame &frame) {
  frame.t_us = esp_timer_get_time();
  frame.flags = 0;
  bool okay = true;

  // Deliberately trigger every sensor first so their conversions overlap.
  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    if (!sensor_found[i]) {
      frame.flags |= 1UL << (FLAG_NOT_FOUND_BASE + i);
      okay = false;
      continue;
    }
    frame.sensor[i].start_status = startMeasurement(i);
    if (statusHasError(frame.sensor[i].start_status)) {
      frame.flags |= 1UL << (FLAG_STATUS_ERROR_BASE + i);
      okay = false;
    }
    if ((frame.sensor[i].start_status & STATUS_SED) != 0)
      frame.flags |= 1UL << (FLAG_SED_BASE + i);
  }

  delayMicroseconds(TXYZ_CONVERSION_US);

  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    if (!sensor_found[i]) continue;
    frame.sensor[i].read_status = readMeasurement(i, frame.sensor[i]);
    if (statusHasError(frame.sensor[i].read_status)) {
      frame.flags |= 1UL << (FLAG_STATUS_ERROR_BASE + i);
      okay = false;
    }
    if ((frame.sensor[i].read_status & STATUS_SED) != 0)
      frame.flags |= 1UL << (FLAG_SED_BASE + i);
    if (isSaturated(frame.sensor[i])) {
      frame.flags |= 1UL << (FLAG_SATURATION_BASE + i);
      okay = false;
    }
  }
  return okay;
}

bool configureOne(size_t i) {
  if (!mlx[i].begin_SPI(SENSOR_CS_PINS[i], &SPI)) return false;

  const bool configured =
      mlx[i].setGain(SENSOR_GAIN) &&
      mlx[i].setResolution(MLX90393_X, SENSOR_RESOLUTION) &&
      mlx[i].setResolution(MLX90393_Y, SENSOR_RESOLUTION) &&
      mlx[i].setResolution(MLX90393_Z, SENSOR_RESOLUTION) &&
      mlx[i].setOversampling(SENSOR_OSR) && mlx[i].setFilter(SENSOR_FILTER) &&
      mlx[i].setTrigInt(false);
  if (!configured) return false;

  uint16_t conf1 = 0;
  uint16_t conf3 = 0;
  const uint8_t status1 = readRegister(i, MLX90393_CONF1, conf1);
  const uint8_t status3 = readRegister(i, MLX90393_CONF3, conf3);
  const bool register_match = ((conf1 >> 4) & 0x07) == SENSOR_GAIN &&
                              ((conf3 >> 5) & 0x03) == SENSOR_RESOLUTION &&
                              ((conf3 >> 7) & 0x03) == SENSOR_RESOLUTION &&
                              ((conf3 >> 9) & 0x03) == SENSOR_RESOLUTION &&
                              ((conf3 >> 2) & 0x07) == SENSOR_FILTER &&
                              (conf3 & 0x03) == SENSOR_OSR;
  return !statusHasError(status1) && !statusHasError(status3) && register_match;
}

size_t configureSensors() {
  const uint32_t saved_hz = spi_hz;
  spi_hz = DEFAULT_SPI_HZ;
  size_t found = 0;
  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    sensor_found[i] = configureOne(i);
    if (sensor_found[i]) ++found;
  }
  spi_hz = saved_hz;
  return found;
}

void printHelp() {
  Serial.println();
  Serial.println("Dermis MLX90393 array test menu");
  Serial.println("  p  probe/reset and bus-isolation test");
  Serial.println("  c  dump configuration and scale");
  Serial.println("  n  10-second noise test");
  Serial.println("  s  stream CSV until any key");
  Serial.println("  r  1/2/4/8 MHz rate sweep");
  Serial.println("  t  read temperature");
  Serial.println("  l  HX711 load-cell mode (any key exits)");
  Serial.println("  f  select next runtime SPI clock (1/2/4/8/10 MHz)");
  Serial.println("  ?  this help");
  Serial.printf("Current SPI clock: %lu Hz\n", static_cast<unsigned long>(spi_hz));
}

void probeCommand() {
  Serial.println("# PROBE: exit, reset, exit, and read CONF1");
  bool all_ok = true;
  size_t found = 0;

  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    const uint8_t ex_before = simpleCommand(i, CMD_EX);
    delay(1);
    const uint8_t reset = simpleCommand(i, CMD_RT);
    delay(2);
    const uint8_t ex_after = simpleCommand(i, CMD_EX);
    uint16_t conf1 = 0;
    const uint8_t read_status = readRegister(i, MLX90393_CONF1, conf1);
    const bool present = !statusHasError(ex_after) &&
                         !statusHasError(read_status) &&
                         (conf1 & 0x000F) == MLX90393_HALL_CONF;
    sensor_found[i] = present;
    if (present) ++found;
    all_ok &= present;

    Serial.printf("S%u CS=%d %s ex-before=", static_cast<unsigned>(i + 1),
                  SENSOR_CS_PINS[i], present ? "FOUND" : "NOT FOUND");
    printStatus(ex_before);
    Serial.print(" reset=");
    printStatus(reset);
    Serial.print(" ex-after=");
    printStatus(ex_after);
    Serial.print(" rr=");
    printStatus(read_status);
    Serial.printf(" CONF1=0x%04X\n", conf1);
  }

  Serial.println("# BUS ISOLATION: volatile scratch tags (no EEPROM write)");
  bool isolation_ok = true;
  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    if (!sensor_found[i]) continue;
    const uint16_t tag = static_cast<uint16_t>(0xA501U + i);
    const uint8_t status = writeRegister(i, 0x0A, tag);
    if (statusHasError(status)) isolation_ok = false;
  }
  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    if (!sensor_found[i]) continue;
    uint16_t value = 0;
    const uint8_t status = readRegister(i, 0x0A, value);
    const uint16_t expected = static_cast<uint16_t>(0xA501U + i);
    const bool unique = !statusHasError(status) && value == expected;
    isolation_ok &= unique;
    Serial.printf("S%u tag=0x%04X expected=0x%04X status=",
                  static_cast<unsigned>(i + 1), value, expected);
    printStatus(status);
    Serial.printf(" %s\n", unique ? "PASS" : "FAIL (CS short/floating?)");
  }

  // With every CS high, a correctly tri-stated MISO follows weak internal
  // pulls. A driven/stuck line fails this contention check.
  for (size_t i = 0; i < SENSOR_COUNT; ++i) digitalWrite(SENSOR_CS_PINS[i], HIGH);
  pinMode(PIN_SPI_MISO, INPUT_PULLUP);
  delayMicroseconds(20);
  const int pulled_high = digitalRead(PIN_SPI_MISO);
  pinMode(PIN_SPI_MISO, INPUT_PULLDOWN);
  delayMicroseconds(20);
  const int pulled_low = digitalRead(PIN_SPI_MISO);
  pinMode(PIN_SPI_MISO, INPUT);
  const bool miso_released = pulled_high == HIGH && pulled_low == LOW;
  isolation_ok &= miso_released;
  Serial.printf("MISO release check: pull-up=%d pull-down=%d %s\n", pulled_high,
                pulled_low, miso_released ? "PASS" : "FAIL (contention/stuck)");

  const size_t configured = configureSensors();
  all_ok &= isolation_ok && configured == SENSOR_COUNT;
  Serial.printf("PROBE SUMMARY: %s (%u/4 found, isolation=%s, configured=%u/4)\n",
                all_ok ? "PASS" : "FAIL", static_cast<unsigned>(found),
                isolation_ok ? "PASS" : "FAIL",
                static_cast<unsigned>(configured));
}

const char *gainName(uint8_t gain) {
  static const char *const names[] = {"5x", "4x", "3x", "2.5x", "2x",
                                     "1.67x", "1.33x", "1x"};
  return gain < 8 ? names[gain] : "invalid";
}

void configCommand() {
  bool all_ok = true;
  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    uint16_t conf1 = 0;
    uint16_t conf2 = 0;
    uint16_t conf3 = 0;
    const uint8_t s1 = readRegister(i, MLX90393_CONF1, conf1);
    const uint8_t s2 = readRegister(i, MLX90393_CONF2, conf2);
    const uint8_t s3 = readRegister(i, MLX90393_CONF3, conf3);
    const bool ok = sensor_found[i] && !statusHasError(s1) &&
                    !statusHasError(s2) && !statusHasError(s3);
    all_ok &= ok;
    const uint8_t gain = (conf1 >> 4) & 0x07;
    const uint8_t res_x = (conf3 >> 5) & 0x03;
    const uint8_t res_y = (conf3 >> 7) & 0x03;
    const uint8_t res_z = (conf3 >> 9) & 0x03;
    const uint8_t filter = (conf3 >> 2) & 0x07;
    const uint8_t osr = conf3 & 0x03;
    const uint8_t osr2 = (conf3 >> 12) & 0x03;
    Serial.printf(
        "S%u %s CONF1=0x%04X CONF2=0x%04X CONF3=0x%04X gain=%s(%u) "
        "res[x,y,z]=%u,%u,%u OSR=%u OSR2=%u filter=%u "
        "scale[uT/LSB]=%.3f,%.3f,%.3f\n",
        static_cast<unsigned>(i + 1), ok ? "PASS" : "FAIL", conf1, conf2,
        conf3, gainName(gain), gain, res_x, res_y, res_z, osr, osr2, filter,
        XY_UT_PER_LSB, XY_UT_PER_LSB, Z_UT_PER_LSB);
  }
  Serial.println("Digital range: X/Y +/-78.74 mT, Z +/-126.87 mT.");
  Serial.println("WARNING: standard ABA-0xx analog saturation onset is about 50 mT.");
  Serial.printf("CONFIG SUMMARY: %s\n", all_ok ? "PASS" : "FAIL");
}

void noiseCommand() {
  Serial.println("# NOISE: keep every patch untouched for 10 seconds...");
  OnlineStats stats[SENSOR_COUNT][3];
  for (size_t i = 0; i < SENSOR_COUNT; ++i)
    for (size_t axis = 0; axis < 3; ++axis) stats[i][axis].reset();

  uint32_t frames = 0;
  uint32_t errors = 0;
  uint32_t saturation = 0;
  const int64_t start_us = esp_timer_get_time();
  while (esp_timer_get_time() - start_us < 10000000LL) {
    Frame frame{};
    if (!acquireFrame(frame)) ++errors;
    for (size_t i = 0; i < SENSOR_COUNT; ++i) {
      if (!sensor_found[i]) continue;
      stats[i][0].add(frame.sensor[i].x_ut);
      stats[i][1].add(frame.sensor[i].y_ut);
      stats[i][2].add(frame.sensor[i].z_ut);
      if ((frame.flags & (1UL << (FLAG_SATURATION_BASE + i))) != 0)
        ++saturation;
    }
    ++frames;
  }
  const double elapsed_s = (esp_timer_get_time() - start_us) / 1000000.0;
  static const char axis_name[] = {'X', 'Y', 'Z'};
  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    for (size_t axis = 0; axis < 3; ++axis) {
      const OnlineStats &v = stats[i][axis];
      Serial.printf("S%u%c n=%lu mean=%.3f uT std=%.3f uT min=%.3f uT max=%.3f uT\n",
                    static_cast<unsigned>(i + 1), axis_name[axis],
                    static_cast<unsigned long>(v.count), v.mean, v.stddev(),
                    v.minimum, v.maximum);
    }
  }
  const bool pass = errors == 0 && saturation == 0;
  Serial.printf("NOISE SUMMARY: %s frames=%lu rate=%.1f fps errors=%lu saturation=%lu\n",
                pass ? "PASS" : "FAIL", static_cast<unsigned long>(frames),
                frames / elapsed_s, static_cast<unsigned long>(errors),
                static_cast<unsigned long>(saturation));
}

void emitFrame(const Frame &frame) {
  char line[384];
  const int length = snprintf(
      line, sizeof(line),
      "%lld,%.3f,%.3f,%.3f,%.2f,%.3f,%.3f,%.3f,%.2f,"
      "%.3f,%.3f,%.3f,%.2f,%.3f,%.3f,%.3f,%.2f,0x%08lX\n",
      static_cast<long long>(frame.t_us), frame.sensor[0].x_ut,
      frame.sensor[0].y_ut, frame.sensor[0].z_ut, frame.sensor[0].t_c,
      frame.sensor[1].x_ut, frame.sensor[1].y_ut, frame.sensor[1].z_ut,
      frame.sensor[1].t_c, frame.sensor[2].x_ut, frame.sensor[2].y_ut,
      frame.sensor[2].z_ut, frame.sensor[2].t_c, frame.sensor[3].x_ut,
      frame.sensor[3].y_ut, frame.sensor[3].z_ut, frame.sensor[3].t_c,
      static_cast<unsigned long>(frame.flags));
  if (length > 0) {
    const size_t bytes = static_cast<size_t>(length) < sizeof(line)
                             ? static_cast<size_t>(length)
                             : sizeof(line) - 1;
    Serial.write(reinterpret_cast<const uint8_t *>(line), bytes);
  }
}

void streamCommand() {
  while (Serial.available()) Serial.read();
  Serial.println("#BEGIN,t_us,s1x_uT,s1y_uT,s1z_uT,s1t_C,s2x_uT,s2y_uT,s2z_uT,s2t_C,s3x_uT,s3y_uT,s3z_uT,s3t_C,s4x_uT,s4y_uT,s4z_uT,s4t_C,flags");
  uint32_t interval_frames = 0;
  uint32_t interval_errors = 0;
  uint32_t interval_saturation = 0;
  uint32_t total_frames = 0;
  uint32_t total_errors = 0;
  uint32_t total_saturation = 0;
  int64_t interval_start = esp_timer_get_time();

  while (Serial.available() == 0) {
    Frame frame{};
    if (!acquireFrame(frame)) {
      ++interval_errors;
      ++total_errors;
    }
    for (size_t i = 0; i < SENSOR_COUNT; ++i)
      if ((frame.flags & (1UL << (FLAG_SATURATION_BASE + i))) != 0) {
        ++interval_saturation;
        ++total_saturation;
      }
    emitFrame(frame);
    ++interval_frames;
    ++total_frames;

    const int64_t now = esp_timer_get_time();
    if (now - interval_start >= 1000000LL) {
      const double seconds = (now - interval_start) / 1000000.0;
      Serial.printf("#RATE,fps=%.1f,errors=%lu,saturation=%lu,spi_hz=%lu\n",
                    interval_frames / seconds,
                    static_cast<unsigned long>(interval_errors),
                    static_cast<unsigned long>(interval_saturation),
                    static_cast<unsigned long>(spi_hz));
      interval_frames = 0;
      interval_errors = 0;
      interval_saturation = 0;
      interval_start = now;
    }
  }
  while (Serial.available()) Serial.read();
  Serial.println("#END");
  const bool pass = total_errors == 0 && total_saturation == 0;
  Serial.printf("STREAM SUMMARY: %s frames=%lu errors=%lu saturation=%lu "
                "(stopped by host key)\n",
                pass ? "PASS" : "FAIL", static_cast<unsigned long>(total_frames),
                static_cast<unsigned long>(total_errors),
                static_cast<unsigned long>(total_saturation));
}

void rateSweepCommand() {
  constexpr uint32_t rates[] = {1000000, 2000000, 4000000, 8000000};
  bool pass = true;
  Serial.println("# RATE SWEEP: acquisition only (CSV output disabled)");
  for (const uint32_t rate : rates) {
    spi_hz = rate;
    uint32_t frames = 0;
    uint32_t errors = 0;
    const int64_t start = esp_timer_get_time();
    while (esp_timer_get_time() - start < 2000000LL) {
      Frame frame{};
      if (!acquireFrame(frame)) ++errors;
      ++frames;
    }
    const double seconds = (esp_timer_get_time() - start) / 1000000.0;
    Serial.printf("RATE %lu MHz: %.1f fps, frames=%lu, errors=%lu %s\n",
                  static_cast<unsigned long>(rate / 1000000), frames / seconds,
                  static_cast<unsigned long>(frames),
                  static_cast<unsigned long>(errors), errors == 0 ? "PASS" : "FAIL");
    pass &= errors == 0;
  }
  spi_hz = DEFAULT_SPI_HZ;
  Serial.printf("RATE SWEEP SUMMARY: %s (runtime clock restored to 1 MHz)\n",
                pass ? "PASS" : "FAIL");
}

void temperatureCommand() {
  bool pass = true;
  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    if (!sensor_found[i]) {
      Serial.printf("S%u NOT FOUND\n", static_cast<unsigned>(i + 1));
      pass = false;
      continue;
    }
    Sample sample{};
    sample.start_status = startMeasurement(i, AXIS_T);
    delayMicroseconds(T_ONLY_CONVERSION_US);
    sample.read_status = readMeasurement(i, sample, AXIS_T);
    const bool ok = !statusHasError(sample.start_status) &&
                    !statusHasError(sample.read_status);
    pass &= ok;
    Serial.printf("S%u %s raw=%u temperature=%.2f C start=",
                  static_cast<unsigned>(i + 1), ok ? "PASS" : "FAIL",
                  sample.raw_t, sample.t_c);
    printStatus(sample.start_status);
    Serial.print(" read=");
    printStatus(sample.read_status);
    Serial.println();
  }
  Serial.printf("TEMPERATURE SUMMARY: %s\n", pass ? "PASS" : "FAIL");
}

void nextFrequencyCommand() {
  constexpr uint32_t rates[] = {1000000, 2000000, 4000000, 8000000, 10000000};
  size_t index = 0;
  for (size_t i = 0; i < sizeof(rates) / sizeof(rates[0]); ++i) {
    if (rates[i] == spi_hz) {
      index = (i + 1) % (sizeof(rates) / sizeof(rates[0]));
      break;
    }
  }
  spi_hz = rates[index];
  if (spi_hz > MAX_SPI_HZ) spi_hz = MAX_SPI_HZ;
  Serial.printf("SPI CLOCK: %lu Hz (PASS)\n", static_cast<unsigned long>(spi_hz));
}

void loadCellCommand() {
  Serial.println("# LOAD CELL MODE: MLX acquisition paused");
  Serial.printf("# HX711 wiring: DOUT=GPIO%d SCK=GPIO%d VCC=3.3V\n",
                PIN_HX711_DOUT, PIN_HX711_SCK);
  Serial.println("# Remove all force now; taring in 2 seconds...");

  // Put each MLX90393 in idle mode and deselect every device. Leave the SPI
  // peripheral initialized so the existing MLX driver instances remain valid;
  // no MLX transactions occur while this blocking command is running.
  for (size_t i = 0; i < SENSOR_COUNT; ++i) {
    if (sensor_found[i]) simpleCommand(i, CMD_EX);
    digitalWrite(SENSOR_CS_PINS[i], HIGH);
  }
  load_cell.begin(PIN_HX711_DOUT, PIN_HX711_SCK);
  delay(2000);
  while (Serial.available()) Serial.read();

  if (!load_cell.wait_ready_timeout(2000)) {
    Serial.println("LOAD CELL ERROR: HX711 not ready. Check 3.3V, GND, DOUT, and SCK.");
    load_cell.power_down();
    const size_t restored = configureSensors();
    Serial.printf("MLX RESTORE: configured %u/4 sensors\n",
                  static_cast<unsigned>(restored));
    return;
  }

  constexpr uint8_t TARE_SAMPLES = 20;
  const int64_t tare_start_us = esp_timer_get_time();
  const int64_t zero = load_cell.read_average(TARE_SAMPLES);
  const int64_t tare_elapsed_us = esp_timer_get_time() - tare_start_us;
  Serial.printf("# TARE,zero_raw=%lld,samples=%u\n",
                static_cast<long long>(zero), TARE_SAMPLES);

  // Even at 80 samples/s, 20 conversions require about 250 ms. A zero result
  // completed almost instantly means DOUT is held low and the library shifted
  // 24 zero bits rather than receiving conversions from an HX711.
  if (zero == 0 && tare_elapsed_us < 100000) {
    Serial.printf(
        "LOAD CELL ERROR: DOUT stuck LOW/all-zero data "
        "(20 reads in %lld us; valid HX711 maximum is 80 Hz).\n",
        static_cast<long long>(tare_elapsed_us));
    Serial.println(
        "# Check HX711 3.3V-GND, common GND, DT->GPIO1, and SCK->GPIO2.");
    load_cell.power_down();
    const size_t restored = configureSensors();
    Serial.printf("MLX RESTORE: configured %u/4 sensors (%s)\n",
                  static_cast<unsigned>(restored),
                  restored == SENSOR_COUNT ? "PASS" : "FAIL");
    return;
  }
  Serial.println("#BEGIN_LOAD_CELL,t_us,raw,delta_counts");

  uint32_t samples = 0;
  uint32_t timeouts = 0;
  int64_t minimum = INT64_MAX;
  int64_t maximum = INT64_MIN;
  const int64_t start_us = esp_timer_get_time();

  while (Serial.available() == 0) {
    if (!load_cell.wait_ready_timeout(250)) {
      ++timeouts;
      Serial.println("#HX711_TIMEOUT");
      continue;
    }
    const int64_t raw = load_cell.read();
    const int64_t delta = raw - zero;
    if (delta < minimum) minimum = delta;
    if (delta > maximum) maximum = delta;
    ++samples;
    Serial.printf("%lld,%lld,%lld\n",
                  static_cast<long long>(esp_timer_get_time()),
                  static_cast<long long>(raw),
                  static_cast<long long>(delta));
  }
  while (Serial.available()) Serial.read();

  const double elapsed_s =
      static_cast<double>(esp_timer_get_time() - start_us) / 1000000.0;
  load_cell.power_down();
  Serial.println("#END_LOAD_CELL");
  Serial.printf(
      "LOAD CELL SUMMARY: samples=%lu rate=%.1f Hz timeouts=%lu "
      "delta_min=%lld delta_max=%lld units=ADC_counts\n",
      static_cast<unsigned long>(samples),
      elapsed_s > 0.0 ? samples / elapsed_s : 0.0,
      static_cast<unsigned long>(timeouts),
      static_cast<long long>(samples > 0 ? minimum : 0),
      static_cast<long long>(samples > 0 ? maximum : 0));

  // Restore the MLX bus and configuration before returning to the menu.
  driveAllCsHigh();
  const size_t restored = configureSensors();
  Serial.printf("MLX RESTORE: configured %u/4 sensors (%s)\n",
                static_cast<unsigned>(restored),
                restored == SENSOR_COUNT ? "PASS" : "FAIL");
}

}  // namespace

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(250);

  // HARD REQUIREMENT: all CS pins are output-high before the SPI peripheral is
  // initialized, preventing any MLX90393 from latching the wrong bus protocol.
  driveAllCsHigh();
  SPI.begin(PIN_SPI_SCLK, PIN_SPI_MISO, PIN_SPI_MOSI, -1);

  Serial.println();
  Serial.println("Dermis MLX90393 array bring-up firmware");
  const size_t found = configureSensors();
  Serial.printf("Startup: configured %u/4 sensors at %lu Hz (%s)\n",
                static_cast<unsigned>(found),
                static_cast<unsigned long>(spi_hz),
                found == SENSOR_COUNT ? "PASS" : "FAIL - run p");
  printHelp();
}

void loop() {
  if (!Serial.available()) {
    delay(1);
    return;
  }
  const char command = static_cast<char>(Serial.read());
  switch (command) {
    case 'p': probeCommand(); break;
    case 'c': configCommand(); break;
    case 'n': noiseCommand(); break;
    case 's': streamCommand(); break;
    case 'r': rateSweepCommand(); break;
    case 't': temperatureCommand(); break;
    case 'l': loadCellCommand(); break;
    case 'f': nextFrequencyCommand(); break;
    case '?': printHelp(); break;
    case '\r':
    case '\n': break;
    default:
      Serial.printf("Unknown command '%c'. Press ? for help.\n", command);
      break;
  }
}
