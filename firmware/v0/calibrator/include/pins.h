#pragma once

#include <Arduino.h>

constexpr int PIN_SPI_SCLK = 18;
constexpr int PIN_SPI_MISO = 13;
constexpr int PIN_SPI_MOSI = 14;

constexpr size_t SENSOR_COUNT = 4;
constexpr int SENSOR_CS_PINS[SENSOR_COUNT] = {21, 4, 16, 17};

constexpr int PIN_HX711_DOUT = 42;
constexpr int PIN_HX711_SCK = 2;
