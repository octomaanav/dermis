#pragma once

#include <Arduino.h>

// Dermis wiring. Keep every hardware pin in this file.
constexpr int PIN_SPI_SCLK = 18;
constexpr int PIN_SPI_MISO = 14;
constexpr int PIN_SPI_MOSI = 13;

constexpr int PIN_CS_S1 = 21;
constexpr int PIN_CS_S2 = 4;
constexpr int PIN_CS_S3 = 16;
constexpr int PIN_CS_S4 = 17;

constexpr size_t SENSOR_COUNT = 4;
constexpr int SENSOR_CS_PINS[SENSOR_COUNT] = {
    PIN_CS_S1,
    PIN_CS_S2,
    PIN_CS_S3,
    PIN_CS_S4,
};

