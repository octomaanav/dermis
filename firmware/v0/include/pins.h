#pragma once

#include <Arduino.h>

// Dermis wiring. Keep every hardware pin in this file.
constexpr int PIN_SPI_SCLK = 18;
constexpr int PIN_SPI_MISO = 13;
constexpr int PIN_SPI_MOSI = 14;

constexpr int PIN_CS_S1 = 21;
constexpr int PIN_CS_S2 = 4;
constexpr int PIN_CS_S3 = 16;
constexpr int PIN_CS_S4 = 17;

// One BMI160 shares SCLK/MISO/MOSI with the MLX90393 array.
constexpr size_t IMU_COUNT = 1;
constexpr int IMU_CS_PINS[IMU_COUNT] = {9};

// HX711 load-cell amplifier. Power the board from 3.3 V, not 5 V.
constexpr int PIN_HX711_DOUT = 1;
constexpr int PIN_HX711_SCK = 2;

constexpr size_t SENSOR_COUNT = 4;
constexpr int SENSOR_CS_PINS[SENSOR_COUNT] = {
    PIN_CS_S1,
    PIN_CS_S2,
    PIN_CS_S3,
    PIN_CS_S4,
};

