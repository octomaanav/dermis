# Stored MLX90393 firmware

`main.cpp` is the previous combined MLX90393-array and HX711 test firmware.
It is outside `src/`, so PlatformIO does not compile it while the bare HX711
test is active.

To restore it, swap this file with `../src/main.cpp` and set
`monitor_speed = 921600` in `../platformio.ini`. Also restore this dependency:

`adafruit/Adafruit MLX90393@2.0.5`
