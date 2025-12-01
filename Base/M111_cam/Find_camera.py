from arena_api.system import system

device_infos = system.device_infos

if not device_infos:
    print("No devices found.")
else:
    print("Raw device info:\n", device_infos[0])
