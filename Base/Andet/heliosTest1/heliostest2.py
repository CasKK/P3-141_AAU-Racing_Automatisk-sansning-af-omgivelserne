from arena_api.system import system

devices = system.create_device()
print(f"Found {len(devices)} devices")
if devices:
    dev = devices[0]
    print("Camera:", dev.nodemap.get_node('DeviceModelName').value)