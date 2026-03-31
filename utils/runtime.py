import os


def normalize_device_config(device_cfg):
    if device_cfg is None:
        return []

    if not isinstance(device_cfg, (list, tuple)):
        device_cfg = [device_cfg]

    normalized = []
    for device in device_cfg:
        if isinstance(device, str):
            device = device.strip()
        normalized.append(str(device))
    return normalized


def configure_visible_devices(args):
    devices = normalize_device_config(args.get("device", []))
    args["device"] = devices

    if devices and all(device != "-1" for device in devices):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    return args
