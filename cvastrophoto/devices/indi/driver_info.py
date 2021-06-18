def canonic_driver_name(name):
    return name.lower().strip()


DYNAMIC_PULSE_SUPPORT = {
    'guide simulator',
    'ccd simulator',
    'telescope simulator',
}


def has_dynamic_pulse_support(device):
    cname = canonic_driver_name(device.properties.get('DRIVER_INFO', [''])[0])
    return cname in DYNAMIC_PULSE_SUPPORT


def _flatten(value):
    if isinstance(value, dict):
        return _flatten(list(value.values()))
    elif isinstance(value, (list, tuple)):
        return list(value)
    else:
        return [value]


def lookup_override(overrides, *values):
    for svalues in values:
        for value in _flatten(svalues):
            if value in overrides:
                return overrides[value]


BAUD_RATE_OVERRIDES = {
    # device name or driver name to baud override
}


def default_baud_rate(device_port):
    if '/ttyUSB' in device_port:
        return 115200
    else:
        return 9600


def guess_baud_rate(device_name, device_port, driver_info):
    override = lookup_override(BAUD_RATE_OVERRIDES, driver_info, device_port)
    if override:
        return override

    return default_baud_rate(device_port)
