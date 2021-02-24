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
