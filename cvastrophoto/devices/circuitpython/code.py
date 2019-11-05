import time
import microcontroller
import board
import digitalio
from adafruit_circuitplayground.express import cpx

cpx.pixels.auto_write = False

ra_n_out = digitalio.DigitalInOut(board.D3)
dec_n_out = digitalio.DigitalInOut(board.D2)
dec_p_out = digitalio.DigitalInOut(board.D0)
ra_p_out = digitalio.DigitalInOut(board.D1)

ra_n_out.switch_to_output()
ra_p_out.switch_to_output()
dec_n_out.switch_to_output()
dec_p_out.switch_to_output()

max_rate = 0.5  # Lower rates increase control finesse, higher rates allow higher drift compensation
led_brightness = 0.02  # Very dim, better for preserving night vision
led_changed_brightness = 0.1  # Stronger after changing something, for more visibility
led_changed_time = 2  # Time after a change it will remain brighter
change_step = 0.01  # Smaller steps mean finer control, but slower changes

ra_rate = dec_rate = 0
ra_period = dec_period = 2
ra_state = dec_state = False
whichselected = 0  # 0=ra, 1=dec

ra_rate = 0.15

def pixel_intensity(rate, pixnum):
    if rate < 0:
        rate = -rate
        pixnum = 9 - pixnum
    rate /= max_rate
    lo = pixnum * 0.1
    hi = (pixnum + 1) * 0.1
    return max(0, min(1, (rate - lo) / (hi - lo)))

def update_pixels(now, last_rate_change):
    if now < last_rate_change + led_changed_time:
        cpx.pixels.brightness = led_changed_brightness
        selected = whichselected
        has_focus = True
    else:
        cpx.pixels.brightness = led_brightness
        selected = 2
        has_focus = False

    for pixnum in range(10):
        ralevel = max(0, min(255, int(pixel_intensity(ra_rate, pixnum) * 255)))
        declevel = max(0, min(255, int(pixel_intensity(dec_rate, pixnum) * 255)))
        cpx.pixels[pixnum] = (ralevel, declevel, (ralevel, declevel, 0)[selected])

    cpx.pixels.show()

    return has_focus

def update_output():
    cpx.red_led = ra_state or dec_state

    ra_n_out.value = ra_state and (ra_rate < 0)
    ra_p_out.value = ra_state and (ra_rate > 0)
    dec_n_out.value = dec_state and (dec_rate < 0)
    dec_p_out.value = dec_state and (dec_rate > 0)

def main():
    global ra_state, dec_state, ra_rate, dec_rate, ra_period, dec_period
    global whichselected

    monotonic = time.monotonic

    start = monotonic()
    ra_base = dec_base = last_rate_change = input_debounce = start
    has_focus = True

    update_pixels(start, last_rate_change)
    update_output()

    while True:
        now = monotonic()
        state_change = False

        if ra_rate:
            new_ra_state = (now - ra_base) < (abs(ra_rate) * ra_period)
        else:
            new_ra_state = False

        if dec_rate:
            new_dec_state = (now - dec_base) < (abs(dec_rate) * dec_period)
        else:
            new_dec_state = False

        if now > (ra_base + ra_period):
            ra_base += ra_period
        if now > (dec_base + dec_period):
            dec_base += dec_period

        if new_ra_state != ra_state:
            ra_state = new_ra_state
            state_change = True

        if new_dec_state != dec_state:
            dec_state = new_dec_state
            state_change = True

        if state_change:
            update_output()

        if now > input_debounce:
            rate_change = False
            if cpx.touch_A1:
                input_debounce = now + 0.1
                if whichselected:
                    dec_rate = min(max_rate, dec_rate + change_step)
                else:
                    ra_rate = min(max_rate, ra_rate + change_step)
                rate_change = True
            elif cpx.touch_A2:
                input_debounce = now + 0.1
                if whichselected:
                    dec_rate = max(-max_rate, dec_rate - change_step)
                else:
                    ra_rate = max(-max_rate, ra_rate - change_step)
                rate_change = True
            elif cpx.touch_A3:
                whichselected = 1 - whichselected
                rate_change = True

            input_debounce = now + 0.25

            if rate_change or has_focus:
                has_focus = update_pixels(now, last_rate_change)

            if rate_change:
                last_rate_change = now

                if abs(ra_rate) > change_step:
                    if abs(ra_rate) * max_rate * ra_period < 0.1:
                        ra_period *= 2
                    elif abs(ra_rate) * max_rate * ra_period > 0.25:
                        ra_period *= 0.5

                if abs(dec_rate) > change_step:
                    if abs(dec_rate) * max_rate * dec_period < 0.1:
                        dec_period *= 2
                    elif abs(dec_rate) * max_rate * dec_period > 0.25:
                        dec_period *= 0.5

                print(
                    "Updated ra", ra_rate, "period", ra_period,
                    "dec", dec_rate, "period", dec_period)

main()
