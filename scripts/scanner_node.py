#!/usr/bin/env python3

from pynput import keyboard
import rospy
import time


class BarCodeScanner():
    def __init__(self):
        self._listener = keyboard.Listener(on_press=self.key_press_callback)
        self._listener.start()
        self._barcode = ""

    def key_press_callback(self, e):
        try:
            e_char = e.char
        except Exception as e:
            print(f"Key not converted for {e}")
            return
        if not e_char.isnumeric():
            self._barcode = ""
        else:
            self._barcode += e_char

    def barcode(self):
        return self._barcode

    def run(self):
        while True:
            if len(self._barcode) > 14:
                self._barcode = ""
            if len(self._barcode) > 2:
                rospy.set_param("/barcode", self.barcode())
            time.sleep(0.5)


if __name__ == "__main__":
    s = BarCodeScanner()
    s.run()