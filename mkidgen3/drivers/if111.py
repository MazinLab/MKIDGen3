#!/usr/bin/env python

#
# IF111 - An Interface Board for ZCU111
# * draft support package
#

import serial
import time
import json

class IF111(object):

    timeout = 300   # in 10ms
    baudrate = 115200
    message = ''
    setting = {'dac_att':[None, None], 'adc_att':[None, None], 'lo_freq': [None, None]}

    def __init__(self, port):
        self.port = port
        self.ser = serial.Serial()
    
    def open(self, port=None):
        if port is not None:
            self.port = port
            
        self.close()
        self.ser = serial.Serial(port=self.port, baudrate=self.baudrate)
        
        return self.ser.is_open
    
    def close(self):
        try:
            if self.ser.is_open:
                self.ser.close()
        except:
            pass
    
    def send_cmd(self, cmd):
        try:
            # check and open the serial port
            if not self.ser.is_open:
                if not self.open():
                    return None
            
            self.ser.reset_input_buffer()
            self.ser.write(cmd)
            timeout = self.timeout
            while self.ser.in_waiting <= 0:
                time.sleep(0.01) 
                timeout -= 1
                if timeout == 0:
                    self.message = "IF board timeout"
                    return None                    
        except:
            self.message = "IF board error"
            print(self.message)
            return None
            
        self.message = self.ser.read(self.ser.in_waiting).decode("utf-8")
        
        return self.message
        
    def set_power(self, on):
        if on:
            return self.send_cmd(b'p\n') is not None
        else:
            return self.send_cmd(b'x\n') is not None

    def set_lo_hz(self, freq_hz, ch=0):
        freq_set = b'f%d %d\n' % (int(ch), int(freq_hz))
        self.setting['lo_freq'][ch] = freq_hz
        return self.send_cmd(freq_set) is not None
        
    def set_lo_mhz(self, freq_mhz, ch=0):
        return self.set_lo_hz(freq_mhz*1000000, ch)
    
    def set_dac_att(self, att_db, ch=0):
        dac_set = b'd%d %d\n' % (int(ch), int(att_db))
        self.setting['dac_att'][ch] = att_db
        return self.send_cmd(dac_set) is not None
        
    def set_adc_att(self, att_db, ch=0):
        adc_set = b'a%d %d\n' % (int(ch), int(att_db))
        self.setting['adc_att'][ch] = att_db
        return self.send_cmd(adc_set) is not None

    def get_lo_hz(self, ch=0):
        return self.setting['lo_freq'][ch]
        
    def dump(self):
        if self.send_cmd(b'v\n') is not None:
            print(self.message)
        else:
            print("IF111 timeout")

    def update(self):
        ret = True

        for i in range(2):
            if self.setting['lo_freq'][i] is not None:
                ret = ret and self.set_lo_hz(int(self.setting['lo_freq'][i]), i)
            if self.setting['adc_att'][i] is not None:
                ret = ret and self.set_adc_att(int(self.setting['adc_att'][i]), i)
            if self.setting['dac_att'][i] is not None:
                ret = ret and self.set_dac_att(int(self.setting['dac_att'][i]), i)

        return ret        

    def to_json(self):
        return json.dumps(self.setting)

    def from_json(self, json_str):
        self.setting  = json.loads(json_str)
        return self.update() 
        
    def from_setting(self, setting):
        self.setting = setting
        return self.update()
