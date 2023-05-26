import RPi.GPIO as GPIO
import dht11
import bmpsensor
import time
import mq2

# initialize GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
#GPIO.cleanup()

# read data using Pin GPIO21 
instance = dht11.DHT11(pin=21)

while True:
    result = instance.read()
    temp,pres,alt=bmpsensor.readBmp180()
    COlevel= mq2.readadc(0, 11, 10, 9, 8)
    conc=str("%.2f"%((COlevel/1024.)*3.3))
    if result.is_valid():
        print("Temp: %d C" % result.temperature +' '+"Humid: %d %%" % result.humidity +' '+"Pressure: %d Pa" % pres +' '+"Altitude: %d m" % alt +' '+"Current Gas AD vaule: %s V" % conc)
        #'+"Current Gas AD vaule: %d V" %conc                                                                                     

    time.sleep(1)
GPIO.cleanup() 