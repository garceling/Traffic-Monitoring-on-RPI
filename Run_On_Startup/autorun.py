# Write your code here :-)
from gpiozero import Button
from time import sleep
from tkinter.messagebox import *

button=Button(4)

while True:
    if button.is_pressed:
        print("Pressed")
    else:
        print("Released")
    sleep(1)

    showinfo(title="Greetings", message="Pressed")