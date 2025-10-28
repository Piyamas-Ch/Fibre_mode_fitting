from select_region_of_interest import select_region_of_interest
from gaussian_fitting import gaussian_fitting
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
import os

def input_fit_file():
    class WideInputDialog(simpledialog.Dialog):
        def body(self, master):
            tk.Label(master, text="Please enter fit file path:").pack(anchor="w", padx=10, pady=10)
            self.entry = tk.Entry(master, width=60)  # width in characters
            self.entry.pack(padx=10)
            return self.entry

        def apply(self):
            self.result = self.entry.get()

    root = tk.Tk()
    root.withdraw() # Hide the main window

    input_flag = 1
    while input_flag == 1:
        dialog = WideInputDialog(root, title="Input")
        user_fit_file = dialog.result
        if user_fit_file is None:
            # If user close window or click cancel, stop input value
            input_flag = 0
        elif os.path.isfile(user_fit_file) == False:
            # If user input fit file that does not exist, warning to user input again
            messagebox.showwarning("Warning", "File does not exist. Please input again.")
        else:
            # If user input existing fit file, stop input value
            input_flag = 0

    root.destroy() # Close the hidden root window

    return user_fit_file

def read_config(config_file):
    # Read config file to get current value of radius, sensor size (pixel) and sensor magnification
    with open(config_file, "r") as f:
        config_name = []
        config_value = []
        for line in f:
            words = line.strip()
            config_name.append(words.split("=")[0])
            config_value.append(words.split("=")[1])

    return config_name, config_value

def get_user_input(config_name, config_value, config_file):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Request user input radius value
    user_r = simpledialog.askfloat("Input", f"x_c = {x_c}, y_c = {y_c}\nPlease enter radius:", initialvalue=config_value[0])
    if user_r is None:
        # If user close window or click cancel, stop input value
        user_sensor_pixel = None
        user_sensor_M = None
    else:
        # Request user input sensor size (pixel) value
        user_sensor_pixel = simpledialog.askfloat("Input", "Please enter sensor pixel:", initialvalue=config_value[1])
        if user_sensor_pixel is None:
            # If user close window or click cancel, stop input value
            user_sensor_M = None
        else:
            # Reqeust user input sensor magnification value
            user_sensor_M = simpledialog.askfloat("Input", "Please enter sensor magnification:", initialvalue=config_value[2])
            if user_sensor_M is not None:
                # If user input all value, check value changing
                # Update config value to config file, if config value was changed
                if user_r != config_value[0] or user_sensor_pixel != config_value[1] or user_sensor_M != config_value[2]:
                    with open(config_file, "w") as file:
                        file.write(f"{config_name[0]}={user_r}\n")
                        file.write(f"{config_name[1]}={user_sensor_pixel}\n")
                        file.write(f"{config_name[2]}={user_sensor_M}\n")

    root.destroy() # Close the hidden root window

    return user_r, user_sensor_pixel, user_sensor_M

config_file = 'config.txt'
# Request uset input fit file path
fit_file = input_fit_file()
if fit_file is not None:
    # Reqeust user select region of interest in fit file
    x_c, y_c = select_region_of_interest(fit_file)
    if x_c != 0 and y_c != 0:
        # Read config file to get current value of radius, sensor size (pixel) and sensor magnification
        config_name, config_value = read_config(config_file)
        # Reqeust user input radius, sensor size (pixel) and sensor magnification
        r, sensor_pixel, sensor_M = get_user_input(config_name, config_value, config_file)
        if r is not None and sensor_pixel is not None and sensor_M is not None:
            # Start Gaussian fitting
            gaussian_fitting(fit_file, r, [sensor_pixel, sensor_M], x_c, y_c)
