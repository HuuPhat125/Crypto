# **1. Finding Flags** 
## **1.1 Problem**
Each challenge is designed to help introduce you to a new piece of cryptography. Solving a challenge will require you to find a "flag".
These flags will usually be in the format crypto{y0ur_f1rst_fl4g}. The flag format helps you verify that you found the correct solution.
Try submitting this flag into the form below to solve your first challenge.
## **1.2 Solution**
crypto{y0ur_f1rst_fl4g}

# **2. Great Snakes**
## **1.1 Problem**
Modern cryptography involves code, and code involves coding. CryptoHack provides a good opportunity to sharpen your skills.
Of all modern programming languages, Python 3 stands out as ideal for quickly writing cryptographic scripts and attacks. For more information about why we think Python is so great for this, please see the FAQ.
Run the attached Python script and it will output your flag.
Challenge files:
  - great_snakes.py
Resources:
  - Downloading Python
## **2.2 Solution**
Download great_snakes.py file and run it with Python 3.

# **3. Network Attacks**
## **2.3 Solution**
Download telnetlib_example.py and change "clothes" in line 31 to "flag"

```py
b"Welcome to netcat's flag shop!\n"
b'What would you like to buy?\n'
b"I only speak JSON, I hope that's ok.\n"
b'\n'
{'flag': 'crypto{sh0pp1ng_f0r_fl4g5}'}
```

