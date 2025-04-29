import time

with open("output.txt", "a") as f:
    while True:
        f.write("Je suis encore en train de tourner...\n")
        f.flush()  # Force l'écriture immédiate
        time.sleep(5)
