# plot_demo_raw.py
import os
from data_loader import load_raw_bags, filter_bags
from plotting import plot_time_series

def main():
    root = os.getenv("STAT_AI_DATA", "/Users/zeynepoztunc/Downloads/Sensor_STWIN")

    bags = load_raw_bags(root, limit=200, verbose=False)
    print("Total bags loaded:", len(bags))

    # Example: KO acc sensors → plot all axes
    ko_acc_bags = filter_bags(bags, sensor_type="acc", belt_status="KO_LOW_4mm")
    print("KO acc bags:", [b["sensor"] for b in ko_acc_bags])
    if ko_acc_bags:
        plot_time_series(ko_acc_bags[0], axis="y")

    # # Example: temp sensors → axis ignored
    temp_bags = filter_bags(bags, sensor_type="temp")
    print("Temp sensors:", [b["sensor"] for b in temp_bags])
    if temp_bags:
        plot_time_series(temp_bags[1])   # no axis needed

    # # Example: mic sensors
    mic_bags = filter_bags(bags, sensor_type="mic")
    print("Mic sensors:", [b["sensor"] for b in mic_bags])
    if mic_bags:
        plot_time_series(mic_bags[0])
if __name__ == "__main__":
    main()

