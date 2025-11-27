# plot_demo_raw.py
import os
from data_loader import load_raw_bags, filter_bags,group_by_sensor_name
from plotting import plot_time_series,plot_frequency_spectrum

def main():
    root = os.getenv("STAT_AI_DATA", "/Users/zeynepoztunc/Downloads/Sensor_STWIN")

    bags = load_raw_bags(root, limit=200, verbose=False)
    print("Total bags loaded:", len(bags))

    # Example: KO acc sensors → plot all axes
    ko_acc_bags = filter_bags(bags, sensor_type="acc", belt_status="KO_LOW_4mm")

    #group filtered sensor by their names (such as iis3dwb_acc', 'iis2dh_acc', 'ism330dhcx_acc' )
    grouped_acc = group_by_sensor_name(ko_acc_bags)#this is a dictionary

    representatives = {
        name: cycles[0] #key = sensor name and value = the first recording for that sensor
        for name, cycles in grouped_acc.items()
    }

    for bag in representatives.values():
        plot_time_series(bag)

    print("KO acc bags:", [b["sensor"] for b in ko_acc_bags])
    print("grouped acc sensors (grouped):", list(grouped_acc.keys()))


    # Example: temp sensors → axis ignored
    temp_bags = filter_bags(bags, sensor_type="temp",belt_status="KO_LOW_4mm")
    grouped_temp = group_by_sensor_name(temp_bags)#this is a dictionary

    representatives = {
        name: cycles[0] #key = sensor name and value = the first recording for that sensor
        for name, cycles in grouped_temp.items()
    }
    for bag in representatives.values():
        plot_time_series(bag)

    print();
    print("Temp sensors (grouped):", list(grouped_temp.keys()))



    # # Example: mic sensors
    print();
    mic_bags = filter_bags(bags, sensor_type="mic",belt_status="KO_LOW_4mm")
    grouped_mic = group_by_sensor_name(mic_bags)#this is a dictionary

    representatives = {
        name: cycles[0] #key = sensor name and value = the first recording for that sensor
        for name, cycles in grouped_mic.items()
    }
    for bag in representatives.values():
        plot_time_series(bag)

    print();
    print("mic sensors (grouped):", list(grouped_mic.keys()))


    #ko_acc_bags = filter_bags(bags, sensor_type="acc", belt_status="KO_LOW_4mm")
    #if ko_acc_bags:
    #    plot_frequency_spectrum(ko_acc_bags[0], axis="x")   # FFT for x-axis
    

if __name__ == "__main__":
    main()

