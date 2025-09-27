# 📊 Sensor Data for Belt Condition Anomaly Detection

This project analyzes sensor data from the **EQTQ-NEXT capper system** to detect anomalies in belt regulation during capping piston operations.

---

## 📦 Dataset Overview

The dataset was acquired using the **ST STEVAL-STWINKT1B** development board, which includes multiple onboard sensors.

Sensors used:

- `IIS3DWB`: Ultra-wideband 3-axis vibration sensor
- `ISM330DHCX`: 3D accelerometer + 3D gyroscope
- `IIS2DH`: Low-power 3-axis accelerometer
- `IIS2MDC`: 3-axis magnetometer
- `IMP34DT05` & `IMP23ABSU`: MEMS microphones
- `HTS221`: Temperature and humidity sensor
- `LPS22HH`, `STTS751`: Pressure and temperature sensors

---

## 🗂 Folder Structure

The extracted dataset contains two main subdirectories under `Sensor_STWIN/`, each representing a capping piston operating condition:

Each subfolder includes four **belt conditions**:

- `OK`: Newly adjusted belt
- `KO_HIGH_2mm`: Over-tightened belt
- `KO_LOW_2mm`: Slightly loose belt
- `KO_LOW_4mm`: Significantly loose belt

Inside each condition, data is grouped by RPM (e.g., `PMI_50rpm`, `PMS_100rpm`, etc.) and includes:

- Raw sensor data (`*.dat`)
- Metadata (`Acquisitioninfo.json`, `DeviceConfig.json`)

---

## 🛠 Usage Instructions

Due to size constraints (~4 GB), the dataset is **not stored in this GitHub repository**.

### 🔗 Download Link

[Download Sensor_STWIN ZIP (Google Drive)](https://drive.google.com/file/d/1kTVLvjYt6VXaEUNkX_yk3yqUrmNlWBDV/view?usp=sharing)

After downloading:

1. Unzip the archive into the data directory of the project.
2. You should now have a folder named `Sensor_STWIN/` under the data directory.

---
### 🔗 Download Link (Converted Dataset)

[Download Converted Sensor_STWIN Dataset (Google Drive)](https://drive.google.com/file/d/1B5StbnPc7qC2LkN_QUzkrDD5vxLPIFg1/view?usp=drive_link)

> ⚠️ Note: This archive contains the dataset with **all `.dat` files already converted to `.csv`**.  
> To save space:
> - Each CSV file is further compressed as `.csv.gz`.  
> - The whole dataset is packaged into a `.tar.gz` archive.  

After downloading:
1. Extract the `.tar.gz` archive (double-click on macOS, or run `tar -xvzf file.tar.gz` in terminal).  
2. Inside, you’ll find the folder `converted_csv/` containing `.csv.gz` files.  
3. You can open the compressed CSV files directly in Python using pandas:  

   ```python
   import pandas as pd
   df = pd.read_csv("path/to/file.csv.gz")
   print(df.head())
   
---
## 🧪 Data Analysis Tools

Use the official [ST Sensor SDK](https://www.st.com/en/embedded-software/fp-sns-datalog1.html) to extract and analyze `.dat` files.

Recommended notebooks:

- `nb_hsdatalog_core.ipynb`: Direct analysis in Python
- `nb_hsdatalog_converters.ipynb`: Convert to `.csv` for use in Excel, MATLAB, etc.

---

## 📄 References

- STMicroelectronics STEVAL-STWINKT1B Eval Board: [Product Page](https://www.st.com/en/evaluation-tools/steval-stwinkt1b.html)
- Function Pack for Sensor Data Logging: [FP-SNS-DATALOG1](https://www.st.com/en/embedded-software/fp-sns-datalog1.html)
- Internal guide: `readmeEnglish.txt` (included in this repo)
