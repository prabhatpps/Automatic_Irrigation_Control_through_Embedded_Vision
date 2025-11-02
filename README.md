# Automatic Irrigation Control through Embedded Vision

A sophisticated IoT-based smart irrigation system that leverages **computer vision** and **deep learning** to automate watering decisions based on real-time soil moisture classification. This system uses a **Raspberry Pi 5** with an embedded **EfficientNetV2-L** neural network model to classify soil moisture into four categories (dry, low, medium, high) and intelligently control an irrigation pump.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation Guide](#installation-guide)
- [Model Training](#model-training)
- [Deployment on Raspberry Pi](#deployment-on-raspberry-pi)
- [Automation & Auto-Start Setup](#automation--auto-start-setup)
- [Usage Instructions](#usage-instructions)
- [Troubleshooting](#troubleshooting)
- [Performance Metrics](#performance-metrics)
- [Future Enhancements](#future-enhancements)

---

## Project Overview

Traditional irrigation systems are either manually operated or based on simple timer-based controllers, leading to water waste and inconsistent crop health. This project addresses these challenges by implementing an **autonomous vision-based irrigation controller** that:

1. **Captures soil images** continuously using the Raspberry Pi Camera Module
2. **Processes images** in real-time using a pre-trained deep learning model
3. **Classifies soil moisture levels** with high accuracy
4. **Makes intelligent irrigation decisions** based on majority voting from multiple predictions
5. **Controls the pump** with adaptive watering durations based on soil dryness level
6. **Displays real-time status** on an I2C LCD display
7. **Provides visual feedback** through RGB LED indicators

### Why Computer Vision for Irrigation?

- **Non-invasive**: No need for physical soil moisture sensors that degrade over time
- **Contextual Understanding**: Vision captures plant health indicators alongside soil color
- **Scalable**: Single camera can monitor multiple plants or large areas
- **Cost-effective**: Leverages affordable Raspberry Pi and camera hardware
- **AI-powered**: Deep learning models adapt to varying soil types and lighting conditions

---

## Key Features

✅ **Real-time soil moisture classification** using EfficientNetV2-L (4-class classification)  
✅ **Majority voting mechanism** (5 consecutive image captures) for robust predictions  
✅ **Adaptive pump control** with variable irrigation durations:
   - Dry soil: 10 seconds of irrigation
   - Low moisture: 5 seconds of irrigation
   - Medium/High moisture: No irrigation
   
✅ **I2C LCD Display** showing real-time classification results and system status  
✅ **Dual RGB LED indicators** for visual feedback (Green = Irrigating, Red = Standby)  
✅ **Relay-based pump control** for high-voltage/high-current pump systems  
✅ **GPIO-based control** for flexible hardware integration  
✅ **Error handling & graceful shutdown** with proper resource cleanup  
✅ **Configurable cycle time** (default: 20-second intervals between analyses)  
✅ **Cross-platform model deployment** (trained on GPU, infers on Raspberry Pi)  

---

## System Architecture

### Hardware Block Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Raspberry Pi 5                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Camera     │  │  I2C LCD     │  │  GPIO Pins   │       │
│  │   Module     │  │  Display     │  │  (17,22,27)  │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │               │
│  GPIO:  │             I2C:│            GPIO:|               │
│  CAM/   │           SDA:2 │         17-Pump |               │
│  DISP0  │           SCL:3 │          22-LED |               │
│         │                 │          27-LED |               │
└─────────┼─────────────────┼─────────────────┼───────────────┘
          │                 │                 │
          │                 │                 │
    ┌─────▼──┐         ┌────▼────┐      ┌─────▼─────┐
    │Camera  │         │  16x2   │      │ 3V Relay  │
    │5MP     │         │  LCD    │      │ Module    │
    │        │         │         │      │           │
    └────────┘         └─────────┘      └─────┬─────┘
                                              │
                                         ┌────▼──────────────┐
                                         │ Irrigation Pump   │
                                         │ (via Power Relay) │
                                         └───────────────────┘
```

### Software Architecture

```
Input: Soil Image
         ↓
    Image Preprocessing (224×224 resize, normalization)
         ↓
    EfficientNetV2-L Model Inference
         ↓
    4-Class Output: [dry, low, medium, high]
         ↓
    Majority Voting (5 consecutive predictions)
         ↓
    Decision Logic
         ├─ High/Medium Moisture → No Action
         ├─ Low Moisture → Pump ON (5s)
         └─ Dry Soil → Pump ON (10s)
         ↓
    GPIO Control → Relay → Pump
         ↓
    LCD Display Status Update
         ↓
    LED Indicator Feedback
         ↓
    Cycle Delay (20 seconds)
```

---

## Hardware Requirements

### Essential Components

| Component | Quantity | Purpose | Notes |
|-----------|----------|---------|-------|
| Raspberry Pi 5 | 1 | Main processor | 8GB RAM recommended |
| Pi Camera Module 5MP | 1 | Soil image capture | CSI/DSI ribbon cable required |
| 16×2 I2C LCD Display | 1 | Real-time status display | I2C address: 0x27 (may vary) |
| 3V Relay Module | 1 | Pump control switching | Single-channel or multi-channel |
| Water Pump | 1 | Irrigation actuation | DC 12V or AC recommended |
| Power Supply | 1 | System power | 5V USB-C for Pi, separate supply for pump |
| Jumper Wires | ~20 | Connections | Male-to-female and male-to-male |
| Breadboard | 1 | Prototyping/testing | Optional if using permanent connections |

### Optional Components

- **RGB LED Indicators** (2x): Visual feedback (Green for irrigation, Red for standby)
- **Motion Sensor**: Detect water presence in soil
- **Temperature Sensor**: For adaptive irrigation logic
- **WiFi Dongle**: If Pi 5 WiFi fails (Pi 5 has built-in WiFi)
- **Real-Time Clock (RTC)**: For accurate timestamping without internet

### Pin Configuration Summary

| Component | GPIO Pin | Functionality |
|-----------|----------|---------------|
| Relay Control | GPIO 17 | Pump activation |
| Green LED | GPIO 27 | Irrigation status |
| Red LED | GPIO 22 | Standby status |
| I2C SDA | GPIO 2 | LCD data line |
| I2C SCL | GPIO 3 | LCD clock line |
| Camera | CSI0 | Image capture via ribbon cable |

---

## Software Requirements

### System Dependencies

```bash
# Raspberry Pi OS (Bullseye or later)
- Python 3.9+
- CUDA 12.1 (optional, for GPU acceleration on Pi 5)
- pip package manager

# Required Python Libraries
- torch >= 2.0.0           # PyTorch deep learning framework
- torchvision >= 0.15.0    # Computer vision utilities
- picamera2 >= 0.3.8       # Raspberry Pi camera interface
- pillow >= 9.0.0          # Image processing
- RPLCD >= 1.3.0           # I2C LCD display control
- gpiozero >= 2.0.0        # GPIO pin control
- numpy >= 1.21.0          # Numerical computing
- scikit-learn >= 1.0.0    # For model evaluation metrics (training only)
- matplotlib >= 3.4.0      # Plotting (training only)
- seaborn >= 0.11.0        # Statistical visualization (training only)
```

### System Specifications

- **Minimum Storage**: 8GB (model + OS + dependencies)
- **Recommended Storage**: 32GB+ (for model checkpoints and logs)
- **RAM**: 4GB minimum, 8GB recommended
- **Internet Connection**: Required for initial setup and dependency installation
- **Operating System**: Raspberry Pi OS (Bullseye or later, 64-bit recommended)

---

## Installation Guide

### Step 1: Update and Upgrade System

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Step 2: Install System Dependencies

```bash
# Install Python development tools
sudo apt-get install -y python3-dev python3-pip git

# Install I2C tools for LCD communication
sudo apt-get install -y i2c-tools python3-smbus

# Install GPIO and hardware libraries
sudo apt-get install -y libatlas-base-dev libopenjp2-7 libtiff-libjasper-dev libtiffxx5

# Install additional development packages
sudo apt-get install -y build-essential cmake pkg-config libjasper-dev libharfbuzz0b libwebp6 libtiff5
```

### Step 3: Enable Camera and I2C Interfaces

```bash
# Enable Camera and I2C via raspi-config
sudo raspi-config

# Navigate to: Interfacing Options > Camera (Enable)
# Navigate to: Interfacing Options > I2C (Enable)
# Reboot after enabling
```

### Step 4: Create Project Directory

```bash
mkdir -p ~/irrigation_project
cd ~/irrigation_project
```

### Step 5: Install Python Dependencies

Create a `requirements.txt` file:

```txt
torch==2.0.1
torchvision==0.15.1
pillow==10.0.0
RPLCD==1.3.0
gpiozero==2.0.1
numpy==1.24.3
picamera2==0.3.8
scipy==1.11.0
```

Install dependencies:

```bash
pip3 install --upgrade pip setuptools wheel
pip3 install -r requirements.txt
```

### Step 6: Verify I2C LCD Address

```bash
sudo i2cdetect -y 1
```

This will show connected I2C devices. Look for your LCD address (usually 0x27 or 0x3f).

### Step 7: Test GPIO Setup

```python
from gpiozero import OutputDevice
import time

# Test pump relay (GPIO 17)
pump = OutputDevice(17)
pump.on()
time.sleep(2)
pump.off()

print("Pump control test successful!")
```

---

## Model Training

### Dataset Preparation

Organize your soil moisture image dataset in the following structure:

```
data/
├── train/
│   ├── dry/
│   ├── low/
│   ├── medium/
│   └── high/
├── val/
│   ├── dry/
│   ├── low/
│   ├── medium/
│   └── high/
└── test/
    ├── dry/
    ├── low/
    ├── medium/
    └── high/
```

**Recommended splits**: 70% training, 15% validation, 15% testing

### Training Script

Use the provided `EfficientNet.py` script on a system with GPU support (recommended):

```bash
# On a GPU-equipped machine (not Raspberry Pi)
python3 EfficientNet.py
```

**Training Configuration** (from provided code):
- Model: EfficientNetV2-L with transfer learning
- Batch Size: 8
- Epochs: 25
- Learning Rate: 1e-4 (with ReduceLROnPlateau scheduler)
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Data Augmentation: RandomResizedCrop, RandomFlip, ColorJitter, RandomRotation

**Training Output**:
- `best_soil_model.pth`: Saved best model checkpoint
- Training/validation loss and accuracy curves
- Test set confusion matrix and classification report

### Model Evaluation Metrics

The training script generates:
- **Per-class precision, recall, F1-score**
- **Macro and weighted averages**
- **Overall accuracy**
- **Confusion matrix visualization**

Example output:
```
Test Accuracy: 0.9235
Macro P/R/F1: 0.8934 / 0.8876 / 0.8904
Weighted P/R/F1: 0.9210 / 0.9235 / 0.9222
```

### Transfer to Raspberry Pi

After training, copy the model to Raspberry Pi:

```bash
scp best_soil_model.pth pi@<pi-ip>:~/irrigation_project/
```

---

## Deployment on Raspberry Pi

### Step 1: Copy Deployment Script

Place the `irrigate.py` script in your project directory:

```bash
cp irrigate.py ~/irrigation_project/
```

### Step 2: Hardware Wiring

**Connections to Raspberry Pi GPIO:**

| Device | Pin | Connection |
|--------|-----|------------|
| Relay IN | GPIO 17 | Control signal |
| Green LED | GPIO 27 | Anode (+), GND for cathode |
| Red LED | GPIO 22 | Anode (+), GND for cathode |
| I2C LCD SDA | GPIO 2 | I2C data bus |
| I2C LCD SCL | GPIO 3 | I2C clock bus |
| Camera | CSI0 | 15-pin ribbon cable |

**Power Connections:**
- Relay VCC: 3.3V (Pi 3V3 pin)
- Relay GND: Pi GND
- Pump Power: Separate 12V supply (through relay contacts)
- Pi Camera: 3.3V from Pi

### Step 3: Test Deployment Script

```bash
cd ~/irrigation_project
python3 irrigate.py
```

You should see:
- Pi camera initialization
- LCD display showing "Starting Soil Analysis Loop"
- Real-time image classification output
- LED indicators blinking
- Pump triggering based on soil moisture

**Exit with**: `Ctrl+C` (gracefully shuts down and cleans up GPIO)

### Step 4: Verify LCD Communication

If LCD shows garbage or errors:

```bash
# Find I2C address
sudo i2cdetect -y 1

# Update I2C address in irrigate.py if different
# Change: lcd = CharLCD('PCF8574', 0x27)
# to:     lcd = CharLCD('PCF8574', <your_address>)
```

---

## Automation & Auto-Start Setup

### Method 1: systemd Service (Recommended)

**systemd** is the modern standard for managing services on Linux and provides:
- Automatic restart on failure
- Integration with system logging (`journalctl`)
- Dependency management
- Easy enable/disable

#### Create systemd Service File

```bash
sudo nano /etc/systemd/system/irrigation.service
```

Add the following content:

```ini
[Unit]
Description=Automatic Irrigation Control via Embedded Vision
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/irrigation_project
ExecStart=/usr/bin/python3 /home/pi/irrigation_project/irrigate.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Key Parameters Explained:**
- `Type=simple`: Service runs in foreground (best for continuous monitoring)
- `Restart=always`: Automatically restarts if script crashes
- `RestartSec=10`: Wait 10 seconds before restarting
- `StandardOutput=journal`: Logs go to system journal (viewable with `journalctl`)
- `User=pi`: Runs as 'pi' user (required for GPIO access)

#### Enable and Start Service

```bash
# Reload systemd daemon
sudo systemctl daemon-reload

# Enable service to auto-start on boot
sudo systemctl enable irrigation.service

# Start service immediately
sudo systemctl start irrigation.service

# Check service status
sudo systemctl status irrigation.service

# View logs (last 50 lines)
sudo journalctl -u irrigation.service -n 50

# View live logs (real-time)
sudo journalctl -u irrigation.service -f
```

#### Service Management Commands

```bash
# Stop the service
sudo systemctl stop irrigation.service

# Restart the service
sudo systemctl restart irrigation.service

# Disable auto-start (but keep installed)
sudo systemctl disable irrigation.service

# View full service status and recent logs
systemctl status irrigation.service

# Check if service is running
systemctl is-active irrigation.service
```

### Method 2: crontab (Alternative)

If you prefer cron (simpler but less robust):

```bash
# Edit crontab
crontab -e

# Add this line to run at boot (after 60 seconds to allow system startup):
@reboot sleep 60 && /usr/bin/python3 /home/pi/irrigation_project/irrigate.py > /home/pi/irrigation_project/irrigate.log 2>&1 &
```

**Drawbacks of crontab for this use case:**
- No automatic restart if script crashes
- Manual logging required
- Less suitable for long-running processes

### Method 3: rc.local (Legacy)

Not recommended for modern systems, but possible:

```bash
sudo nano /etc/rc.local
```

Add before `exit 0`:
```bash
/usr/bin/python3 /home/pi/irrigation_project/irrigate.py &
```

---

## Usage Instructions

### Initial Setup Checklist

- [ ] Raspberry Pi 5 with latest OS and updates installed
- [ ] All dependencies installed from `requirements.txt`
- [ ] Camera module connected and enabled
- [ ] I2C LCD display connected and I2C enabled
- [ ] GPIO relay module wired to GPIO 17
- [ ] LED indicators wired to GPIO 27 (green) and GPIO 22 (red)
- [ ] Water pump connected through relay contacts
- [ ] Power supplies properly connected (5V for Pi, 12V for pump)
- [ ] Model file `best_soil_model.pth` copied to project directory
- [ ] `irrigate.py` deployment script in project directory

### Runtime Operation

**Manual Start:**
```bash
cd ~/irrigation_project
python3 irrigate.py
```

**Auto-Start on Boot:**
Simply reboot your Raspberry Pi; the systemd service will automatically start the irrigation control system:
```bash
sudo reboot
```

### Expected LCD Output

During normal operation, you'll see the following on the LCD display:

```
Line 1: Img1: dry        (or: low, medium, high)
Line 2: Majority: dry    (or: low, medium, high)
---
Line 1: Majority: dry
Line 2: Count: 3/5
---
Line 1: Soil Dry
Line 2: Pump ON 10s
---
Line 1: Moisture OK
Line 2: Pump OFF
```

### LED Indicator Meanings

| Green LED | Red LED | Status |
|-----------|---------|--------|
| ON | OFF | Pump actively irrigating |
| OFF | ON | System standby (monitoring) |
| Both OFF | Both OFF | System startup or error |

### Cycle Timing

**Default Cycle Interval: 20 seconds**

1. **Seconds 0-5**: Capture 5 consecutive soil images
2. **Seconds 5-6**: Calculate majority class
3. **Seconds 6-16**: Execute pump control (if needed)
4. **Seconds 16-20**: Wait before next cycle

**Total cycle time: 20 seconds**

---

## Troubleshooting

### Camera Not Working

**Problem**: `RuntimeError: Cannot open camera` or camera initialization fails

**Solutions**:
```bash
# Check if camera is detected
vcgencmd get_camera

# Should output: supported=1 detected=1

# If not detected, enable camera interface:
sudo raspi-config
# Interfacing Options > Camera > Enable

# Verify camera with libcamera
libcamera-hello

# Check CSI port connection (reseat ribbon cable if needed)
```

### I2C LCD Display Not Responding

**Problem**: `IOError: [Errno 121] Remote I/O error` or LCD shows garbage

**Solutions**:
```bash
# Verify I2C address
sudo i2cdetect -y 1

# If address is different from 0x27, update irrigate.py:
# Line: lcd = CharLCD('PCF8574', 0x27)
# Change to your address (e.g., 0x3f)

# Enable I2C:
sudo raspi-config
# Interfacing Options > I2C > Enable

# Test I2C communication:
python3 -c "from RPLCD.i2c import CharLCD; lcd = CharLCD('PCF8574', 0x27); lcd.write_string('Test')"
```

### Pump Not Activating

**Problem**: Relay doesn't click or pump doesn't turn on

**Solutions**:
```bash
# Test GPIO 17 control manually
python3 << 'EOF'
from gpiozero import OutputDevice
import time
relay = OutputDevice(17)
print("Turning relay ON...")
relay.on()
time.sleep(2)
relay.off()
print("Relay test complete")
EOF

# Check GPIO permissions
groups pi  # Should include 'gpio' group
# If not:
sudo usermod -a -G gpio pi

# Verify relay module jumper settings (for 3.3V logic)
# Check relay GND connection to Pi GND

# Test with higher current pump (if relay is working):
# Verify pump power supply voltage
```

### Model Loading Fails

**Problem**: `RuntimeError: Attempting to deserialize object on a CUDA device`

**Solutions**:
```bash
# In irrigate.py, ensure device detection is correct:
# Change from:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# To:
device = torch.device('cpu')  # Force CPU inference on Pi

# Verify model file exists:
ls -lh best_soil_model.pth

# Re-copy model if corrupted:
scp best_soil_model.pth pi@<pi-ip>:~/irrigation_project/
```

### High CPU/Memory Usage

**Problem**: Script crashes or becomes very slow

**Solutions**:
```bash
# Monitor resource usage during runtime
top

# Reduce batch processing (currently processes 5 images):
# Edit irrigate.py, reduce loop from 5 to 3:
# for i in range(3):  # Changed from 5

# Increase cycle delay to reduce frequency:
# Change: time.sleep(20)  # At end of main loop
# To:     time.sleep(60)  # Check every 60 seconds instead
```

### Service Not Starting on Boot

**Problem**: `systemctl status` shows service in failed state

**Solutions**:
```bash
# Check systemd logs for detailed error
journalctl -u irrigation.service -n 100

# Verify service file syntax
systemd-analyze verify irrigation.service

# Ensure absolute paths in service file:
# ✗ Wrong: ExecStart=/home/pi/irrigate.py
# ✓ Correct: ExecStart=/usr/bin/python3 /home/pi/irrigation_project/irrigate.py

# Check file permissions:
sudo chmod 644 /etc/systemd/system/irrigation.service

# Reload and try again:
sudo systemctl daemon-reload
sudo systemctl enable irrigation.service
sudo systemctl start irrigation.service
```

### GPIO Permission Denied

**Problem**: `PermissionError: [Errno 13] Permission denied`

**Solution**:
```bash
# Add 'pi' user to GPIO group
sudo usermod -a -G gpio pi

# Log out and log back in (or reboot) for group membership to take effect
exit
# Then SSH back in
```

### Intermittent Camera Captures or Crashes

**Problem**: Script crashes periodically or camera fails unexpectedly

**Solutions**:
```bash
# Enable verbose logging in irrigate.py
# Add after imports:
import logging
logging.basicConfig(level=logging.DEBUG)

# Check for thermal throttling:
vcgencmd measure_temp

# Ensure adequate power supply (5V, ≥3A recommended)
vcgencmd measure_volts

# If voltage drops below 4.75V, supply is insufficient
```

---

## Performance Metrics

### Model Performance

Based on the training configuration provided:

| Metric | Expected Value            |
|--------|---------------------------|
| Test Accuracy | ~92-95%                   |
| Macro Average F1-Score | ~0.89-0.91                |
| Weighted Average F1-Score | ~0.92-0.94                |
| Inference Time per Image | ~800-1200ms (Pi 5, CPU)   |
| Model Size | ~470MB (EfficientNetV2-L) |

### System Performance

| Parameter | Typical Value |
|-----------|---------------|
| Image Capture Latency | 50-150ms |
| Preprocessing Time | 20-50ms |
| Model Inference (1 image) | 800-1200ms |
| 5-image majority voting cycle | 4-7 seconds |
| Total cycle time (with pump) | 6-25 seconds |
| CPU Usage | 40-60% during inference |
| Memory Usage | 350-450MB |
| Power Consumption | 4-6W (system), 20-30W (pump on) |

### Optimization Tips

1. **Quantization**: Convert model to FP16 or INT8 for faster inference
2. **Model Pruning**: Remove less important weights to reduce size
3. **Batch Processing**: Process multiple images in parallel if GPU available
4. **Image Optimization**: Downscale images before feeding to model (currently 224×224)

---

## Future Enhancements

### Planned Features

- [ ] **WiFi/Cloud Integration**: Send data to remote server for monitoring
- [ ] **Web Dashboard**: Real-time visualization of soil moisture trends
- [ ] **Machine Learning**: Adaptive thresholds based on historical data
- [ ] **Multi-Zone Irrigation**: Control multiple pumps for different areas
- [ ] **Weather API Integration**: Adjust watering based on rainfall predictions
- [ ] **Mobile App**: Remote control and notifications
- [ ] **Edge Analytics**: Detect anomalies (pest damage, disease indicators)
- [ ] **Data Logging**: Store classification history to CSV/database
- [ ] **Multi-camera Support**: Cover larger areas with multiple Pi cameras
- [ ] **Thermal Imaging**: Additional sensor for plant stress detection

### Model Improvements

- [ ] Fine-tuning on plant-specific soil types
- [ ] Adding seasonal variation handling
- [ ] Ensemble methods for robustness
- [ ] Real-time model updates via federated learning
- [ ] Lightweight model variants (MobileNet, SqueezeNet) for faster inference

### Hardware Enhancements

- [ ] Add soil moisture sensor for ground truth validation
- [ ] Temperature and humidity sensors for better decision making
- [ ] Flow meter to measure actual water usage
- [ ] Solar panel for off-grid operation
- [ ] Backup battery system for reliability

---

## Project Structure

```
irrigation_project/
├── README.md                    # This file
├── irrigate.py                  # Main deployment script (runs on Pi)
├── EfficientNet.py              # Model training script (runs on GPU machine)
├── best_soil_model.pth          # Trained model weights
├── requirements.txt             # Python dependencies
├── data/                        # Training dataset (optional, not on Pi)
│   ├── train/
│   ├── val/
│   └── test/
├── logs/                        # System logs
├── configs/                     # Configuration files
│   └── irrigation_service       # systemd service file
└── docs/                        # Documentation
    ├── INSTALLATION.md
    ├── TROUBLESHOOTING.md
    └── API.md
```

---

## Support & Contact

For issues, questions, or suggestions:

- **GitHub Issues**: [Create an issue on GitHub](https://github.com/prabhatpps/Automatic_Irrigation_Control_through_Embedded_Vision/issues)
- **Email**: [prpandey192@gmail.com](mailto:prpandey192@gmail.com)
- **Documentation**: See [`/docs`](/docs) directory

---

## Acknowledgments

- **EfficientNet Model**: Tan & Le, 2021 (EfficientNetV2: Smaller Models and Faster Training)
- **PyTorch & Torchvision**: Meta AI Research
- **Raspberry Pi Foundation**: Hardware and community support
- **Open Source Community**: gpiozero, picamera2, RPLCD maintainers

---

## Disclaimer

⚠️ **Safety Notice**: This system involves electrical components, water, and automated equipment. Ensure proper safety measures:

- Use waterproof enclosures for electronics
- Implement emergency stop mechanisms
- Test all electrical connections before deployment
- Never operate with wet hands
- Ensure proper circuit protection and fuses
- Follow local electrical codes and regulations
- Supervise system during initial testing phases

**The authors are not responsible for any damage, injury, or loss resulting from the use of this project.**

---

**Last Updated**: November 2, 2025  
**Version**: 1.0  
**Status**: Active Development