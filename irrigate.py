import time
import torch
from collections import Counter
from torchvision import transforms
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from picamera2 import Picamera2
from PIL import Image
from RPLCD.i2c import CharLCD
from gpiozero import OutputDevice

# ---- Model/Preprocessing Setup ----
NUM_CLASSES = 4
model_path = "best_soil_model.pth"
class_idx = {0: 'dry', 1: 'high', 2: 'low', 3: 'medium'}

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# ---- Camera Setup ---
picam2 = Picamera2()
camera_config = picam2.create_still_configuration()
picam2.configure(camera_config)
picam2.start()

# ---- I2C LCD Setup (adjust address as needed) ----
lcd = CharLCD('PCF8574', 0x27)

def lcd_print(msg, line=0):
    lcd.clear()
    lcd.cursor_pos = (line, 0)
    lcd.write_string(msg[:16])  # 16 chars max per line

# ---- GPIO Setup using gpiozero ----
PUMP_PIN = 17  # GPIO pin for pump control
LED_GREEN = 27
LED_RED = 22
pump = OutputDevice(PUMP_PIN, active_high=True, initial_value=False)
led_on = OutputDevice(LED_GREEN, active_high=True, initial_value=False)
led_off = OutputDevice(LED_RED, active_high=True, initial_value=False)

def control_pump(duration):
    lcd_print(f'Pump ON: {duration}s', 1)
    pump.on()
    led_on.on()
    led_off.off()
    time.sleep(duration)
    pump.off()
    led_on.off()
    led_off.on()
    lcd_print('Pump OFF', 1)

def classify_img(image_pil):
    img_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = outputs.max(1)
    return class_idx[pred.item()]

try:
    lcd_print("Starting Soil", 0)
    lcd_print("Analysis Loop", 1)
    while True:
        led_on.off()
        led_off.on()
        results = []
        for i in range(5):
            arr = picam2.capture_array()
            pil_img = Image.fromarray(arr)
            cls = classify_img(pil_img)
            results.append(cls)
            lcd_print(f'Img{i+1}: {cls}', 0)
            time.sleep(1)

        counted = Counter(results)
        majority, count = counted.most_common(1)[0]
        lcd_print(f'Majority: {majority}', 0)
        lcd_print(f'Count:{count}/5', 1)

        if majority in ['high', 'medium']:
            lcd_print("Moisture OK", 0)
            lcd_print("Pump OFF", 1)
        elif majority == 'low':
            lcd_print("Moisture Low", 0)
            lcd_print("Pump ON 5s", 1)
            control_pump(5)
        elif majority == 'dry':
            lcd_print("Soil Dry", 0)
            lcd_print("Pump ON 10s", 1)
            control_pump(10)
        else:
            lcd_print("Unknown Class", 0)
            lcd_print("No action", 1)

        time.sleep(5)

except KeyboardInterrupt:
    lcd.clear()
    lcd_print("Exiting...", 0)
    pump.off()
    led_on.off()
    led_off.on()
    picam2.close()

except Exception as e:
    lcd.clear()
    lcd_print("Error occurred", 0)
    lcd_print(str(e)[:16], 1)
    pump.off()
    led_on.off()
    led_off.on()
    picam2.close()

