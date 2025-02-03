from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('input_data/08fd33_0.mp4', save=True)
print(results[0])

print("/n Boxes:")
for box in results[0].boxes:
    print(box)