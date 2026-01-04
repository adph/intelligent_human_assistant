import json, os, time
import pyautogui

PATH = os.environ.get("COORDINATES_JSON_PATH", "coordinates.json")

def load():
    if os.path.exists(PATH):
        with open(PATH, "r") as f:
            return json.load(f)
    return {}

def save(d):
    with open(PATH, "w") as f:
        json.dump(d, f, indent=2)

def main():
    coords = load()
    print(f"Saving to: {os.path.abspath(PATH)}")
    while True:
        name = input("\nElement name (blank to quit): ").strip()
        if not name:
            break
        print("Hover mouse over the target in 3 seconds...")
        time.sleep(3)
        x, y = pyautogui.position()
        coords[name] = {"x": int(x), "y": int(y)}
        save(coords)
        print(f"Recorded {name}: x={x}, y={y}")

if __name__ == "__main__":
    main()
