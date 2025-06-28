from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.utils import get_color_from_hex
from kivy.animation import Animation
# ───────────── Imports ─────────────
import os
import numpy as np
import random
import pickle
import string
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.utils import to_categorical

# ───────────── Constants ─────────────
SEQ_LENGTH = 10
EPOCHS = 60
BATCH_SIZE = 128
MODEL_PATH = "model.h5"
MAPPING_PATH = "char_to_index.pkl"
DATA_PATH = "english_words.txt"

Window.clearcolor = get_color_from_hex("#1e1e2e")

# ───────────── Dataset + Training ─────────────
def load_or_create_dataset():
    base_chars = (
        string.ascii_letters +
        string.digits +
        "!@#$%^&*()_+-=[]{}|;:',.<>/?`~\\\"="
    )

    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            for _ in range(30000):
                pwd = ''.join(random.choices(base_chars, k=random.randint(8, 16)))
                f.write(pwd + "\n")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip().lower() for line in f if line.strip()]
        text = ''.join(lines)

    chars = sorted(list(set(text)))
    char_to_index = {ch: i for i, ch in enumerate(chars)}
    index_to_char = {i: ch for ch, i in char_to_index.items()}

    X, y = [], []
    for i in range(0, len(text) - SEQ_LENGTH):
        seq = text[i:i + SEQ_LENGTH]
        label = text[i + SEQ_LENGTH]
        X.append([char_to_index[c] for c in seq])
        y.append(char_to_index[label])

    X = np.reshape(X, (len(X), SEQ_LENGTH, 1)) / float(len(chars))
    y = to_categorical(y)

    return X, y, char_to_index, index_to_char, chars

def train_model(X, y, chars):
    model = Sequential()
    model.add(LSTM(256, input_shape=(SEQ_LENGTH, 1)))
    model.add(Dense(len(chars), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    model.save(MODEL_PATH)
    print("✅ Model trained and saved.")
    return model

def load_model_and_mappings():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPING_PATH):
        print("⚙️ Training model from scratch...")
        X, y, char_to_index, index_to_char, chars = load_or_create_dataset()
        model = train_model(X, y, chars)
        with open(MAPPING_PATH, "wb") as f:
            pickle.dump((char_to_index, index_to_char), f)
    else:
        model = load_model(MODEL_PATH)
        with open(MAPPING_PATH, "rb") as f:
            char_to_index, index_to_char = pickle.load(f)
        chars = list(char_to_index.keys())

    return model, char_to_index, index_to_char, chars

# ───────────── Password Generator ─────────────
def generate_password(length=12):
    model, char_to_index, index_to_char, chars = load_model_and_mappings()
    seed = ''.join(random.choices(chars, k=SEQ_LENGTH))
    result = seed

    for _ in range(length - SEQ_LENGTH):
        x = np.reshape([char_to_index[c] for c in seed], (1, SEQ_LENGTH, 1)) / float(len(chars))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        char = index_to_char[index]
        result += char
        seed = result[-SEQ_LENGTH:]

    return result[:length]

# ───────────── Kivy UI ─────────────
class StylishPasswordApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=20, **kwargs)
        self.length = 12

        self.header = Label(
            text="[b]ML-Based Password Generator[/b]",
            markup=True,
            font_size=28,
            size_hint=(1, 0.2),
            color=get_color_from_hex("#cdd6f4")
        )
        self.add_widget(self.header)

        self.output = TextInput(
            text="",
            font_size=20,
            size_hint=(1, 0.3),
            readonly=True,
            background_color=get_color_from_hex("#313244"),
            foreground_color=get_color_from_hex("#f5e0dc"),
            cursor_blink=False,
            halign='center'
        )
        self.add_widget(self.output)

        self.slider_label = Label(
            text=f"Password Length: {self.length}",
            font_size=18,
            color=get_color_from_hex("#a6adc8")
        )
        self.add_widget(self.slider_label)

        self.slider = Slider(min=8, max=32, value=self.length, step=1)
        self.slider.bind(value=self.on_slider_value_change)
        self.add_widget(self.slider)

        self.generate_btn = Button(
            text="Generate Password",
            font_size=20,
            size_hint=(1, 0.3),
            background_color=get_color_from_hex("#0AC6A3"),
            color=get_color_from_hex("#ededf5"),
            bold=True
        )
        self.generate_btn.bind(on_press=self.animate_button)
        self.add_widget(self.generate_btn)

    def animate_button(self, instance):
        original_color = instance.background_color
        anim = (
            Animation(background_color=(0.2, 1, 0.7, 1), duration=0.1) +
            Animation(background_color=original_color, duration=0.2)
        )
        anim.bind(on_complete=lambda *_: self.generate_password(instance))
        anim.start(instance)


    def on_slider_value_change(self, instance, value):
        self.length = int(value)
        self.slider_label.text = f"Password Length: {self.length}"

    def generate_password(self, instance):
        password = generate_password(self.length)
        self.output.text = password

class MLPasswordApp(App):
    def build(self):
        self.title = "ML Password Generator"
        return StylishPasswordApp()

if __name__ == "__main__":
    MLPasswordApp().run()
