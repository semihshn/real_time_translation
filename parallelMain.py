import os
import speech_recognition as sr
from transformers import pipeline
from pynput import keyboard
import threading
import queue

os.environ["TOKENIZERS_PARALLELISM"] = "false"

translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="tur_Latn",
    tgt_lang="eng_Latn"
)

r = sr.Recognizer()
audio_queue = queue.Queue()

exit_flag = False

def on_press(key):
    global exit_flag
    try:
        if key.char == 'q':
            print("Çıkış yapılıyor...")
            exit_flag = True
            return False
    except AttributeError:
        pass

keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()

def microphone_listener():
    with sr.Microphone() as source:
        print("Ortam gürültüsü ayarlanıyor, lütfen bekleyin...")
        r.adjust_for_ambient_noise(source, duration=1)
        r.pause_threshold = 1.0
        r.non_speaking_duration = 1.0
        print("Artık konuşabilirsiniz... (Çıkmak için 'q' tuşuna basın)")

        while not exit_flag:
            try:
                audio = r.listen(source)
                audio_queue.put(audio)
            except Exception as e:
                print("Dinleme hatası:", e)

def processing_thread():
    while not exit_flag or not audio_queue.empty():
        try:
            audio = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            text = r.recognize_google(audio, language="tr-TR")
            print("Algılanan Metin (Türkçe):", text)
            translation = translator(text)[0]['translation_text']
            print("İngilizce Çeviri:", translation)
        except sr.UnknownValueError:
            print("Sesi anlayamadım. Lütfen tekrar deneyin.")
        except sr.RequestError as e:
            print("Servise bağlanırken hata oluştu; {0}".format(e))
        except Exception as ex:
            print("İşleme sırasında bir hata oluştu:", ex)
        finally:
            audio_queue.task_done()

listener_thread = threading.Thread(target=microphone_listener)
processor_thread = threading.Thread(target=processing_thread)

listener_thread.start()
processor_thread.start()

listener_thread.join()
processor_thread.join()

keyboard_listener.join()
