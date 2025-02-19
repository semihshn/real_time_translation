import speech_recognition as sr
from transformers import pipeline
import keyboard  # Klavye tuşlarını kontrol etmek için

# Türkçe'den İngilizce'ye çeviri için pipeline'ı yükleyin
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tr-en")

# SpeechRecognition için tanıyıcıyı oluşturun
r = sr.Recognizer()

with sr.Microphone() as source:
    print("Ortam gürültüsü ayarlanıyor, lütfen bekleyin...")
    r.adjust_for_ambient_noise(source, duration=1)
    print("Artık konuşabilirsiniz... (Çıkmak için 'q' tuşuna basın)")

    while True:
        # Kullanıcı 'q' tuşuna bastıysa döngüyü kır
        if keyboard.is_pressed('q'):
            print("Çıkış yapılıyor...")
            break

        try:
            print("\nDinleniyor...")
            # Sesi dinler; phrase_time_limit ile her bir dinleme süresini sınırlandırabilirsiniz (ör. 5 saniye)
            audio = r.listen(source, phrase_time_limit=99)
            # Google Speech Recognition kullanarak sesi metne çevirin
            text = r.recognize_google(audio, language="tr-TR")
            print("Algılanan Metin:", text)

            # Çeviri işlemi
            translation = translator(text)[0]['translation_text']
            print("İngilizce Çeviri:", translation)
        except sr.UnknownValueError:
            print("Sesi anlayamadım. Lütfen tekrar deneyin.")
        except sr.RequestError as e:
            print("Servise bağlanırken hata oluştu; {0}".format(e))
        except Exception as ex:
            # Diğer beklenmedik hataları yakalayarak uygulamanın kapanmasını önleyin
            print("Bir hata oluştu:", ex)
