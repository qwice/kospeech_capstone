import os
import speech_recognition as sr
import pyaudio
import wave

recognizer = sr.Recognizer()
microphone = sr.Microphone()

# 오디오 캡처 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1  # 모노
RATE = 16000  # 샘플링 레이트 (Hz)
CHUNK = 1024  # 버퍼 크기
RECORD_SECONDS = 3  # 3초간 녹음
OUTPUT_DIR = "/Users/gangjiyeon/Downloads/capstone/kospeech"  # 저장할 디렉토리
OUTPUT_FILENAME = "output.pcm"  
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)  # 전체 경로

# 특정 키워드 리스트
KEYWORDS = ["헤이 우니", "혜윤이"]

# 마이크를 사용하여 음성 입력을 듣고 인식합니다.
with microphone as source:
    print("마이크 대기 중...")
    recognizer.adjust_for_ambient_noise(source)  # 노이즈 조ㅇ
    
    # 키워드가 들어올 떄까지 녹음
    while True:
        print("녹음중!")
        audio = recognizer.listen(source)
        
        # 음성-> 텍스트 변환
        try:
            recognized_text = recognizer.recognize_google(audio, language="ko-KR")
            print("인식된 텍스트:", recognized_text)
            
            for keyword in KEYWORDS:
                if keyword in recognized_text:
                    audio_data = audio.get_raw_data(convert_rate=RATE, convert_width=2)
                    
                    # PCM 데이터를 WAV 파일로 저장
                    with wave.open(OUTPUT_PATH, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(audio_data)
                    
                    print(f"{OUTPUT_PATH}에 오디오를 저장했습니다.")
                    break 
            else:
                continue  # 키워드가 들어올 때까지 녹음 진행
            break  # 키워드가 들어오면 녹음 중지
        except sr.UnknownValueError:
            print("음성 인식 불가")
        except sr.RequestError as e:
            print(f"에러: {e}")
