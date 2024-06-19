import torch
import requests
import torchaudio
import numpy as np
from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)

model = torch.load('/Users/gangjiyeon/Downloads/capstone/kospeech/outputs/2024-06-19/00-18-58/model.pt', map_location=torch.device('cpu'))
if isinstance(model, torch.nn.DataParallel):
    model = model.module
model.eval()

vocab = KsponSpeechVocabulary('/Users/gangjiyeon/Downloads/capstone/kospeech/data/vocab/kspon_sentencepiece.csv')

def parse_audio(audio_path: str, sample_rate: int = 16000, num_channels: int = 1) -> torch.Tensor:
    waveform = torch.from_numpy(np.fromfile(audio_path, dtype=np.int16).reshape(-1, num_channels).T).float()
    
    # Normalize waveform to the range [-1, 1]
    waveform /= 32768.0
    
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=waveform,
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)

def pcm_to_text(file_path: str, sample_rate: int = 16000, num_channels: int = 1) -> str:
    feature = parse_audio(file_path, sample_rate, num_channels)
    input_length = torch.LongTensor([len(feature)])

    with torch.no_grad():
        if isinstance(model, ListenAttendSpell):
            model.encoder.device = torch.device('cpu')
            model.decoder.device = torch.device('cpu')
            y_hats = model.recognize(feature.unsqueeze(0), input_length)
        elif isinstance(model, DeepSpeech2):
            model.device = torch.device('cpu')
            y_hats = model.recognize(feature.unsqueeze(0), input_length)
        elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
            y_hats = model.greedy_search(feature.unsqueeze(0), input_length)
        else:
            raise ValueError("Unsupported model type")

    sentence = vocab.label_to_string(y_hats.cpu().detach().numpy()[0])
    return sentence

def send_text_to_server(text):
    url = "http://localhost:4000/audio/text"
    data = {"text": text}
    response = requests.post(url, json=data)

file_path = "/Users/gangjiyeon/Downloads/capstone/kospeech/output.pcm"
text = pcm_to_text(file_path, sample_rate=16000, num_channels=1)
print(f"서버로 전송한 텍스트: {text}")
send_text_to_server(text)
