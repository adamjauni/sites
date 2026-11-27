import json, sys, os
import soundfile as sf
from faster_whisper import WhisperModel

# Multilingual model (works for Arabic). Override with WHISPER_MODEL / WHISPER_LANG, e.g. WHISPER_LANG=en.
MODEL = os.environ.get('WHISPER_MODEL', 'large-v3-turbo')
LANG = os.environ.get('WHISPER_LANG', 'ar') or None
try:
    m = WhisperModel(MODEL, device='cuda', compute_type='float16'); print('cuda', MODEL)
except Exception as e:
    print('cuda failed, using cpu:', e); m = WhisperModel(MODEL, device='cpu', compute_type='int8')
for n in sys.argv[1:]:
    wav = f'tr/{n}.wav'
    try:
        # Load audio ourselves: newer PyAV versions break faster-whisper's built-in file decoding.
        audio, sr = sf.read(wav, dtype='float32')
        assert sr == 16000, f'{wav} must be 16 kHz mono'
        if audio.ndim > 1: audio = audio.mean(axis=1)
        segs, _ = m.transcribe(audio, word_timestamps=True, vad_filter=False, beam_size=5, language=LANG)
        out = [{'s': round(s.start, 2), 'e': round(s.end, 2), 't': s.text, 'w': [[round(w.start, 2), round(w.end, 2), w.word] for w in s.words]} for s in segs]
    except Exception as e:
        print('fail', n, e); raise
    json.dump(out, open(f'tr/{n}.json', 'w'), ensure_ascii=False)
    with open(f'tr/{n}.txt', 'w', encoding='utf-8') as f:
        for s in out: f.write(f"{s['s']:7.2f}-{s['e']:7.2f} {s['t']}\n")
    print('done', n, flush=True)
