"""
Viterbox - Vietnamese Text-to-Speech
"""
from .tts import Viterbox, TTSConds, postprocess_audio, normalize_loudness

__version__ = "1.0.0"
__all__ = ["Viterbox", "TTSConds", "postprocess_audio", "normalize_loudness"]
